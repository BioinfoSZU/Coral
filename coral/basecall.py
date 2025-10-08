import os
import math
import shutil
import copy
import pod5
import torch
from glob import glob
from ont_fast5_api.fast5_interface import get_fast5_file
from multiprocessing import Process, set_start_method, Value
from argparse import ArgumentParser
from pathlib import Path
from .util import default_basecall_config, MODEL_DIR
from .download import check_and_download
from .caller import Basecaller


def collect_all_read_ids(input_path, use_pod5=False):
    print(f"[INFO] Collecting read IDs from {'pod5' if use_pod5 else 'fast5'} input...")
    ret = []
    if not use_pod5:
        fast5_paths = [Path(x) for x in glob(input_path + "/" + "**/*.fast5", recursive=True)]
        for filename in fast5_paths:
            with get_fast5_file(filename, 'r') as f5_fh:
                ret.extend(f5_fh.get_read_ids())
    else:
        pod5_reader = pod5.DatasetReader(input_path)
        for record in pod5_reader.reads():
            ret.append(record.read_id)
    return ret


def add_arguments(parser: ArgumentParser):
    parser.add_argument('--input', type=str, default=None, required=True, help="Directory containing fast5 files or Single pod5 file (default: %(default)s)")
    parser.add_argument('--output', type=str, default=None, required=True, help="Output directory (default: %(default)s)")
    parser.add_argument('--kit', choices=['RNA002', 'RNA004'], default=None, required=True, help="RNA002 or RNA004 sequencing kit (default: %(default)s)")
    parser.add_argument('--fast', action='store_true', default=False, help="Use FAST mode that outputs k consecutive bases per step (default: %(default)s)")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device id (default: %(default)s)")
    parser.add_argument('--gpus', type=str, default=None, help="Comma-separated GPU device ids for multi-gpu basecalling, e.g. 0,1,2 (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=500, help="Larger batch size will use more GPU memory (default: %(default)s)")
    parser.add_argument('--beam-size', type=int, default=None, help="Beam size (default: %(default)s)")
    parser.add_argument('--prefix', type=str, default="coral", help="Filename prefix of basecaller output (default: %(default)s)")
    parser.add_argument('--seed', type=int, default=40, help="Seed for random number generators (default: %(default)s)")
    parser.add_argument('--no-deterministic', action='store_true', default=False, help="Disable CUDNN deterministic algorithm (default: %(default)s)")
    parser.add_argument('--parse-fast5-meta', action='store_true', default=False, help="Parse multi-fast5 meta data (default: %(default)s)")
    parser.add_argument('--reads-file', type=str, default=None, help="Basecalling solely on the reads listed in file, with one ID per line (default: %(default)s)")
    parser.add_argument('--keep-split-reads', action='store_true', default=False, help="Keep temporary split read files (default: %(default)s)")


def _setup_torch_device_and_seed(gpu: int, seed: int, no_deterministic: bool):
    torch.set_default_device(f'cuda:{gpu}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    if not no_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_single(args):
    _setup_torch_device_and_seed(args.gpu, args.seed, args.no_deterministic)

    select_model = args.kit + ("_FAST" if args.fast else "")
    check_and_download(select_model)

    use_pod5 = os.path.isfile(args.input) and args.input.endswith(".pod5")

    caller = Basecaller(
        model_name=select_model,
        fast5_dir=args.input,
        output_dir=args.output,
        gpu=args.gpu,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        output_name=args.prefix,
        use_pod5=use_pod5,
        parse_fast5_meta=args.parse_fast5_meta,
        reads_file=args.reads_file,
        verbose=True,
    )

    caller.run()
    caller.clear()
    print(f"Done")


def _split_reads_file(reads, n_parts, out_dir):
    total = len(reads)
    chunk_size = math.ceil(total / n_parts)
    split_paths = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_parts):
        start, end = i * chunk_size, min((i + 1) * chunk_size, total)
        chunk = reads[start:end]
        split_path = out_dir / f"reads_part_{i}.txt"
        with open(split_path, 'w') as f:
            f.write("\n".join(chunk) + "\n")
        split_paths.append(str(split_path))
    return split_paths


def run_multi(args):
    select_model = args.kit + ("_FAST" if args.fast else "")
    check_and_download(select_model)

    gpu_list = [int(x.strip()) for x in args.gpus.split(',') if x.strip()]
    print(f"[INFO] Running on GPUs: {gpu_list}")

    use_pod5 = os.path.isfile(args.input) and args.input.endswith(".pod5")

    all_reads = None
    if not args.reads_file:
        all_reads = collect_all_read_ids(args.input, use_pod5)
    else:
        with open(args.reads_file) as f:
            all_reads = [line.strip() for line in f]

    split_dir = Path(args.output) / "reads_split"
    split_paths = _split_reads_file(all_reads, len(gpu_list), split_dir)

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Spawn workers
    processes = []
    output_prefixes = []
    for pid, (gpu_idx, split_path) in enumerate(zip(gpu_list, split_paths)):
        worker_args = copy.deepcopy(args)
        worker_args.gpu = gpu_idx
        worker_args.reads_file = split_path
        worker_args.prefix = f"{args.prefix}_gpu{gpu_idx}"
        output_prefixes.append(worker_args.prefix)

        p = Process(target=_worker_entrypoint, args=(worker_args, pid))
        p.start()
        processes.append((p, gpu_idx))

    for p, gpu_idx in processes:
        p.join()
        print(f"[INFO] Worker on GPU {gpu_idx} exited with {p.exitcode}")

    if not args.keep_split_reads:
        shutil.rmtree(split_dir, ignore_errors=True)

    with open(os.path.join(args.output, args.prefix + ".fasta"), "w") as outfile:
        for prefix in output_prefixes:
            with open(os.path.join(args.output, prefix + ".fasta"), "r") as infile:
                shutil.copyfileobj(infile, outfile)
            os.remove(os.path.join(args.output, prefix + ".fasta"))

    with open(os.path.join(args.output, args.prefix + "_fail_reads.txt"), "w") as outfile:
        for prefix in output_prefixes:
            with open(os.path.join(args.output, prefix + "_fail_reads.txt"), "r") as infile:
                shutil.copyfileobj(infile, outfile)
            os.remove(os.path.join(args.output, prefix + "_fail_reads.txt"))

    print('Done')


def _worker_entrypoint(args, pid):
    try:
        _setup_torch_device_and_seed(args.gpu, args.seed, args.no_deterministic)

        select_model = args.kit + ("_FAST" if args.fast else "")
        use_pod5 = os.path.isfile(args.input) and args.input.endswith(".pod5")

        caller = Basecaller(
            model_name=select_model,
            fast5_dir=args.input,
            output_dir=args.output,
            gpu=args.gpu,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            output_name=args.prefix,
            use_pod5=use_pod5,
            parse_fast5_meta=args.parse_fast5_meta,
            reads_file=args.reads_file,
            verbose=True if pid == 0 else False,  # Show model-info only in the first process
        )

        caller.run()
        caller.clear()
        print(f"[GPU {args.gpu}] Finished successfully")
    except Exception as e:
        import traceback
        print(f"[GPU {args.gpu}] ERROR: {e}")
        traceback.print_exc()
        raise


def run(args):
    if args.gpus:
        run_multi(args)
    else:
        run_single(args)
