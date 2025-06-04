import os
import torch
from argparse import ArgumentParser
from pathlib import Path
from .util import default_basecall_config, MODEL_DIR
from .download import check_and_download
from .caller import Basecaller


def add_arguments(parser: ArgumentParser):
    parser.add_argument('--input', type=str, default=None, required=True, help="Directory containing fast5 files or Single pod5 file (default: %(default)s)")
    parser.add_argument('--output', type=str, default=None, required=True, help="Output directory (default: %(default)s)")
    parser.add_argument('--kit', choices=['RNA002', 'RNA004'], default=None, required=True, help="RNA002 or RNA004 sequencing kit (default: %(default)s)")
    parser.add_argument('--fast', action='store_true', default=False, help="Use FAST mode that outputs k consecutive bases per step (default: %(default)s)")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device id (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=500, help="Larger batch size will use more GPU memory (default: %(default)s)")
    parser.add_argument('--beam-size', type=int, default=None, help="Beam size (default: %(default)s)")
    parser.add_argument('--prefix', type=str, default="coral", help="Filename prefix of basecaller output (default: %(default)s)")
    parser.add_argument('--seed', type=int, default=40, help="Seed for random number generators (default: %(default)s)")
    parser.add_argument('--no-deterministic', action='store_true', default=False, help="Disable CUDNN deterministic algorithm (default: %(default)s)")
    parser.add_argument('--parse-fast5-meta', action='store_true', default=False, help="Parse multi-fast5 meta data (default: %(default)s)")
    parser.add_argument('--reads-file', type=str, default=None, help="Basecalling solely on the reads listed in file, with one ID per line (default: %(default)s)")

def run(args):
    torch.set_default_device('cuda:{}'.format(args.gpu))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    if not args.no_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    select_model = args.kit
    if args.fast:
        select_model = select_model + "_FAST"
    check_and_download(select_model)

    use_pod5 = False
    if os.path.isfile(args.input) and os.path.basename(args.input).endswith(".pod5"):
        use_pod5 = True

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
    print('Done')
