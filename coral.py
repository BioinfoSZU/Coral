import os
import sys
import argparse
import numpy as np
import torch
import time
import torch.backends.cudnn as cudnn
from util import trim, normalisation
from model import CoralModel, decoding_backtracking
from tqdm import tqdm
from glob import glob
from pathlib import Path
from Bio import SeqIO
from ont_fast5_api.fast5_interface import get_fast5_file
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process, Event


def fast5_process(fast5_dir, chunk_queue, end_event, args):
    torch.set_num_threads(1)
    fast5_paths = [Path(x) for x in glob(fast5_dir + "/" + "**/*.fast5", recursive=True)]
    if len(fast5_paths) <= 0:
        raise RuntimeError('There are no fast5 in directory {}'.format(fast5_dir))

    read_id_filter = None
    if args.read_id_path is not None:
        read_id_filter = set()
        with open(args.read_id_path, 'r') as f:
            for line in f:
                read_id_filter.add(line.strip())
        print('There are {} reads in read_id path'.format(len(read_id_filter)))

    fasta_path = os.path.join(args.output_dir, f'{args.prefix}.fasta')
    processed_reads = None
    if os.path.exists(fasta_path):
        processed_reads = set([str(seq.id) for seq in SeqIO.parse(fasta_path, 'fasta')])
        # print('Exist processed reads: ', len(processed_reads))

    for filename in fast5_paths:
        with get_fast5_file(filename, 'r') as f5_fh:
            ids = f5_fh.get_read_ids()
            for read_id in ids:
                if read_id_filter is not None and read_id not in read_id_filter:
                    continue
                if processed_reads is not None and read_id in processed_reads:
                    continue

                while chunk_queue.qsize() >= 100:
                    time.sleep(1)

                read = f5_fh.get_read(read_id)
                raw = read.handle[read.raw_dataset_name][:]
                channel_info = read.handle[read.global_key + 'channel_id'].attrs
                offset = int(channel_info['offset'])
                scaling = channel_info['range'] / channel_info['digitisation']
                scaled = np.array(scaling * (raw + offset), dtype=np.float32)
                shift, scale = normalisation(scaled)
                trimmed_samples = trim(scaled, threshold=scale * 2.4 + shift)
                signal = (scaled[trimmed_samples:] - shift) / scale
                signal = torch.from_numpy(signal)
                T = len(signal)
                start_pos = 0
                chunks = []
                while start_pos < T:
                    end_pos = min(start_pos + args.chunksize, T)
                    if (T - end_pos) < args.chunksize // 2:
                        end_pos = T
                    chunks.append(signal[None, start_pos: end_pos])
                    start_pos = end_pos
                for chunk_id, chunk in enumerate(chunks):
                    chunk_queue.put((read_id, len(chunks), chunk_id, chunk))

    chunk_queue.put(None)
    end_event.wait()


def decoding_process(chunk_queue, write_queue, end_event, args):
    def torch_setting(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('high')
    torch_setting(40)
    torch.set_num_threads(2)

    model = CoralModel()
    model = model.cuda(args.gpu)
    if args.model is None:
        args.model = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'model', 'weights.pth.tar')
        print('Using default trained model')
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage.cuda(args.gpu))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    chunks_meta = []
    chunks = []
    chunk_lengths = []

    is_ending = False
    with torch.inference_mode():
        while not is_ending:
            if chunk_queue.qsize() == 0:
                continue

            chunk_data = chunk_queue.get()
            if chunk_data is None:
                is_ending = True

            if chunk_data is not None:
                read_id, chunks_size, chunk_id, chunk = chunk_data
                chunks_meta.append((read_id, chunks_size, chunk_id))
                chunks.append(chunk)
                chunk_lengths.append(chunk.size(1))

            if len(chunks) < args.batch_size and not is_ending:
                continue

            if len(chunks) > 0:
                max_chunk_length = max(chunk_lengths)
                chunks_tensor = torch.cat([
                    torch.nn.functional.pad(tensor, (0, max_chunk_length - tensor.size(1))) for tensor in chunks
                ], dim=0).cuda(args.gpu, non_blocking=True)
                chunk_lengths_tensor = torch.tensor(chunk_lengths, dtype=torch.long).cuda(args.gpu, non_blocking=True)

                tgt_idx_arr, beam_idx_arr, bp, predicts, Q = model.max_decoding_batch(
                    chunks_tensor,
                    chunk_lengths_tensor,
                )

                write_queue.put((chunks_meta, tgt_idx_arr, beam_idx_arr, bp, predicts, Q))
                chunks_meta = []
                chunks = []
                chunk_lengths = []

    write_queue.put(None)
    end_event.wait()


def writing_process(write_queue, end_event, args):
    torch.set_num_threads(1)

    fasta_path = os.path.join(args.output_dir, f'{args.prefix}.fasta')
    if os.path.exists(fasta_path):
        fasta = open(fasta_path, 'a')
    else:
        fasta = open(fasta_path, 'w')

    nucleotides = np.array(['A', 'C', 'G', 'T'], dtype='<U1')
    cache = {}
    while True:
        if write_queue.qsize() == 0:
            continue
        chunk_decoding_result = write_queue.get()
        if chunk_decoding_result is None:
            break

        chunks_meta, tgt_idx_arr, beam_idx_arr, bp_matrix, predicts_matrix, Q_matrix = chunk_decoding_result
        for meta, tgt_idx, beam_idx, bp, predicts, Q in zip(
                chunks_meta, tgt_idx_arr, beam_idx_arr, bp_matrix, predicts_matrix, Q_matrix
        ):
            read_id, chunks_size, chunk_id = meta
            if read_id not in cache:
                cache[read_id] = [[''] * chunks_size, 0, chunks_size, [0.0] * chunks_size]
            seq_data, seq_mean_quality = decoding_backtracking(tgt_idx, beam_idx, bp, predicts, Q)
            cache[read_id][3][chunk_id] = seq_mean_quality
            cache[read_id][0][chunk_id] = ''.join(nucleotides[seq_data])
            cache[read_id][1] += 1

        erase_rid_list = []
        for rid, results in cache.items():
            if results[1] == results[2]:
                fasta.write('>{}\n{}\n'.format(rid, ''.join(results[0])[::-1]))
                fasta.flush()
                erase_rid_list.append(rid)
        for rid in erase_rid_list:
            cache.pop(rid)

    if len(cache) != 0:
        print('Warning: There may be some error in basecalling ??')
    fasta.close()
    print('Basecalling finished')
    end_event.set()


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description='Coral: a basecaller for nanopore direct-RNA sequencing')
    parser.add_argument('input_dir', type=str, help='Directory containing fast5 files')
    parser.add_argument('output_dir', type=str, help='Output directory')
    parser.add_argument('--prefix', type=str, default='called', help='Filename prefix of basecaller output (default: called)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id (default: 0)')
    parser.add_argument('--read-id-path', type=str, default=None,
                        help='Basecalling solely on the reads listed in file, with one ID per line (default: None)')
    parser.add_argument('--chunksize', type=int, default=20000,
                        help="Length of signal chunk (default: 20000)")
    parser.add_argument('--batch-size', type=int, default=4,
                        help="Larger batch size will use more GPU memory (default:4)")
    parser.add_argument('--model', type=str, default=None,
                        help='Path of model checkpoint file (default: model/weights.pth.tar)')
    parser.add_argument('--version', action='version', version='%(prog)s v1.0')
    __args = parser.parse_args()

    if not os.path.isdir(__args.input_dir):
        raise RuntimeError('Input directory does not exist')

    if not os.path.exists(__args.output_dir):
        os.makedirs(__args.output_dir)

    __chunk_queue = Queue()
    __write_queue = Queue()
    __end_event = Event()
    p1 = Process(target=fast5_process, args=(__args.input_dir, __chunk_queue, __end_event, __args,))
    p2 = Process(target=decoding_process, args=(__chunk_queue, __write_queue, __end_event, __args,))
    p3 = Process(target=writing_process, args=(__write_queue, __end_event, __args,))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
