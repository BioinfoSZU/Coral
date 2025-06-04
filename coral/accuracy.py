import sys
import os
import numpy as np
import re
import pysam
from tqdm import tqdm
from argparse import ArgumentParser


def get_read_stats(cigar_splitted, strand):
    match, mismatch, insertion, deletion, n_clips = [], [], [], [], [0, 0]
    for index in range(0, len(cigar_splitted)-1, 2):
        count, symbol = int(cigar_splitted[index]), cigar_splitted[index+1]
        if symbol == "=":
            match.append(int(count))
        elif symbol == "X":
            mismatch.append(int(count))
        elif symbol == "I":
            insertion.append(int(count))
        elif symbol == "D":
            deletion.append(int(count))
        elif symbol == "S":
            if strand == "+":
                if index == 0:
                    n_clips[0] += count
                else:
                    n_clips[1] += count
            if strand == "-":
                if index == 0:
                    n_clips[1] += count
                else:
                    n_clips[0] += count
    return match, mismatch, insertion, deletion, n_clips


def get_sample_stats(infile, return_counts=False):
    print("Processing sample:", infile)
    seqs = read_samfile(infile)
    read_stats, read_q, empirical_q = {}, [], []
    n_seqs = len(seqs)

    read_stats["metrics"] = np.empty((n_seqs, 3), dtype=np.float64)
    read_stats["acc"] = np.empty(n_seqs, dtype=np.float64)

    read_stats["align_ratio"] = np.empty(n_seqs, dtype=np.float64)
    read_stats["original_read_length"] = np.empty(n_seqs, dtype=np.int64)
    read_stats["aligned_read_length"] = np.empty(n_seqs, dtype=np.int64)
    read_stats["n_clips"] = np.empty((n_seqs, 2), dtype=np.float64)

    mismatches, insertions, deletions = [], [], []

    for seq_num in range(n_seqs):
        name, flag, start, cigar, sample_seq, mapq, chromo, qs_base = seqs[seq_num]

        strand = '-' if int(flag) & 0x10 else '+'

        cigar_splitted = re.split('([^0-9])', cigar.strip())
        match, mismatch, insertion, deletion, n_clips = get_read_stats(cigar_splitted, strand=strand)

        if return_counts:
            mismatches += mismatch
            insertions += insertion
            deletions += deletion

        match, mismatch, insertion, deletion = np.sum(match), np.sum(mismatch), np.sum(insertion), np.sum(deletion)
        align_ratio = 1 - (n_clips[0] + n_clips[1]) / len(sample_seq)
        read_l = match + mismatch + insertion + deletion

        read_stats["metrics"][seq_num, 0] = mismatch / read_l
        read_stats["metrics"][seq_num, 1] = insertion / read_l
        read_stats["metrics"][seq_num, 2] = deletion / read_l
        read_stats["acc"][seq_num] = match / read_l

        read_stats["align_ratio"][seq_num] = align_ratio
        read_stats["aligned_read_length"][seq_num] = match + mismatch + insertion
        read_stats["n_clips"][seq_num, :] = n_clips

    if return_counts:
        return read_stats, np.array(mismatches), np.array(insertions), np.array(deletions)
    else:
        return read_stats


def read_samfile(
    samfile,
):
    seqs, supp_count, unmap_cnt = [], 0, 0
    sf = pysam.AlignmentFile(samfile, "r")
    for read in sf.fetch():
        if read.is_unmapped:
            unmap_cnt += 1
            continue
        if read.is_supplementary:
            supp_count += 1
            continue
        if read.query_sequence is None:
            print("FAIL:", read.qname, read.reference_name)
            continue

        name = read.query_name
        flag = read.flag
        start = read.reference_start + 1
        mapq = read.mapping_quality
        cigar = read.cigarstring
        seq = read.query_sequence
        chromo = read.reference_name
        qscore = read.query_qualities
        seqs.append((name, flag, start, cigar, seq, mapq, chromo, qscore))

    if supp_count > 0:
        print("The number of supplementary alignments in this sample is ", supp_count)
    if unmap_cnt > 0:
        print('The number of unmapped reads in this sample is ', unmap_cnt)
    return seqs


def add_arguments(parser: ArgumentParser):
    parser.add_argument('--samfile', type=str, help='sam file path created by minimap2')


def run(args):
    if args.samfile is None or not os.path.isfile(args.samfile):
        print('samfile does not exist')
        sys.exit(0)

    stat, mis, ins, dels = get_sample_stats(args.samfile, return_counts=True)
    print('accuracy  (median/mean): {:.2f}% / {:.2f}%'.format(
        np.median(stat['acc']) * 100.0,
        np.mean(stat['acc']) * 100.0
    ))
    print('mismatch  (median/mean): {:.2f}% / {:.2f}%'.format(
        np.median(stat['metrics'][:, 0]) * 100.0,
        np.mean(stat['metrics'][:, 0]) * 100.0,
    ))
    print('insertion (median/mean): {:.2f}% / {:.2f}%'.format(
        np.median(stat['metrics'][:, 1]) * 100.0,
        np.mean(stat['metrics'][:, 1]) * 100.0,
    ))
    print('deletion  (median/mean): {:.2f}% / {:.2f}%'.format(
        np.median(stat['metrics'][:, 2]) * 100.0,
        np.mean(stat['metrics'][:, 2]) * 100.0,
    ))
    print('read length  (median/mean): {:.0f} / {:.0f}'.format(
        int(np.median(stat['aligned_read_length'])),
        int(np.mean(stat['aligned_read_length'])),
    ))
