import re
import math
import parasail
import numpy as np
import torch
import json
import contextlib
import numpy.core.multiarray
from packaging import version
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR


split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

basecall_config_f = Path(__file__).parent / "configs/basecall_config.json"
with open(basecall_config_f) as basecall_config_file_handle:
    default_basecall_config = json.load(basecall_config_file_handle)


def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    allowlist = [np.core.multiarray._reconstruct, np.core.multiarray.scalar, np.ndarray, np.dtype]  # np.core is allowed in numpy.__version__ < 2.0
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [bytes]
    allowlist += [type(np.dtype(np.int32))]
    allowlist += [type(np.dtype(np.uint32))]
    allowlist += [type(np.dtype(np.float32))]
    allowlist += [type(np.dtype(np.float64))]

    return torch.serialization.safe_globals(allowlist)


def trim(_signal, window_size=40, threshold=2.4, min_trim=10, min_elements=3, max_samples=8000, max_trim=0.3):
    seen_peak = False
    num_windows = min(max_samples, len(_signal)) // window_size

    for pos in range(num_windows):
        __start = pos * window_size + min_trim
        __end = __start + window_size
        window = _signal[__start: __end]
        if len(window[window > threshold]) > min_elements or seen_peak:
            seen_peak = True
            if window[-1] > threshold:
                continue
            if __end >= min(max_samples, len(_signal)) or __end / len(_signal) > max_trim:
                return min_trim
            return __end

    return min_trim


def normalisation(sig, norm_params=None):
    if 'mean' in norm_params and 'stdev' in norm_params:
        return norm_params['mean'], norm_params['stdev']
    qa, qb = np.quantile(sig, [norm_params['quantile_a'], norm_params['quantile_b']])
    _shift = max(10, norm_params['shift_multiplier'] * (qa + qb))
    _scale = max(1.0, norm_params['scale_multiplier'] * (qb - qa))
    return _shift, _scale


def convert_seq_to_str(s, alphabet):
    return "".join([alphabet[c] for c in s])


def convert_ints_to_qual(qs, ch_base=33):
    return ''.join([chr(q + ch_base) for q in qs])


def get_padded_zero_tensor(tensor: torch.Tensor, new_size):
    assert tensor.shape[0] <= new_size
    ret = tensor.new_zeros((new_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    ret[:len(tensor)] = tensor
    return ret


def alignment_process(query_seq, ref_seq, rid, verbose=False):
    alignment = parasail.sw_trace_striped_32(query_seq, ref_seq, 8, 4, parasail.dnafull)

    if alignment is None:
        return None, None

    cigstr = alignment.cigar.decode.decode()
    rstart = alignment.cigar.beg_ref
    qstart = alignment.cigar.beg_query

    first = re.search(split_cigar, cigstr)
    if first is None or len(cigstr) == 0:
        if verbose:
            print(
                f'Error when parsing alignment: rid={rid} cigstr={cigstr}, rstart={rstart}, qstart={qstart}, query={query_seq} ref={ref_seq}'
            )
        return None, None

    first_count, first_op = first.groups()

    if first_op == 'I':
        qstart += int(first_count)

    if first_op == 'D':
        rstart = int(first_count)

    alignment_region_len = len(alignment.traceback.comp)
    half = alignment_region_len // 2
    query_half_len = len(alignment.traceback.query[:half].replace('-', ''))
    ref_half_len = len(alignment.traceback.ref[:half].replace('-', ''))
    return qstart + query_half_len, rstart + ref_half_len


def cosine_decay_schedule(y0, y1):
    return lambda t: y1 + 0.5 * (y0 - y1) * (np.cos(t * np.pi) + 1.0)


def linear_schedule(y0, y1):
    return lambda t: y0 + (y1 - y0) * t


def piecewise_schedule(knots, funcs):
    def f(t):
        i = np.searchsorted(knots, t)
        t0 = 0.0 if i == 0 else knots[i - 1]
        t1 = 1.0 if i == len(knots) else knots[i]
        return funcs[i]((t - t0) / (t1 - t0))
    return f


def func_scheduler(optimizer, func, total_steps, warmup_steps=None, warmup_ratio=0.1, start_step=0):
    if warmup_steps:
        y0 = func(0.0)
        func = piecewise_schedule(
            [warmup_steps / total_steps],
            [linear_schedule(warmup_ratio * y0, y0), func]
        )
    return LambdaLR(optimizer, (lambda step: func((step + start_step) / total_steps)))


def linear_warmup_cosine_decay(end_ratio=0.01, warmup_steps=500, **kwargs):
    return lambda optimizer, train_loader, epochs, last_epoch: func_scheduler(
        optimizer=optimizer,
        func=cosine_decay_schedule(1.0, end_ratio),
        total_steps=epochs * len(train_loader),
        warmup_steps=warmup_steps,
        start_step=last_epoch * len(train_loader),
    )

