import numpy as np


def trim(signal, window_size=40, threshold=2.4, min_trim=10, min_elements=3, max_samples=8000, max_trim=0.3):
    seen_peak = False
    num_windows = min(max_samples, len(signal)) // window_size

    for pos in range(num_windows):
        start = pos * window_size + min_trim
        end = start + window_size
        window = signal[start:end]
        if len(window[window > threshold]) > min_elements or seen_peak:
            seen_peak = True
            if window[-1] > threshold:
                continue
            if end >= min(max_samples, len(signal)) or end / len(signal) > max_trim:
                return min_trim
            return end

    return min_trim


def normalisation(sig):
    q20, q90 = np.quantile(sig, [0.2, 0.9])
    shift = max(10, 0.51 * (q20 + q90))
    scale = max(1.0, 0.53 * (q90 - q20))
    return shift, scale
