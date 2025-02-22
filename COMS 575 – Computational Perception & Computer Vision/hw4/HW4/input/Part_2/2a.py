import cmath
import numpy as np


# Part 2a: FFT
def recursive_fft(input_x):
    x = np.asarray(input_x, dtype=float)
    N = x.shape[0]

    if N is 1:
        return x
    else:
        even = recursive_fft(x[::2])
        odd = recursive_fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        # combine two halves
        return np.concatenate([even + factor[:N // 2] * odd, even + factor[N // 2:] * odd])
                     