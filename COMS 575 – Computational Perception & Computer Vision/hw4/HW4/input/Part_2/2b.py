import cmath
import numpy as np


# Part 2b: IFFT
def recursive_sub_ifft(input_x):
    x = np.asarray(input_x, dtype=float)
    N = x.shape[0]

    if N is 1:
        return x
    else:
        even = recursive_sub_ifft(x[::2])
        odd = recursive_sub_ifft(x[1::2])
        # numpy arange(N) gets the list of N
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        # combine two halves using numpy to combine them as it iterates through the halves
        return np.concatenate([even + factor[:N // 2] * odd, even + factor[N // 2:] * odd])


# call this method, so the values will be normalized at the end
def ifft(input_x):
    return recursive_sub_ifft(input_x) / len(input_x)