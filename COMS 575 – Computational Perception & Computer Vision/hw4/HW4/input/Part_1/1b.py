import cmath
import numpy as np


# Part 1b: Slow IDFT
def slow_idft(input_X):
    N = len(input_X)
    out = []
    for k in range(N):
        z = complex(0)
        for n in range(N):
            exponent = 2j * cmath.pi * k * n / N
            z += input_X[n] * cmath.exp(exponent)
        out.append(1 / N * z)
    return out