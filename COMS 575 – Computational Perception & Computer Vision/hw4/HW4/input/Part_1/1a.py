import cmath
import numpy as np


# Part 1a: Slow DFT
def slow_dft(input_x):
    N = len(input_x)
    out = []
    for k in range(N):
        z = complex(0)
        for n in range(N):
            exponent = 2j * cmath.pi * k * n / N
            z += input_x[n] * cmath.exp(-exponent)
        out.append(z)
    return out