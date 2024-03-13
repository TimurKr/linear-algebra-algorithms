#!/usr/bin/env python3

import os

import numpy as np

from utils.algorithms import lu
from utils.io import export_matrix, export_vector, read_matrix, read_vector


def main():
    # Read the input matrix "matica.txt"
    A = read_matrix("inputs/matrix.txt", os.path.dirname(__file__))
    b = read_vector("inputs/vector.txt", os.path.dirname(__file__))

    n = len(A)

    x = np.zeros(n)
    w = 1

    while True:
        x_prev = x.copy()
        for i in range(n):
            x[i] = (1 - w) * x[i] + w * (b[i] - np.dot(A[i], x)) / A[i, i]
        if np.linalg.norm(x - x_prev) < 10**-6:
            break

        print("TODO: Dorobiť")

        # Loops forever...
        break 

    return

if __name__ == "__main__":
    main()
