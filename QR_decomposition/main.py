#!/usr/bin/env python3

import os
import sys
import numpy as np

from utils.io import export_matrix, export_vector, read_matrix, read_vector


def main():
    
    # Read the input matrix "matica.txt"
    matrix = read_matrix("inputs/matrix.txt", os.path.dirname(__file__))
    print(matrix)

    I = np.eye(len(matrix))


    for k in range(len(matrix)):    # Not working...
        x = np.transpose(matrix)[k][k:]
        print(x)
        u_k = np.sign(x[0]) * np.linalg.norm(x) * I[1] + x
        u_k = u_k / np.linalg.norm(u_k)
        matrix[k:][k:] -= 2 * u_k*(u_k * matrix[k:][k:])

    # Export the matrix "matica.txt" to "matica_out.txt"
    export_matrix(matrix, "outputs/matrix.txt", os.path.dirname(__file__))

    return

if __name__ == "__main__":
    main()
