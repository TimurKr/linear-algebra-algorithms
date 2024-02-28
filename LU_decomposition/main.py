#!/usr/bin/env python3

import os
import numpy as np

from utils.io import export_matrix, export_vector, read_matrix, read_vector


def main():
    
    # Read the input matrix "matica.txt"
    matrix = read_matrix("inputs/matrix.txt", os.path.dirname(__file__))
    # Read the input vector "ps.txt"
    vector = read_vector("inputs/vector.txt", os.path.dirname(__file__))

    I = np.eye(len(matrix))

    for k in range(len(matrix)):
        for j in range(k+1, len(matrix)):
            I[j][k] = matrix[j][k] / matrix[k][k]
            matrix[j][k:] -= I[j][k] * matrix[k][k:]

    # Export the matrix "matica.txt" to "matica_out.txt"
    export_matrix(matrix, "outputs/matrix.txt", os.path.dirname(__file__))
    # Export the vector "ps.txt" to "ps_out.txt"
    export_vector(vector, "outputs/vector.txt", os.path.dirname(__file__))

    return

if __name__ == "__main__":
    main()
