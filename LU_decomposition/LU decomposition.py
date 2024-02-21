#!/usr/bin/env python3

import os
import numpy as np

def read_matrix(filePath: str) -> np.ndarray:
    with open(os.path.join(os.path.dirname(__file__), filePath), "r") as file:
        matrix = [[float(e) for e in row.split() if e] for row in file]
    return np.array(matrix)

def read_vector(filePath: str) -> np.ndarray:
    with open(os.path.join(os.path.dirname(__file__), filePath), "r") as file:
        vector = [float(e) for e in file.read().split() if e]
    return np.array(vector)

def export_matrix(matrix: np.ndarray, filePath: str) -> None:
    with open(os.path.join(os.path.dirname(__file__), filePath), "w") as file:
        for row in matrix:
            file.write(" ".join([str(e)[:11].ljust(12) for e in row]) + "\n")
    return

def export_vector(vector: np.ndarray, filePath: str) -> None:
    with open(os.path.join(os.path.dirname(__file__), filePath), "w") as file:
        file.write("\n".join([str(v)[:8].ljust(8) for v in vector]))
    return 

def identity_matrix(n: int) -> np.ndarray:
    return np.eye(n)

def main():
    
    # Read the input matrix "matica.txt"
    matrix = read_matrix("inputs/matrix.txt")
    # Read the input vector "ps.txt"
    vector = read_vector("inputs/vector.txt")

    I = identity_matrix(len(matrix))

    for k in range(len(matrix)):
        for j in range(k+1, len(matrix)):
            I[j][k] = matrix[j][k] / matrix[k][k]
            matrix[j][k:] -= I[j][k] * matrix[k][k:]

    # Export the matrix "matica.txt" to "matica_out.txt"
    export_matrix(matrix, "outputs/matrix.txt")
    # Export the vector "ps.txt" to "ps_out.txt"
    export_vector(vector, "outputs/vector.txt")

    return

if __name__ == "__main__":
    main()
