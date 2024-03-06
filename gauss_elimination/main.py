#!/usr/bin/env python3

import os
import sys
import numpy as np
from LU_decomposition.main import lu

from utils.io import export_matrix, export_vector, read_matrix, read_vector

def forward_substitution(M: np.ndarray, y: np.ndarray):
    n = len(M)
    x = np.zeros_like(y, dtype=np.double)
    
    for i in range(n):
        x[i] = (y[i] - np.dot(M[i, :i], x[:i])) / M[i, i]
        
    return x

def backward_substitution(M: np.ndarray, y: np.ndarray):
    n = len(M)
    x = np.zeros_like(y, dtype=np.double)
    
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(M[i, i+1:], x[i+1:])) / M[i, i]
        
    return x

def main():
    
    # Read the input matrix "matica.txt"
    matrix = read_matrix("inputs/matrix.txt", os.path.dirname(__file__))

    L, U = lu(matrix)
    
    # Read the input vector "vektor.txt"
    vector = read_vector("inputs/vector.txt", os.path.dirname(__file__))

    y = forward_substitution(L, vector)
    x = backward_substitution(U, y)

    # Export the matrix "matica.txt" to "matica_out.txt"
    export_vector(x, "outputs/vector.txt", os.path.dirname(__file__))

    return

if __name__ == "__main__":
    main()
