#!/usr/bin/env python3

import os
import numpy as np

from utils.io import export_matrix, export_vector, read_matrix, read_vector

def lu(A: np.ndarray):
    
    #Get the number of rows
    n = len(A)
    
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    
    #Loop over rows
    for i in range(n):
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i]
        
    return L, U



def main():
    
    # Read the input matrix "matica.txt"
    A = read_matrix("inputs/matrix.txt", os.path.dirname(__file__))

    L, U = lu(A)

    export_matrix(L, "outputs/L.txt", os.path.dirname(__file__))
    export_matrix(U, "outputs/U.txt", os.path.dirname(__file__))

    return

if __name__ == "__main__":
    main()
