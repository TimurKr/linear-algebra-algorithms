#!/usr/bin/env python3

import os
import time

from utils.algorithms import crs_multiplication, lu, multiplication
from utils.io import export_matrix, export_vector, read_matrix, read_matrix_crs, read_vector

def main():
    
    # Read the input matrix "matica.txt"
    A = read_matrix("inputs/large/matrix.txt", os.path.dirname(__file__))
    vals, col_ind, row_ptr = read_matrix_crs("inputs/large/matrix.txt", os.path.dirname(__file__))

    x = read_vector("inputs/large/vector.txt", os.path.dirname(__file__))

    print("\nCorrect answer base on numpy:", flush=True)
    start_time = time.time()
    print(A @ x)
    end_time = time.time()
    print("\nElapsed time for numpy multiplication:")
    print(end_time - start_time)

    print("\nElapsed time for CRS multiplication:", flush=True)
    start_time = time.time()
    y = crs_multiplication(vals, col_ind, row_ptr, x)
    end_time = time.time()
    print(end_time - start_time)
    print("\nResult of CRS multiplication:")
    print(y)

    print("\nElapsed time for multiplication:", flush=True)
    start_time = time.time_ns()
    y = multiplication(A, x)
    end_time = time.time()
    print(end_time - start_time)
    print("\nResult of multiplication:")
    print(y, "\n")
    
    # L, U = lu(A)

    # export_matrix(L, "outputs/L.txt", os.path.dirname(__file__))
    # export_matrix(U, "outputs/U.txt", os.path.dirname(__file__))

    return

if __name__ == "__main__":
    main()
