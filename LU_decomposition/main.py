#!/usr/bin/env python3

import os

from utils.algorithms import lu
from utils.io import export_matrix, export_vector, read_matrix, read_vector


def main():
    
    # Read the input matrix "matica.txt"
    A = read_matrix("inputs/matrix.txt", os.path.dirname(__file__))

    L, U = lu(A)

    export_matrix(L, "outputs/L.txt", os.path.dirname(__file__))
    export_matrix(U, "outputs/U.txt", os.path.dirname(__file__))

    return

if __name__ == "__main__":
    main()
