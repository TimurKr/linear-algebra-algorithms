#!/usr/bin/env python3

import os
import sys
import numpy as np
from LU_decomposition.main import lu
from utils.algorithms import backward_substitution, forward_substitution

from utils.io import export_matrix, export_vector, read_matrix, read_vector

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
