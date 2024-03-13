#!/usr/bin/env python3

import os

import numpy as np
from utils.io import export_vector, read_matrix, read_vector


def main():
    # Read the input matrix "matica.txt"
    A = read_matrix("inputs/matrix.txt", os.path.dirname(__file__))
    b = read_vector("inputs/vector.txt", os.path.dirname(__file__))

    x = np.zeros_like(b)

    maxit = 10000
    minnorm = 1e-6
    minnormdif = 1e-6

    for i in range(maxit):
        
        r = - (np.dot(A, x) - b)
        alpha = np.dot(r, r) / np.dot(r, np.dot(A, r))
        x = x + alpha * r

        prev_norm = norm if 'norm' in locals() else np.linalg.norm(x)
        norm = np.linalg.norm(r)
        if (abs(norm - prev_norm) < minnormdif):
            print(f"\nHOTOVO!\nRozdiel noriem je mensi ako {minnormdif}")
            break
        if (norm < minnorm):
            print(f"\nHOTOVO!\nNorma je mensia ako {minnorm}")
            break

        if (i % 10 == 0):
            print(f"Iteracia {i}\t\tNorma chyby: {norm}\t\tRozidel noriem chýb: {abs(norm - prev_norm)}", flush=True)
    else:
        print(f"\nHOTOVO!\nPresiahlo sa {maxit} iteracii")

    print(f"Vysledok: {x}")
    print(f"Počet iterácii: {i}")
    print(f"Norma chyby: {norm}")

    export_vector(x, "outputs/steepest_decent.txt", os.path.dirname(__file__))    

    return

if __name__ == "__main__":
    main()
