#!/usr/bin/env python3

import os
import time

import numpy as np
from utils.algorithms import crs_multiplication
from utils.io import export_vector, read_matrix, read_matrix_crs, read_vector

maxit = 10000
minnorm = 0.001
minnormdif = 1e-6

def main():
    """
    This function compares the performence of performing the BiCGSTAB algorithm
    to solve a linear system of equations.

    1. Using the full matrix and numpy.
    2. Using the matrix in CRS format and custom matrix multiplication.
    """

    # starttime = time.time()
    
    # A = read_matrix("inputs/large/matrix.txt", os.path.dirname(__file__))
    # b = read_vector("inputs/large/vector.txt", os.path.dirname(__file__))

    # readtime = time.time()

    # r = [b]
    # p = [b]
    # x = [np.zeros_like(b)]

    # print("\n\nStarting BiCGSTAB algorithm with numpy...\n", flush=True)

    # for i in range(maxit):

    #     alpha = (r[i] @ r[0]) / ((A @ p[i]) @ r[0])
    #     s = r[i] - alpha * (A @ p[i])
    #     w = ((A @ s) @ s) / ((A @ s) @ (A @ s))
    #     x.append(x[i] + alpha * p[i] + w * s)
    #     r.append(s - w * (A @ s))
    #     beta = ((r[i+1] @ r[0]) / (r[i] @ r[0])) * (alpha / w)
    #     p.append(r[i+1] + beta * (p[i] - w * (A @ p[i])))

    #     prev_norm = norm if 'norm' in locals() else np.linalg.norm(r[0])
    #     norm = np.linalg.norm(r[-1])
    #     if (abs(norm - prev_norm) < minnormdif):
    #         print(f"\nHOTOVO!\tRozdiel noriem je mensi ako {minnormdif}")
    #         break
    #     if (norm < minnorm):
    #         print(f"\nHOTOVO!\tNorma je mensia ako {minnorm}")
    #         break

    #     if (i % 10 == 0):
    #         print(f"Iteracia {i}\t\tNorma chyby: {norm}\t\tRozidel noriem chýb: {abs(norm - prev_norm)}", flush=True)
    # else:
    #     print(f"\nHOTOVO!\tPresiahlo sa {maxit} iteracii")

    # print(f"\nVysledok: {x[-1]}")
    # print(f"Počet iterácii: {i}")
    # print(f"Norma chyby: {norm}")

    # endtime = time.time()
    # print("\n----------------------")
    # print(f"Reading time: {readtime - starttime}")
    # print(f"Execution time: {endtime - readtime}")
    # print("----------------------\n")

    # export_vector(x[-1], "outputs/bicgstab.txt", os.path.dirname(__file__))

    # CRS
    starttime = time.time()
    
    vals, col_ind, row_ptr = read_matrix_crs("inputs/large/matrix.txt", os.path.dirname(__file__))
    b = read_vector("inputs/large/vector.txt", os.path.dirname(__file__))

    readtime = time.time()
    print(f"Reading time: {readtime - starttime}")


    r = [b]
    p = [b]
    x = [np.zeros_like(b)]

    print("\n\n-------------------------------------------\n\n")
    print("Starting BiCGSTAB algorithm with crs_multiplication...\n", flush=True)

    for i in range(maxit):
        alpha = np.dot(r[i], r[0]) / np.dot(crs_multiplication(vals, col_ind, row_ptr, p[i]), r[0])
        s = r[i] - alpha * crs_multiplication(vals, col_ind, row_ptr, p[i])
        w = np.dot(crs_multiplication(vals, col_ind, row_ptr, s), s) / np.dot(crs_multiplication(vals, col_ind, row_ptr, s), crs_multiplication(vals, col_ind, row_ptr, s))
        x.append(x[i] + alpha * p[i] + w * s)
        r.append(s - w * crs_multiplication(vals, col_ind, row_ptr, s))
        beta = (np.dot(r[i+1], r[0]) / np.dot(r[i], r[0])) * (alpha / w)
        p.append(r[i+1] + beta * (p[i] - w * crs_multiplication(vals, col_ind, row_ptr, p[i])))

        prev_norm = norm if 'norm' in locals() else np.linalg.norm(r[0])
        norm = np.linalg.norm(r[-1])
        if (abs(norm - prev_norm) < minnormdif):
            print(f"\nHOTOVO!\tRozdiel noriem je mensi ako {minnormdif}")
            break
        if (norm < minnorm):
            print(f"\nHOTOVO!\tNorma je mensia ako {minnorm}")
            break

        if (i % 10 == 0):
            print(f"Iteracia {i}\t\tNorma chyby: {norm}\t\tRozidel noriem chýb: {abs(norm - prev_norm)}", flush=True)
    else:
        print(f"\nHOTOVO!\tPresiahlo sa {maxit} iteracii")

    print(f"\nVysledok: {x[-1]}")
    print(f"Počet iterácii: {i}")
    print(f"Norma chyby: {norm}")

    endtime = time.time()
    print("\n----------------------")
    print(f"Reading time: {readtime - starttime}")
    print(f"Execution time: {endtime - readtime}")
    print("----------------------\n")

    export_vector(x[-1], "outputs/bicgstab_crs.txt", os.path.dirname(__file__))

    return

if __name__ == "__main__":
    main()
