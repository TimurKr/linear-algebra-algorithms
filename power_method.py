import os
import numpy as np

from utils.algorithms import crs_multiplication, lu, multiplication
from utils.io import export_matrix, export_vector, read_matrix, read_matrix_crs, read_vector

maxit = 100
tol = 1e-4

def max_eigen_val(A, x):

    for k in range(maxit):
        x_temp = multiplication(A, x)

        lambda_k = max(x_temp)

        x = x_temp / lambda_k

        r = multiplication(A, x) - lambda_k * x        

        if (np.linalg.norm(r) < tol):
            break

    return lambda_k, k

def min_eigen_val(A, x):

    for k in range(maxit):
        x_temp = np.linalg.solve(A, x)

        lambda_k = max(x_temp)

        x = x_temp / lambda_k

        r = np.linalg.solve(A, x) - lambda_k * x        

        if (np.linalg.norm(r) < tol):
            break

    return 1/lambda_k, k

def main():
    x = read_vector("inputs/vector.txt", os.path.dirname(__file__))
    
    A = read_matrix("inputs/matrix.txt", os.path.dirname(__file__))

    print("Power method:")
    print("Correct answer base on numpy:")
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues.sort()
    for i in eigenvalues:
        print(i)
    print()

    eig_val, num_it = max_eigen_val(A, x)

    print(f"Maximálna vlastná hodnota: {eig_val} - za {num_it} iterácií")

    eig_val, num_it = min_eigen_val(A, x)

    print(f"Minimálna vlastná hodnota: {eig_val} - za {num_it} iterácií")


if __name__ == "__main__":
    main()