import os
import numpy as np


def read_matrix(filePath: str, dir: str) -> np.ndarray:
  with open(os.path.join(dir, filePath), "r") as file:
    matrix = [[float(e) for e in row.split() if e] for row in file]
  return np.array(matrix, dtype=float)

def read_matrix_crs(filePath: str, dir: str):
    val: list[float] = []
    col_ind: list[int] = []
    row_ptr: list[int] = []

    # Add code here to populate val, col_ind, and row_ind arrays
    with open(os.path.join(dir, filePath), "r") as file:
        for row in file:
            row_ptr.append(len(val))
            for i, e in enumerate(row.split()):
                if abs(float(e)) > 1e-10:
                    val.append(float(e))
                    col_ind.append(i)
    return val, col_ind, row_ptr


def read_vector(filePath: str, dir: str) -> np.ndarray:
    with open(os.path.join(dir, filePath), "r") as file:
        vector = [float(e) for e in file.read().split() if e]
    return np.array(vector)

def export_matrix(matrix: np.ndarray, filePath: str, dir: str) -> None:
    with open(os.path.join(dir, filePath), "w") as file:
        for row in matrix:
            file.write(" ".join([str(e)[:11].ljust(12) for e in row]) + "\n")
    return

# def export_matrix_crs(val: list, col_ind: list, row_ptr: list, filePath: str, dir: str) -> None:
#     with open(os.path.join(dir, filePath), "w") as file:
#         for i, e in enumerate(row_ptr):
#             r = 
            
    # return

def export_vector(vector: np.ndarray, filePath: str, dir: str) -> None:
    with open(os.path.join(dir, filePath), "w") as file:
        file.write("\n".join([str(v)[:8].ljust(8) for v in vector]))
    return 

    