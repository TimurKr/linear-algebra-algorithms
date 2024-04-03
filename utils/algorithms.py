import numpy as np

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

def multiplication(A: np.ndarray, x: np.ndarray):
    n = len(A)
    y = np.zeros(n, dtype=np.double)
    
    for i in range(n):
        y[i] = sum(A[i, j] * x[j] for j in range(n))
        
    return y

def dot(a: np.ndarray, b: np.ndarray):
    return sum(ai * bi for ai, bi in zip(a, b))

def crs_multiplication(vals: list[float], col_ind: list[int], row_ptr: list[int], x: np.ndarray):
    n = len(row_ptr)
    y = np.zeros(n, dtype=np.double)
    
    # if n > 10:
    #     print(f"CRS for large matrix, size: {n}x{n}", flush=True)
    #     print(f"Doing {len(vals)} multiplications instead of {n**2}", flush=True)

    for i in range(n-1):
        y[i] = sum(vals[j] * x[col_ind[j]] for j in range(row_ptr[i], row_ptr[i+1]))
    y[-1] = sum(vals[j] * x[col_ind[j]] for j in range(row_ptr[i+1], len(vals)))
        
    return y