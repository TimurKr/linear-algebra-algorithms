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