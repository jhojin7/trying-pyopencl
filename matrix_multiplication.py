from timeit import timeit
import pyopencl as cl
import numpy as np
import torch

def init_mat(x,y):
    """Initialize Matrices\n
    A * B = C, np.matmul(A,B,C2)"""
    A,B,C,C2 = None,None,None,None
    A = np.random.rand(x,y)
    B = np.random.rand(x,y)
    C = np.zeros((x,y))
    C2 = np.zeros((x,y))
    return A,B,C,C2  

def matmul_seq(A,B,C,C2)->bool:
    """Matrix Multiplication. Sequential version.\n
    returns bool C==C2
    """
    # using for loops
    for a in range(C.shape[0]):
        for b in range(C.shape[1]):
            for k in range(C.shape[1]):
                C[a,b] += A[a,k] * B[k,b]
    # using np.matmul
    np.matmul(A,B,C2)
    # check returned values
    return (C.all() == C2.all())

setup = """
A,B,C,C2 = init_mat(32,32)
"""
stmt = """
global check
check = matmul_seq(A,B,C,C2)
""" # use global for values that are needed in main

if __name__ == "__main__":
    # do 10 tests
    for _ in range(10):
        check = None
        time = timeit(setup=setup,stmt=stmt,globals=globals(),number=1)
        print("Sequential:",time, "\tCheck:",check)

# Sequential: 0.04553819999999997         Check: True
# Sequential: 0.04522880000000007         Check: True
# Sequential: 0.04114210000000007         Check: True
# Sequential: 0.03388379999999991         Check: True
# Sequential: 0.030907999999999936        Check: True
# Sequential: 0.030723999999999974        Check: True
# Sequential: 0.03270020000000007         Check: True
# Sequential: 0.030240999999999962        Check: True
# Sequential: 0.03482999999999992         Check: True
# Sequential: 0.04426520000000034         Check: True