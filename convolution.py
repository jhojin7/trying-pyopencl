from pytools import F
from getimages import _imshow
from matrix_multiplication import init_mat, matmul_seq
from timeit import timeit
import pyopencl as cl
import numpy as np
import torch
import json

SAMPLE_BIN = "./cifar10/sample_bin_data.json"
IMG = "./cifar10/__DEER_muntjac_s_001000.png"

#### traversing a 3d array
def init_3darray(_shape:np.shape, _dtype:np.dtype):
    # -> np.ndarray.astype(np.int64):
    """initialize int array with _shape(k,i,j) and type _dype"""
    K,I,J = _shape
    arr = np.zeros(_shape).astype(_dtype)
    cnt = 1
    for k in range(K):
        for i in range(I):
            for j in range(J):
                arr[k,i,j] = cnt
                cnt += 1
        #     print(rgb,arr[rgb,i,:])
        # print()
    return arr

def convolution(A,filter):
    # # how to get a window for conv
    # print(A[0])
    # for i in range(3+2):
    #     for j in range(3+2):
    #         print(A[0,i:i+3,j:j+3])

    C = np.zeros_like(A)
    paddedA = np.pad(A,1)[1:IMGSHAPE[0]+1]
    # convolve
    for k in range(paddedA.shape[0]):
        for i in range(filter.shape[1]+2):
            for j in range(filter.shape[0]+2):
                # print(A[k,i:i+3,j:j+3])
                tmpA = paddedA[k,i:i+3,j:j+3]
                tmpC = np.zeros_like(filter)
                # print(tmpA.shape, tmpC.shape, filter.shape)
                # print(tmpA)
                np.matmul(tmpA,filter,tmpC)
                C[k,i,j] = tmpC.sum()
    return C

if __name__=="__main__":
    IMGSHAPE = (3,5,5)
    A = init_3darray(IMGSHAPE, np.int64)
    B = np.array([[1,1,1],[1,1,1],[1,1,1]],
        dtype=np.int64) # (3,3)
    C = convolution(A,B)
    print(C)
    # C2 = np.zeros_like(A)
    # print(C2)