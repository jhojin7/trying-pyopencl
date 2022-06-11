from timeit import timeit
import json

import pyopencl as cl
import numpy as np
import torch

import getimages

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

    C = np.zeros_like(A)#,dtype="float64")
    paddedA = np.pad(A,1)[1:A.shape[0]+1]
    # print(paddedA.shape)
    # convolve
    for k in range(paddedA.shape[0]):
        for i in range(A.shape[1]):
            for j in range(A.shape[2]):
                # print(A[k,i:i+3,j:j+3])
                tmpA = paddedA[k,i:i+3,j:j+3]
                tmpC = np.zeros_like(filter)
                # print(tmpA.shape, tmpC.shape, filter.shape)
                # print(tmpA)
                np.matmul(tmpA,filter,tmpC)
                C[k,i,j] = tmpC.sum()/np.count_nonzero(filter)
    # print((tmpA @ filter).sum(), tmpC.sum(), np.dot(tmpA,filter).sum())

    return C

if __name__=="__main__":
    """input data"""
    fname, data = getimages.get_images(20)
    print(data.shape)
    getimages._imshow(data)
    # A = init_3darray(data.shape, np.int64)
    # A.fill(1)

    """filters"""
    B = np.array([[0,0,1],
                  [1,0,0],
                  [0,1,1]], dtype=np.int64) # (3,3)
    ones = np.ones_like(B)
    intensifiedSharpen = np.array(
    [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.int0) # (3,3)
    laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")
    sharper = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")
    horiz = np.array((
    [-1, -2, -1],
    [0,0,0],
    [1,2,1]), dtype="int")

    """conv"""
    FILTER = B
    C = convolution(data,FILTER)
    print(C.shape)
    getimages._imshow(C)