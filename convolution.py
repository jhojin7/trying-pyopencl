from timeit import timeit
import json
from time import time

import pyopencl as cl
import numpy as np
from tqdm import tqdm

import getimages

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

def conv_opencl(A,filter):
    from matrix_multiplication import Context
    mf = cl.mem_flags
    # build program and kernel
    CL = Context()
    kernel = CL.build_matmul()

    C = np.zeros_like(A)#,dtype="float64")
    # filter buffer allocation
    buf_filt = cl.Buffer(CL.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
        size=0, hostbuf=filter) # ro, copy host memory
    # convolve
    paddedA = np.pad(A,1)[1:A.shape[0]+1]
    for k in range(paddedA.shape[0]):
        for i in range(A.shape[1]):
            for j in range(A.shape[2]):
                tmpA = paddedA[k,i:i+3,j:j+3].flatten()
                tmpC = np.empty_like(filter)
                # np.matmul(tmpA,filter,tmpC)
                # C[k,i,j] = tmpC.sum()/np.count_nonzero(filter)

                # initialize buffers
                buf_tmpA = cl.Buffer(CL.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=tmpA) # ro, copy host memory
                # make and set arguments
                buf_tmpC = cl.Buffer(CL.ctx, mf.WRITE_ONLY, tmpC.nbytes) # wo
                # row_tmpA = np.uint32(3)
                # col_tmpA = np.uint32(3)
                # col_filt = np.uint32(filter.shape[1])

                # matmul
                kernel(CL.queue, tmpA.shape, None, buf_tmpA, buf_filt, buf_tmpC,
                    np.uint32(3), np.uint32(3), np.uint32(3))
                    # row_tmpA, col_tmpA, col_filt) # parse Clang integer
                cl.enqueue_copy(CL.queue, tmpC, buf_tmpC)
                C[k,i,j] = tmpC.sum()/np.count_nonzero(filter)
    return C

if __name__=="__main__":
    """filters"""
    ones = np.ones((3,3),dtype=np.uint32)

    """input data"""
    fname, data = getimages.get_images(20)
    print(data.shape, type(data[0,0,0]))
    # getimages._imshow(data)

    """conv"""
    FILTER = ones
    start = time()
    C = convolution(data,FILTER)
    end = time()
    print(C.shape, "seq time:", end-start)
    start = time()
    xC = conv_opencl(data,FILTER)
    end = time()
    print(xC.shape, "cl time:", end-start)
    print(C[0,0,:])
    print(xC[0,0,:])
    # getimages._imshow(C)