from matrix_multiplication import init_mat, matmul_seq
from timeit import timeit
import pyopencl as cl
import numpy as np
import torch
import json

A,B,C,C2 = init_mat(32,32)
FILTER = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
# print(A,B,C,C2,FILTER,sep="\n")
STRIDE = 1

SAMPLE_BIN = "./cifar10/sample_bin_data.json"
IMG = "./cifar10/__DEER_muntjac_s_001000.png"

imgsdata = None
with open(SAMPLE_BIN,"r") as f:
    imgsdata = json.load(f) 
    f.close()

imgdata = imgsdata[IMG]
filter = FILTER
# print(type(imgdata),type(imgdata[0][0][0]))

# def matmul_1Darray_seq(img)

print(imgdata[0][0])
# def conv(img,img_dim,
#     filt,filt_dim):
#     matmul_seq(A,B)
