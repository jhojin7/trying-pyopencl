import pyopencl as cl
import numpy as np
import torch

matX = torch.rand(32*32)
matY = torch.rand(32*32)
matZ = torch.zeros(32*32)

for i in range(32):
    for j in range(32):
        matZ[32*i+j] = matX[32*i+j]\
            + matY[32*i+j]

print(matZ == matX+matY)
