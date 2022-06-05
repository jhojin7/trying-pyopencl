# https://documen.tician.de/pyopencl/
import numpy as np
import pyopencl as cl
from time import time

def opencl(a_np, b_np):
    # https://stackoverflow.com/a/26395800/3413574
    platform = cl.get_platforms()[0] 
    device = platform.get_devices()[0]
    ctx = cl.Context([device])

    # ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    prg = cl.Program(ctx, """
    __kernel void sum(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
    int gid = get_global_id(0);
    res_g[gid] = a_g[gid] + b_g[gid];
    }
    """).build()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    # knl = prg.sum  # Use this Kernel object for repeated calls
    # knl(queue, a_np.shape, None, a_g, b_g, res_g)
    prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)
    return res_np

def seq(a_np, b_np):
    # return a_np+b_np
    res_np = np.zeros(len(a_np))
    for i in range(len(a_np)):
        res_np[i] = a_np[i] + b_np[i]
    return res_np

if __name__=="__main__":
    N = 10**7
    a_np = np.random.rand(N).astype(np.float32)
    b_np = np.random.rand(N).astype(np.float32)

    # OpenCL
    # Check on CPU with Numpy:
    start_t = time()
    res_np = opencl(a_np, b_np)
    end_t = time()
    print("OpenCL: ",end_t-start_t)
    print(res_np - (a_np + b_np))

    # Sequential
    start_t = time()
    res_np = seq(a_np, b_np)
    end_t = time()
    print("Sequential: ",end_t-start_t)
    print(res_np - (a_np + b_np))

    # print(np.linalg.norm(res_np - (a_np + b_np)))
    assert np.allclose(res_np, a_np + b_np)

"""
OpenCL:  0.15827703475952148
[0. 0. 0. ... 0. 0. 0.]
Sequential:  3.5307586193084717
[0. 0. 0. ... 0. 0. 0.]
"""