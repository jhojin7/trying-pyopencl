from turtle import pos
import pyopencl as cl
import numpy as np
from timeit import timeit

def init_mat(x,y):
    """Initialize Matrices\n
    A * B = C, np.matmul(A,B,C2)"""
    A,B,C,C2 = None,None,None,None
    # need .astype(np.float32) for Clang
    A = np.random.rand(x,y).astype(np.float32)
    B = np.random.rand(x,y).astype(np.float32)
    C = np.zeros((x,y)).astype(np.float32)
    C2 = np.zeros((x,y)).astype(np.float32)
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

class Context:
    """Initialize OpenCL context"""
    def __init__(self):
        self.platform = cl.get_platforms()[0] 
        self.device = self.platform.get_devices()[0]
        self.ctx = cl.Context([self.device])
        # ctx = cl.create_some_context()
        # cl.enable_debugging(self.platform)
        self.queue = cl.CommandQueue(self.ctx)

    def print_info(self):
        print(self.platform)
        print(self.device)
        print(self.ctx)
    
    def build_matmul(self):
        # program object
        program = cl.Program(self.ctx, r"""
        __kernel void matmul(
            __global int* A, __global int* B, __global int* C,
            int ROW_A, int COL_A, int COL_B) {
            int i= get_global_id(1);
            int j= get_global_id(0);
            int k;
            C[i * COL_B + j] = 0;
            for (k = 0; k < COL_A; k++){
                C[i * COL_B + j] += A[i * COL_A + k] * B[k * COL_B + j];
                //printf("%d %d %d\n",C[i * COL_B + j],A[i * COL_A + k], B[k * COL_B + j]);
            }
        }
        """).build()

        # kernel object
        return program.matmul

def matmul_opencl(A,B,_context:Context)->bool:
    ctx = _context.ctx
    queue = _context.queue

    # initialize buffers
    mf = cl.mem_flags
    buf_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
            size=0, hostbuf=A) # ro, copy host memory
    buf_B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
            size=0, hostbuf=B) # ro, copy host memory
    buf_C = cl.Buffer(ctx, mf.WRITE_ONLY, A.nbytes) # wo

    # program object
    prg = cl.Program(ctx, r"""
    __kernel void matmul(
        __global float* A, __global float* B, __global float* C,
        int ROW_A, int COL_A, int COL_B) {
        int i= get_global_id(1);
        int j= get_global_id(0);
        int k;
        C[i * COL_B + j] = 0.0f;
        for (k = 0; k < COL_A; k++){
            C[i * COL_B + j] += A[i * COL_A + k] * B[k * COL_B + j];
        }
    }
    """).build()

    # kernel object
    kernel = prg.matmul
    # make and set arguments
    s = np.uint32(A.shape[0])
    kernel(queue, A.shape, None, buf_A, buf_B, buf_C,s,s,s) # parse Clang integer

    C = np.empty_like(A)
    cl.enqueue_copy(queue, C, buf_C)
    # return np.array_equal(C,np.matmul(A,B))
    return (C.all() == np.matmul(A,B).all())

if __name__ == "__main__":
############################################################
    dims = (3,3)# set matrix dims
    repeat = 32*32*3
############################################################
    times1, times2 = [], []
    ctx = Context()
    setup = f"A,B,C,C2 = init_mat({dims[0]},{dims[1]})" 
    for _ in range(repeat):
        check = None
        time = timeit(setup=setup,
                stmt="global check; check = matmul_seq(A,B,C,C2)",
                globals=globals(),number=1)
        times1.append(time)
        print("Sequential:\t{:>15.10}".format(time), "Check:",check)

    # for _ in range(repeat):
        check = None
        time = timeit(setup=setup,
                stmt="global check;check = matmul_opencl(A,B,ctx)",
                globals=globals(),number=1)
        times2.append(time)
        print("Parallel:\t{:>15.10}".format(time), "Check:",check)
    
    # find avg and compare runtime
    avg1 = sum(times1)/len(times1)
    avg2 = sum(times2)/len(times2)
    print("dims:{}, repeat:{}, Sequential Avg:{:.10}, Parallel Avg:{:.10}, diff:{:.10}" 
        .format(dims, repeat, avg1, avg2, abs(avg1-avg2)))