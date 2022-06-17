import numpy as np
from timeit import timeit
import asyncio
setup, stmt = "", ""

n = 10**4
mat = np.random.rand(n,n)
print(mat.shape)
async def rowsum(mat, i):
    row_s = 0
    for j in range(n):
        row_s += mat[i,j]
    return row_s
    
async def matsum(mat):
    s = 0
    tasks = []
    for i in range(n):
        task = asyncio.create_task(rowsum(mat,i))
        tasks.append(task)
    s = sum(await asyncio.gather(*tasks))
    print(s)
    return s

t = timeit("print(sum([sum(mat[x,:]) for x in range(n)]))", setup, globals=globals(), number=1)
t1 = timeit("asyncio.run(matsum(mat))", setup, globals=globals(), number=1)
t2 = timeit("print(mat.sum())", setup, globals=globals(), number=1)
t3 = timeit("""s = 0
for i in range(n):
    for j in range(n):
        s += mat[i,j]
print(s)
""", setup, globals=globals(), number=1)
print(t,t1, t2, t3)

