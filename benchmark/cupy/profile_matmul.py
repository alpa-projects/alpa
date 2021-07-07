import time

import cupy as cp


def benchmark(n, k, m):
    warmup = 2
    number = 100

    a = cp.ones((n, k), "float32")
    b = cp.ones((k, m), "float32")
    for i in range(warmup):
        c = a @ b

    cp.cuda.Device(0).synchronize()
    tic = time.time()
    for i in range(number):
        cp.dot(a, b, c)
    cp.cuda.Device(0).synchronize()
    toc = time.time()

    complexity = n * k * m
    cost = (toc - tic) / number
    shape = (n, k, m)

    print(f"{shape}, {complexity}, {cost:3f}")

benchmark(4096, 2304, 9216)
benchmark(4096, 9216, 2304)

benchmark(8192, 2304, 4608)
benchmark(8192, 4608, 2304)

