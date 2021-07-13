import time

import cupy as cp

def benchmark(n, k, m, dtype):
    warmup = 2
    number = 100

    a = cp.ones((n, k), dtype)
    b = cp.ones((k, m), dtype)
    for i in range(warmup):
        c = a @ b

    cp.cuda.Device(0).synchronize()
    tic = time.time()
    for i in range(number):
        cp.dot(a, b, c)
    cp.cuda.Device(0).synchronize()
    toc = time.time()

    total_flops = 2 * n * k * m
    cost = (toc - tic) / number
    shape = (n, k, m, dtype)

    print(f"shape: {shape}, TFLOP: {total_flops / 1e12:.2f}, "
          f"cost: {cost:3f}, "
          f"TFLOPS : {total_flops / cost / 1e12:.2f}""")

benchmark(1024, 1024, 1024, "float32")
benchmark(4096, 4096, 4096, "float32")
benchmark(8192, 8192, 8192, "float32")
benchmark(4096, 4096, 4096, "float16")

#benchmark(4096, 2304, 9216, "float32")
#benchmark(4096, 9216, 2304, "float32")
#
#benchmark(8192, 2304, 4608, "float32")
#benchmark(8192, 4608, 2304, "float32")

