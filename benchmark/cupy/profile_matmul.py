"""Profile peak TFLOPS on matrix multiplications."""
import time
import cupy as cp

def benchmark(n, k, m, dtype, init_method="ones"):
    warmup = 5
    number = 50

    if init_method == "zeros":
        a = cp.zeros((n, k), dtype)
        b = cp.zeros((k, m), dtype)
    elif init_method == "full":
        a = cp.full((n, k), 1e-7, dtype)
        b = cp.full((k, m), 1e-7, dtype)
    elif init_method == "nans":
        a = cp.full((n, k), cp.nan, dtype)
        b = cp.full((k, m), cp.nan, dtype)
    elif init_method == "ones":
        a = cp.ones((n, k), dtype)
        b = cp.ones((k, m), dtype)
    elif init_method == "ones+randn":
        a = cp.ones((n, k), dtype)
        b = cp.ones((k, m), dtype)
        ratio = 2
        a[0:n//ratio, :] = cp.random.randn(n//ratio, k).astype(dtype)
        b[0:k//ratio, :] = cp.random.randn(k//ratio, m).astype(dtype)
    elif init_method == "randn":
        a = cp.random.randn(n, k).astype(dtype)
        b = cp.random.randn(k, m).astype(dtype)
    elif init_method == "uniform":
        a = cp.random.uniform(-1, 1, (n, k)).astype(dtype)
        b = cp.random.uniform(-1, 1, (k, m)).astype(dtype)
    elif init_method == "uniform+":
        a = cp.random.uniform(0, 1, (n, k)).astype(dtype)
        b = cp.random.uniform(0, 1, (k, m)).astype(dtype)
    else:
        raise ValueError(f"Invalid method: {init_method}")
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

    print(f"shape: {shape}, init_method: {init_method:>8}, "
          f"TFLOP: {total_flops / 1e12:.2f}, "
          f"cost: {cost:3f}, "
          f"TFLOPS : {total_flops / cost / 1e12:.2f}""")


for n in [8192]:
    for init_method in ["nans", "full", "zeros", "ones",
                        "randn", "uniform", "uniform+", "ones+randn"]:
        benchmark(n, n, n, "float16", init_method)
