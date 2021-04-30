import numpy as np
import cupy as cp

from util import benchmark_func

def test_matmul():
    N = M = 1024
    K = 1 << 20

    a = cp.empty((N, K), dtype=cp.float32)
    b = cp.empty((K, M), dtype=cp.float32)
    cp.cuda.Device(0).synchronize()

    def matmul():
        c = a @ b
        cp.cuda.Device(0).synchronize()

    costs = benchmark_func(matmul) * 1000
    print("Mean Cost: %.3f ms (std: %.3f ms)" % (np.mean(costs), np.std(costs)))


if __name__ == "__main__":
    test_matmul()

