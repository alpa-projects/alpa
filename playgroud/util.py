import time

import numpy as np

def benchmark_func(func, warmup=1, repeat=3):
    for i in range(warmup):
        func()

    costs = []
    for i in range(repeat):
        tic = time.time()
        func()
        costs.append(time.time() - tic)

    return np.array(costs)

