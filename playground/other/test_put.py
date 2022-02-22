import time

import jax
import ray
import numpy as np


def benchmark_ray(x):
    array = np.ones((x,), dtype=np.float32)

    # warm up
    for i in range(2):
        ray.put(array)

    # benchmark
    tic = time.time()
    for i in range(10):
        ray.put(array)
    print(f"size: {x}, time: {time.time() - tic:.2f}")


def benchmark_jax_put(x):
    batch = np.ones((x,), dtype=np.float32)

    # warm up
    for i in range(2):
        tmp = jax.device_put(batch)
    tmp.block_until_ready()

    # benchmark
    tic = time.time()
    y = [None] * 10
    for i in range(10):
        y[i] = jax.device_put(batch)
        #y[i] = None
        #y[i].block_until_ready()
    print(f"size: {x}, time: {time.time() - tic:.2f}")


for i in range(10):
    benchmark_ray(8192 * 28 * 28 * 1)

for i in range(10):
    benchmark_jax_put(8192 * 28 * 28 * 1)
