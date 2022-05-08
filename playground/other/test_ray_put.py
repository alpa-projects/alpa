import time

import jax
import ray
import numpy as np

MB = 1024**2
GB = 1024**3


def benchmark_ray(x):
    array = np.ones((x,), dtype=np.float32)
    warmup = 0
    number = 1

    # warm up
    for i in range(warmup):
        ray.put(array)

    # benchmark
    tic = time.time()
    for i in range(number):
        ray.put(array)
    cost = time.time() - tic

    size = np.prod(array.shape) * array.dtype.itemsize
    bandwidth = size / (cost / number)
    print(f"size: {size/MB:.2f} MB, bandwidth: {bandwidth/MB:.2f} MB")


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


for i in [1, 64, 128, 512, 1024]:
    benchmark_ray(i * MB)
for i in [1, 64, 128, 512, 1024]:
    benchmark_ray(i * MB)
for i in [1, 64, 128, 512, 1024]:
    benchmark_ray(i * MB)

#for i in range(10):
#    benchmark_jax_put(8192 * 28 * 28 * 1)
