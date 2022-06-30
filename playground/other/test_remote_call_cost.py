import time

from alpa.device_mesh import Mesh
import numpy as np
import ray

ray.init(address="auto")
worker = ray.remote(num_gpus=1)(Worker).remote()

latencies = []
for i in range(1000):
    tic = time.time()
    ray.get(worker.check_alive.remote())
    latency = time.time() - tic
    print(f"{i}, latency: {latency * 1e3:.2f} ms")
