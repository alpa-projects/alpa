# check gpu devices
import os

import jax.numpy as jnp
import ray


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
ray.init(num_gpus=2, num_cpus=4)


@ray.remote(num_gpus=1, num_cpus=2)
class Runner:
    def __init__(self, name):
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        self.name = name
        self.a = None
        self.b = None

    def compute(self):
        print(type(self.a))
        print(type(self.b))
        c = jnp.matmul(self.a, self.b)
        print(type(c))
        return c

    def set(self, refs):
        arrays = ray.get(refs)
        print(arrays)
        # a = ray.get(a_ref)
        # print(a)
        # print(type(a))
        self.a = jnp.asarray(arrays[0])
        # b = ray.get(b_ref)
        # print(b)
        # print(type(b))
        self.b = jnp.asarray(arrays[1])


workers = []
workers.append(Runner.remote(name="0"))
workers.append(Runner.remote(name="1"))

a = jnp.ones([3, 4])
b = jnp.ones([4, 5])
a_ref = ray.put(a)
b_ref = ray.put(b)
worker = workers[0]
worker.set.remote([a_ref, b_ref])
c_ref = worker.compute.remote()
c_result = ray.get(c_ref)

worker = workers[1]
worker.set.remote([a_ref, b_ref])
c_ref = worker.compute.remote()
c_result = ray.get(c_ref)
print(c_result)
