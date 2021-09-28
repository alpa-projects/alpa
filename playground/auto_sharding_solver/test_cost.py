import numpy as np

from cluster_env import ClusterEnvironment

def s(*shape):
    return np.prod(shape) * 4

env = ClusterEnvironment(np.ones((8, 1)), [1, 1], [0.02, 0.02], 0)

a = env.all_reduce_cost(s(16, 14, 14, 8192)) + env.all_reduce_cost(s(16, 28, 28, 2048)) + \
    env.all_to_all_cost(s(16, 28, 28, 4096))

print(a)


b = env.all_gather_cost(s(16, 28, 28, 4096)) + env.all_gather_cost(s(1, 1, 4096, 8192))
print(b)


