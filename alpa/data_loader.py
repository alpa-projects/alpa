""""Distributed data loaders for loading data into device meshes."""
import collections
import itertools

import jax
from jax.interpreters import pxla
from jax._src.abstract_arrays import ShapedArray
import numpy as np

from alpa.device_mesh import PhysicalDeviceMesh, DistributedArray, _shard_array


class DataLoader:

    def __init__(self,
                 input_iter,
                 sharding_specs,
                 physical_mesh=None,
                 prefetch_size=1):
        self.input_iter = input_iter
        self.sharding_specs = sharding_specs
        self.prefetch_size = prefetch_size

        if physical_mesh is None:
            self.physical_mesh = PhysicalDeviceMesh()
        else:
            self.physical_mesh = physical_mesh

        self.queue = collections.deque()
        self.first_iter = True

    def enqueue(self, num_batches):
        for batch in itertools.islice(self.input_iter, num_batches):
            flatten_args, tree = jax.tree_flatten(batch)

            # Cache meta info
            if self.first_iter:
                self.first_iter = False
                self.avals = [
                    ShapedArray(a.shape, a.dtype) for a in flatten_args
                ]
                self.indices = [
                    tuple(spec.indices(aval.shape))
                    for spec, aval in zip(self.sharding_specs, self.avals)
                ]

            new_args = self.physical_mesh.shard_args_to_arrays(
                self.avals, self.indices, self.sharding_specs, flatten_args)
            self.queue.append(jax.tree_unflatten(tree, new_args))

    def __iter__(self):
        if self.prefetch_size:
            self.enqueue(self.prefetch_size)
            while self.queue:
                yield self.queue.popleft()
                self.enqueue(1)
        else:
            while True:
                self.enqueue(1)
                if self.queue:
                    yield self.queue.popleft()
                else:
                    break
