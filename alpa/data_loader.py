""""Distributed data loaders for loading data into device meshes."""
import collections
import itertools

import jax
from jax.interpreters import pxla
from jax._src.abstract_arrays import ShapedArray
import numpy as np

from alpa.device_mesh import DistributedArray, _shard_array


class DataLoader:

    def __init__(self,
                 input_iter,
                 sharding_specs,
                 devices=None,
                 physical_mesh=None,
                 prefetch_size=1):
        self.input_iter = input_iter
        self.sharding_specs = sharding_specs
        self.devices = devices
        self.physical_mesh = physical_mesh
        self.prefetch_size = prefetch_size

        self.queue = collections.deque()

        self.first_iter = True

        if self.devices is not None:
            self.shard_func = self.shard_to_local_devices
        elif self.physical_mesh is not None:
            self.shard_func = self.shard_to_physical_mesh
        else:
            self.devices = jax.local_devices()
            self.shard_func = self.shard_to_local_devices

    def shard_to_local_devices(self, batch, aval, sharding_spec, indices):
        shards = [batch[indices[k]] for k in range(len(self.devices))]
        buffers = pxla.device_put(shards, self.devices)
        return pxla._ShardedDeviceArray(aval, sharding_spec, buffers)

    def shard_to_physical_mesh(self, batch, aval, sharding_spec, indices):
        buffers = _shard_array(batch, self.physical_mesh, indices)
        return DistributedArray(self.physical_mesh, aval, sharding_spec,
                                buffers, indices)

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

            new_args = []
            for j, a in enumerate(flatten_args):
                new_args.append(
                    self.shard_func(a, self.avals[j],
                                    self.sharding_specs[j],
                                    self.indices[j]))
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
