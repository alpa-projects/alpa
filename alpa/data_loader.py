""""Distributed data loaders for loading data into device meshes."""
import collections

from jax.interpreters import pxla
from jax._src.abstract_arrays import ShapedArray
import numpy as np

from alpa.device_mesh import DistributedArray, _shard_array


class DataLoader:

    def __init__(self,
                 arrays,
                 sharding_specs,
                 batch_size,
                 physical_mesh=None,
                 devices=None,
                 shuffle=False,
                 prefetch_size=1):
        self.arrays = arrays
        self.sharding_specs = sharding_specs
        self.batch_size = batch_size
        self.devices = devices
        self.physical_mesh = physical_mesh
        self.shuffle = shuffle
        self.prefetch_size = prefetch_size

        # Cache meta info
        self.avals = [
            ShapedArray((self.batch_size,) + a.shape[1:], a.dtype)
            for a in self.arrays
        ]
        self.indices = [
            tuple(spec.indices(aval.shape))
            for spec, aval in zip(self.sharding_specs, self.avals)
        ]
        self.indices_shape = [
            (len(index), len(index[0])) for index in self.indices
        ]
        self.indices_flatten = [
            tuple(x.__reduce__()[1]
                  for x in np.ravel(index))
            for index in self.indices
        ]

        # Iterator status
        self.pt = 0
        self.queue = collections.deque()
        self.perm = None

        if self.devices is not None:
            self.shard_func = self.shard_to_local_devices
        if self.physical_mesh is not None:
            self.shard_func = self.shard_to_physical_mesh

    def shard_to_local_devices(self, batch, aval, sharding_spec, indices):
        shards = [batch[indices[k]] for k in range(len(self.devices))]
        buffers = pxla.device_put(shards, self.devices)
        return pxla._ShardedDeviceArray(aval, sharding_spec, buffers)

    def shard_to_physical_mesh(self, batch, aval, sharding_spec, indices):
        buffers = _shard_array(batch, self.physical_mesh, indices)
        return DistributedArray(self.physical_mesh, aval, sharding_spec,
                                buffers, indices)

    def enqueue(self, num_batches):
        batch_size = self.batch_size

        for i in range(num_batches):
            if self.pt + batch_size < len(self.arrays[0]):
                iter_ret = []
                for j, a in enumerate(self.arrays):
                    if self.shuffle:
                        batch = a[self.perm[self.pt:self.pt + batch_size]]
                    else:
                        batch = a[self.pt:self.pt + batch_size]

                    iter_ret.append(
                        self.shard_func(batch, self.avals[j],
                                        self.sharding_specs[j],
                                        self.indices[j]))

                self.queue.append(iter_ret)

                self.pt += batch_size

    def __iter__(self):
        self.pt = 0
        if self.shuffle:
            self.perm = np.random.permutation(len(self.arrays[0]))

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
