""""Distributed data loaders for loading data into device meshes."""
import collections
import itertools

import jax
from jax.interpreters import pxla
from jax._src.abstract_arrays import ShapedArray
import numpy as np
import ray

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


# The global executable and buffer counter.
mesh_data_loader_counter = 0


def next_mesh_data_loader_uuid():
    """Return the next uuid of a mesh data loader."""
    global mesh_data_loader_counter
    mesh_data_loader_counter = (mesh_data_loader_counter + 1) % (1 << 60)
    return mesh_data_loader_counter



class MeshDriverDataLoader:
    def __init__(self,
                 input_iter_func,
                 sharding_specs,
                 physical_mesh=None,
                 prefetch_size=1):
        self.input_iter = input_iter
        self.sharding_specs = sharding_specs
        self.prefetch_size = prefetch_size
        self.physical_mesh = physical_mesh

        self.uuid = next_mesh_data_loader_uuid()

    def __del__(self):
        physical_mesh = self.physical_mesh
        if physical_mesh.workers is None or not ray.is_initialized():
            return

        for i in range(physical_mesh.num_hosts):
            physical_mesh.workers[i].delete_data_loader.remote(self.uuid)


class MeshWorkerDataLoader:
    def __init__(self,
                 mesh_host_worker,
                 input_iter,
                 output_uuids,
                 shard_indices,
                 prefetch_size):
        self.input_iter = input_iter
        self.output_uuids = output_uuids
        self.indices = indices
        self.prefetch_size = prefetch_size

        self.devices = mesh_host_worker.local_devices
        self.buffers = mesh_host_worker.buffers

        # A queue for prefetching
        self.queue = collections.deque()

    def enqueue(self, num_batches):
        for args in itertools.islice(self.input_iter, num_batches):
            batch = []
            for i in range(len(args)):
                shards = [
                    args[i][self.shard_indices[i][k]]
                    for k in range(len(self.devices))
                ]

                batch.append([jax.device_put(x, d) for x, d in zip(shards, self.devices)])

            self.queue.append(batch)

    def pop_left(self):
        batch = self.queue.popleft()
        for i, shards in enumerate(batch):
            for uuid, shard in zip(self.output_uuids[i], shards):
                self.buffers[uuid] = shard

    def __iter__(self):
        if self.prefetch_size:
            self.enqueue(self.prefetch_size)
            while self.queue:
                yield self.pop_left()
                self.enqueue(1)
        else:
            while True:
                self.enqueue(1)
                if self.queue:
                    yield self.pop_left()
                else:
                    break
