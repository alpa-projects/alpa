""""Distributed data loaders for loading data into device meshes."""
import collections
import itertools

import jax
from jax.interpreters import pxla, xla
import numpy as np
import ray

from alpa.device_mesh import LocalPhysicalDeviceMesh, DistributedArray
from alpa.mesh_executable import create_remote_buffer_refs


class DataLoader:
    """A driver-only dataloader that loads data on the driver process and
    sends the data to all workers."""

    def __init__(self,
                 input_iter,
                 sharding_specs,
                 physical_mesh=None,
                 prefetch_size=1):
        self.input_iter = input_iter
        self.sharding_specs = sharding_specs
        self.prefetch_size = prefetch_size

        if physical_mesh is None:
            self.physical_mesh = LocalPhysicalDeviceMesh()
        else:
            self.physical_mesh = physical_mesh

        self.queue = collections.deque()
        self.first_iter = True
        self.avals = None
        self.indices = None

    def enqueue(self, num_batches):
        for batch in itertools.islice(self.input_iter, num_batches):
            flatten_args, tree = jax.tree_flatten(batch)

            # Cache meta info
            if self.first_iter:
                self.first_iter = False
                self.avals = [xla.abstractify(a) for a in flatten_args]
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


def get_num_devices_for_whole_batch(sharding_spec, batch_dim=0):
    """Get the number of devices for a whole batch."""
    num_devices = 1
    for sharding in sharding_spec.sharding:
        if isinstance(sharding, pxla.Chunked):
            num_devices *= np.prod(sharding.chunks)

    for assignment in sharding_spec.mesh_mapping:
        if isinstance(assignment, pxla.Replicated):
            num_devices *= assignment.replicas

    sharding = sharding_spec.sharding[batch_dim]

    num_data_chunk = 1
    if isinstance(sharding, pxla.Chunked):
        num_data_chunk = np.prod(sharding.chunks)

        # Assert the data chunk is mapped to the first dim of device mesh
        for assignment in sharding_spec.mesh_mapping:
            if isinstance(assignment, pxla.ShardedAxis):
                assert assignment.axis == 0
                break

    return num_devices / num_data_chunk


class MeshDriverDataLoader:
    """The driver part of a distributed data loader. The driver part creates distributed
    arrays and sends commands to let workers load the data in parallel."""

    def __init__(self,
                 batch_size,
                 num_samples,
                 input_iter_func,
                 avals,
                 sharding_specs,
                 physical_mesh,
                 prefetch_size=1):
        indices = [
            tuple(np.ravel(spec.indices(aval.shape)))
            for spec, aval in zip(sharding_specs, avals)
        ]

        self.uuid = next_mesh_data_loader_uuid()
        self.physical_mesh = physical_mesh

        # Create output DisributedArray
        self.output_uuids = []
        self.output_arrays = []
        for i in range(len(avals)):
            buf_refs, buf_uuids = create_remote_buffer_refs(physical_mesh)
            self.output_uuids.append(buf_uuids)
            self.output_arrays.append(
                DistributedArray(physical_mesh, avals[i], sharding_specs[i],
                                 buf_refs))

        # Create worker part data loaders
        self.worker_data_loaders = []
        self.num_batches = num_samples // batch_size

        for i in range(physical_mesh.num_hosts):
            host_output_uuids = []
            host_indices = []
            for j in range(len(avals)):
                batch_size = avals[j].shape[0]
                num_devices_for_one_batch = get_num_devices_for_whole_batch(
                    sharding_specs[j])
                num_hosts_for_one_batch = max(
                    1, num_devices_for_one_batch /
                    physical_mesh.num_devices_per_host)
                assert float(num_hosts_for_one_batch).is_integer(
                ), f"{num_hosts_for_one_batch}"
                num_hosts_for_one_batch = int(num_hosts_for_one_batch)

                batch_size_per_host = batch_size / (physical_mesh.num_hosts /
                                                    num_hosts_for_one_batch)
                assert batch_size_per_host.is_integer()
                batch_size_per_host = int(batch_size_per_host)

                num_samples_per_host = self.num_batches * batch_size_per_host

                start = (i // num_hosts_for_one_batch) * num_samples_per_host
                end = (
                    (i // num_hosts_for_one_batch) + 1) * num_samples_per_host

                host_output_uuids.append(
                    self.output_uuids[j][i * physical_mesh.num_devices_per_host:
                                         (i + 1) *
                                         physical_mesh.num_devices_per_host])
                host_indices.append([])
                for k in range(physical_mesh.num_devices_per_host):
                    device_id = i * physical_mesh.num_devices_per_host + k
                    tmp_indices = list(indices[j][device_id])
                    if tmp_indices[0].start is not None:
                        tmp_indices[0] = slice(
                            tmp_indices[0].start -
                            i // num_hosts_for_one_batch * batch_size_per_host,
                            tmp_indices[0].stop -
                            i // num_hosts_for_one_batch * batch_size_per_host,
                            tmp_indices[0].step)
                    host_indices[-1].append(tuple(tmp_indices))

            args = (input_iter_func, (start, end, batch_size_per_host),
                    host_output_uuids, host_indices, prefetch_size)
            physical_mesh.workers[i].put_data_loader.remote(self.uuid, *args)

    def __iter__(self):
        # Create the iterators on workers
        for w in self.physical_mesh.workers:
            w.data_loader_iter.remote(self.uuid)

        # Yield next batch
        for _ in range(self.num_batches):
            for w in self.physical_mesh.workers:
                w.data_loader_next.remote(self.uuid)
            for a in self.output_arrays:
                a.flush()
            yield self.output_arrays

    def __del__(self):
        physical_mesh = self.physical_mesh
        if physical_mesh.workers is None or not ray.is_initialized():
            return

        for i in range(physical_mesh.num_hosts):
            physical_mesh.workers[i].delete_data_loader.remote(self.uuid)


class MeshWorkerDataLoader:
    """The worker part of a distributed data loader. The driver part creates distributed
    arrays and sends commands to let workers load the data in parallel."""

    def __init__(self, mesh_host_worker, input_iter_func, input_iter_args,
                 output_uuids, shard_indices, prefetch_size):
        self.input_iter = input_iter_func(*input_iter_args)
        self.output_uuids = output_uuids
        self.shard_indices = shard_indices
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
                buffers = [
                    jax.device_put(x, d) for x, d in zip(shards, self.devices)
                ]
                batch.append(buffers)

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
