"""Cross mesh resharding for pipeline parallelism."""
from itertools import chain
import logging
import math
from typing import List, Any

import numpy as np
from jax.interpreters import pxla
import ray

import alpa.collective as col
from alpa.device_mesh import (DistributedArray, ReshardingAllGatherSpec,
                              ReshardingRecvSpec, ReshardingTileSpec, ReshardingBroadcastSpec)
from alpa.mesh_executable import RemoteBufferRef
from alpa.global_env import global_config
from alpa.pipeline_parallel.computation import XlaShardedPipelineComputation
from alpa.pipeline_parallel.resharding_tensor import (VirtualDistributedArray,
                                                      TileSlice,
                                                      unflatten_tile_index)
from alpa.util import OrderedSet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

resharding_task_counter = 0


def next_resharding_task_uuid():
    """Generate the next resharding task uuid."""
    global resharding_task_counter
    resharding_task_counter = (resharding_task_counter + 1) % (1 << 60)
    return resharding_task_counter


def _suffix_prod(ary):
    out = [1]
    for e in reversed(ary):
        out.append(out[-1] * e)
    return list(reversed(out))


def _get_chunk_value(spec):
    if isinstance(spec, pxla.Chunked):
        return int(np.prod(spec.chunks))
    return 1


def _add_chunk(spec, chunk):
    if isinstance(spec, pxla.Chunked):
        return pxla.Chunked(spec.chunks + [chunk])
    return pxla.Chunked([chunk])


def _get_chunk_prefixsum(shardings):
    chunk_cnt = 0
    chunk_prefixsum = []
    for dim_sharding in shardings:
        chunk_prefixsum.append(chunk_cnt)
        if isinstance(dim_sharding, pxla.Chunked):
            chunk_cnt += len(dim_sharding.chunks)
    return chunk_prefixsum


def _get_mesh_mapping(shardings, init_mesh_mapping, squeezed_mesh_mapping):
    chunk_prefixsum = _get_chunk_prefixsum(shardings)
    mesh_mapping = []
    for mesh_dim, mapping in enumerate(squeezed_mesh_mapping):
        prev_mapping = init_mesh_mapping[mesh_dim]
        if mapping is None:
            mesh_mapping.append(prev_mapping)
            continue
        replicas = 1
        if isinstance(prev_mapping, pxla.Replicated):
            replicas = prev_mapping.replicas
        for (tensor_dim, chunk_idx) in mapping:
            mesh_mapping.append(
                pxla.ShardedAxis(chunk_prefixsum[tensor_dim] + chunk_idx))
            replicas //= shardings[tensor_dim].chunks[chunk_idx]
        if replicas > 1:
            mesh_mapping.append(pxla.Replicated(replicas))
    return mesh_mapping


def _reduce_chunk(spec, tensor_dim, remove_last_chunks):
    """
    Rewrite the sharding spec on given dimension by removing the last n chunks
    on that dimension.
    """

    def add_or_merge(mesh_mapping, replicas):
        if (len(mesh_mapping) > 0 and
                isinstance(mesh_mapping[-1], pxla.Replicated)):
            replicas *= mesh_mapping[-1].replicas
            mesh_mapping[-1] = pxla.Replicated(replicas)
        else:
            mesh_mapping.append(pxla.Replicated(replicas))

    # get new chunks
    cur_chunks = spec.sharding[tensor_dim].chunks
    new_chunks = cur_chunks[:-1 * remove_last_chunks]
    new_chunks = (pxla.Chunked(new_chunks)
                  if len(new_chunks) > 0 else pxla.NoSharding())
    shardings = list(spec.sharding)
    shardings[tensor_dim] = new_chunks
    # axis->(tensor dim, chunk idx)
    chunk_axis_to_tensor_dim = []
    for t_dim, dim_spec in enumerate(spec.sharding):
        if isinstance(dim_spec, pxla.Chunked):
            for chunk_idx in range(len(dim_spec.chunks)):
                chunk_axis_to_tensor_dim.append((t_dim, chunk_idx))
    # (tensor dim, chunk idx)->new axis
    new_prefix_sum = _get_chunk_prefixsum(shardings)
    remove_idx = len(spec.sharding[tensor_dim].chunks) - remove_last_chunks
    mesh_mapping = []
    for mapping in spec.mesh_mapping:
        if isinstance(mapping, pxla.Replicated):
            add_or_merge(mesh_mapping, mapping.replicas)
            continue
        assert isinstance(mapping, pxla.ShardedAxis)
        t_dim, chunk_idx = chunk_axis_to_tensor_dim[mapping.axis]
        if t_dim == tensor_dim and chunk_idx >= remove_idx:
            replicas = spec.sharding[t_dim].chunks[chunk_idx]
            add_or_merge(mesh_mapping, replicas)
        else:
            axis = new_prefix_sum[t_dim] + chunk_idx
            mesh_mapping.append(pxla.ShardedAxis(axis))
    # new spec
    new_spec = pxla.ShardingSpec(shardings, mesh_mapping)
    return new_spec


def _allgather_logical_mesh_from_spec(sharding_spec):
    chunks = list(
        chain(*[
            spec.chunks
            for spec in sharding_spec.sharding
            if isinstance(spec, pxla.Chunked)
        ]))
    return tuple(
        chunks[m.axis] if isinstance(m, pxla.ShardedAxis) else m.replicas
        for m in sharding_spec.mesh_mapping)


def _allgather_dim_offset(length, offset, chunks, local_allgather_idx):
    """
    given length of the tensor dim, offset of each mesh dim and corresponding
    chunk values(as well as mesh length), return offset of the allgather
    """
    assert len(offset) == len(chunks)
    # TODO(yonghao): step should run only once for a dim
    step = _suffix_prod(chunks)
    assert length % step[0] == 0
    atom_len = length // step[0]
    allgather_offset = 0
    for chunk_idx in range(local_allgather_idx, len(chunks)):
        allgather_offset += step[chunk_idx + 1] * offset[chunk_idx]
    return atom_len * allgather_offset


def _signle_tensor_dim_allgather_groups(mesh_shape, allgather_dims):
    """given an N-dim mesh shape, group all receivers into allgather groups."""

    def iter_tool(depth, length, steps):
        if len(steps) == 0:
            yield 0
            return
        local_delta = [1] * (length[depth] - 1) + [-1 * (length[depth] - 1)]
        offset = 0
        for delta in local_delta:
            if depth == len(steps) - 1:
                yield offset
            else:
                sub_iter = iter_tool(depth + 1, length, steps)
                for o in sub_iter:
                    yield offset + o
            offset += steps[depth] * delta

    steps = _suffix_prod(mesh_shape)
    offset_steps = []
    offset_length = []
    group_steps = []
    group_length = []
    for d, l in enumerate(mesh_shape):
        if d in allgather_dims:
            offset_steps.append(steps[d + 1])
            offset_length.append(l)
        else:
            group_steps.append(steps[d + 1])
            group_length.append(l)
    group_iter = iter_tool(0, group_length, group_steps)
    offsets = list(iter_tool(0, offset_length, offset_steps))
    groups = []
    for group_idx in group_iter:
        receivers = []
        for offset in offsets:
            receivers.append(offset + group_idx)
        groups.append(receivers)
    return groups


class _AllgatherSpec:

    def __init__(self, allgather_chunk_indices):
        """allgather chunk indices: a list of (tensor_dim, chunk_index, _)"""
        self._allgather_chunk_num = {}
        allgather_chunk_indices = sorted(allgather_chunk_indices)
        cur_tensor_dim = -1
        cur_chunk_idx = -1
        for (tensor_dim, chunk_idx, _) in allgather_chunk_indices:
            if tensor_dim == cur_tensor_dim:
                assert chunk_idx == cur_chunk_idx + 1
                self._allgather_chunk_num[tensor_dim] += 1
            else:
                cur_tensor_dim = tensor_dim
                assert tensor_dim not in self._allgather_chunk_num
                self._allgather_chunk_num[tensor_dim] = 1
            cur_chunk_idx = chunk_idx

    def mapped_mesh_dim(self, tensor_dim, sharding_spec):
        """Get the mapped mesh dim of a tensor dim's allgather chunks."""
        shardings = sharding_spec.sharding
        prefix_sum = _get_chunk_prefixsum(shardings)

        allgather_chunks = self._allgather_chunk_num[tensor_dim]
        start_idx = len(shardings[tensor_dim].chunks) - allgather_chunks
        start_axis = prefix_sum[tensor_dim] + start_idx
        mesh_dims = [None] * allgather_chunks
        for d, mapping in enumerate(sharding_spec.mesh_mapping):
            if isinstance(mapping, pxla.ShardedAxis):
                idx = mapping.axis - start_axis
                if 0 <= idx < allgather_chunks:
                    mesh_dims[idx] = d
        return mesh_dims

    def start_idx(self, tensor_dim, sharding_spec):
        dim_sharding = sharding_spec.sharding[tensor_dim]
        return len(dim_sharding.chunks) - self._allgather_chunk_num[tensor_dim]

    @property
    def allgather_dims(self):
        return list(self._allgather_chunk_num.keys())

    def post_allgather(self, tensor_dim):
        ret = _AllgatherSpec([])
        ret._allgather_chunk_num = dict(self._allgather_chunk_num)
        ret._allgather_chunk_num.pop(tensor_dim)
        return ret


class ReshardingTask:
    """
    A task that addresses cross-mesh resharding between two meshes.

    Args:
        task_spec (ReshardingTaskSpec): the task spec of this task.
        collective_group (CollectiveGroup): the collective group information.
        src_mesh (PhysicalMesh): the source mesh to send.
        dst_mesh (PhysicalMesh): the destination mesh to receive.
    """

    def __init__(self, task_spec, collective_group, src_mesh, dst_mesh):
        self.task_spec: ReshardingTaskSpec = task_spec
        self.collective_group = collective_group
        self.src_mesh = src_mesh
        self.dst_mesh = dst_mesh

    @property
    def is_local_allgather_task(self):
        """If this task involves a post scatter-allgather task."""
        return self.task_spec.strategy.is_local_allgather


class EagerReshardingTask(ReshardingTask):
    """An eager resharding task.

    It does not put task info into remote workers. Instead, it provides
    a do() interface to execute the task immediately.
    """

    def do(self, src_array):
        """According to the task_spec, launch send/recv operations eagerly.

        Used in centralized distributed runtime.

        Args:
            src_array (DistributedArray): the source array to be resharded.
        """
        if src_array.device_mesh != self.src_mesh:
            raise RuntimeError(f"The src array locates on a different "
                               f"mesh `{src_array.device_mesh}` than "
                               f"self.src_mesh `{self.src_mesh}`.")

        bufs: List[Any] = [None] * len(self.task_spec.dst_indices)
        device_str_to_buf_map = {}
        for i, (dst_tile, src_tiles, indices_in_dst_tiles) in enumerate(
                self.task_spec.dst_tile_to_src_tiles_map):
            # Loop over each dst tile for this shard
            s = self.task_spec.strategy[i]
            # strategy is len(dst_tile.device_strs) by len(src_tiles)
            for replica_index, receiver in enumerate(
                    dst_tile.replica_device_strs):
                # loop over this replica (hence a specific destination gpu device)
                senders = [
                    s[replica_index][src_tile_index]
                    for src_tile_index, src_tile in enumerate(src_tiles)
                ]
                device_str_to_buf_map[
                    receiver] = self.same_destination_group_send_recv(
                        src_array, senders, src_tiles, dst_tile,
                        indices_in_dst_tiles, receiver)
        # Assemble the buffer based on the order present in indices
        for i, device_str in enumerate(
                self.task_spec.dst.device_mesh.device_strs):
            # for each replica
            bufs[self.task_spec.dst.device_str_to_flat_index[
                device_str]] = device_str_to_buf_map[device_str]

        # Now construct the distributed array
        dst_array = DistributedArray(self.dst_mesh, src_array.aval,
                                     self.task_spec.dst_sharding_spec, bufs,
                                     self.task_spec.dst_indices)
        return dst_array

    def same_destination_group_send_recv(self, src_array, senders, src_tiles,
                                         dst_tile, indices_in_dst_tiles,
                                         receiver):
        """P2P Communication accounting for multiple senders and one receiver (a destination tile)."""
        receiver_host_id = self.collective_group.device_str_to_host_id_map[
            receiver]
        receiver_device_id = self.collective_group.device_str_to_device_id_map[
            receiver]
        receiver_worker = self.collective_group.device_str_to_mesh_worker_map[
            receiver]
        result_buf = RemoteBufferRef(self.dst_mesh, receiver_host_id,
                                     receiver_device_id)
        # Put an empty buffer first.
        ray.get(
            receiver_worker.put_non_zero_buffer.remote(result_buf.uuid,
                                                       result_buf.device_id,
                                                       dst_tile.tile_shape,
                                                       result_buf.dtype))
        receiver_rank, receiver_gpu_idx = self.collective_group.device_str_to_rank_map[
            receiver]
        for i, sender in enumerate(senders):
            # send is a device_str in src_mesh
            # we need to find out its mesh_worker, and the corresponded sender remotebuf (uuid-indexed).
            sender_buf = src_array.remote_buffers[
                self.task_spec.src.device_str_to_flat_index[sender]]
            sender_worker = self.collective_group.device_str_to_mesh_worker_map[
                sender]
            # assert sender_buf.device_id == i
            sender_rank, sender_gpu_idx = self.collective_group.device_str_to_rank_map[
                sender]
            # launch NCCL send/recv
            tile = src_tiles[i]
            indices_in_dst_tile = indices_in_dst_tiles[i]
            send_done_ref = sender_worker.send_tile.remote(
                sender_buf.uuid, tile.offset, receiver_rank, receiver_gpu_idx,
                self.collective_group.group_name)
            recv_done_ref = receiver_worker.recv_tile.remote(
                result_buf.uuid, result_buf.device_id, indices_in_dst_tile,
                sender_rank, sender_gpu_idx, self.collective_group.group_name)
            ray.get([send_done_ref, recv_done_ref])
        return result_buf


class SymbolicReshardingTask(ReshardingTask):
    """A symbolic resharding task that puts task info in remote workers."""

    def __init__(self, task_spec, collective_group, src_mesh, dst_mesh):
        super().__init__(task_spec, collective_group, src_mesh, dst_mesh)
        # Dict of worker -> ((offset, rank, gpu index))
        self._sender_tasks = {w: [] for w in self.src_mesh.workers}
        # Dict of worker -> ((indices, rank, gpu index))
        self._receiver_tasks = {w: [] for w in self.dst_mesh.workers}
        # Dict of worker -> ((device_ids), (device_strs), (slices))
        self._allgather_tasks = {host: [] for host in self.dst_mesh.workers}

        self.sender_uuid_plan = []
        self.receiver_uuid_plan = []
        self.send_worker_task_ids = {}
        self.recv_worker_task_ids = {}
        self.allgather_worker_task_ids = {}

        # generate the above states
        self._compile()

        # create communicators
        if global_config.eagerly_create_communicators:
            self._create_resharding_communicators()
        # print(self.__str__()+"\n")

    @property
    def sender_tasks(self):
        """Return sender sub-tasks."""
        return self._sender_tasks

    @property
    def receiver_tasks(self):
        """Return receiver sub-tasks."""
        return self._receiver_tasks

    @property
    def allgather_tasks(self):
        """Return allgahter sub-tasks."""
        return self._allgather_tasks

    def _allgather_receiver_offset(self, receiver, logical_mesh_shape):
        if not self.task_spec.allgather_spec:
            return -1
        host_idx, device_idx = self.collective_group.device_str_to_rank_map[
            receiver]
        if host_idx >= self.collective_group.src_mesh.num_hosts:
            host_idx = host_idx - self.collective_group.src_mesh.num_hosts
        flatten_idx = host_idx * self.dst_mesh.num_devices_per_host + device_idx
        offsets = []
        for mesh_dim in reversed(logical_mesh_shape):
            offsets.append(flatten_idx % mesh_dim)
            flatten_idx //= mesh_dim
        return list(reversed(offsets))

    def _indices_in_dst_post_allgather(self,
                                       indices,
                                       mesh_idx,
                                       dst_spec=None,
                                       allgather_spec=None):
        if not self.task_spec.allgather_spec:
            return indices
        dst_spec = dst_spec or self.task_spec.dst_sharding_spec
        allgather_spec = allgather_spec or self.task_spec.allgather_spec

        indices = list(indices)
        shape = self.task_spec.aval.shape
        for tensor_dim in allgather_spec.allgather_dims:
            allgather_start_idx = allgather_spec.start_idx(tensor_dim, dst_spec)
            mapped_chunks = dst_spec.sharding[tensor_dim].chunks[
                allgather_start_idx:]
            mapped_offset = [
                mesh_idx[idx]
                for idx in allgather_spec.mapped_mesh_dim(tensor_dim, dst_spec)
            ]
            dim_offset = _allgather_dim_offset(shape[tensor_dim], mapped_offset,
                                               mapped_chunks,
                                               allgather_start_idx)
            indices[tensor_dim] = slice(dim_offset + indices[tensor_dim].start,
                                        dim_offset + indices[tensor_dim].stop,
                                        None)

        for idx, tensor_slice in enumerate(indices):
            if tensor_slice.start is None:
                assert tensor_slice.stop is None
                indices[idx] = slice(0, shape[idx], None)
        return indices

    def _compile(self):
        """
        Generate all send, recv, and allgather tasks.

        This function does the following:
        (1) generate send, recv, and allgather tasks (if needed),
        (2) put all tasks to their corresponding MeshHostWorkers.
        (3) pre-generate NCCL communicators for those tasks.
        """
        self._compile_send_recv_tasks()
        if self.is_local_allgather_task:
            self._compile_allgather_tasks()

        # put send and recv tasks
        task_dones = []
        for worker, task in self.sender_tasks.items():
            uuid = next_resharding_task_uuid()
            self.send_worker_task_ids[worker] = uuid
            task_dones.append(
                worker.put_resharding_send_task.remote(
                    uuid, task, self.collective_group.group_name))
        for worker, task in self.receiver_tasks.items():
            uuid = next_resharding_task_uuid()
            self.recv_worker_task_ids[worker] = uuid
            task_dones.append(
                worker.put_resharding_recv_task.remote(
                    uuid, task, self.collective_group.group_name))
        ray.get(task_dones)

        # put allgather tasks
        task_dones = []
        for worker, task in self._allgather_tasks.items():
            uuid = next_resharding_task_uuid()
            self.allgather_worker_task_ids[worker] = uuid
            task_dones.append(
                worker.put_resharding_allgather_task.remote(uuid, task))
        ray.get(task_dones)

    def _create_resharding_communicators(self):
        """Create the NCCL communicators in advance."""
        communicator_params = set()
        for worker, recv_tasks in self.receiver_tasks.items():
            dst_rank = self.collective_group.worker_to_rank_map[worker]
            for recv_task in recv_tasks:
                dst_gpu_idx = recv_task.device_id
                tile_specs = recv_task.tile_specs
                for tile_spec in tile_specs:
                    src_rank = tile_spec.rank
                    src_gpu_idx = tile_spec.gpu_idx
                    param = (src_rank, src_gpu_idx, dst_rank, dst_gpu_idx)
                    if param not in communicator_params:
                        communicator_params.add(param)

        # now init the communicators
        group_name = self.collective_group.group_name
        for param in communicator_params:
            task_dones = []
            src_rank, src_gpu_idx, dst_rank, dst_gpu_idx = param
            src_worker = self.collective_group.mesh_workers[src_rank]
            dst_worker = self.collective_group.mesh_workers[dst_rank]
            nccl_uid = ray.get(src_worker.generate_nccl_uid.remote(group_name))
            task_dones.append(
                src_worker.init_p2p_communicator.remote(group_name, src_rank,
                                                        src_gpu_idx, dst_rank,
                                                        dst_gpu_idx, nccl_uid))
            task_dones.append(
                dst_worker.init_p2p_communicator.remote(group_name, dst_rank,
                                                        dst_gpu_idx, src_rank,
                                                        src_gpu_idx, nccl_uid))
            ray.get(task_dones)

    def _compile_send_recv_tasks(self):
        """Generate all send/recv tasks."""
        logical_mesh_shape = _allgather_logical_mesh_from_spec(
            self.task_spec.dst_sharding_spec)
        for i, (dst_tile, src_tiles, indices_in_dst_tiles) in enumerate(
                self.task_spec.dst_tile_to_src_tiles_map):
            spec_plan = self.task_spec.strategy.per_spec_plans[i]
            for replica_index, receiver in enumerate(
                    dst_tile.replica_device_strs):
                # Get args for an empty buffer
                receiver_device_id = (
                    self.collective_group.device_str_to_device_id_map[receiver])
                receiver_worker = (self.collective_group.
                                   device_str_to_mesh_worker_map[receiver])
                dtype = self.task_spec.src.aval.dtype
                # Get args for send/recv
                senders = [
                    spec_plan[replica_index][src_tile_index]
                    for src_tile_index, _ in enumerate(src_tiles)
                ]
                self.receiver_uuid_plan.append(receiver)
                receiver_rank, receiver_gpu_idx = (
                    self.collective_group.device_str_to_rank_map[receiver])
                recv_tile_specs = []
                mesh_idx = self._allgather_receiver_offset(
                    receiver, logical_mesh_shape)
                for sender_idx, sender in enumerate(senders):
                    # Sender's task
                    sender_worker = self.collective_group.device_str_to_mesh_worker_map[
                        sender]
                    self._sender_tasks[sender_worker].append(
                        ReshardingTileSpec(src_tiles[sender_idx].offset,
                                           receiver_rank, receiver_gpu_idx))
                    self.sender_uuid_plan.append(sender)
                    # Receiver's task
                    sender_rank, sender_gpu_idx = \
                        self.collective_group.device_str_to_rank_map[sender]
                    indices_in_dst_tile = self._indices_in_dst_post_allgather(
                        indices_in_dst_tiles[sender_idx], mesh_idx)
                    recv_tile_specs.append(
                        ReshardingTileSpec(indices_in_dst_tile, sender_rank,
                                           sender_gpu_idx))
                receiver_task = ReshardingRecvSpec(receiver_device_id,
                                                   dst_tile.tile_shape, dtype,
                                                   recv_tile_specs)
                self._receiver_tasks[receiver_worker].append(receiver_task)

    def _compile_allgather_tasks(self):
        """
        Compile allgather tasks on destination mesh. Allgather tasks are
        created with the order described below:

        1. Iterate all tensor dimensions that has chunks inserted by
        _rewrite_allgather_spec;
        2. The rewritten sharding spec corresponds to a logical mesh shape, and
        each receiver has its own index with respect to the mesh shape. Let the
        index be (x_0,x_1,...,x_n) and the mesh shape is (l_0,l_1,...,l_n);
        3. The iterated tensor dimension corresponds to some logical mesh
        dimensions, assume these dimensions are D, a subset of [n]. The index is
        categorized into two parts. The first is (x_{i_0},x_{i_1},...x_{i_m}),
        where {i_0,i_1,...,i_m}=D while the second is the rest. The first part
        is named group index while the second is allgather index.
        4. Devices with the same group index forms an allgather group, and the
        allgather index indicates their order inside the group.
        """
        dst_spec = self.task_spec.dst_sharding_spec
        tensor_shape = self.task_spec.dst.tensor_shape
        allgather_spec = self.task_spec.allgather_spec
        tmp_allgather_spec = self.task_spec.allgather_spec
        # use reverse order to keep the result always continuous.
        for tensor_dim in reversed(allgather_spec.allgather_dims):
            indices = pxla.spec_to_indices(tensor_shape, dst_spec)[0]
            # The mesh shape is changed as a tensor dim's sharding is changed
            mesh_shape = _allgather_logical_mesh_from_spec(dst_spec)
            mesh_dims = allgather_spec.mapped_mesh_dim(tensor_dim, dst_spec)
            # Get allgather groups
            allgather_groups = _signle_tensor_dim_allgather_groups(
                mesh_shape, mesh_dims)
            post_dst_spec = _reduce_chunk(dst_spec, tensor_dim, len(mesh_dims))
            # For each group, get device ids, tensor slices and other allgather config
            for group_ids in allgather_groups:
                host_idx = group_ids[0] // self.dst_mesh.num_devices_per_host
                host = self.dst_mesh.workers[host_idx]
                device_ids = [
                    gid % self.dst_mesh.num_devices_per_host
                    for gid in group_ids
                ]
                tensor_slices = []
                for receiver_idx in group_ids:
                    flatten_idx = receiver_idx
                    mesh_idx = []
                    for mesh_dim in reversed(mesh_shape):
                        mesh_idx.append(flatten_idx % mesh_dim)
                        flatten_idx //= mesh_dim
                    mesh_idx = list(reversed(mesh_idx))
                    tensor_slices.append(
                        self._indices_in_dst_post_allgather(
                            indices, mesh_idx, dst_spec, tmp_allgather_spec))
                # the tensor dim's sharding is changed, use it for output slice.
                output_slice = list(tensor_slices[0])
                output_slice[tensor_dim] = slice(
                    0, self.task_spec.aval.shape[tensor_dim] /
                    _get_chunk_value(post_dst_spec.sharding[tensor_dim]), None)
                self._allgather_tasks[host].append(
                    ReshardingAllGatherSpec(device_ids, tensor_slices,
                                            output_slice))
            # change the tensor dim's sharding and the remained allgather spec.
            dst_spec = _reduce_chunk(dst_spec, tensor_dim, len(mesh_dims))
            tmp_allgather_spec = tmp_allgather_spec.post_allgather(tensor_dim)
        return self._allgather_tasks

    # FIXME(Hao): test the function below; it might be buggy.
    def do_prepared(self, src_array, profiling=False):
        """Execute a task which has been put in the remote workers."""
        send_buf_uuids = {host: [] for host in self.src_mesh.workers}
        recv_buf_uuids = {host: [] for host in self.dst_mesh.workers}

        bufs: List[Any] = [None] * len(self.task_spec.dst_indices)
        device_str_to_buf_map = {}

        for receiver in self.receiver_uuid_plan:
            receiver_host_id = self.collective_group.device_str_to_host_id_map[
                receiver]
            receiver_device_id = self.collective_group.device_str_to_device_id_map[
                receiver]
            receiver_worker = self.collective_group.device_str_to_mesh_worker_map[
                receiver]
            result_buf = RemoteBufferRef(self.dst_mesh, receiver_host_id,
                                         receiver_device_id)
            recv_buf_uuids[receiver_worker].append(result_buf.uuid)
            device_str_to_buf_map[receiver] = result_buf

        for sender in self.sender_uuid_plan:
            sender_worker = self.collective_group.device_str_to_mesh_worker_map[
                sender]
            send_buf = src_array.remote_buffers[
                self.task_spec.src.device_str_to_flat_index[sender]]
            send_buf_uuids[sender_worker].append(send_buf.uuid)

        results = []
        if profiling:
            for worker, uuid in self.send_worker_task_ids.items():
                results.append(
                    worker.profile_resharding_send_task.remote(
                        uuid, send_buf_uuids[worker]))
            for worker, uuid in self.recv_worker_task_ids.items():
                results.append(
                    worker.profile_resharding_recv_task.remote(
                        uuid, recv_buf_uuids[worker]))
        else:
            for worker, uuid in self.send_worker_task_ids.items():
                results.append(
                    worker.run_resharding_send_task.remote(
                        uuid, send_buf_uuids[worker]))
            for worker, uuid in self.recv_worker_task_ids.items():
                results.append(
                    worker.run_resharding_recv_task.remote(
                        uuid, recv_buf_uuids[worker]))
            logger.debug("Precompiled tasks launched.")
            ray.get(results)
        for device_str in self.task_spec.dst.device_mesh.device_strs:
            # for each replica
            bufs[self.task_spec.dst.device_str_to_flat_index[
                device_str]] = device_str_to_buf_map[device_str]
        # Now construct the distributed array
        dst_array = DistributedArray(self.dst_mesh, src_array.aval,
                                     self.task_spec.dst_sharding_spec, bufs,
                                     self.task_spec.dst_indices)
        if profiling:
            return results
        return dst_array

    def __str__(self):
        return (f"ReshardingTask(shape:{self.task_spec.aval.shape},\n"
                f"{self.task_spec.src_sharding_spec} ->\n"
                f"{self.task_spec.dst_sharding_spec})")


class CommunicatorConfig:
    """Config used to initilize broadcast communicator."""

    def __init__(self, comm_key):
        self.comm_key = comm_key
        self.workers = []
        self.device_ids = []

    def add(self, worker, device_id):
        self.workers.append(worker)
        self.device_ids.append(device_id)

    def __hash__(self):
        return hash((self.comm_key, tuple(self.workers), tuple(self.device_ids)))

    def __eq__(self, other):
        if not isinstance(other, CommunicatorConfig):
            return False
        elif self.comm_key != other.comm_key:
            return False
        elif len(self.workers) != len(other.workers):
            return False

        for i in range(len(self.workers)):
            if self.workers[i] != other.workers[i] or self.device_ids[i] != other.device_ids[i]:
                return False

        return True


class SymbolicBroadcastReshardingTask(ReshardingTask):
    """A Broadcast based symbolic resharding task that puts task info in remote workers."""

    def __init__(self, task_spec, collective_group, src_mesh, dst_mesh):
        super().__init__(task_spec, collective_group, src_mesh, dst_mesh)
        # task is a dict: (i, src_tile_index)->ReshardingBroadcastSpec
        self._broadcast_tasks = {host: {} for host in self.src_mesh.workers + self.dst_mesh.workers}
        self.broadcast_worker_task_ids = {}
        self.communicator_configs = set()

        # generate the above states
        self._compile()

        # create communicators
        if global_config.eagerly_create_communicators:
            self._create_resharding_communicators()
        # print(self.__str__()+"\n")

    @property
    def broadcast_tasks(self):
        """Return broadcast sub-tasks."""
        return self._broadcast_tasks

    def _compile(self):
        """
        Generate all broadcast tasks.

        This function does the following:
        (1) generate broadcast tasks (if needed),
        (2) put all tasks to their corresponding MeshHostWorkers.
        (3) pre-generate NCCL communicators for those tasks.
        """
        self._compile_broadcast_tasks()

        # put broadcast tasks on corresponding mesh workers
        task_dones = []
        for worker, task in self._broadcast_tasks.items():
            uuid = next_resharding_task_uuid()
            self.broadcast_worker_task_ids[worker] = uuid
            # print(worker, uuid, task)
            task_dones.append(
                worker.put_resharding_broadcast_task.remote(uuid, task, self.collective_group.group_name)
            )
        ray.get(task_dones)

    def _compile_broadcast_tasks(self):
        """Compile broadcast tasks."""
        dtype = self.task_spec.src.aval.dtype        
        for i, (dst_tile, src_tiles, indices_in_dst_tiles) in enumerate(self.task_spec.dst_tile_to_src_tiles_map):
            spec_plan = self.task_spec.strategy.per_spec_plans[i]
            for src_tile_index, (src_tile, indices_in_dst_tile) in enumerate(zip(src_tiles, indices_in_dst_tiles)):
                sender = spec_plan[src_tile_index]
                sender_worker = self.collective_group.device_str_to_mesh_worker_map[sender]
                broadcast_group = (i, src_tile_index)
                devices = [sender] + dst_tile.replica_device_strs
                comm_key = "$".join(devices)
                world_size = len(devices)

                comm_config = CommunicatorConfig(comm_key)

                group_spec = self._broadcast_tasks[sender_worker].setdefault(
                    broadcast_group, ReshardingBroadcastSpec(comm_key=comm_key, 
                                                             world_size=world_size, 
                                                             devices_ids=[self.collective_group.
                                                                          device_str_to_device_id_map[sender]],
                                                             devices_global_rank=[0], 
                                                             tensor_slices=[src_tile.offset], 
                                                             recv_tile_shape=src_tile.tile_shape, 
                                                             dtype=dtype)
                )
                comm_config.add(sender_worker, self.collective_group.device_str_to_device_id_map[sender])

                for replica_index, receiver in enumerate(dst_tile.replica_device_strs):
                    receiver_worker = (self.collective_group.device_str_to_mesh_worker_map[receiver])
                    group_spec = self._broadcast_tasks[receiver_worker].setdefault(
                        broadcast_group, ReshardingBroadcastSpec(comm_key=comm_key, 
                                                                 world_size=world_size,
                                                                 devices_ids=[], 
                                                                 devices_global_rank=[], 
                                                                 tensor_slices=[], 
                                                                 recv_tile_shape=dst_tile.tile_shape, 
                                                                 dtype=dtype)
                    )

                    group_spec.devices_ids.append(self.collective_group.device_str_to_device_id_map[receiver])
                    group_spec.devices_global_rank.append(1 + replica_index)
                    group_spec.tensor_slices.append(indices_in_dst_tile)
                    comm_config.add(receiver_worker, self.collective_group.device_str_to_device_id_map[receiver])

                self.communicator_configs.add(comm_config)
        return self._broadcast_tasks

    def _create_resharding_communicators(self):
        """Create the NCCL communicators for broadcast in advance."""
        group_name = self.collective_group.group_name
        for config in self.communicator_configs:
            task_dones = []
            worker_to_devices_and_global_ranks = {}
            world_size = len(config.workers)
            for global_rank, (worker, device_id) in enumerate(zip(config.workers, config.device_ids)):
                if worker not in worker_to_devices_and_global_ranks:
                    worker_to_devices_and_global_ranks[worker] = {"device_ids": [], "global_ranks": []}
                worker_to_devices_and_global_ranks[worker]["device_ids"].append(device_id)
                worker_to_devices_and_global_ranks[worker]["global_ranks"].append(global_rank)

            sender_worker = config.workers[0]
            nccl_uid = ray.get(sender_worker.generate_nccl_uid.remote(group_name))

            for worker, devices_info in worker_to_devices_and_global_ranks.items():
                task_dones.append(worker.init_broadcast_communicator.remote(group_name, 
                                                                            config.comm_key, 
                                                                            world_size, 
                                                                            devices_info["device_ids"], 
                                                                            devices_info["global_ranks"], 
                                                                            nccl_uid)
                                  )
                task_dones.append(worker.init_broadcast_communicator.remote(group_name, 
                                                                            config.comm_key, 
                                                                            world_size, 
                                                                            devices_info["device_ids"], 
                                                                            devices_info["global_ranks"], 
                                                                            nccl_uid)
                                  )
            ray.get(task_dones)

    def __str__(self):
        return (f"Broadcast based ReshardingTask(shape:{self.task_spec.aval.shape},\n"
                f"{self.task_spec.src_sharding_spec} ->\n"
                f"{self.task_spec.dst_sharding_spec})")


class CollectiveGroup:
    """
    A class for setting up real NCCL groups.

    Args:
        device_strs (List[str]): list of device strs in this group.
        src_mesh (PhysicalDeviceMesh): the source physical mesh.
        dst_mesh (PhysicalDeviceMesh): the destination physical mesh.
    """

    def __init__(self, device_strs, src_mesh, dst_mesh):
        self.device_strs = device_strs
        self.src_mesh = src_mesh
        self.dst_mesh = dst_mesh

        # generate a group name
        self.group_name = ",".join(self.device_strs)

        # construct a device str -> rank: (process_rank, gpu_index) map
        self.device_str_to_rank_map = {}
        self.device_str_to_mesh_worker_map = {}
        self.device_str_to_host_id_map = {}
        self.device_str_to_device_id_map = {}
        self.worker_to_rank_map = {}

        # arranged following the rank order
        num_host = len(self.src_mesh.host_ips) + len(self.dst_mesh.host_ips)
        self.mesh_workers: List[Any] = [None] * num_host
        for i, _ in enumerate(src_mesh.host_ips):
            self.mesh_workers[i] = self.src_mesh.workers[i]
            for j in range(src_mesh.num_devices_per_host):
                device_str = self.src_mesh.device_strs[
                    i * src_mesh.num_devices_per_host + j]
                self.device_str_to_rank_map[device_str] = (i, j)
                self.device_str_to_mesh_worker_map[
                    device_str] = self.src_mesh.workers[i]
                self.device_str_to_host_id_map[device_str] = i
                self.device_str_to_device_id_map[device_str] = j
        for i, _ in enumerate(dst_mesh.host_ips):
            self.mesh_workers[
                i + len(self.src_mesh.host_ips)] = self.dst_mesh.workers[i]
            for j in range(dst_mesh.num_devices_per_host):
                device_str = self.dst_mesh.device_strs[
                    i * src_mesh.num_devices_per_host + j]
                self.device_str_to_rank_map[device_str] = (
                    i + len(src_mesh.host_ips), j)
                self.device_str_to_mesh_worker_map[
                    device_str] = self.dst_mesh.workers[i]
                self.device_str_to_host_id_map[device_str] = i
                self.device_str_to_device_id_map[device_str] = j

        self.worker_to_rank_map = {
            worker: r for r, worker in enumerate(self.mesh_workers)
        }

    def instantiate(self):
        """Instantiate the collective group in Ray lazily."""
        options = {
            "group_name": self.group_name,
            "world_size": len(self.mesh_workers),
            "ranks": [i for i, _ in enumerate(self.mesh_workers)],
            "backend": "nccl"
        }
        col.create_collective_group(self.mesh_workers, **options)

    def instantiate_now(self):
        """Instantiate the collective group eagerly (but not communicators)."""
        world_size = len(self.mesh_workers)
        task_dones = []
        logger.debug(
            "Trying to create ray.collective groups among participants.")
        for rank, worker in enumerate(self.mesh_workers):
            task_dones.append(
                worker.init_collective_group.remote(world_size, rank, "nccl",
                                                    self.group_name))
        ray.get(task_dones)
        logger.debug(f"The group {self.group_name} has been created.")

    def destroy(self):
        """Destroy the NCCL collective group at exit."""
        logger.debug(f"Recycling the collective group: {self.group_name}.")
        for worker in self.mesh_workers:
            # This remote call will remove ray named actors (hence it is necessary)
            ray.get(worker.destroy_collective_group.remote(self.group_name))
        # Destroy the declared named actor in ray
        self._destroy_info_actor()

    def _destroy_info_actor(self):
        name = "info_" + self.group_name
        try:
            store = ray.get_actor(name)
            ray.kill(store)
        except ValueError:
            pass


class ReshardingTaskSpec:
    """
    A helper class specifies how to perform cross-mesh resharding for two arrays.

    Args:
        src_array (VirtualDistributedArray): the source VirtualDistributedArray.
        dst_array (VirtualDistributedArray): the destination VirtualDistributedArray.
    """

    def __init__(self, src_array, dst_array, local_chunks):
        self.src = src_array
        self.dst = dst_array
        self._dst_tile_to_src_tiles_map = None
        self._strategy = None
        self._local_chunks = local_chunks
        self.allgather_spec = (_AllgatherSpec(local_chunks)
                               if local_chunks else None)

    @property
    def src_sharding_spec(self):
        """Return the sharding spec of the source array."""
        return self.src.sharding_spec

    @property
    def dst_sharding_spec(self):
        """Return the sharding spec of the destination array."""
        return self.dst.sharding_spec

    @property
    def aval(self):
        """Return the abstract value of the array."""
        return self.src.aval

    @property
    def src_indices(self):
        """Return the sharding (flattened) indices of the source array."""
        return self.src.indices

    @property
    def dst_indices(self):
        """Return the sharding (flattened) indices of the destination array."""
        return self.dst.indices

    @property
    def dst_tile_to_src_tiles_map(self):
        """
        Map from dst_tile to all corresponding src TileSlices.

        It is a list of length len(dst.tiles), each element is a 3-element tuple
        (dst_tile, src_tile_slices, indices_in_dst_tile):
        - dst_tile: a tile from dst.tiles
        - src_tile_slices: a list of TileSlice objects from src, corresponding to this dst_tile
        - indices_in_dst_tile: a list of slicers. Each slicer is a list of slice objects, corresponding to
            a TileSlice in src_tile_slices, representing the indices of this TileSlice in dst_tile.
        """
        if not self._dst_tile_to_src_tiles_map:
            self._dst_tile_to_src_tiles_map = self.generate_src_dst_map()
        return self._dst_tile_to_src_tiles_map

    def generate_src_dst_map(self):
        """
        Analyzes the src and dst array and generate the dst_tile_to_src_tiles_map.

        It aims to tell the needed collective group and communication pattern.

        Returns:
            dst_tile_to_src_tiles_map (tuple[tile, tileslices, indices]):
                see the docstring of `dst_tile_to_src_tiles_map`.
        """
        dst_tile_to_src_tiles_map = []
        for tile in self.dst.tiles.flatten():
            # loop over each tile
            src_tile_slices, indices_in_dst_tile = self._look_up_dst_tile_from_src(
                tile)
            dst_tile_to_src_tiles_map.append(
                (tile, src_tile_slices, indices_in_dst_tile))
        return dst_tile_to_src_tiles_map

    def _look_up_dst_tile_from_src(self, tile):
        """
        Look up all related tiles from the source array for a given destination tile.

        See the docstring in dst_tile_to_src_tiles_map() for more details.
        """
        # For each dim in the dst tile, find all the related tiles, and ragged values on that dim in src_tiles.
        # To record that, for each dim, we make a tuple containing the first and last index of tiles in src array
        # that intersects with the dst tile: Shards between [start, end) are involved; Left included, right not
        # included.
        related_tile_start_end = [tuple()] * self.src.tensor_rank

        # Meanwhile, for each dim, for the first and end tile, we make a tuple recording the slicing offset:
        # - start_shard_offset: [start_shard_offset: ] on that dim is activated.
        # - end_shard_offset: [:end_sharding_offset] on that dim is activated.
        related_tile_offset = [tuple()] * self.src.tensor_rank

        for i, dim in enumerate(self.src.tensor_shape):
            tile_length, ragged = divmod(dim, self.src.tile_shape[i])
            assert not ragged
            start_tile, start_tile_offset = divmod(tile.indices[i].start,
                                                   tile_length)
            end_tile, end_tile_offset = divmod(tile.indices[i].stop,
                                               tile_length)
            # if falling on the middle a src tile, increase the index of the final tile by 1.
            if end_tile_offset:
                end_tile = end_tile + 1
            # if falling on the end of a src tile, the offset should be [0: tile_length]
            if end_tile_offset == 0:
                end_tile_offset = tile_length
            related_tile_start_end[i] = (start_tile, end_tile)
            related_tile_offset[i] = (start_tile_offset, end_tile_offset)

        # count the number of tile slices
        num_src_tileslices = 1
        for start, end in related_tile_start_end:
            num_src_tileslices = num_src_tileslices * (end - start)

        src_tileslices = []
        indices_in_dst_tile = []
        for tileslice_index in range(num_src_tileslices):
            tile_index_relative = unflatten_tile_index(
                tileslice_index,
                [end - start for start, end in related_tile_start_end])
            tile_index_absolute = [
                start + tile_index_relative[dim_index]
                for dim_index, (start, end) in enumerate(related_tile_start_end)
            ]
            # depending on its index, calculate a slice for it
            offsets = []
            indices = []
            # loop over each dimension
            for i, r in enumerate(tile_index_absolute):
                start, end = related_tile_start_end[i]
                tile_length_on_this_dim = self.src.tiles[tuple(
                    tile_index_absolute)].tile_shape[i]
                if r == start and r == end - 1:
                    # the dst tile is smaller or equal to the src tile
                    left_offset = related_tile_offset[i][0]
                    right_offset = related_tile_offset[i][1]
                    offsets.append(slice(left_offset, right_offset))
                    indices.append(slice(0, tile.tile_shape[i]))  # all included
                elif r == start:
                    # meaning it is the first involved tile, and not the last
                    offset = related_tile_offset[i][0]
                    offsets.append(slice(offset, tile_length_on_this_dim))
                    indices.append(slice(0, tile_length_on_this_dim - offset))
                elif r == end - 1:
                    # meaning it is the last involved tile, and not the first
                    offset = related_tile_offset[i][1]
                    offsets.append(slice(0, offset))
                    indices.append(
                        slice(tile.tile_shape[i] - offset, tile.tile_shape[i]))
                else:
                    # meaning it is a fully involved tile
                    offset = related_tile_offset[i][0]
                    offsets.append(slice(0, tile_length_on_this_dim))
                    left_in_dst_tile = (
                        tile_length_on_this_dim - offset +
                        (tile_index_relative[i] - 1) * tile_length_on_this_dim)
                    right_in_dst_tile = left_in_dst_tile + tile_length_on_this_dim
                    indices.append(slice(left_in_dst_tile, right_in_dst_tile))
            # construct a new tile slice
            this_tileslice = TileSlice(
                self.src.tiles[tuple(tile_index_absolute)], offset=offsets)
            src_tileslices.append(this_tileslice)
            indices_in_dst_tile.append(indices)
        return src_tileslices, indices_in_dst_tile

    def set_resharding_strategy(self, strategy):
        """Now the strategy is np.array(dtype=str) to specify connections between src tiles and dst tile."""
        self._strategy = strategy

    @property
    def strategy(self):
        """Return the communication strategy for this resharding task spec."""
        if not self._strategy:
            raise RuntimeError(
                "Generate and set strategy in the cross-mesh communicator first."
            )
        return self._strategy

    def get_participant_device_strs(self):
        """Identify all participant device strs (for NCCL setup) in this task spec."""
        if not self._strategy:
            raise RuntimeError("Generate and set strategy first.")
        device_strs = OrderedSet()
        # senders
        for tile_strategy in self.strategy.per_spec_plans:
            device_strs = device_strs | OrderedSet(
                tile_strategy.flatten().tolist())
        # receivers
        for tile in self.dst.tiles.flatten():
            device_strs = device_strs | OrderedSet(tile.replica_device_strs)
        return device_strs

    def __str__(self):
        ret_str = ""
        ret_str += f"{self.src_sharding_spec} -> {self.dst_sharding_spec};"
        if self.allgather_spec:
            ret_str += " allgather: "
            for allgather_chunk_info in self._local_chunks:
                shard_dim, shard_axis, extra_sharding = allgather_chunk_info
                ret_str += (
                    f"(shard at tensor dim {shard_dim}, " +
                    f"shard at mesh axis {shard_axis}, chunked value {extra_sharding}), "
                )
        return ret_str


class ReshardingStrategy:
    """A data class for storing resharding communication information.

    Args:
        per_spec_plans (List[np.ndarray]): `per_spec_plan` is a list a np array, with
            length as len(spec.dst_tile_to_src_tiles_map), each array is with shape
            [len(dst_tile.devices), len(src_tiles)]; it specifies for each replica of
            a dst tile, how it should get the data from src_tiles (src tile replicas).
        is_local_allgather (bool): if this strategy involves post allgather operations.
    """

    def __init__(self, per_spec_plans, is_local_allgather):
        self.per_spec_plans = per_spec_plans
        self.is_local_allgather = is_local_allgather


class CrossMeshCommunicator:
    """
    Communicator for cross-mesh resharding.

    Given the pipeline schedule and stages, the class analyzes them and generates:
    - resharding specs (see docstring of `ReshardingTaskSpec`),
    - resharding strategies (see docstring of `ReshardingStrategy`).
    This communicator only takes care of compilation-time work, and does not get involved
    with physical meshes, buffer creations, or other runtime work.

    Args:
        sharded_stages (Sequence[XlaShardedPipelineComputation]): list of stages to
            form the pipeline.
        schedule (Any): the pipelining schedule for these stages.
    """

    def __init__(self, sharded_stages, schedule):
        if not isinstance(sharded_stages, list):
            raise RuntimeError("Require a list of stages.")
        for s in sharded_stages:
            if not isinstance(s, XlaShardedPipelineComputation):
                raise RuntimeError("Require a list of sharded stages.")
        # Do not mutate
        self._sharded_stages = sharded_stages
        self._schedule = schedule
        self.resharding_specs = None

        # Loads for load balancing.
        self._sender_loads = {
            device_str: 0 for mesh in self._schedule.meshes
            for device_str in mesh.device_strs
        }
        self._receiver_loads = {
            device_str: 0 for mesh in self._schedule.meshes
            for device_str in mesh.device_strs
        }

        # Initialize all resharding specs
        self._create_resharding_specs()
        # Generate a send/recv strategies for all resharding tasks by looking at their load.
        for _, _, var_spec_map in self.task_spec_iter():
            for _, spec in var_spec_map.items():
                if global_config.resharding_mode == "send_recv":
                    strategy = self._generate_send_recv_resharding_strategy_by_loads(spec,
                                                                                     self._sender_loads,
                                                                                     self._receiver_loads)
                else:
                    strategy = self._generate_broadcast_resharding_strategy_by_loads(spec,
                                                                                     self._sender_loads,
                                                                                     self._receiver_loads)
                spec.set_resharding_strategy(strategy)

    @property
    def num_mesh(self):
        """Number of meshes in the schedule."""
        return self._schedule.num_mesh

    @staticmethod
    def _rewrite_allgather_spec(sharding_spec: pxla.ShardingSpec, mesh,
                                var_shape):
        """
        Given a sharding spec, if use_scatter_gather is turned on and the tensor
        corresponding to the spec is not fully sharded, the function rewrite the
        spec to a fully-sharded one, and return info of added chunks.

        The rewrite is by steps below:
        1. Iterate all logical mesh dimensions(m_dim) along which the tensor is
        replicated;
        2. Iterate all tensor dimensions(t_dim). If the length of the tensor on
        t_dim and the number of replicas on m_dim have a common divisor greater
        than 1, an extra chunk is appended on t_dim;
        3. When there is no replicas on m_dim, the iteration terminates.
        """

        local_chunks = []
        if not global_config.use_scatter_gather:
            return sharding_spec, local_chunks

        # check whether the tensor is fully sharded.
        replicated_mesh_dim = []
        mesh_dim_to_chunk_axis = {}
        for mesh_dim, dim_mapping in enumerate(sharding_spec.mesh_mapping):
            if isinstance(dim_mapping, pxla.Replicated):
                replicated_mesh_dim.append((mesh_dim, dim_mapping.replicas))
            else:
                dim_mapping: pxla.ShardedAxis
                mesh_dim_to_chunk_axis[mesh_dim] = dim_mapping.axis
        if len(replicated_mesh_dim) == 0:
            return sharding_spec, local_chunks
        assert len(replicated_mesh_dim) == 1, "Only support 1D and 2D mesh"

        # create chunk axis to tensor dim mapping
        chunk_axis_to_tensor_dim = []
        for tensor_dim, dim_spec in enumerate(sharding_spec.sharding):
            if isinstance(dim_spec, pxla.Chunked):
                for chunk_idx in range(len(dim_spec.chunks)):
                    chunk_axis_to_tensor_dim.append((tensor_dim, chunk_idx))

        # TODO(yonghao): Support allgather cross node for communication balance.
        # Check whether allgather should be cross node.
        node_mesh_mapping = sharding_spec.mesh_mapping[0]
        node_chunk = 1
        if isinstance(node_mesh_mapping, pxla.ShardedAxis):
            tensor_dim, _ = chunk_axis_to_tensor_dim[node_mesh_mapping.axis]
            node_chunk = _get_chunk_value(sharding_spec.sharding[tensor_dim])
        if node_chunk < mesh.num_hosts:
            return sharding_spec, local_chunks

        sharding = list(sharding_spec.sharding)
        squeezed_mesh_mapping = [
            None if isinstance(dim_mapping, pxla.Replicated) else
            [chunk_axis_to_tensor_dim[dim_mapping.axis]]
            for dim_mapping in sharding_spec.mesh_mapping
        ]
        for (mesh_dim, replica) in replicated_mesh_dim:
            dim_local_mapping = []
            for tensor_dim, dim_sharding in enumerate(sharding):
                chunked_value = _get_chunk_value(dim_sharding)
                chunked_len = var_shape[tensor_dim] // chunked_value
                new_chunk = math.gcd(replica, chunked_len)
                if new_chunk == 1:
                    continue
                sharding[tensor_dim] = _add_chunk(dim_sharding, new_chunk)
                chunk_idx = len(sharding[tensor_dim].chunks) - 1
                local_chunks.append((tensor_dim, chunk_idx, new_chunk))
                dim_local_mapping.append((tensor_dim, chunk_idx))

                replica //= new_chunk
                if replica == 1:
                    break
            if replica != 1:
                logger.warning(
                    "ReshardingTask is not fully sharded, this causes redundant communication."
                )
            if len(dim_local_mapping) != 0:
                squeezed_mesh_mapping[mesh_dim] = dim_local_mapping

        mesh_mapping = _get_mesh_mapping(sharding, sharding_spec.mesh_mapping,
                                         squeezed_mesh_mapping)
        new_sharding_spec = pxla.ShardingSpec(sharding, mesh_mapping)
        # sorted by (tensor dim, chunk idx, mesh dim)
        return new_sharding_spec, local_chunks

    def _create_resharding_specs(self):
        stages = self._sharded_stages
        meshes = self._schedule.meshes
        num_stage = len(self._sharded_stages)
        stage_placements = [
            list(self._schedule.stage_placement(i))[0] for i in range(num_stage)
        ]
        deps = self._schedule.dependency
        assert deps.shape[0] == num_stage
        assert deps.shape[1] == num_stage

        # Note(Hao): resharding_specs is num_mesh x num_mesh matrix
        # Each element is a dict: the name of variables are keys, ReshardingSpec are values.
        self.resharding_specs = [
            [{} for _ in range(self.num_mesh)] for _ in range(self.num_mesh)
        ]

        # find stages that will communicate
        pairs = np.argwhere(deps > 0)
        for i in range(pairs.shape[0]):
            # for each pair of stages that are dependent,
            src_stage_index = pairs[i][1]
            src_stage = stages[src_stage_index]
            dst_stage_index = pairs[i][0]
            dst_stage = stages[dst_stage_index]
            src_mesh_index = stage_placements[src_stage_index]
            dst_mesh_index = stage_placements[dst_stage_index]
            src_mesh = meshes[src_mesh_index]
            dst_mesh = meshes[dst_mesh_index]

            # we only take care of cross-mesh sharding.
            if src_mesh_index == dst_mesh_index:
                continue

            # find out variables that need resharding, and get their
            # (1) out_sharding_spec in the src stage
            # (2) in_sharding_spec in the destination stage.
            resharding_vars, out_var_indices, in_var_indices = (
                self._args_between(src_stage, dst_stage))
            out_sharding_specs = src_stage.output_sharding_specs
            in_sharding_specs = dst_stage.input_sharding_specs

            # Make a ReshardSpec for each VirtualDistributedArray
            for var, out_var_index, in_var_index in zip(resharding_vars,
                                                        out_var_indices,
                                                        in_var_indices):
                src_sharding_spec = out_sharding_specs[out_var_index]
                dst_sharding_spec = in_sharding_specs[in_var_index]
                # TODO(yonghao): to keep some consistency, record both
                # dst sharding spec before and after allgather

                if global_config.resharding_mode == "send_recv":
                    dst_sharding_spec, local_chunks = self._rewrite_allgather_spec(
                        dst_sharding_spec, dst_mesh, var.aval.shape)
                else:
                    local_chunks = None

                src_array = VirtualDistributedArray(
                    device_mesh=src_mesh,
                    aval=var.aval,
                    sharding_spec=src_sharding_spec)
                dst_array = VirtualDistributedArray(
                    device_mesh=dst_mesh,
                    aval=var.aval,
                    sharding_spec=dst_sharding_spec)
                task_spec = ReshardingTaskSpec(src_array, dst_array,
                                               local_chunks)
                self.resharding_specs[src_mesh_index][dst_mesh_index][repr(
                    var)] = task_spec

    def task_spec_iter(self):
        """A convenient iterator over all activated task specs."""
        for i in range(self.num_mesh):
            for j in range(self.num_mesh):
                if not self.resharding_specs[i][j]:
                    continue
                yield i, j, self.resharding_specs[i][j]

    @staticmethod
    def _generate_send_recv_resharding_strategy_by_loads(
            spec, src_loads, dst_loads):
        """Generate the resharding strategy by balancing loads."""
        is_local_allgather = spec.allgather_spec is not None
        per_spec_plans = []
        for dst_tile, src_tileslices, _ in spec.dst_tile_to_src_tiles_map:
            # plan is a 2D array
            per_spec_plan = np.empty(
                (len(dst_tile.replica_device_strs), len(src_tileslices)),
                dtype=object)
            for receiver_idx, receiver in enumerate(
                    dst_tile.replica_device_strs):
                for src_tileslice_idx, src_tileslice in enumerate(
                        src_tileslices):
                    loads = {
                        sender: src_loads[sender]
                        for sender in src_tileslice.replica_device_strs
                    }
                    sender = min(loads, key=loads.get)
                    per_spec_plan[receiver_idx][src_tileslice_idx] = sender
                    # upload load on-the-fly
                    src_loads[sender] += src_tileslice.slice_size
                    dst_loads[receiver] += src_tileslice.slice_size
            per_spec_plans.append(per_spec_plan)
        strategy = ReshardingStrategy(per_spec_plans, is_local_allgather)
        return strategy

    @staticmethod
    def _generate_broadcast_resharding_strategy_by_loads(spec, src_loads, dst_loads):
        """
            Generate the broadcast-based resharding strategy by balancing loads.
            For each tile, I not only allow one source to provide the tile.
        """
        #TODO(hexu): (1) allow for multiple sources. (2) update load on the fly.
        per_spec_plans = []
        dst_loads = None
        for _, src_tileslices, _ in spec.dst_tile_to_src_tiles_map:
            per_spec_plan = np.empty((len(src_tileslices),), dtype=object)

            for src_tileslice_idx, src_tileslice in enumerate(src_tileslices):
                loads = {
                    sender: src_loads[sender]
                    for sender in src_tileslice.replica_device_strs
                }
                sender = min(loads, key=loads.get)

                per_spec_plan[src_tileslice_idx] = sender
                src_loads[sender] += src_tileslice.slice_size
            per_spec_plans.append(per_spec_plan)
        strategy = ReshardingStrategy(per_spec_plans, None)
        return strategy

    @staticmethod
    def _args_between(src_stage, dst_stage):
        """Find the variable exchanged between stages."""
        resharding_vars = []
        src_indices = []
        dst_indices = []
        for i, var in enumerate(src_stage.outvars):
            if var in dst_stage.invars:
                resharding_vars.append(var)
                src_indices.append(i)
                dst_indices.append(dst_stage.invars.index(var))
        return resharding_vars, src_indices, dst_indices
