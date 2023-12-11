"""Cross mesh resharding for pipeline parallelism."""
from abc import ABC, abstractmethod
from collections import namedtuple
import logging
import math
import random
import time
from typing import List, Any

from jax.interpreters import pxla
import numpy as np
import ray

import alpa.collective as col
from alpa.device_mesh import (DistributedArray, RemoteArrayRef,
                              ReshardingRecvSpec, ReshardingSendSpec,
                              ReshardingTileSpec, ReshardingBroadcastSpec,
                              _device_mesh_put_dummy, device_id_to_str, VirtualWorker)
from alpa.global_env import global_config
from alpa.mesh_executable import (UtilMeshWorkerExecutable,
                                  next_mesh_executable_uuid)
from alpa.pipeline_parallel.computation import XlaShardedPipelineComputation
from alpa.pipeline_parallel.resharding_tensor import (VirtualDistributedArray,
                                                      TileSlice,
                                                      unflatten_tile_index)
from alpa.util import OrderedSet, compile_allgather

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

resharding_task_counter = 0


def next_resharding_task_uuid():
    """Generate the next resharding task uuid."""
    global resharding_task_counter
    resharding_task_counter = (resharding_task_counter + 1) % (1 << 60)
    return resharding_task_counter


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

        remote_ref = _device_mesh_put_dummy(src_array.aval, self.dst_mesh,
                                            self.task_spec.dst_indices, 1)  # pylint: disable=protected-access
        for i, (dst_tile, src_tiles, indices_in_dst_tiles) in enumerate(
                self.task_spec.dst_tile_to_src_tiles_map):
            # Loop over each dst tile for this shard
            s = self.task_spec.strategy[i]
            # strategy is len(dst_tile.device_strs) by len(src_tiles)
            for replica_index, receiver in enumerate(
                    dst_tile.replica_device_strs):
                # loop over this replica (hence a specific destination gpu
                # device)
                senders = [
                    s[replica_index][src_tile_index]
                    for src_tile_index, src_tile in enumerate(src_tiles)
                ]
                self.same_destination_group_send_recv(src_array, senders,
                                                      src_tiles,
                                                      indices_in_dst_tiles,
                                                      receiver, remote_ref.uuid)

        # Now construct the distributed array
        dst_array = DistributedArray(self.dst_mesh, src_array.aval,
                                     self.task_spec.dst_sharding_spec,
                                     remote_ref, self.task_spec.dst_indices)
        return dst_array

    def same_destination_group_send_recv(self, src_array, senders, src_tiles,
                                         indices_in_dst_tiles, receiver, uuid):
        """P2P Communication accounting for multiple senders and one receiver
        (a destination tile)."""
        receiver_device_id = self.collective_group.device_str_to_device_id_map[
            receiver]
        receiver_worker = self.collective_group.device_str_to_mesh_worker_map[
            receiver]
        # Put an empty buffer first.
        receiver_rank, receiver_gpu_idx = (
            self.collective_group.device_str_to_rank_map[receiver])
        for i, sender in enumerate(senders):
            # send is a device_str in src_mesh
            # we need to find out its mesh_worker, and the corresponded sender
            # remotebuf (uuid-indexed).
            sender_worker = self.collective_group.device_str_to_mesh_worker_map[
                sender]
            # assert sender_buf.device_id == i
            sender_rank, sender_gpu_idx = (
                self.collective_group.device_str_to_rank_map[sender])
            # launch NCCL send/recv
            tile = src_tiles[i]
            indices_in_dst_tile = indices_in_dst_tiles[i]
            send_done_ref = sender_worker.send_tile.remote(
                src_array.remote_ref.uuid, tile.offset, receiver_rank,
                receiver_gpu_idx, self.collective_group.group_name)
            recv_done_ref = receiver_worker.recv_tile.remote(
                uuid, receiver_device_id, indices_in_dst_tile, sender_rank,
                sender_gpu_idx, self.collective_group.group_name)
            ray.get([send_done_ref, recv_done_ref])


class SymbolicReshardingTask(ReshardingTask):
    """A symbolic resharding task that puts task info in remote workers."""

    def __init__(self, task_spec, collective_group, src_mesh, dst_mesh):
        super().__init__(task_spec, collective_group, src_mesh, dst_mesh)
        # Dict of worker -> ((offset, rank, gpu index))
        self._sender_tasks = {w: [] for w in self.src_mesh.workers}
        # Dict of worker -> ((indices, rank, gpu index))
        self._receiver_tasks = {w: [] for w in self.dst_mesh.workers}
        self.allgather_uuid = None

        self.send_worker_task_ids = {}
        self.recv_worker_task_ids = {}

        self.task_dones = []

        # generate the above states
        self._compile()
        # print(self.__str__()+"\n")

    @property
    def sender_tasks(self):
        """Return sender sub-tasks."""
        return self._sender_tasks

    @property
    def receiver_tasks(self):
        """Return receiver sub-tasks."""
        return self._receiver_tasks

    def _compile(self):
        """
        Generate all send, recv, and allgather tasks.

        This function does the following:
        (1) generate send, recv, and allgather tasks (if needed),
        (2) put all tasks to their corresponding MeshHostWorkers.
        (3) pre-generate NCCL communicators for those tasks.
        """
        self._compile_send_recv_tasks()
        if not global_config.debug_with_pipeshard_runtime:
            self.put_all_tasks()

    def put_all_tasks(self):
        """
        Put all send, recv and allgather tasks to their MeshHostWorkers
        """
        # put send and recv tasks
        task_dones = []
        temp_worker = None
        for worker, task in self.sender_tasks.items():
            uuid = next_resharding_task_uuid()
            if isinstance(worker, VirtualWorker):
                for actor, idx in self.collective_group.worker_to_rank_map.items():
                    if idx == worker.index:
                        temp_worker = actor
                worker = temp_worker
            self.send_worker_task_ids[worker] = uuid

            if not isinstance(worker, VirtualWorker):
                task_dones.append(
                    worker.put_resharding_send_task.remote(
                        uuid, task, self.collective_group.group_name))
        for worker, task in self.receiver_tasks.items():
            uuid = next_resharding_task_uuid()
            if isinstance(worker, VirtualWorker):
                for actor, idx in self.collective_group.worker_to_rank_map.items():
                    if idx == worker.index:
                        temp_worker = actor
                worker = temp_worker
            self.recv_worker_task_ids[worker] = uuid
            if not isinstance(worker, VirtualWorker):
                task_dones.append(
                    worker.put_resharding_recv_task.remote(
                        uuid, task, self.collective_group.group_name))
        if len(task_dones) > 0:
            ray.get(task_dones)

        # put allgather tasks
        task_dones = []
        if self.is_local_allgather_task:
            self.allgather_uuid = uuid = next_mesh_executable_uuid()
            task_spec = self.task_spec
            hlo = compile_allgather(task_spec.aval.shape, task_spec.aval.dtype,
                                    task_spec.dst_sharding_spec,
                                    task_spec.final_dst_spec,
                                    np.prod(self.dst_mesh.shape))

            for worker in self.dst_mesh.workers:
                if isinstance(worker, VirtualWorker):
                    for actor, idx in self.collective_group.worker_to_rank_map.items():
                        if idx == worker.index:
                            temp_worker = actor
                    worker = temp_worker
                if not isinstance(worker, VirtualWorker):
                    task_dones.append(
                        worker.put_executable.remote(uuid, UtilMeshWorkerExecutable,
                                                     hlo))
        if len(task_dones) > 0:
            ray.get(task_dones)

    def create_resharding_communicators(self):
        """Create the NCCL communicators in advance."""
        communicator_params = set()
        for worker, recv_tasks in self.receiver_tasks.items():
            if isinstance(worker, VirtualWorker):
                dst_rank = worker.index
            else:
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
        task_dones = []
        for param in communicator_params:
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
        dtype = self.task_spec.src.aval.dtype

        # print("order: ", self.task_spec.strategy.order)
        for i, k, j in self.task_spec.strategy.order:
            spec_plan = self.task_spec.strategy.per_spec_plans[i]
            dst_tile, src_tiles, indices_in_dst_tiles = (
                self.task_spec.dst_tile_to_src_tiles_map[i])
            replica_index, receiver = k, dst_tile.replica_device_strs[k]
            _, _, indices_in_dst_tile = (j, src_tiles[j],
                                         indices_in_dst_tiles[j])

            # Get args for an empty buffer
            receiver_device_id = (
                self.collective_group.device_str_to_device_id_map[receiver])
            receiver_worker = (
                self.collective_group.device_str_to_mesh_worker_map[receiver])
            dtype = self.task_spec.src.aval.dtype
            # Get args for send/recv
            senders = [
                spec_plan[replica_index][src_tile_index]
                for src_tile_index, _ in enumerate(src_tiles)
            ]
            receiver_rank, receiver_gpu_idx = (
                self.collective_group.device_str_to_rank_map[receiver])
            recv_tile_specs = []
            for sender_idx, sender in enumerate(senders):
                # Sender's task
                sender_worker = (
                    self.collective_group.device_str_to_mesh_worker_map[sender])
                src_device_id = (
                    self.collective_group.device_str_to_device_id_map[sender])
                self._sender_tasks[sender_worker].append(
                    ReshardingSendSpec(
                        src_device_id,
                        ReshardingTileSpec(src_tiles[sender_idx].offset,
                                           receiver_rank, receiver_gpu_idx)))
                # Receiver's task
                sender_rank, sender_gpu_idx = \
                    self.collective_group.device_str_to_rank_map[sender]
                indices_in_dst_tile = indices_in_dst_tiles[sender_idx]
                recv_tile_specs.append(
                    ReshardingTileSpec(indices_in_dst_tile, sender_rank,
                                       sender_gpu_idx))
            receiver_task = ReshardingRecvSpec(receiver_device_id,
                                               dst_tile.tile_shape, dtype,
                                               recv_tile_specs)
            self._receiver_tasks[receiver_worker].append(receiver_task)

    # FIXME(Hao): test the function below; it might be buggy.
    def do_prepared(self, src_array, profiling=False):
        """Execute a task which has been put in the remote workers."""

        result_ref = RemoteArrayRef(self.dst_mesh)

        results = []
        if profiling:
            for worker, uuid in self.send_worker_task_ids.items():
                results.append(
                    worker.profile_resharding_send_task.remote(
                        uuid, src_array.remote_ref.uuid))
            for worker, uuid in self.recv_worker_task_ids.items():
                results.append(
                    worker.profile_resharding_recv_task.remote(
                        uuid, result_ref.uuid))
        else:
            for worker, uuid in self.send_worker_task_ids.items():
                results.append(
                    worker.run_resharding_send_task.remote(
                        uuid, src_array.remote_ref.uuid))
            for worker, uuid in self.recv_worker_task_ids.items():
                results.append(
                    worker.run_resharding_recv_task.remote(
                        uuid, result_ref.uuid))
            logger.debug("Precompiled tasks launched.")
            ray.get(results)
        # Now construct the distributed array
        dst_array = DistributedArray(self.dst_mesh, src_array.aval,
                                     self.task_spec.dst_sharding_spec,
                                     result_ref, self.task_spec.dst_indices)
        if profiling:
            return results
        return dst_array

    def __str__(self):
        return (f"ReshardingTask(shape: {self.task_spec.aval.shape}, "
                f"mesh_id: {self.src_mesh.mesh_id}->{self.dst_mesh.mesh_id},\n"
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
        return hash(
            (self.comm_key, tuple(self.workers), tuple(self.device_ids)))

    def __eq__(self, other):
        if not isinstance(other, CommunicatorConfig):
            return False
        elif self.comm_key != other.comm_key:
            return False
        elif len(self.workers) != len(other.workers):
            return False

        for i in range(len(self.workers)):
            if (self.workers[i] != other.workers[i] or
                    self.device_ids[i] != other.device_ids[i]):
                return False

        return True


class SymbolicBroadcastReshardingTask(ReshardingTask):
    """A Broadcast based symbolic resharding task that puts task info in remote
    workers."""

    def __init__(self, task_spec, collective_group, src_mesh, dst_mesh):
        super().__init__(task_spec, collective_group, src_mesh, dst_mesh)
        # task is a dict: (i, src_tile_index)->ReshardingBroadcastSpec
        self._broadcast_tasks = {
            host: {} for host in self.src_mesh.workers + self.dst_mesh.workers
        }
        self.broadcast_worker_task_ids = {}
        self.communicator_configs = set()

        # generate the above states
        self._compile()
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

        if not global_config.debug_with_pipeshard_runtime:
            self.put_all_tasks()

    def put_all_tasks(self):
        """Put all tasks to their corresponding MeshHostWorkers."""
        task_dones = []
        for worker, task in self._broadcast_tasks.items():
            uuid = next_resharding_task_uuid()
            if isinstance(worker, VirtualWorker):
                for actor, idx in self.collective_group.worker_to_rank_map.items():
                    if idx == worker.index:
                        temp_worker = actor
                worker = temp_worker
            self.broadcast_worker_task_ids[worker] = uuid
            # print(worker, uuid, task)
            if not isinstance(worker, VirtualWorker):
                task_dones.append(
                    worker.put_resharding_broadcast_task.remote(
                        uuid, task, self.collective_group.group_name))

        ray.get(task_dones)

    def _compile_broadcast_tasks(self):
        """Compile broadcast tasks."""
        dtype = self.task_spec.src.aval.dtype

        # print("order: ", self.task_spec.strategy.order)
        for i, j in self.task_spec.strategy.order:
            spec_plan = self.task_spec.strategy.per_spec_plans[i]
            dst_tile, src_tiles, indices_in_dst_tiles = (
                self.task_spec.dst_tile_to_src_tiles_map[i])
            src_tile, indices_in_dst_tile = (src_tiles[j],
                                             indices_in_dst_tiles[j])

            sender = spec_plan[j]
            sender_worker = (
                self.collective_group.device_str_to_mesh_worker_map[sender])
            broadcast_group = (i, j)
            devices = [sender] + dst_tile.replica_device_strs
            comm_key = "$".join(devices)
            world_size = len(devices)

            comm_config = CommunicatorConfig(comm_key)

            group_spec = self._broadcast_tasks[sender_worker].setdefault(
                broadcast_group,
                ReshardingBroadcastSpec(comm_key=comm_key,
                                        world_size=world_size,
                                        devices_ids=[
                                            self.collective_group.
                                            device_str_to_device_id_map[sender]
                                        ],
                                        devices_global_rank=[0],
                                        tensor_slices=[src_tile.offset],
                                        recv_tile_shape=src_tile.tile_shape,
                                        dtype=dtype))
            comm_config.add(
                sender_worker,
                self.collective_group.device_str_to_device_id_map[sender])

            for replica_index, receiver in enumerate(
                    dst_tile.replica_device_strs):
                receiver_worker = (self.collective_group.
                                   device_str_to_mesh_worker_map[receiver])
                group_spec = self._broadcast_tasks[receiver_worker].setdefault(
                    broadcast_group,
                    ReshardingBroadcastSpec(comm_key=comm_key,
                                            world_size=world_size,
                                            devices_ids=[],
                                            devices_global_rank=[],
                                            tensor_slices=[],
                                            recv_tile_shape=dst_tile.tile_shape,
                                            dtype=dtype))

                group_spec.devices_ids.append(
                    self.collective_group.device_str_to_device_id_map[receiver])
                group_spec.devices_global_rank.append(1 + replica_index)
                group_spec.tensor_slices.append(indices_in_dst_tile)
                comm_config.add(
                    receiver_worker,
                    self.collective_group.device_str_to_device_id_map[receiver])

            self.communicator_configs.add(comm_config)

        return self._broadcast_tasks

    def create_resharding_communicators(self):
        """Create the NCCL communicators for broadcast in advance."""
        group_name = self.collective_group.group_name
        for config in self.communicator_configs:
            task_dones = []
            worker_to_devices_and_global_ranks = {}
            world_size = len(config.workers)
            for global_rank, (worker, device_id) in enumerate(
                    zip(config.workers, config.device_ids)):
                if worker not in worker_to_devices_and_global_ranks:
                    worker_to_devices_and_global_ranks[worker] = {
                        "device_ids": [],
                        "global_ranks": []
                    }
                worker_to_devices_and_global_ranks[worker]["device_ids"].append(
                    device_id)
                worker_to_devices_and_global_ranks[worker][
                    "global_ranks"].append(global_rank)

            sender_worker = config.workers[0]
            nccl_uid = ray.get(
                sender_worker.generate_nccl_uid.remote(group_name))

            for worker, devices_info in (
                    worker_to_devices_and_global_ranks.items()):
                task_dones.append(
                    worker.init_broadcast_communicator.remote(
                        group_name, config.comm_key, world_size,
                        devices_info["device_ids"],
                        devices_info["global_ranks"], nccl_uid))
            ray.get(task_dones)

    def __str__(self):
        return (f"B-ReshardingTask(shape: {self.task_spec.aval.shape}, "
                f"mesh_id: {self.src_mesh.mesh_id}->{self.dst_mesh.mesh_id},\n"
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
        self.instantiated = False
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
                    i * dst_mesh.num_devices_per_host + j]
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
        if self.instantiated:
            return
        options = {
            "group_name": self.group_name,
            "world_size": len(self.mesh_workers),
            "ranks": [i for i, _ in enumerate(self.mesh_workers)],
            "backend": "nccl"
        }
        col.create_collective_group(self.mesh_workers, **options)
        self.instantiated = True

    def instantiate_now(self):
        """Instantiate the collective group eagerly (but not communicators)."""
        if self.instantiated:
            return
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
        self.instantiated = True

    def destroy(self):
        """Destroy the NCCL collective group at exit."""
        logger.debug(f"Recycling the collective group: {self.group_name}.")
        for worker in self.mesh_workers:
            # This remote call will remove ray named actors (hence it is
            # necessary)
            ray.get(worker.destroy_collective_group.remote(self.group_name))
        # Destroy the declared named actor in ray
        self._destroy_info_actor()
        self.instantiated = False

    def _destroy_info_actor(self):
        name = "info_" + self.group_name
        try:
            store = ray.get_actor(name)
            ray.kill(store)
        except ValueError:
            pass


class ReshardingTaskSpec:
    """
    A helper class specifies how to perform cross-mesh resharding for two
    arrays.

    Args:
        src_array (VirtualDistributedArray): the source VirtualDistributedArray.
        dst_array (VirtualDistributedArray): the destination
            VirtualDistributedArray.
    """

    def __init__(self, src_array, dst_array, final_dst_spec):
        self.src = src_array
        self.dst = dst_array
        self._dst_tile_to_src_tiles_map = None
        self._strategy = None
        self.final_dst_spec = final_dst_spec

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
        - src_tile_slices: a list of TileSlice objects from src, corresponding
            to this dst_tile
        - indices_in_dst_tile: a list of slicers. Each slicer is a list of slice
            objects, corresponding to
            a TileSlice in src_tile_slices, representing the indices of this
            TileSlice in dst_tile.
        """
        if not self._dst_tile_to_src_tiles_map:
            self._dst_tile_to_src_tiles_map = self.generate_src_dst_map()
        return self._dst_tile_to_src_tiles_map

    def generate_src_dst_map(self):
        """
        Analyzes the src and dst array and generate the
        dst_tile_to_src_tiles_map.

        It aims to tell the needed collective group and communication pattern.

        Returns:
            dst_tile_to_src_tiles_map (tuple[tile, tileslices, indices]):
                see the docstring of `dst_tile_to_src_tiles_map`.
        """
        dst_tile_to_src_tiles_map = []
        for tile in self.dst.tiles.flatten():
            # loop over each tile
            src_tile_slices, indices_in_dst_tile = (
                self._look_up_dst_tile_from_src(tile))
            dst_tile_to_src_tiles_map.append(
                (tile, src_tile_slices, indices_in_dst_tile))
        return dst_tile_to_src_tiles_map

    def _look_up_dst_tile_from_src(self, tile):
        """
        Look up all related tiles from the source array for a given destination
        tile.

        See the docstring in dst_tile_to_src_tiles_map() for more details.
        """
        # For each dim in the dst tile, find all the related tiles, and ragged
        # values on that dim in src_tiles.
        # To record that, for each dim, we make a tuple containing the first and
        # last index of tiles in src array that intersects with the dst tile:
        # Shards between [start, end) are involved; Left included, right not
        # included.
        related_tile_start_end = [tuple()] * self.src.tensor_rank

        # Meanwhile, for each dim, for the first and end tile, we make a tuple
        # recording the slicing offset:
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
            # if falling on the middle a src tile, increase the index of the
            # final tile by 1.
            if end_tile_offset:
                end_tile = end_tile + 1
            # if falling on the end of a src tile, the offset should be
            # [0: tile_length]
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
                    right_in_dst_tile = (left_in_dst_tile +
                                         tile_length_on_this_dim)
                    indices.append(slice(left_in_dst_tile, right_in_dst_tile))
            # construct a new tile slice
            this_tileslice = TileSlice(
                self.src.tiles[tuple(tile_index_absolute)], offset=offsets)
            src_tileslices.append(this_tileslice)
            indices_in_dst_tile.append(indices)
        return src_tileslices, indices_in_dst_tile

    def set_resharding_strategy(self, strategy):
        """Now the strategy is np.array(dtype=str) to specify connections
        between src tiles and dst tile."""
        self._strategy = strategy

    @property
    def strategy(self):
        """Return the communication strategy for this resharding task spec."""
        if not self._strategy:
            raise RuntimeError(
                "Generate and set strategy in the cross-mesh communicator "
                "first.")
        return self._strategy

    def generate_naive_order(self, mode):
        """Return the naive order to submit resharding tasks."""

        order = []
        if mode == "sendrecv":
            for i, (dst_tile, src_tiles,
                    _) in enumerate(self.dst_tile_to_src_tiles_map):
                for k, _ in enumerate(dst_tile.replica_device_strs):
                    for j, _ in enumerate(src_tiles):
                        order.append((i, k, j))
        elif mode == "broadcast":
            for i, (_, src_tiles,
                    _) in enumerate(self.dst_tile_to_src_tiles_map):
                for j, _ in enumerate(src_tiles):
                    order.append((i, j))
        else:
            raise NotImplementedError

        return order

    def get_participant_device_strs(self):
        """Identify all participant device strs (for NCCL setup) in this task
        spec."""
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
        ret_str += f"{self.src_sharding_spec} -> {self.dst_sharding_spec}"
        if self.final_dst_spec != self.dst_sharding_spec:
            ret_str += f" -(allgather)-> {self.final_dst_spec}"
        ret_str += ";"
        return ret_str


class ReshardingStrategy:
    """A data class for storing resharding communication information.

    Args:
        mode (str): Two choices:["sendrecv", "broadcast"].
        per_spec_plans (List[np.ndarray]): `per_spec_plan` is a list a np array,
            with length as len(spec.dst_tile_to_src_tiles_map), each array is
            with shape [len(dst_tile.devices), len(src_tiles)]; it specifies for
            each replica of a dst tile, how it should get the data from
            src_tiles (src tile replicas).
        order (List[Tuple(int, ...)]): in which order we should submit
            these nccl communication operation into cuda stream. When mode
            is "sendrecv", order is of type List[Tuple(int, int)];
            Otherwise, order is of type List[Tuple(int, int, int)].
        is_local_allgather (bool): if this strategy involves post allgather
            operations.
    """

    def __init__(self, mode, per_spec_plans, order, is_local_allgather):
        self.mode = mode
        self.per_spec_plans = per_spec_plans
        self.order = order
        self.is_local_allgather = is_local_allgather


class CrossMeshCommunicator:
    """
    Communicator for cross-mesh resharding.

    Given the pipeline schedule and stages, the class analyzes them and
    generates:
    - resharding specs (see docstring of `ReshardingTaskSpec`),
    - resharding strategies (see docstring of `ReshardingStrategy`).
    This communicator only takes care of compilation-time work, and does not
    get involved with physical meshes, buffer creations, or other runtime work.

    Args:
        sharded_stages (Sequence[XlaShardedPipelineComputation]): list of stages
            to form the pipeline.
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
        # Generate a send/recv strategies for all resharding tasks by looking
        # at their load.
        for src_mesh_idx, dst_mesh_idx, var_spec_map in self.task_spec_iter():
            for _, spec in var_spec_map.items():
                if global_config.resharding_mode == "send_recv":
                    strategy = (self._generate_send_recv_resharding_strategy(
                        spec, self._schedule.meshes[src_mesh_idx],
                        self._schedule.meshes[dst_mesh_idx]))
                else:
                    strategy = (self._generate_broadcast_resharding_strategy(
                        spec, self._schedule.meshes[src_mesh_idx],
                        self._schedule.meshes[dst_mesh_idx]))
                spec.set_resharding_strategy(strategy)

    @property
    def num_mesh(self):
        """Number of meshes in the schedule."""
        return self._schedule.num_mesh

    @staticmethod
    def _rewrite_allgather_spec(sharding_spec, dst_num_hosts, var_shape):
        """
        Given a sharding spec, if use_local_allgather is on and the tensor
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

        if not global_config.use_local_allgather:
            return sharding_spec
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
            return sharding_spec
        assert len(replicated_mesh_dim) == 1, "Only support 1D and 2D mesh"

        # create chunk axis to tensor dim mapping
        chunk_axis_to_tensor_dim = []
        for tensor_dim, dim_spec in enumerate(sharding_spec.sharding):
            if isinstance(dim_spec, pxla.Chunked):
                for chunk_idx in range(len(dim_spec.chunks)):
                    chunk_axis_to_tensor_dim.append((tensor_dim, chunk_idx))

        # TODO(yonghao): add a global config for wheter cross-node allgather is
        # allowed
        node_mesh_mapping = sharding_spec.mesh_mapping[0]
        node_chunk = 1
        if isinstance(node_mesh_mapping, pxla.ShardedAxis):
            tensor_dim, _ = chunk_axis_to_tensor_dim[node_mesh_mapping.axis]
            node_chunk = _get_chunk_value(sharding_spec.sharding[tensor_dim])
        if node_chunk < dst_num_hosts:
            return sharding_spec

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
                dim_local_mapping.append((tensor_dim, chunk_idx))

                replica //= new_chunk
                if replica == 1:
                    break
            if replica != 1:
                logger.warning(
                    "ReshardingTask is not fully sharded, this causes "
                    "redundant communication.")
            if len(dim_local_mapping) != 0:
                squeezed_mesh_mapping[mesh_dim] = dim_local_mapping

        mesh_mapping = _get_mesh_mapping(sharding, sharding_spec.mesh_mapping,
                                         squeezed_mesh_mapping)
        new_sharding_spec = pxla.ShardingSpec(sharding, mesh_mapping)
        # sorted by (tensor dim, chunk idx, mesh dim)
        return new_sharding_spec

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
        # Each element is a dict: the name of variables are keys, ReshardingSpec
        # are values.
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

                final_dst_spec = dst_sharding_spec
                if global_config.resharding_mode == "send_recv":
                    dst_sharding_spec = self._rewrite_allgather_spec(
                        dst_sharding_spec, dst_mesh.num_hosts, var.aval.shape)

                src_array = VirtualDistributedArray(
                    device_mesh=src_mesh,
                    aval=var.aval,
                    sharding_spec=src_sharding_spec)
                dst_array = VirtualDistributedArray(
                    device_mesh=dst_mesh,
                    aval=var.aval,
                    sharding_spec=dst_sharding_spec)
                task_spec = ReshardingTaskSpec(src_array, dst_array,
                                               final_dst_spec)
                self.resharding_specs[src_mesh_index][dst_mesh_index][
                    var] = task_spec

    def task_spec_iter(self):
        """A convenient iterator over all activated task specs."""
        for i in range(self.num_mesh):
            for j in range(self.num_mesh):
                if not self.resharding_specs[i][j]:
                    continue
                yield i, j, self.resharding_specs[i][j]

    @staticmethod
    def get_resources_info_in_mesh(mesh):
        device_strs = []
        device_host_map = {}
        nic_constraints = []

        for i in range(mesh.num_hosts):
            ip = mesh.host_info[i]["NodeManagerAddress"]
            one_nic_constraint = []
            for device in mesh.devices[i]:
                device_str = device_id_to_str(ip, device)
                device_strs.append(device_str)
                one_nic_constraint.append(device_str)
                #TODO: Here we assume there is only one NIC in one host.
                device_host_map[device_str] = ip
            nic_constraints.append(one_nic_constraint)
        return device_strs, device_host_map, nic_constraints

    @staticmethod
    def _get_hardware_info_for_loadbalance(src_mesh, dst_mesh):
        src_mesh_devices, src_device_host_map, src_nic_constraints = (
            CrossMeshCommunicator.get_resources_info_in_mesh(src_mesh))
        dst_mesh_devices, dst_device_host_map, dst_nic_constraints = (
            CrossMeshCommunicator.get_resources_info_in_mesh(dst_mesh))
        device_host_map = {**src_device_host_map, **dst_device_host_map}
        nic_constraints = src_nic_constraints + dst_nic_constraints
        return (src_mesh_devices, dst_mesh_devices, device_host_map,
                nic_constraints)

    @staticmethod
    def _generate_send_recv_resharding_strategy_by_loads(
            spec: ReshardingTaskSpec, src_loads, dst_loads):
        """Generate the resharding strategy by balancing loads."""
        is_local_allgather = spec.final_dst_spec != spec.dst_sharding_spec
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

        strategy = ReshardingStrategy("sendrecv", per_spec_plans,
                                      spec.generate_naive_order("sendrecv"),
                                      is_local_allgather)
        return strategy

    def _generate_send_recv_resharding_strategy(self, spec: ReshardingTaskSpec,
                                                src_mesh, dst_mesh):
        if global_config.resharding_loadbalance_mode == "normal":
            strategy = (self._generate_send_recv_resharding_strategy_by_loads(
                spec, self._sender_loads, self._receiver_loads))
        elif global_config.resharding_loadbalance_mode == "no_loadbalance":
            strategy = (
                self._generate_send_recv_resharding_strategy_by_no_load(spec))
        elif global_config.resharding_loadbalance_mode in ([
                "loadbalance_size", "loadbalance_order"
        ]):
            strategy = self.\
            _generate_send_recv_resharding_strategy_by_loadbalance(
                spec, src_mesh, dst_mesh)
        else:
            raise NotImplementedError()
        return strategy

    def _generate_broadcast_resharding_strategy(self, spec: ReshardingTaskSpec,
                                                src_mesh, dst_mesh):
        if global_config.resharding_loadbalance_mode == "normal":
            strategy = (self._generate_broadcast_resharding_strategy_by_loads(
                spec, self._sender_loads, self._receiver_loads))
        elif global_config.resharding_loadbalance_mode == "no_loadbalance":
            strategy = (
                self._generate_broadcast_resharding_strategy_by_no_load(spec))
        elif global_config.resharding_loadbalance_mode in [
                "loadbalance_size", "loadbalance_order"
        ]:
            strategy = (
                self._generate_broadcast_resharding_strategy_by_loadbalance(
                    spec, src_mesh, dst_mesh))
        else:
            raise NotImplementedError()
        return strategy

    @staticmethod
    def _generate_send_recv_resharding_strategy_by_no_load(
            spec: ReshardingTaskSpec):
        """Generate the resharding strategy by balancing loads."""
        is_local_allgather = spec.final_dst_spec != spec.dst_sharding_spec
        per_spec_plans = []
        for dst_tile, src_tileslices, _ in spec.dst_tile_to_src_tiles_map:
            # plan is a 2D array
            per_spec_plan = np.empty(
                (len(dst_tile.replica_device_strs), len(src_tileslices)),
                dtype=object)
            for receiver_idx, _ in enumerate(dst_tile.replica_device_strs):
                for src_tileslice_idx, src_tileslice in enumerate(
                        src_tileslices):
                    sender = src_tileslice.replica_device_strs[0]
                    # Choose an arbitrary sender without considering loads
                    per_spec_plan[receiver_idx][src_tileslice_idx] = sender
            per_spec_plans.append(per_spec_plan)

        strategy = ReshardingStrategy("sendrecv", per_spec_plans,
                                      spec.generate_naive_order("sendrecv"),
                                      is_local_allgather)
        return strategy

    @staticmethod
    def _generate_send_recv_resharding_strategy_by_loadbalance(
            spec, src_mesh, dst_mesh):
        """
            Generate the send/recv-based resharding strategy by balancing
            loads and along time.
        """

        # pre-process
        src_mesh_devices, dst_mesh_devices, device_host_map, nic_constraints = (
            CrossMeshCommunicator._get_hardware_info_for_loadbalance(
                src_mesh, dst_mesh))

        works = []
        for i, (dst_tile, src_tileslices,
                _) in enumerate(spec.dst_tile_to_src_tiles_map):
            for receiver in dst_tile.replica_device_strs:
                for j, src_tileslice in enumerate(src_tileslices):
                    senders = src_tileslice.replica_device_strs
                    data_size = src_tileslice.tile_size
                    works.append(
                        SingleReshardingLoadBalancingWork(
                            senders, [receiver], data_size))

        # solve and get solution
        task = ReshardingLoadBalancingTaskSolver(src_mesh_devices,
                                                 dst_mesh_devices,
                                                 device_host_map, works,
                                                 nic_constraints)

        sol_assigned_sender, sol_order = task.solve()

        # post-process
        per_spec_plans = []
        rank_to_idx = []
        cnt = 0
        for i, (dst_tile, src_tileslices,
                _) in enumerate(spec.dst_tile_to_src_tiles_map):
            per_spec_plan = np.empty(
                (len(dst_tile.replica_device_strs), len(src_tileslices)),
                dtype=object)
            for k, receiver in enumerate(dst_tile.replica_device_strs):
                for j, src_tileslice in enumerate(src_tileslices):
                    sender = sol_assigned_sender[cnt]
                    per_spec_plan[k][j] = sender
                    rank_to_idx.append((i, k, j))
                    cnt += 1
            per_spec_plans.append(per_spec_plan)

        order = [rank_to_idx[i] for i in sol_order]
        is_local_allgather = spec.final_dst_spec != spec.dst_sharding_spec
        strategy = ReshardingStrategy("sendrecv", per_spec_plans, order,
                                      is_local_allgather)
        return strategy

    @staticmethod
    def _generate_broadcast_resharding_strategy_by_no_load(
            spec: ReshardingTaskSpec):
        """
            Generate the broadcast-based resharding strategy by balancing
            loads. For each tile, I not only allow one source to provide
            the tile.
        """
        # pylint: disable=unused-argument
        per_spec_plans = []
        for _, src_tileslices, _ in spec.dst_tile_to_src_tiles_map:
            per_spec_plan = np.empty((len(src_tileslices),), dtype=object)

            for src_tileslice_idx, src_tileslice in enumerate(src_tileslices):
                per_spec_plan[
                    src_tileslice_idx] = src_tileslice.replica_device_strs[0]
            per_spec_plans.append(per_spec_plan)
        strategy = ReshardingStrategy("broadcast", per_spec_plans,
                                      spec.generate_naive_order("broadcast"),
                                      None)
        return strategy

    @staticmethod
    def _generate_broadcast_resharding_strategy_by_loadbalance(
            spec, src_mesh, dst_mesh):
        """
            Generate the broadcast-based resharding strategy by balancing
            loads and along time.
        """

        # pre-process
        src_mesh_devices, dst_mesh_devices, device_host_map, nic_constraints = (
            CrossMeshCommunicator._get_hardware_info_for_loadbalance(
                src_mesh, dst_mesh))

        works = []
        for i, (dst_tile, src_tileslices,
                _) in enumerate(spec.dst_tile_to_src_tiles_map):
            for j, src_tileslice in enumerate(src_tileslices):
                senders = src_tileslice.replica_device_strs
                receivers = dst_tile.replica_device_strs
                data_size = src_tileslice.tile_size
                works.append(
                    SingleReshardingLoadBalancingWork(senders, receivers,
                                                      data_size))

        # solve and get solution
        task = ReshardingLoadBalancingTaskSolver(src_mesh_devices,
                                                 dst_mesh_devices,
                                                 device_host_map, works,
                                                 nic_constraints)

        sol_assigned_sender, sol_order = task.solve()

        # post-process
        per_spec_plans = []
        rank_to_idx = []
        cnt = 0
        for i, (dst_tile, src_tileslices,
                _) in enumerate(spec.dst_tile_to_src_tiles_map):
            per_spec_plan = np.empty((len(src_tileslices),), dtype=object)
            for j, src_tileslice in enumerate(src_tileslices):
                sender = sol_assigned_sender[cnt]
                per_spec_plan[j] = sender
                rank_to_idx.append((i, j))
                cnt += 1
            per_spec_plans.append(per_spec_plan)

        order = [rank_to_idx[i] for i in sol_order]
        strategy = ReshardingStrategy("broadcast", per_spec_plans, order, None)
        return strategy

    @staticmethod
    def _generate_broadcast_resharding_strategy_by_loads(
            spec, src_loads, dst_loads):
        """
            Generate the broadcast-based resharding strategy by balancing loads.
            For each tile, I not only allow one source to provide the tile.
        """
        # pylint: disable=unused-argument
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
        strategy = ReshardingStrategy("broadcast", per_spec_plans,
                                      spec.generate_naive_order("broadcast"),
                                      None)
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


SingleReshardingLoadBalancingWork = namedtuple(
    "SingleReshardingLoadBalancingWork", ["senders", "receivers", "data_size"])
SingleAbstractedLoadBalancingWork = namedtuple(
    "SingleAbstractedLoadBalancingWork",
    ["sender_ids", "receiver_ids", "duration"])


class ReshardingLoadBalancingTaskSolver:
    """This is class of solver for load balancing problem"""

    def __init__(self,
                 src_mesh_devices,
                 dst_mesh_devices,
                 device_host_map,
                 works,
                 nic_contraints,
                 host_bridge_contraints=None):
        """We define the load balancing problem in resharding problem.
        Here both send_recv and broadcast based implementation could
        be formulated in this way.

        Args:
            src_mesh_devices: All gpus in src mesh.
            dst_mesh_devices: All gpus in dst mesh.
            device_host_map: a map from device to its corresponding host.
            works (List[SingleReshardingLoadBalancingWork]): all works to
                be scheduled in this task.
            nic_contraints (List[List[device]]): each list[device] contains
                a set of devices that competes for the same NIC.
                Now I assmue sender and receiver do not share NIC.
                The assumption is met in nic_contraints.
                I assume these constraints are disjoint sets.
        """
        self.src_mesh_devices = src_mesh_devices
        self.dst_mesh_devices = dst_mesh_devices
        self.all_devices = list(
            set(src_mesh_devices).union(set(dst_mesh_devices)))
        self.device_host_map = device_host_map
        self.works = works
        self.nic_contraints = nic_contraints
        self.host_bridge_contraints = host_bridge_contraints

        # self.print_task()

    def solve(self):
        """
            Return two data
            1. The first List[device] represents which sender to choose
               for each work.
            2. The second List[int] represents the order to execute
               these works.
        """

        # Deal with the case when a src device share the same NIC with a tar
        # device. Now I assmue they do not share NIC. The assumption is met
        # in nic_contraints so we do not need to deal with it in this method.

        tmp_device_to_worker_id_map = {
            device: idx for idx, device in enumerate(self.all_devices)
        }
        for nic_contraint in self.nic_contraints:
            min_id = min(
                tmp_device_to_worker_id_map[device] for device in nic_contraint)
            for device in nic_contraint:
                tmp_device_to_worker_id_map[device] = min_id

        device_to_worker_id_map = {}
        worker_id_to_devices = {}
        n_workers = 0
        for idx, device in enumerate(self.all_devices):
            if tmp_device_to_worker_id_map[device] == idx:
                device_to_worker_id_map[device] = n_workers
                worker_id_to_devices[n_workers] = [device]
                n_workers += 1
            else:
                group_head_device = self.all_devices[
                    tmp_device_to_worker_id_map[device]]
                worker_id = device_to_worker_id_map[group_head_device]
                device_to_worker_id_map[device] = worker_id
                worker_id_to_devices[worker_id].append(device)

        abstract_works = []
        for work in self.works:
            sender_ids = set()
            for sender in work.senders:
                sender_ids.add(device_to_worker_id_map[sender])
            sender_ids = list(sender_ids)
            sender_ids.sort()
            receiver_ids = set()
            for receiver in work.receivers:
                receiver_ids.add(device_to_worker_id_map[receiver])
            receiver_ids = list(receiver_ids)
            receiver_ids.sort()
            time_spent = work.data_size

            abstract_works.append(
                SingleAbstractedLoadBalancingWork(sender_ids, receiver_ids,
                                                  time_spent))

        if global_config.resharding_loadbalance_mode == "loadbalance_size":
            task = LoadBalancingOverSizeTaskSolver(n_workers, abstract_works)
        else:
            if global_config.loadbalance_order_algo == "search":
                task = LoadBalancingTaskSolverSearchAlgo(
                    n_workers, abstract_works)
            else:
                task = LoadBalancingTaskSolverGreedyAlgo(
                    n_workers, abstract_works)

        sol_assigned_sender_id, sol_order = task.solve()

        sol_assigned_sender = []
        for work, worker_id in zip(self.works, sol_assigned_sender_id):
            selected_sender = None
            for sender in work.senders:
                if device_to_worker_id_map[sender] == worker_id:
                    selected_sender = sender
                    break
            assert selected_sender is not None
            sol_assigned_sender.append(selected_sender)
        return sol_assigned_sender, sol_order

    def print_task(self):
        print("\nTask[START]")
        print(f"src_mesh_devices: {self.src_mesh_devices}")
        print(f"dst_mesh_devices: {self.dst_mesh_devices}")
        print(f"device_host_map: {self.device_host_map}")
        print("works:")
        for work in self.works:
            print(work)
        print("nic_contraints:")
        for contraint in self.nic_contraints:
            print(contraint)
        print("Task[END]\n")


class AbstractedLoadBalancingTaskSolver(ABC):
    """This is class of solver for abstracted load balancing problem"""

    def __init__(self, n_workers, works):
        """We abstract the load balancing problem into this mathematically
        clear form.

        Args:
            n_workers (int): The total number of single threaded
                workers in this loadbalancing task.
            works (List[SingleAbstractedLoadBalancingWork]): all works to
                be scheduled in this task.
        """
        self.n_workers = n_workers
        self.n_works = len(works)
        self.works = works
        self.loads = [0 for _ in range(n_workers)]

        # self.print_task()

    @abstractmethod
    def solve(self):
        """
            Return two list[int] of length n_works
            1. The first represents which sender to choose for each work.
            2. The second represents the order to execute these works.
        """
        raise NotImplementedError

    def print_task(self):
        print("AbstractedTask[START]")
        print(f"n_workers: {self.n_workers}")
        print("works:")
        for work in self.works:
            print(work)
        print("AbstractedTask[END]")


class LoadBalancingTaskSolverGreedyAlgo(AbstractedLoadBalancingTaskSolver):
    """Implementation of load balance: use randomized greedy algorithm"""

    def find_one_random_concurrent_set_of_works(self, works_ids):
        """This method finds one set of works that could be run
           concurrently.

        Args:
            works_ids (List[int]): The ids of works that could be
                selected.

        Returns:
            one_concurrent_works_ids (list[int]): The ids of works
                selected in this method.
            one_concurrent_selected_senders (list[int]): The assigned
                senders for the selected works.
        """

        def probability_of_being_selected(loads):
            # these weights could be more carefully tuned.
            max_weight = max(loads)
            weights = [max_weight - weight + 1 for weight in loads]
            return weights

        used = [False for _ in range(self.n_workers)]
        perm = np.random.permutation(np.array(works_ids))
        one_concurrent_works_ids = []
        one_concurrent_selected_senders = []
        for i in perm:
            work = self.works[i]
            receivers_availability = True
            for receiver in work.receiver_ids:
                if used[receiver]:
                    receivers_availability = False
                    break
            if not receivers_availability:
                continue

            available_senders = []
            for sender in work.sender_ids:
                if not used[sender]:
                    available_senders.append(sender)
            if not available_senders:
                continue

            weights = probability_of_being_selected(
                [self.loads[sender] for sender in available_senders])
            selected_sender = random.choices(available_senders,
                                             weights=weights)[0]

            used[selected_sender] = True
            for receiver in work.receiver_ids:
                used[receiver] = True

            one_concurrent_works_ids.append(i)
            one_concurrent_selected_senders.append(selected_sender)
        return one_concurrent_works_ids, one_concurrent_selected_senders

    def find_best_concurrent_set_of_works(self, works_ids, n_rounds=100):
        """
            One simple strategy is that everytime we choose the maximum number
            of works and minimize std and put them into the sequence.
            The simple logic behind is to maximize concurrency.

        Args:
            works_ids (List[int]): All available works waiting for running.
            n_rounds (int, optional): The number of rounds to run for finding
                the best set of works. Defaults to 100.
        """

        def calc_std(data):
            ave = sum(data) / len(data)
            std = (sum((x - ave)**2 for x in data) / len(data))**0.5
            return std

        # def calc_max(A):
        #     return max(A)

        max_num = 0
        min_std = None
        best_concurrent_works_ids = []
        best_concurrent_selected_senders = []
        for _ in range(n_rounds):
            one_concurrent_works_ids, one_concurrent_selected_senders = \
                self.find_one_random_concurrent_set_of_works(works_ids)
            num = len(one_concurrent_works_ids)
            if num < max_num:
                continue

            loads = list(self.loads)
            for work_id, selected_sender in zip(
                    one_concurrent_works_ids, one_concurrent_selected_senders):
                loads[selected_sender] += self.works[work_id].duration

            # here we could use different criterions
            std = calc_std(loads)  # calc_max(loads)
            # std = calc_std(
            # [self.works[i].duration for i in range(one_concurrent_works_ids)]
            # )

            if num > max_num or (num == max_num and std < min_std):
                max_num = num
                min_std = std
                best_concurrent_works_ids = one_concurrent_works_ids
                best_concurrent_selected_senders = (
                    one_concurrent_selected_senders)
        return best_concurrent_works_ids, best_concurrent_selected_senders

    def solve(self):
        sol_assigned_sender_id = [None for _ in range(len(self.works))]
        sol_order = []
        while True:
            available_works_ids = [
                i for i in range(len(self.works)) if i not in sol_order
            ]
            best_concurrent_works_ids, best_concurrent_selected_senders = \
                self.find_best_concurrent_set_of_works(available_works_ids)

            for work_id, sender_id in zip(best_concurrent_works_ids,
                                          best_concurrent_selected_senders):
                sol_order.append(work_id)
                sol_assigned_sender_id[work_id] = sender_id
                self.loads[sender_id] += self.works[work_id].duration

            if len(sol_order) == len(self.works):
                break

        assert None not in sol_assigned_sender_id

        return sol_assigned_sender_id, sol_order


class LoadBalancingTaskSolverSearchAlgo(AbstractedLoadBalancingTaskSolver):
    """Implementation of load balance: use search algorithm with pruning"""

    def __init__(self, n_workers, works):
        super().__init__(n_workers, works)

        self.sol_assigned_sender_id = [None for _ in range(len(self.works))]
        self.sol_order = []
        self.minimal_finish_time = None

        self.cur_assigned_sender_id = [None for _ in range(len(self.works))]
        self.cur_order = []

        self.start_time = time.time()
        self.search_time_threshold = 1

    def evaluate_one_solution(self, assigned_sender_id, order):
        """Given current task-sender assigment and order to submit
           these tasks, this method return the finishing time of each
           receiver for the current schedule as solution.
           To get the finishing time, this method just simulates the
           whole process.

        Args:
            assigned_sender_id: This variable contains idx of sender
                for each task.
            order: The order to submit different tasks.

        Returns:
            current_time (list[int]): the time for each receiver
                after finishing all the tasks assigned to it.
        """
        current_time = [0 for _ in range(self.n_workers)]

        for i in order:
            work = self.works[i]
            sender_id = assigned_sender_id[i]
            mx_time = max([current_time[sender_id]] + [
                current_time[receiver_id] for receiver_id in work.receiver_ids
            ])
            current_time[sender_id] = mx_time + work.duration
            for receiver_id in work.receiver_ids:
                current_time[receiver_id] = mx_time + work.duration
        return current_time

    def heuristic(self, current_time, remained_work_ids):
        """ Given the current time for each receiver to finish
            its assigned works, and the remained work to be
            assigned, this method estimate the minimal amount
            of time to finish all works. If the minimal amount
            of time to finish all works is still longer than
            current best solution, then we could prune the current
            search branch.

        Args:
            current_time (list[int]): the time for each receiver
                after finishing all the tasks assigned to it.
            remained_work_ids (list[int]): The ids of works remained
                to be assigned to workers.

        Returns:
            int: the minimal amount of time to finish all works
                with current assignment and order schedule.
        """
        remained_time_lowerbound = [0 for _ in range(self.n_workers)]
        for i in remained_work_ids:
            work = self.works[i]
            sender_id_with_mintime = -1
            for sender_id in work.sender_ids:
                if sender_id_with_mintime == -1:
                    sender_id_with_mintime = sender_id
                elif (remained_time_lowerbound[sender_id] +
                      current_time[sender_id] <
                      remained_time_lowerbound[sender_id_with_mintime] +
                      current_time[sender_id_with_mintime]):
                    sender_id_with_mintime = sender_id
            # heuristic function could be continuely improved.
            remained_time_lowerbound[sender_id_with_mintime] += work.duration
            for receiver_id in work.receiver_ids:
                remained_time_lowerbound[receiver_id] += work.duration

        max_time = max(
            x + y for x, y in zip(remained_time_lowerbound, current_time))
        return max_time

    def dfs(self, depth):
        """This is the Depth First Search function
           to search the order of submitting works
           and sender for each work.

        Args:
            depth (int): The depth of the DFS; In other
            words, we are deciding the depth_th task in
            order array.
        """
        if time.time() - self.start_time > self.search_time_threshold:
            return

        current_time = self.evaluate_one_solution(self.cur_assigned_sender_id,
                                                  self.cur_order)

        if depth == len(self.works):
            finish_time = max(current_time)
            if (self.minimal_finish_time is None or
                    finish_time < self.minimal_finish_time):
                self.minimal_finish_time = finish_time
                self.sol_assigned_sender_id = list(self.cur_assigned_sender_id)
                self.sol_order = list(self.cur_order)
            return

        remained_work_ids = [
            i for i in range(len(self.works)) if i not in self.cur_order
        ]

        heuristic = self.heuristic(current_time, remained_work_ids)
        if (self.minimal_finish_time is not None and
                heuristic > self.minimal_finish_time):
            return

        for i in remained_work_ids:
            self.cur_order.append(i)
            work = self.works[i]
            for sender_id in work.sender_ids:
                self.cur_assigned_sender_id[i] = sender_id
                self.dfs(depth + 1)
            self.cur_assigned_sender_id[i] = None
            self.cur_order.pop()

    def solve(self):

        self.dfs(depth=0)

        assert None not in self.sol_assigned_sender_id

        return self.sol_assigned_sender_id, self.sol_order


class LoadBalancingOverSizeTaskSolver(AbstractedLoadBalancingTaskSolver):
    """Implementation of load balance: only consider workers' workloads"""

    def __init__(self, n_workers, works):
        super().__init__(n_workers, works)

        self.sol_assigned_sender_id = [None for _ in range(len(self.works))]
        self.sol_order = []

    def solve(self):
        for i, work in enumerate(self.works):
            loads = {sender: self.loads[sender] for sender in work.sender_ids}
            sender = min(loads, key=loads.get)
            self.sol_assigned_sender_id[i] = sender
            self.loads[sender] += work.duration
            self.sol_order.append(i)

        assert None not in self.sol_assigned_sender_id

        return self.sol_assigned_sender_id, self.sol_order
