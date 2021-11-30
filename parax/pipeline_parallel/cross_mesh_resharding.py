"""Cross mesh resharding for pipeline parallelism."""
from collections.abc import Iterable
from dataclasses import dataclass
import logging
from typing import List

import numpy as np
import ray
import ray.util.collective as col
import jax.linear_util as lu
from jax.interpreters import pxla
from jax.interpreters.pxla import Replicated

from parax.device_mesh import DistributedArray, RemoteBufferRef
from parax.pipeline_parallel.computation import XlaShardedPipelineComputation
from parax.global_env import global_config
from parax.util import OrderedSet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

resharding_task_counter = 0


def next_resharding_task_uuid():
    global resharding_task_counter
    resharding_task_counter = (resharding_task_counter + 1) % (1 << 60)
    return resharding_task_counter


class VirtualDistributedArray:
    """
    Distributed Array without allocating remote buffers.

    VDA wrapper differs from DistributedArray (DA) in that:
    (1) it does not allocate a remote buffer at construction;
    (2) its device_mesh attribute is a virtual mesh (not physical).

    Args:
        device_mesh (VirtualPhysicalMesh): the virtual mesh this VDA locates on.
        aval (aval): shape information about the array.
        sharding_spec (ShardingSpec): sharding spec of this array.
    """

    def __init__(self, *, device_mesh, aval, sharding_spec):
        self.device_mesh = device_mesh
        self.aval = aval
        self.sharding_spec = sharding_spec

        self._indices = None
        self._one_replica_buffer_indices = None
        self._tile_assignments = None
        self._tiles = None

        self._sharding_spec_proto = self.sharding_spec.sharding_proto()

    @property
    def tensor_shape(self):
        """Return the shape of the original tensor."""
        return self.aval.shape

    @property
    def tensor_rank(self):
        """Return the rank of the original tensor."""
        return len(self.tensor_shape)

    @property
    def indices(self):
        """Return the indices of the sharded tensor."""
        if not self._indices:
            self._indices = pxla.spec_to_indices(self.tensor_shape,
                                                 self.sharding_spec)
        return self._indices

    @property
    def tile_assignments(self):
        """Return the device assignment of each tile."""
        if self._tile_assignments is None:
            if self.replicated:
                mesh_flat = np.arange(self.device_mesh.total_devices)
                self._tile_assignments = np.reshape(
                    mesh_flat,
                    self.tile_shape + [self.device_mesh.total_devices])
            else:
                # Generate tile assignments using proto
                proto = self._sharding_spec_proto
                shape = proto.tile_assignment_dimensions
                devices_flat = proto.tile_assignment_devices
                self._tile_assignments = np.reshape(devices_flat, shape)
        return self._tile_assignments

    @property
    def replicated_maxes(self):
        """Return the list of mesh axes for replication."""
        replicated_maxes = []
        for maxis, assignment in enumerate(self.sharding_spec.mesh_mapping):
            if isinstance(assignment, Replicated):
                replicated_maxes.append(maxis)
        return replicated_maxes

    @property
    def tiled(self):
        """Whether this distributed array is fully tiled."""
        if not self.replicated_maxes:
            return True
        return False

    @property
    def replicated(self):
        """Whether this distributed array is fully replicated."""
        if len(self.replicated_maxes) == len(self.sharding_spec.mesh_mapping):
            return True
        return False

    @property
    def partial_tiled(self):
        """Whether this distributed array is mixed sharded and replicated."""
        if self.replicated_maxes and len(self.replicated_maxes) \
                < len(self.sharding_spec.mesh_mapping):
            return True
        return False

    @property
    def tile_shape(self):
        """
        Return the shape of the tiles.

        Each dim of the tile_shape is an integer representing how many tiles are along this dim.
        """
        if self.tiled:
            return self.tile_assignments.shape
        elif self.partial_tiled:
            return self.tile_assignments.shape[:-1]
        else:
            # when fully replicated, the tile shape should be [1, ..., 1, num_devices],
            # with rank = rank(array) + 1
            return [1] * len(self.sharding_spec.sharding)

    @property
    def num_tiles(self):
        """Return the number of tiles of the VDA."""
        return np.prod(self.tile_shape)

    @property
    def tiles(self):
        """Return all the shards of the VDA following their orders."""
        if self._tiles is None:
            # Below are for tiled or partial_tiled.
            num_tiles = np.prod(self.tile_shape)
            # unique tiles (not counting those replicated)
            self._tiles = np.empty(self.tile_shape, dtype=object)
            for tile_index_flat in range(num_tiles):
                # get its index
                tile_index = unflatten_tile_index(tile_index_flat,
                                                  self.tile_shape)
                indices = [None] * len(self.tensor_shape)
                for i, dim in enumerate(self.tensor_shape):
                    tile_size, ragged = divmod(dim, self.tile_shape[i])
                    assert not ragged
                    indices[i] = slice(tile_size * tile_index[i],
                                       tile_size * (tile_index[i] + 1))
                device_ids = self.tile_assignments[tuple(tile_index)]
                if not isinstance(device_ids, Iterable):
                    device_ids = [device_ids]
                else:
                    device_ids = list(device_ids)
                device_strs = [
                    self.device_mesh.device_strs[d] for d in device_ids
                ]
                dst_tile = Tile(index=tile_index,
                                index_flat=tile_index_flat,
                                replica_device_ids=device_ids,
                                replica_device_strs=device_strs,
                                indices=indices)
                self._tiles[tuple(tile_index)] = dst_tile
        return self._tiles

    @property
    def device_str_to_flat_index(self):
        """Maps a device_str to its index in the flattened .indices object."""
        device_str_to_flat_index_map = dict()
        for i, device_str in enumerate(self.device_mesh.device_strs):
            device_str_to_flat_index_map[device_str] = i
        return device_str_to_flat_index_map


VDA = VirtualDistributedArray


# TODO(Hao): maybe we should derive two classes: Eager and Lazy Resharding tasks.
class ReshardingTask:
    """
    A helper class that launch the NCCL communication based on a resharding task spec.

    Args:
        task_spec (ReshardingTaskSpec): the task spec of this task.
        collective_group (CollectiveGroup): the collective group information.
        src_mesh (PhysicalMesh): the source mesh to send.
        dst_mesh (PhysicalMesh): the destiantion mesh to receive.
    """

    def __init__(self, task_spec, collective_group, src_mesh, dst_mesh):
        self.task_spec = task_spec
        self.collective_group = collective_group
        self.src_mesh = src_mesh
        self.dst_mesh = dst_mesh

        # internal states
        self._sender_tasks = None
        self._receiver_tasks = None
        self._has_put_send_recv_tasks = False

    @property
    def has_initialized_send_recv_tasks(self):
        if self._sender_tasks is not None and self._receiver_tasks is not None:
            return True
        return False

    @property
    def has_put_send_recv_tasks(self):
        return self._has_put_send_recv_tasks

    @property
    def sender_tasks(self):
        if not self.has_initialized_send_recv_tasks:
            raise RuntimeError("Sender tasks have not been initialized.")
        return self._sender_tasks

    @property
    def receiver_tasks(self):
        if not self.has_initialized_send_recv_tasks:
            raise RuntimeError("Receiver tasks have not been initialized.")
        return self._receiver_tasks

    def do(self, src_array):
        """According to the task_spec, launch send/recv operations.

        Used in dynamic mode.

        Args:
            src_array (DistributedArray): the source array to be resharded.
        """
        if src_array.device_mesh != self.src_mesh:
            raise RuntimeError("The src array locates on a different mesh `{}` "
                               "than self.src_mesh `{}`.".format(
                                   src_array.device_mesh, self.src_mesh))

        bufs = [None] * len(self.task_spec.dst_indices)
        device_str_to_buf_map = dict()
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
        # construct a remote buf for this tile
        receiver_host_id = self.collective_group.device_str_to_host_id_map[
            receiver]
        receiver_device_id = self.collective_group.device_str_to_device_id_map[
            receiver]
        receiver_worker = self.collective_group.device_str_to_mesh_worker_map[
            receiver]

        dtype = src_array.remote_buffers[0].dtype
        result_buf = RemoteBufferRef(self.dst_mesh,
                                     receiver_host_id,
                                     receiver_device_id,
                                     dtype=dtype)
        # Put an empty buffer first.
        if global_config.pipeline_aggressively_sync:
            ray.get(
                receiver_worker.put_non_zero_buffer.remote(
                    result_buf.uuid, result_buf.device_id, dst_tile.tile_shape,
                    result_buf.dtype))
            logger.debug("We are synchronizing for `put_empty_buffer`.")
        else:
            receiver_worker.put_non_zero_buffer.remote(result_buf.uuid,
                                                       result_buf.device_id,
                                                       dst_tile.tile_shape,
                                                       result_buf.dtype)
            logger.debug("We are NOT synchronizing for `put_empty_buffer`.")

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

            if global_config.pipeline_aggressively_sync:
                ray.get([send_done_ref, recv_done_ref])
                logger.debug(
                    "We are synchronizing for `send_tile`/`recv_tile`.")
            else:
                logger.debug(
                    "We are NOT synchronizing for `send_tile`/`recv_tile`.")
        return result_buf

    def get_send_recv_tasks(self):
        """Init send/recv tasks if not yet."""
        if self.has_initialized_send_recv_tasks:
            return self.sender_tasks, self.receiver_tasks

        self._sender_tasks = {host: list() for host in self.src_mesh.workers}
        self._receiver_tasks = {host: list() for host in self.dst_mesh.workers}

        self.sender_uuid_plan = []
        self.receiver_uuid_plan = []
        for i, (dst_tile, src_tiles, indices_in_dst_tiles) in enumerate(
                self.task_spec.dst_tile_to_src_tiles_map):
            s = self.task_spec.strategy[i]
            for replica_index, receiver in enumerate(
                    dst_tile.replica_device_strs):
                # Get args for an empty buffer
                receiver_device_id = \
                    self.collective_group.device_str_to_device_id_map[receiver]
                receiver_worker = \
                    self.collective_group.device_str_to_mesh_worker_map[receiver]
                dtype = self.task_spec.src.aval.dtype
                receiver_task = [receiver_device_id, dst_tile.tile_shape, dtype]
                # Get args for send/recv
                senders = [
                    s[replica_index][src_tile_index]
                    for src_tile_index, _ in enumerate(src_tiles)
                ]
                self.receiver_uuid_plan.append(receiver)
                receiver_rank, receiver_gpu_idx = \
                    self.collective_group.device_str_to_rank_map[receiver]
                receiver_subtasks = []
                for i, sender in enumerate(senders):
                    # Sender's task
                    tile = src_tiles[i]
                    sender_worker = self.collective_group.device_str_to_mesh_worker_map[
                        sender]
                    self._sender_tasks[sender_worker].append(
                        (tile.offset, receiver_rank, receiver_gpu_idx))
                    self.sender_uuid_plan.append(sender)
                    # Receiver's task
                    sender_rank, sender_gpu_idx = \
                        self.collective_group.device_str_to_rank_map[sender]

                    indices_in_dst_tile = indices_in_dst_tiles[i]
                    receiver_subtasks.append(
                        (indices_in_dst_tile, sender_rank, sender_gpu_idx))
                receiver_task.append(receiver_subtasks)

                self._receiver_tasks[receiver_worker].append(receiver_task)

        # return read-only
        return self.sender_tasks, self.receiver_tasks

    def put_send_recv_tasks(self):
        """Put send recv tasks to remote worker."""
        if self.has_put_send_recv_tasks:
            return
        sender_tasks, receiver_tasks = self.get_send_recv_tasks()
        group_name = self.collective_group.group_name
        self.send_worker_task_ids = dict()
        task_dones = []
        for worker, task in sender_tasks.items():
            uuid = next_resharding_task_uuid()
            self.send_worker_task_ids[worker] = uuid
            task_dones.append(
                worker.put_resharding_send_task.remote(uuid, task, group_name))
        self.recv_worker_task_ids = dict()
        for worker, task in receiver_tasks.items():
            uuid = next_resharding_task_uuid()
            self.recv_worker_task_ids[worker] = uuid
            task_dones.append(
                worker.put_resharding_recv_task.remote(uuid, task, group_name))
        ray.get(task_dones)
        self._has_put_send_recv_tasks = True

    def do_prepared(self, src_array, profiling=False):
        send_buf_uuids = {host: list() for host in self.src_mesh.workers}
        recv_buf_uuids = {host: list() for host in self.dst_mesh.workers}

        bufs = [None] * len(self.task_spec.dst_indices)
        device_str_to_buf_map = dict()

        dtype = self.task_spec.src.aval.dtype
        for receiver in self.receiver_uuid_plan:
            receiver_host_id = self.collective_group.device_str_to_host_id_map[
                receiver]
            receiver_device_id = self.collective_group.device_str_to_device_id_map[
                receiver]
            receiver_worker = self.collective_group.device_str_to_mesh_worker_map[
                receiver]
            result_buf = RemoteBufferRef(self.dst_mesh,
                                         receiver_host_id,
                                         receiver_device_id,
                                         dtype=dtype)
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
            ray.get(results)
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
            if global_config.pipeline_aggressively_sync:
                ray.get(results)
                logger.debug("Using precompiled tasks in sync mode.")
            else:
                logger.debug("Using precomipled tasks in async mode.")

        for i, device_str in enumerate(
                self.task_spec.dst.device_mesh.device_strs):
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
        return f"ReshardingTask(shape:{self.task_spec.aval.shape}, "\
               f"{self.task_spec.src_sharding_spec} -> {self.task_spec.dst_sharding_spec})"


@dataclass
class Tile:
    """
    Representing a full tile (shard) on the original distributed array.

    Args:
        index (List[int]): the index of this shard in the tile_assignments matrix of the VDA.
        index_flat (int): flattend index, row-majored.
        replica_device_ids (List[int]): the device ids this shard is replicated on.
        replica_device_strs (List[str]): the device strs this shard is replicated on.
        indices (List[slice]): a list of slices that expresses its indices in the original array.
    """

    index: List[int]
    index_flat: int
    replica_device_ids: List[int]
    replica_device_strs: List[str]
    indices: List[slice]

    @property
    def tile_size(self):
        """Return the size (number of elements) of the tile."""
        size = 1
        for s in self.indices:
            size = size * (s.stop - s.start)
        return size

    @property
    def tile_shape(self):
        """Return the shape of the tile."""
        return [s.stop - s.start for s in self.indices]


@dataclass
class TileSlice(Tile):
    """
    Representing a slice of a tile of the array using an offset.

    TileSlice subsets Tile, and Tile subsets VDA.

    Args:
        offset (List[slice]): a list of slice objects to represent the offset made on the shard.
    """

    offset: List[slice]

    def __init__(self, tile, offset):
        self.index = tile.index
        self.index_flat = tile.index
        self.replica_device_ids = tile.replica_device_ids
        self.replica_device_strs = tile.replica_device_strs
        self.indices = tile.indices
        self.offset = offset

    @property
    def slice_size(self):
        """Return the size (number of elements) of this tile slice."""
        size = 1
        for o in self.offset:
            size = size * (o.stop - o.start)
        return size


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
        self.device_str_to_rank_map = dict()
        self.device_str_to_mesh_worker_map = dict()
        self.device_str_to_host_id_map = dict()
        self.device_str_to_device_id_map = dict()

        # arranged following the rank order
        num_host = len(self.src_mesh.host_ips) + len(self.dst_mesh.host_ips)
        self.mesh_workers = [None] * num_host
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

    def instantiate(self):
        """Instantiate the collective group in Ray."""
        options = {
            "group_name": self.group_name,
            "world_size": len(self.mesh_workers),
            "ranks": [i for i, _ in enumerate(self.mesh_workers)],
            "backend": "nccl"
        }
        col.create_collective_group(self.mesh_workers, **options)

    def destroy(self):
        """Destroy the nccl collective group at exit."""
        logger.debug("Recycling the collective group: {}.".format(
            self.group_name))
        for worker in self.mesh_workers:
            # This remote call will remove ray named actors (hence it is necessary)
            ray.get(worker.destroy_collective_group.remote(self.group_name))
        # Destroy the declared named actor in ray

        # TODO(Hao): move this part of recycling to ray.util.collective instead of here.
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
        src_array (VirtualDistributedArray): the source distributed array, in virtual.
        dst_array (VirtualDistributedArray): the destination distributed array, in virtual.
    """

    def __init__(self, src_array, dst_array):
        self.src = src_array
        self.dst = dst_array
        self._dst_tile_to_src_tiles_map = None
        self._strategy = None

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
        assert self.src.aval == self.dst.aval
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
                    left_in_dst_tile = tile_length_on_this_dim - offset + \
                        (tile_index_relative[i] - 1) * tile_length_on_this_dim
                    right_in_dst_tile = left_in_dst_tile + tile_length_on_this_dim
                    indices.append(slice(left_in_dst_tile, right_in_dst_tile))
            # construct a new tile slice
            this_tileslice = TileSlice(
                self.src.tiles[tuple(tile_index_absolute)], offset=offsets)
            src_tileslices.append(this_tileslice)
            indices_in_dst_tile.append(indices)
        return src_tileslices, indices_in_dst_tile

    def set_resharding_strategy(self, strategy):
        """Now the strategy is an np.array(dtype=str) to specify connections between src tiles and dst tile."""
        # TODO(Hao): extend the strategy to have a schedule
        self._strategy = strategy

    @property
    def strategy(self):
        """Return the communication strategy for this resharding task spec."""
        if not self._strategy:
            raise RuntimeError("Generate and set strategy first.")
        return self._strategy

    def get_participant_device_strs(self):
        """Identify all participant device strs (for NCCL setup) in this task spec."""
        if not self._strategy:
            raise RuntimeError("Generate and set strategy first.")
        device_strs = OrderedSet()
        # senders
        for tile_strategy in self.strategy:
            device_strs = device_strs | OrderedSet(
                tile_strategy.flatten().tolist())
        # receivers
        for tile in self.dst.tiles.flatten():
            device_strs = device_strs | OrderedSet(tile.replica_device_strs)
        return device_strs


def unflatten_tile_index(index, shape):
    """Unroll a flattend index based on the given shape."""
    unflattened_index = []
    reminder = index
    for i in range(len(shape) - 1):
        cur_index = int(reminder / np.prod(shape[i + 1:]))
        unflattened_index.append(cur_index)
        reminder = reminder - cur_index * np.prod(shape[i + 1:])
    unflattened_index.append(reminder)
    return unflattened_index


class CrossMeshCommunicator:
    """
    Communicator for cross-mesh resharding.

    Given the pipeline schedule and stages, the class analyzes them and generate:
    - resharding specs (see docstring of `ReshardingTaskSpec`)
    - resharding strategies (see docstring of `_generate_resharding_strategy_by_loads()`)

    This communicator only takes care of compilation-time work, and does not get involved
    with physical meshes, buffer creations, or other execution-time work.

    Args:
        sharded_stages (List[XlaShardedPipelineComputation]): list of stages to form the pipeline.
        schedule (Any): the pipelining schedule for these stages.
    """

    def __init__(self, sharded_stages, schedule):
        if not isinstance(sharded_stages, list):
            raise RuntimeError("Require a list of stages.")
        if not all([
                isinstance(s, XlaShardedPipelineComputation)
                for s in sharded_stages
        ]):
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

        # Run a load-balancing algorithm to generate the resharding strategies for each spec
        self._generate_resharding_strategy()

    @property
    def num_mesh(self):
        """Number of meshes in the schedule."""
        return self._schedule.num_mesh

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
            [dict() for _ in range(self.num_mesh)] for _ in range(self.num_mesh)
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
            resharding_vars, out_var_indices, in_var_indices = \
                self._args_between(src_stage, dst_stage)
            out_sharding_specs = src_stage.output_sharding_specs
            in_sharding_specs = dst_stage.input_sharding_specs

            # Make a ReshardSpec for each VDA
            for var, out_var_index, in_var_index in \
                    zip(resharding_vars, out_var_indices, in_var_indices):
                src_sharding_spec = out_sharding_specs[out_var_index]
                dst_sharding_spec = in_sharding_specs[in_var_index]
                src_array = VDA(device_mesh=src_mesh,
                                aval=var.aval,
                                sharding_spec=src_sharding_spec)
                dst_array = VDA(device_mesh=dst_mesh,
                                aval=var.aval,
                                sharding_spec=dst_sharding_spec)
                task_spec = ReshardingTaskSpec(src_array, dst_array)
                self.resharding_specs[src_mesh_index][dst_mesh_index][repr(
                    var)] = task_spec

    def _generate_resharding_strategy(self):
        """
        Generate a send/recv strategies for all resharding tasks by looking at their load.

        For now I simply run a greedy algorithm from the 1st resharding spec to the last.
        """
        for _, _, var_spec_map in self.task_spec_iter():
            for _, spec in var_spec_map.items():
                strategy = self._generate_resharding_strategy_by_loads(spec)
                spec.set_resharding_strategy(strategy)

    def task_spec_iter(self):
        """A convenient iterator over all activated task specs."""
        for i in range(self.num_mesh):
            for j in range(self.num_mesh):
                if not self.resharding_specs[i][j]:
                    continue
                yield i, j, self.resharding_specs[i][j]

    # TODO(Hao): implement another send/recv strategy similar to the scatter-gather optimization in megatron.
    def _generate_resharding_strategy_by_loads(self, spec):
        """
        Generate the resharding strategy for a resharding task spec.

        Strategy is a list a np array, with length as len(spec.dst_tile_to_src_tiles_map)
        each array is with shape [len(dst_tile.devices), len(src_tiles)]; it specifies for each
        replica of a dst tile, how (src tile replicas) it should get the data from src_tiles.
        """
        strategy = []
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
                        sender: self._sender_loads[sender]
                        for sender in src_tileslice.replica_device_strs
                    }
                    sender = min(loads, key=loads.get)
                    per_spec_plan[receiver_idx][src_tileslice_idx] = sender
                    # upload load on-the-fly
                    self._sender_loads[sender] += src_tileslice.slice_size
                    self._receiver_loads[receiver] += src_tileslice.slice_size
            strategy.append(per_spec_plan)
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
