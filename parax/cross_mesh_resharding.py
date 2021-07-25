from dataclasses import dataclass, field
from typing import Union, List, Tuple

import numpy as np
import ray
from jax.interpreters import pxla
from jax.interpreters.pxla import _hashable_index, Replicated
import ray.util.collective as col

from parax.pipeline_stage import XlaShardedPipelineStage
from parax.device_mesh import DistributedArray, RemoteBufferRef


class VirtualDistributedArray:
    """Distributed Array without allocating remote buffers.

    VDA wrapper differs from DistributedArray (DA) in that:
    (1) it does not allocate a remote buffer at construction
    (2) its device_mesh attribute is a virtual mesh (not physical)

    Args:
        device_mesh (VirtualMesh): the virtual mesh this VDA locates on.
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

        self._sharding_spec_proto =self.sharding_spec.sharding_proto()

    @property
    def tensor_shape(self):
        return self.aval.shape

    @property
    def tensor_rank(self):
        return len(self.tensor_shape)

    @property
    def indices(self):
        if not self._indices:
            self._indices = pxla.spec_to_indices(self.tensor_shape, self.sharding_spec)
        return self._indices

    @property
    def tile_assignments(self):
        """Returns a np.array representing sharding along multiple dimensions."""
        if self._tile_assignments is None:
            if self.replicated:
                mesh_flat = np.arange(self.device_mesh.total_devices)
                self._tile_assignments = np.reshape(mesh_flat,
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
        replicated_maxes = []
        for maxis, assignment in enumerate(self.sharding_spec.mesh_mapping):
            if isinstance(assignment, Replicated):
                replicated_maxes.append(maxis)
        return replicated_maxes

    @property
    def tiled(self):
        if not self.replicated_maxes:
            return True
        return False

    @property
    def replicated(self):
        if len(self.replicated_maxes) == len(self.sharding_spec.mesh_mapping):
            return True
        return False

    @property
    def partial_tiled(self):
        if self.replicated_maxes and len(self.replicated_maxes) \
                < len(self.sharding_spec.mesh_mapping):
            return True
        return False

    @property
    def tile_shape(self):
        """TODO(Hao): add some comments"""
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
        return np.prod(self.tile_shape)

    @property
    def tiles(self):
        if self._tiles is None:
            # Below are for tiled or partial_tiled.
            num_tiles = np.prod(self.tile_shape)
            # unique tiles (not counting those replicated)
            self._tiles = np.empty(self.tile_shape, dtype=object)
            for tile_index_flat in range(num_tiles):
                # get its index
                tile_index = unflatten_tile_index(tile_index_flat, self.tile_shape)
                device_ids = list(self.tile_assignments[tuple(tile_index)])
                indices = [None] * len(self.tensor_shape)
                for i, dim in enumerate(self.tensor_shape):
                    tile_size, ragged = divmod(dim, self.tile_shape[i])
                    assert not ragged
                    indices[i] = slice(tile_size * tile_index[i], tile_size * (tile_index[i] + 1))
                # TODO(Hao): check here.
                device_strs = [self.device_mesh.device_strs[d] for d in device_ids]
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


class ReshardingTask:
    def __init__(self, task_spec, collective_group, src_array):
        self.task_spec = task_spec
        self.collective_group = collective_group
        self.src_array = src_array

        self.src_mesh = self.src_array.device_mesh
        if self.src_mesh == collective_group.src_mesh:
            self.dst_mesh = collective_group.dst_mesh
        else:
            self.dst_mesh = collective_group.src_mesh

    def do(self):
        # according to task_spec, launch send/recv operations
        # Hao: some sanity tests
        bufs = [None] * len(self.task_spec.dst_indices)
        device_str_to_buf_map = dict()
        for i, (dst_tile, src_tiles, indices_in_dst_tiles) in enumerate(self.task_spec.dst_tile_to_src_tiles_map):
            # Loop over each dst tile for this shard
            s = self.task_spec.strategy[i]
            # strategy is len(dst_tile.device_strs) by len(src_tiles)
            for replica_index, receiver in enumerate(dst_tile.replica_device_strs):
                # loop over this replica (hence a specific destination gpu device)
                senders = [s[replica_index][src_tile_index]
                           for src_tile_index, src_tile in enumerate(src_tiles)]
                device_str_to_buf_map[receiver] = self.same_destination_group_send_recv(
                    senders, src_tiles, dst_tile, indices_in_dst_tiles, receiver)
            # for each tile, assemble the results and generate a RemoteBufRef


        # assemble the buffer based on the order present in indices
        for i, device_str in enumerate(self.task_spec.dst.device_mesh.device_strs):
            # for each replica
            bufs[self.task_spec.dst.device_str_to_flat_index[device_str]] = device_str_to_buf_map[device_str]

        # Now construct the distributed array
        dst_array = DistributedArray(self.dst_mesh,
                                     self.src_array.aval,
                                     self.task_spec.dst_sharding_spec,
                                     bufs,
                                     self.task_spec.dst_indices)
        return dst_array

    def same_destination_group_send_recv(self,
                                         senders,
                                         src_tiles,
                                         dst_tile,
                                         indices_in_dst_tiles,
                                         receiver):
        # construct a remote buf for this tile
        receiver_host_id = self.collective_group.device_str_to_host_id_map[receiver]
        receiver_device_id = self.collective_group.device_str_to_device_id_map[receiver]
        receiver_worker = self.collective_group.device_str_to_mesh_worker_map[receiver]
        result_buf = RemoteBufferRef(self.dst_mesh, receiver_host_id, receiver_device_id)
        # Put an empty buffer first.
        ray.get(receiver_worker.put_empty_buffer.remote(
            result_buf.uuid, result_buf.device_id, dst_tile.tile_shape))
        receiver_rank, receiver_gpu_idx = self.collective_group.device_str_to_rank_map[receiver]
        for i, sender in enumerate(senders):
            # send is a device_str in src_mesh
            # we need to find out its mesh_worker, and the corresponded sender remotebuf (uuid-indexed).
            sender_buf = self.src_array.remote_buffers[self.task_spec.src.device_str_to_flat_index[sender]]
            sender_worker = self.collective_group.device_str_to_mesh_worker_map[sender]
            assert sender_buf.device_id == int(sender[-1])
            sender_rank, sender_gpu_idx = self.collective_group.device_str_to_rank_map[sender]
            # launch NCCL send/recv
            tile = src_tiles[i]
            indices_in_dst_tile = indices_in_dst_tiles[i]
            send_done_ref = sender_worker.send_tile.remote(sender_buf.uuid,
                                                           tile.offset,
                                                           receiver_rank,
                                                           receiver_gpu_idx,
                                                           self.collective_group.group_name)
            recv_done_ref = receiver_worker.recv_tile.remote(result_buf.uuid,
                                                             result_buf.device_id,
                                                             indices_in_dst_tile,
                                                             sender_rank,
                                                             sender_gpu_idx,
                                                             self.collective_group.group_name)
            ray.get([send_done_ref, recv_done_ref])
        return result_buf


@dataclass
class Tile:
    """Representing a full tile (shard) on the original distributed array.

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
        size = 1
        for s in self.indices:
            size = size * (s.stop - s.start)
        return size

    @property
    def tile_shape(self):
        return [s.stop - s.start for s in self.indices]


@dataclass
class TileSlice(Tile):
    """Representing a slice of a tile of the array using an offset.

    TileSlice \subset Tile \subset VDA.

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
        size = 1
        for o in self.offset:
            size = size * (o.stop - o.start)
        return size


class CollectiveGroup:
    """A class for setting up real NCCL groups.

    Args:
        device_strs (List[str]): list of device strs in this group.
        src_mesh (PhysicalDeviceMesh): the source physical mesh.
        dst_mesh (PhysicalDeviceMesh): the destination physical mesh.
    """
    def __init__(self, device_strs, src_mesh, dst_mesh):
        self.device_strs = list(device_strs)
        self.src_mesh = src_mesh
        self.dst_mesh = dst_mesh

        # generate a group name
        self.group_name = ",".join(self.device_strs)

        self._debug_check()

        # construct a device str -> rank: (process_rank, gpu_index) map
        self.device_str_to_rank_map = dict()
        self.device_str_to_mesh_worker_map = dict()
        self.device_str_to_host_id_map = dict()
        self.device_str_to_device_id_map = dict()

        # arranged following the rank order
        num_host = len(self.src_mesh.host_ips) + len(self.dst_mesh.host_ips)
        self.mesh_workers = [None] * num_host
        for i, host_ip in enumerate(src_mesh.host_ips):
            self.mesh_workers[i] = self.src_mesh.workers[i]
            for j in range(src_mesh.num_devices_per_host):
                device_str = self.src_mesh.device_strs[i * src_mesh.num_devices_per_host + j]
                self.device_str_to_rank_map[device_str] = (i, j)
                self.device_str_to_mesh_worker_map[device_str] = self.src_mesh.workers[i]
                self.device_str_to_host_id_map[device_str] = i
                self.device_str_to_device_id_map[device_str] = j
        for i, host_ip in enumerate(dst_mesh.host_ips):
            self.mesh_workers[i + len(self.src_mesh.host_ips)] = self.dst_mesh.workers[i]
            for j in range(dst_mesh.num_devices_per_host):
                device_str = self.dst_mesh.device_strs[i * src_mesh.num_devices_per_host + j]
                self.device_str_to_rank_map[device_str] = (i + len(src_mesh.host_ips), j)
                self.device_str_to_mesh_worker_map[device_str] = self.dst_mesh.workers[i]
                self.device_str_to_host_id_map[device_str] = i
                self.device_str_to_device_id_map[device_str] = j

    def instantiate(self):
        options = {"group_name": self.group_name,
                   "world_size": len(self.mesh_workers),
                   "ranks": [i for i, _ in enumerate(self.mesh_workers)],
                   "backend": "nccl"}
        col.create_collective_group(self.mesh_workers, **options)

    def _debug_check(self):
        all_device_strs = self.src_mesh.device_strs + self.dst_mesh.device_strs
        # TODO(Hao): incorrect assertion
        assert set(self.device_strs) == set(all_device_strs)


class ReshardingTaskSpec:
    def __init__(self, src_array, dst_array):
        self.src = src_array
        self.dst = dst_array
        self._dst_tile_to_src_tiles_map = None
        self._strategy = None

    @property
    def src_sharding_spec(self):
        return self.src.sharding_spec

    @property
    def dst_sharding_spec(self):
        return self.dst.sharding_spec

    @property
    def aval(self):
        assert self.src.aval == self.dst.aval
        return self.src.aval

    @property
    def src_indices(self):
        return self.src.indices

    @property
    def dst_indices(self):
        """This `indices` is the most original (flattened) one in distributed array."""
        return self.dst.indices

    @property
    def dst_tile_to_src_tiles_map(self):
        """Map from dst_tile to all corresponding src TileSlices.

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
        """This function analyzes the src and dst array.

        It aims to tell the needed collective group and communication pattern.

        Returns:
            strategy (): a
        """
        dst_tile_to_src_tiles_map = []
        for tile in self.dst.tiles.flatten():
            # loop over each tile
            src_tile_slices, indices_in_dst_tile = self._look_up_dst_tile_from_src(tile)
            dst_tile_to_src_tiles_map.append((tile, src_tile_slices, indices_in_dst_tile))
        return dst_tile_to_src_tiles_map

    def _look_up_dst_tile_from_src(self, tile):
        """See the docstring in dst_tile_to_src_tiles_map()."""

        # For each dim in tile, find all the related tiles, and ragged values on that dim in src_tiles.

        # For each dim, we make a tuple recording the first and last index of tiles in src array that intersects
        # with the dst tile. Shards between [start, end) are involved; Left included, right not included.
        related_tile_start_end = [tuple()] * self.src.tensor_rank

        # For each dim, for the first and end tile, we make a tuple recording the slicing offset:
        # - start_shard_offset: [start_shard_offset: ] on that dim is activated.
        # - end_shard_offset: [:end_sharding_offset] on that dim is activated.
        related_tile_offset = [tuple()] * self.src.tensor_rank

        for i, dim in enumerate(self.src.tensor_shape):
            tile_length, ragged = divmod(dim, self.src.tile_shape[i])
            assert not ragged
            start_tile, start_tile_offset = divmod(tile.indices[i].start, tile_length)
            end_tile, end_tile_offset = divmod(tile.indices[i].stop, tile_length)
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
            tile_index_relative = unflatten_tile_index(tileslice_index,
                                                       [end - start for start, end in related_tile_start_end])
            tile_index_absolute = [start + tile_index_relative[dim_index]
                                   for dim_index, (start, end) in enumerate(related_tile_start_end)]
            # depending on its index, calculate a slice for it
            offsets = []
            indices = []
            # loop over each dimension
            for i, r in enumerate(tile_index_absolute):
                start, end = related_tile_start_end[i]
                tile_length_on_this_dim = self.src.tiles[tuple(tile_index_absolute)].tile_shape[i]
                if r == start:
                    # meaning it is the first involved tile
                    offset = related_tile_offset[i][0]
                    offsets.append(slice(offset, tile_length_on_this_dim))
                    indices.append(slice(0, tile_length_on_this_dim - offset))
                elif r == end - 1:
                    # meaning it is the last involved tile
                    offset = related_tile_offset[i][1]
                    offsets.append(slice(0, offset))
                    indices.append(slice(tile.tile_shape[i] - offset, tile.tile_shape[i]))
                else:
                    # meaning it is a fully involved tile
                    offset = related_tile_offset[i][0]
                    offsets.append(slice(0, tile_length_on_this_dim))
                    left_in_dst_tile = tile_length_on_this_dim - offset + \
                                       (tile_index_relative[i] - 1) * tile_length_on_this_dim
                    right_in_dst_tile = left_in_dst_tile + tile_length_on_this_dim
                    indices.append(slice(left_in_dst_tile, right_in_dst_tile))
            # construct a new tile slice
            this_tileslice = TileSlice(self.src.tiles[tuple(tile_index_absolute)], offset=offsets)
            src_tileslices.append(this_tileslice)
            indices_in_dst_tile.append(indices)
        return src_tileslices, indices_in_dst_tile

    def set_resharding_strategy(self, strategy):
        """Now the strategy is an np.array(dtype=str) to specify connections between src tiles and dst tile."""
        # TODO(Hao): extend the strategy to have a schedule
        self._strategy = strategy

    @property
    def strategy(self):
        if not self._strategy:
            raise RuntimeError("Generate and set strategy first.")
        return self._strategy

    def get_participant_device_strs(self):
        """Identify all participant device strs (for NCCL setup) in this task spec."""
        if not self._strategy:
            raise RuntimeError("Generate and set strategy first.")
        device_strs = set()
        for tile_strategy in self.strategy:
            device_strs = device_strs | set(tile_strategy.flatten().tolist())
        return device_strs


def unflatten_tile_index(index, shape):
    """Unroll a flattend index based on the given shape."""
    unflattened_index = []
    reminder = index
    for i in range(len(shape) - 1):
        cur_index =  int(reminder / np.prod(shape[i+1:]))
        unflattened_index.append(cur_index)
        reminder = reminder - cur_index * np.prod(shape[i+1:])
    unflattened_index.append(reminder)
    return unflattened_index


class CrossMeshCommunicator:
    def __init__(self, sharded_stages, schedule):
        if not isinstance(sharded_stages, list):
            raise RuntimeError("Require a list of stages.")
        if not all([isinstance(s, XlaShardedPipelineStage) for s in sharded_stages]):
            raise RuntimeError("Require a list of sharded stages.")

        # Do not mutate
        self._sharded_stages = sharded_stages
        self._schedule = schedule

        self.resharding_specs = None

        # TODO(Hao): implement the cache later.
        self.resharding_cache = dict()

        # Loads for load balancing.
        self._sender_loads = {device_str: 0 for mesh in self._schedule.meshes
                              for device_str in mesh.device_strs}
        self._receiver_loads = {device_str: 0 for mesh in self._schedule.meshes
                                for device_str in mesh.device_strs}

        # Initialize all resharding specs
        self._create_resharding_specs()

        # Run a load-balancing algorithm to generate the resharding strategies for each spec
        self._generate_resharding_strategy()

    @property
    def num_mesh(self):
        return self._schedule.num_mesh

    def _create_resharding_specs(self):
        stages = self._sharded_stages
        meshes = self._schedule.meshes
        num_stage = len(self._sharded_stages)
        stage_placements = [list(self._schedule.stage_placement(i))[0] for i in range(num_stage)]
        deps = self._schedule.dependency
        assert(deps.shape[0] == num_stage)
        assert(deps.shape[1] == num_stage)

        # Note(Hao): resharding_specs is num_mesh x num_mesh matrix
        # Each element is a dict: the name of variables are keys, ReshardingSpec are values.
        self.resharding_specs = [[dict() for i in range(self.num_mesh)]
                                 for j in range(self.num_mesh)]

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
            vars, out_var_indices, in_var_indices = \
                self._args_between(src_stage, dst_stage)
            # out_sharding_specs = src_stage.output_sharding_specs_on_mesh(
            #     src_mesh.get_default_logical_mesh())
            # in_sharding_specs = dst_stage.input_sharding_specs_on_mesh(
            #     dst_mesh.get_default_logical_mesh())
            out_sharding_specs = src_stage.output_sharding_specs
            in_sharding_specs = dst_stage.input_sharding_specs

            # Make a ReshardSpec for each VDA
            for var, out_var_index, in_var_index in zip(vars, out_var_indices, in_var_indices):
                src_sharding_spec = out_sharding_specs[out_var_index]
                dst_sharding_spec = in_sharding_specs[in_var_index]
                src_array = VDA(device_mesh=src_mesh,
                                aval=var.aval,
                                sharding_spec=src_sharding_spec)
                dst_array = VDA(device_mesh=dst_mesh,
                                aval=var.aval,
                                sharding_spec=dst_sharding_spec)
                task_spec = ReshardingTaskSpec(src_array, dst_array)
                self.resharding_specs[src_mesh_index][dst_mesh_index][repr(var)] = task_spec

    def _generate_resharding_strategy(self):
        """Generate a send/recv strategies for all resharding tasks by looking at their load.

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

    def _generate_resharding_strategy_by_loads(self, spec):
        # Strategy is a list a np array, with length as len(spec.dst_tile_to_src_tiles_map)
        # each array is with shape [len(dst_tile.devices), len(src_tiles)]; it specifies for each
        # replica of a dst tile, how (src tile replicas) it should get the data from src_tiles.
        strategy = []
        for dst_tile, src_tileslices, _ in spec.dst_tile_to_src_tiles_map:
            # plan is a 2D array
            per_spec_plan = np.empty((len(dst_tile.replica_device_strs), len(src_tileslices)),
                                     dtype=object)
            for receiver_idx, receiver in enumerate(dst_tile.replica_device_strs):
                for src_tileslice_idx, src_tileslice in enumerate(src_tileslices):
                    loads = {sender: self._sender_loads[sender] for sender in src_tileslice.replica_device_strs}
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
        vars = []
        src_indices = []
        dst_indices = []
        for i, var in enumerate(src_stage.outvars):
            if var in dst_stage.invars:
                vars.append(var)
                src_indices.append(i)
                dst_indices.append(dst_stage.invars.index(var))
        return vars, src_indices, dst_indices

    def pprint_spec(self):
        return NotImplementedError
