"""Tensor classes and utilities used for cross-mesh resharding."""
from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, Any

import numpy as np
from jax.interpreters import pxla
from jax.interpreters.pxla import Replicated, ShardingSpec

from alpa.device_mesh import VirtualPhysicalMesh


def unflatten_tile_index(index, shape):
    """Unroll a flattened index based on the given shape."""
    unflattened_index = []
    reminder = index
    for i in range(len(shape) - 1):
        cur_index = int(reminder / np.prod(shape[i + 1:]))
        unflattened_index.append(cur_index)
        reminder = reminder - cur_index * np.prod(shape[i + 1:])
    unflattened_index.append(reminder)
    return unflattened_index


class VirtualDistributedArray:
    """
    Distributed Array without allocating remote buffers.

    VirtualDistributedArray wrapper differs from DistributedArray in that:
    (1) it does not allocate a remote buffer at construction;
    (2) its device_mesh attribute is a virtual mesh (not physical).

    Args:
        device_mesh (VirtualPhysicalMesh): the virtual mesh this
            VirtualDistributedArray locates on.
        aval (aval): shape information about the array.
        sharding_spec (ShardingSpec): sharding spec of this array.
    """

    def __init__(self, *, device_mesh: VirtualPhysicalMesh, aval,
                 sharding_spec: ShardingSpec):
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
                mesh_flat = np.arange(self.device_mesh.num_devices)
                self._tile_assignments = np.reshape(
                    mesh_flat, self.tile_shape + [self.device_mesh.num_devices])
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
    def num_replicas(self):
        """Number of replicas if replicated or partially tiled."""
        if self.tiled:
            return 1
        else:
            num_replicas = 1
            for _, assignment in enumerate(self.sharding_spec.mesh_mapping):
                if isinstance(assignment, Replicated):
                    num_replicas = num_replicas * assignment.replicas
            return num_replicas

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
        if (self.replicated_maxes and len(self.replicated_maxes) < len(
                self.sharding_spec.mesh_mapping)):
            return True
        return False

    @property
    def tile_shape(self):
        """
        Return the shape of the tiles.

        Each dim of the tile_shape is an integer representing how many tiles are
        along this dim.
        """
        if self.tiled:
            return self.tile_assignments.shape
        elif self.partial_tiled:
            return self.tile_assignments.shape[:-1]
        else:
            # when fully replicated, the tile shape should be
            # [1, ..., 1, num_devices], with rank = rank(array) + 1
            return [1] * len(self.sharding_spec.sharding)

    @property
    def num_tiles(self):
        """Return the number of tiles of the VirtualDistributedArray."""
        return np.prod(self.tile_shape)

    @property
    def tiles(self):
        """Return all the shards of the VirtualDistributedArray following their
        orders."""
        if self._tiles is None:
            # Below are for tiled or partial_tiled.
            num_tiles = np.prod(self.tile_shape)
            # unique tiles (not counting those replicated)
            self._tiles = np.empty(self.tile_shape, dtype=object)
            for tile_index_flat in range(num_tiles):
                # get its index
                tile_index = unflatten_tile_index(tile_index_flat,
                                                  self.tile_shape)
                indices: List[Any] = [None] * len(self.tensor_shape)
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
        device_str_to_flat_index_map = {}
        for i, device_str in enumerate(self.device_mesh.device_strs):
            device_str_to_flat_index_map[device_str] = i
        return device_str_to_flat_index_map


@dataclass
class Tile:
    """
    Representing a full tile (shard) on the original distributed array.

    Args:
        index (List[int]): the index of this shard in the tile_assignments
            matrix of the VirtualDistributedArray.
        index_flat (int): flattend index, row-majored.
        replica_device_ids (List[int]): the device ids this shard is replicated
            on.
        replica_device_strs (List[str]): the device strs this shard is
            replicated on.
        indices (List[slice]): a list of slices that expresses its indices in
            the original array.
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

    TileSlice subsets Tile, and Tile subsets VirtualDistributedArray.

    Args:
        offset (List[slice]): a list of slice objects to represent the offset
            made on the shard.
    """

    offset: List[slice]

    def __init__(self, tile, offset):
        super().__init__(tile.index, tile.index_flat, tile.replica_device_ids,
                         tile.replica_device_strs, tile.indices)
        self.offset = offset

    @property
    def slice_size(self):
        """Return the size (number of elements) of this tile slice."""
        size = 1
        for o in self.offset:
            size = size * (o.stop - o.start)
        return size
