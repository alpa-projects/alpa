"""Serialization utilities for Alpa.
Adapted from
https://flax.readthedocs.io/en/latest/_modules/flax/serialization.html
Add support for DistributedArray and ReplicatedDistributedArray serialization
in Alpa.
"""

import logging
import os
import pickle
from typing import Union, Any, Sequence

from flax.serialization import to_state_dict, from_state_dict
import jax
from jax.interpreters.pxla import ShardingSpec
from jax.core import ShapedArray
from jax._src.tree_util import tree_flatten, tree_leaves, tree_unflatten
from flax.serialization import (to_state_dict, from_state_dict,
                                _ndarray_from_bytes, _ndarray_to_bytes)
import msgpack
import numpy as np
import ray

from alpa.device_mesh import (DistributedArray, ReplicatedDistributedArray,
                              PhysicalDeviceMesh)

PyTree = Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _dfs_pytree(tree, prefix, paths):
    if isinstance(tree, dict):
        for k, v in tree.items():
            _dfs_pytree(v, prefix + "." + str(k), paths)
    elif isinstance(tree, (tuple, list)):
        for i, v in enumerate(tree):
            _dfs_pytree(v, prefix + "." + str(i), paths)
    elif tree is None:
        return
    else:
        # Leaf node
        paths.append(prefix)


def _save_arr(ckpt_dir, arr):
    os.makedirs(ckpt_dir, exist_ok=True)
    shard_name = "0.0"
    metadata = {
        'global_shape': arr.shape,
        'dtype': arr.dtype,
        'shard_names': [shard_name],
        'shard_indices': None,
    }
    with open(os.path.join(ckpt_dir, shard_name), "wb") as datafile:
        np.save(datafile, arr)
    with open(os.path.join(ckpt_dir, ".metadata0"), "wb") as metafile:
        pickle.dump(metadata, metafile)


def load_entire_arr(ckpt_dir, metadatas):
    assert len(metadatas) > 0
    with open(os.path.join(ckpt_dir, metadatas[0]), "rb") as metafile:
        meta = pickle.load(metafile)
    if meta["shard_indices"] is None:
        return np.load(os.path.join(ckpt_dir, meta["shard_names"][0]))
    entire_arr = np.empty(meta["global_shape"], meta['dtype'])
    for metadata in metadatas:
        with open(os.path.join(ckpt_dir, metadata), "rb") as metafile:
            meta = pickle.load(metafile)
        for shard_name, shard_indice in zip(meta["shard_names"], meta["shard_indices"]):
            entire_arr[shard_indice] = np.load(os.path.join(ckpt_dir, shard_name))
    return entire_arr


def save_checkpoint(nfs_dir: Union[str, os.PathLike], target: PyTree,
                    step: int, local_cache_dir=None):
    """Save a checkpoint of the `target` to `path`. 

        Similar to flax.training.checkpoints.save_checkpoint, but support DistributedArrays 
        and ReplicatedDistributedArray in alpa. 
        # TODO (zhongyinmin): copy all the safe-saving stuff from 
        https://flax.readthedocs.io/en/latest/_modules/flax/training/checkpoints.html#save_checkpoint

        Args:
           nfs_dir: shared filesystem path to store checkpoint. 
           target: serializable flax object, usually a trainState.
           step: training step number or other metric number
           local_cache_dir: If not None, `save_checkpoint` will return immediately after each host saving
           its part of the model into this cache directory, and only the metadata file will be saved into
           the `nfs_dir`. You can then use `rsync` to easily merge these these local data into `nfs_dir`.
    """
    # create directories if not exist
    os.makedirs(nfs_dir, exist_ok=True)

    if local_cache_dir is not None:
        os.makedirs(local_cache_dir, exist_ok=True)
        save_dir = local_cache_dir
    else:
        save_dir = nfs_dir

    target = to_state_dict(target)
    flat_dirs = []
    _dfs_pytree(target, "state", flat_dirs)
    flat_target, target_tree = tree_flatten(target)
    flat_metadata = []
    obj_refs = []
    assert(len(flat_dirs) == len(flat_target))
    for arr_dir, x in zip(flat_dirs, flat_target):
        if isinstance(x, (DistributedArray, ReplicatedDistributedArray, np.ndarray, jax.xla.DeviceArray)):
            if isinstance(x, DistributedArray):
                obj_refs.extend(x.save(os.path.join(save_dir, arr_dir), False))
            elif isinstance(x, ReplicatedDistributedArray):
                obj_refs.extend(x.replica.save(os.path.join(save_dir, arr_dir), False))
            elif isinstance(x, (np.ndarray, jax.xla.DeviceArray)):
                _save_arr(os.path.join(save_dir, arr_dir), x)
            flat_metadata.append(arr_dir)
        else:
            flat_metadata.append(x)

    metapath = os.path.join(nfs_dir, f"checkpoint_{step}")
    metadata = tree_unflatten(target_tree, flat_metadata)
    with open(metapath, "wb") as metafile:
        metafile.write(msgpack.packb(metadata))
    ray.get(obj_refs)
    if local_cache_dir is not None:
        logger.warning("Only the metadata has been saved into the shared filesystem, and the other data has\
                        only been saved into the local cache directory. You can use rsync to merge them into\
                        the shared filesystem easily.")


class LoadInfo:
    """
    A wrapper for the loading information.
    """
    def __init__(self, 
                 aval: ShapedArray, 
                 meshes: Sequence[PhysicalDeviceMesh], 
                 specs: Sequence[ShardingSpec]):
        assert len(meshes) == len(specs)
        self.aval = aval
        self.meshes = meshes
        self.specs = specs

    def add_replica(self, mesh, spec):
        self.meshes.append(mesh)
        self.specs.append(spec)

    def get_info(self):
        if self.is_replicated():
            return zip(self.meshes, self.specs)
        else:
            return self.meshes[0], self.specs[0]

    def is_replicated(self):
        return len(self.meshes) > 1

    def __str__(self):
        return f"{self.aval}, {self.meshes[0].mesh_id}, {self.specs[0]}"


def restore_checkpoint(ckpt_dir: Union[str, os.PathLike], step: int,
                       load_info: PyTree):
    """Restore the specified checkpoint from `path`.

        Similar to flax.training.checkpoints.load_checkpoint,
        but support DistributedArrays and ReplicatedDistributedArray in alpa.
        # TODO (zhongyinmin): copy all the safe-loading stuff from
        https://flax.readthedocs.io/en/latest/_modules/flax/training/checkpoints.html#restore_checkpoint

        Args:
            ckpt_dir: directory of checkpoints to restore from. If you do not have a shared filesystem, 
            each host needs a copy of the checkpoint on its local disk at the same path.
            at the same path.
            step: step number to load.
            load_info: shardingSpec and deviceMesh allocation info for loading.
    """
    metapath = os.path.join(ckpt_dir, f"checkpoint_{step}")
    with open(metapath, 'rb') as metafile:
        metadata = from_state_dict(load_info, msgpack.unpackb(metafile.read()))

    state_paths, state_tree = tree_flatten(metadata)
    flat_info = tree_leaves(load_info)
    flat_load_state = []
    obj_refs = []
    for path, info in zip(state_paths, flat_info):
        if info is None:
            logger.warning('Variable is not used, skip loading it')
            flat_load_state.append(None)
        if info.is_replicated():
            meshes, arrays = [], []
            for mesh, spec in info.get_info():
                meshes.append(mesh)
                obj_ref, dist_arr = DistributedArray.load(os.path.join(ckpt_dir, path), info.aval, mesh, spec, False)
                obj_refs.extend(obj_ref)
                arrays.append(dist_arr)
            flat_load_state.append(ReplicatedDistributedArray(meshes, arrays)) 
        else:
            mesh, spec = info.get_info()
            obj_ref, dist_arr = DistributedArray.load(os.path.join(ckpt_dir, path), info.aval, mesh, spec, False)
            obj_refs.extend(obj_ref)
            flat_load_state.append(dist_arr)
    ray.get(obj_refs)
    return tree_unflatten(state_tree, flat_load_state)
