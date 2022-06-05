"""Serialization utilities for Alpa.
Adapted from
https://flax.readthedocs.io/en/latest/_modules/flax/serialization.html
Add support for DistributedArray and ReplicatedDistributedArray serialization
in Alpa.
"""

import logging
import os
import subprocess
from typing import Union, Any, Sequence
import uuid

from flax.serialization import to_state_dict, from_state_dict
import jax
from jax.interpreters.pxla import ShardingSpec
from jax.core import ShapedArray
from jax._src.tree_util import tree_flatten, tree_leaves, tree_unflatten
from flax.serialization import (to_state_dict, from_state_dict,
                                _ndarray_from_bytes, _ndarray_to_bytes)
import msgpack
import numpy as np
import pickle

from alpa.device_mesh import (DistributedArray, ReplicatedDistributedArray,
                              PhysicalDeviceMesh)

PyTree = Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    
    with open(os.path.join(ckpt_dir, f".metadata0"), "wb") as metafile:
        pickle.dump(metadata, metafile)
    

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
           its part of the model into this cache directory. There will be background processes moving these
           local data into `nfs_dir`.
    """
    def _get_save_path():
        # TODO: remove random path
        return uuid.uuid4().hex

    # create directories if not exist
    os.makedirs(nfs_dir, exist_ok=True)

    if local_cache_dir is not None:
        save_dir = local_cache_dir
    else:
        save_dir = nfs_dir

    flat_target, target_tree = tree_flatten(target)
    flat_metadata = []
    background_proc_pool = []
    for x in flat_target:
        if isinstance(x, DistributedArray):
            arr_dir = _get_save_path()
            x.save(os.path.join(save_dir, arr_dir))
            flat_metadata.append(arr_dir)
            ## TODO: background process
            # if local_cache_dir is not None:
            #     p = subprocess.Popen(["mv", save_dir, nfs_dir], stdin=None, 
            #                           stdout=None, stderr=None, shell=False)
            #     background_proc_pool.append(p)

        elif isinstance(x, ReplicatedDistributedArray):
            arr_dir = _get_save_path()
            x.replica.save(os.path.join(save_dir, arr_dir))
            flat_metadata.append(arr_dir)
            ## TODO: background process
        elif isinstance(x, (np.ndarray, jax.xla.DeviceArray)):
            arr_dir = _get_save_path()
            _save_arr(os.path.join(save_dir, arr_dir), x)
            flat_metadata.append(arr_dir)
        else:
            flat_metadata.append(x)

    metapath = os.path.join(nfs_dir, f"checkpoint_{step}")
    metadata = tree_unflatten(target_tree, flat_metadata)
    with open(metapath, "wb") as metafile:
        metafile.write(msgpack.packb(to_state_dict(metadata)))
        

class LoadInfo:
    """
    A wrapper for the loading information.
    """

    def __init__(self, avals: Sequence[ShapedArray],
                 meshes: Sequence[PhysicalDeviceMesh],
                 specs: Sequence[ShardingSpec]):
        assert len(avals) == len(meshes)
        assert len(meshes) == len(specs)
        self.avals = avals
        self.meshes = meshes
        self.specs = specs

    def add_replica(self, aval, mesh, spec):
        self.avals.append(aval)
        self.meshes.append(mesh)
        self.specs.append(spec)

    def get_info(self):
        if self.is_replicated():
            return zip(self.avals, self.meshes, self.specs)
        else:
            return self.avals[0], self.meshes[0], self.specs[0]

    def is_replicated(self):
        return len(self.avals) > 1

    def __str__(self):
        return f'{self.avals[0]}, {self.meshes[0].mesh_id}, {self.specs[0]}'


def restore_checkpoint(ckpt_dir: Union[str, os.PathLike], step: int,
                       load_info: PyTree):
    """Restore the specified checkpoint from `path`.

        Similar to flax.training.checkpoints.load_checkpoint,
        but support DistributedArrays and ReplicatedDistributedArray in alpa.
        # TODO (zhongyinmin): copy all the safe-loading stuff from
        https://flax.readthedocs.io/en/latest/_modules/flax/training/checkpoints.html#restore_checkpoint

        Args:
            ckpt_dir: directory of checkpoints to restore from.
            step: step number to load.
            load_info: shardingSpec and deviceMesh allocation info for loading.
    """
    metapath = os.path.join(ckpt_dir, f"checkpoint_{step}")
    with open(metapath, 'rb') as metafile:
        metadata = from_state_dict(load_info, msgpack.unpackb(metafile.read()))

    state_paths, state_tree = tree_flatten(metadata)
    flat_info = tree_leaves(load_info)
    flat_load_state = []
    for path, info in zip(state_paths, flat_info):
        if info is None:
            logger.warning('Variable is not used, skip loading it')
            flat_load_state.append(None)
        if info.is_replicated():
            meshes, arrays = [], []
            for aval, mesh, spec in info.get_info():
                meshes.append(mesh)
                arrays.append(DistributedArray.load(os.path.join(ckpt_dir, path), aval, mesh, spec))
            flat_load_state.append(ReplicatedDistributedArray(meshes, arrays)) 
        else:
            aval, mesh, spec = info.get_info()
            flat_load_state.append(DistributedArray.load(os.path.join(ckpt_dir, path), aval, mesh, spec))
    return tree_unflatten(state_tree, flat_load_state)
