"""
Serialization utilities for Alpa.
Support DistributedArray and ReplicatedDistributedArray serialization in Alpa.
"""

import logging
import os
import pickle
from typing import Union

from flax.serialization import to_state_dict, from_state_dict
import jax
from jax._src.tree_util import tree_flatten, tree_leaves, tree_unflatten, PyTreeDef
import msgpack
import numpy as np

from alpa.device_mesh import (DistributedArray, ReplicatedDistributedArray,
                              get_global_virtual_physical_mesh,
                              get_global_physical_mesh)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _dfs_pytree(tree, prefix):
    paths = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            paths += _dfs_pytree(v, prefix + "." + str(k))
    elif isinstance(tree, (tuple, list)):
        for i, v in enumerate(tree):
            paths += _dfs_pytree(v, prefix + "." + str(i))
    elif tree is not None:
        # Leaf node
        paths.append(prefix)
    return paths


def _save_unsharded_array(ckpt_dir, arr):
    os.makedirs(ckpt_dir, exist_ok=True)
    shard_name = "shard_0.0"
    metadata = {
        "global_shape": arr.shape,
        "dtype": arr.dtype,
        "shard_names": [shard_name],
        "shard_indices": None,
    }
    with open(os.path.join(ckpt_dir, shard_name), "wb") as datafile:
        np.save(datafile, arr)
    with open(os.path.join(ckpt_dir, "metadata_0"), "wb") as metafile:
        pickle.dump(metadata, metafile)


def load_sharded_array(ckpt_dir, metadatas):
    """
        Used by MeshHostWorker.load_tensor to first load the entire shared
        array from disk.
    """
    assert len(metadatas) > 0
    with open(os.path.join(ckpt_dir, metadatas[0]), "rb") as metafile:
        meta = pickle.load(metafile)
    if meta["shard_indices"] is None:
        return np.load(os.path.join(ckpt_dir, meta["shard_names"][0]))
    entire_array = np.empty(meta["global_shape"], meta["dtype"])
    for metadata in metadatas:
        with open(os.path.join(ckpt_dir, metadata), "rb") as metafile:
            meta = pickle.load(metafile)
        for shard_name, shard_indice in zip(meta["shard_names"],
                                            meta["shard_indices"]):
            entire_array[shard_indice] = np.load(
                os.path.join(ckpt_dir, shard_name))
    return entire_array


def save_checkpoint(ckpt_dir: Union[str, os.PathLike],
                    target: PyTreeDef,
                    step: int,
                    local_cache_dir: Union[str, os.PathLike, None] = None):
    """
        Save a checkpoint of the `target` to `ckpt_dir`.

        If you want to save a model which has been parallelized on multiple
        nodes by alpa, `ckpt_dir` should be a shared filesystem path.
        It is also recommended to provide a `local_cache_dir` on local disk
        to speed up the saving process because `save_checkpoint` will return
        as soon as each node has saved its shard of the model into
        `local_cache_dir`. The DaemonMoveWorkers will then move these local
        shards into `ckpt_dir` in the background.

        If you just want to save a unparallelized model or the model is
        parallellized on a single node, `ckpt_dir` should be a normal
        path on local disk, and the `local_cache_dir` should be None.

        Args:
           ckpt_dir: the directory where this checkpoint will be saved.
           target: serializable flax object, usually a trainState.
           step: training step number or other metric number.
           local_cache_dir: If not None, `ckpt_dir` should be a
           shared filesystem path, and this function will return as soon as
           the shards have been saved to this local directory. DaemonMoveWorkers
           will move these shards into `ckpt_dir` in the background.
    """
    # create directories if not exist
    os.makedirs(ckpt_dir, exist_ok=True)
    if local_cache_dir is not None:
        os.makedirs(local_cache_dir, exist_ok=True)

    target = to_state_dict(target)
    flat_dirs = _dfs_pytree(target, "state")
    flat_target, target_tree = tree_flatten(target)
    flat_metadata = []
    assert (len(flat_dirs) == len(flat_target))
    for arr_dir, x in zip(flat_dirs, flat_target):
        arr_path = os.path.join(ckpt_dir, arr_dir)
        if local_cache_dir is None:
            arr_cache_path = None
        else:
            arr_cache_path = os.path.join(local_cache_dir, arr_dir)
        if isinstance(x, (DistributedArray, ReplicatedDistributedArray,
                          np.ndarray, jax.xla.DeviceArray)):
            if isinstance(x, DistributedArray):
                x.save(arr_path, arr_cache_path)
            elif isinstance(x, ReplicatedDistributedArray):
                x.replica.save(arr_path, arr_cache_path)
            elif isinstance(x, (np.ndarray, jax.xla.DeviceArray)):
                _save_unsharded_array(arr_path, x)
            flat_metadata.append(arr_dir)
        else:
            flat_metadata.append(x)

    metapath = os.path.join(ckpt_dir, f"checkpoint_{step}")
    metadata = tree_unflatten(target_tree, flat_metadata)
    with open(metapath, "wb") as metafile:
        metafile.write(msgpack.packb(metadata))


def restore_checkpoint(ckpt_dir: Union[str, os.PathLike], step: int,
                       placement_specs: PyTreeDef):
    """
        Restore the specified checkpoint from `ckpt_dir` and reshard it
        according to the `placement_specs`.

        Args:
            ckpt_dir: directory of checkpoints to restore from. If you
            do not have a shared filesystem, each host needs a copy of
            the checkpoint on its local disk at the same path.
            step: step number to load.
            placement_specs: shardingSpec and deviceMesh placement info
            for loading.
    """
    metapath = os.path.join(ckpt_dir, f"checkpoint_{step}")
    with open(metapath, "rb") as metafile:
        metadata = from_state_dict(placement_specs,
                                   msgpack.unpackb(metafile.read()))

    state_paths, state_tree = tree_flatten(metadata)
    flat_info = tree_leaves(placement_specs)
    flat_load_state = []
    mesh_group = get_global_virtual_physical_mesh().launched_physical_mesh_group
    physical_mesh = get_global_physical_mesh()

    assert mesh_group is not None or physical_mesh is not None

    for path, info in zip(state_paths, flat_info):
        if info is None:
            logger.warning("Variable is not used, skip loading it")
            flat_load_state.append(None)
        elif mesh_group is None:
            dist_arr = DistributedArray.load(os.path.join(ckpt_dir, path),
                                             info.aval, physical_mesh,
                                             physical_mesh,
                                             info.sharding_specs[0])
            flat_load_state.append(dist_arr)
        elif len(info.mesh_ids) == 1:
            dist_arr = DistributedArray.load(os.path.join(ckpt_dir,
                                                          path), info.aval,
                                             mesh_group[info.mesh_ids[0]],
                                             info.sharding_specs[0])
            flat_load_state.append(dist_arr)
        else:
            meshes, arrays = [], []
            for mesh_id, spec in zip(info.mesh_ids, info.sharding_specs):
                meshes.append(mesh_group[mesh_id])
                dist_arr = DistributedArray.load(os.path.join(ckpt_dir,
                                                              path), info.aval,
                                                 mesh_group[mesh_id], spec)
                arrays.append(dist_arr)
            flat_load_state.append(ReplicatedDistributedArray(meshes, arrays))

    return tree_unflatten(state_tree, flat_load_state)
