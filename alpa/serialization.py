"""Serialization utilities for Alpa.
Adapted from https://flax.readthedocs.io/en/latest/_modules/flax/serialization.html.
Add support for DistributedArray and ReplicatedDistributedArray serialization in Alpa.
"""

import enum
import os
from typing import Union, Any, Sequence
import uuid

from flax.serialization import to_state_dict, from_state_dict, _ndarray_from_bytes, _ndarray_to_bytes
import jax
from jax.interpreters.pxla import ShardingSpec
from jax.core import ShapedArray
from jax._src.tree_util import tree_flatten, tree_leaves, tree_unflatten
import msgpack
import numpy as np

from alpa.device_mesh import DistributedArray, ReplicatedDistributedArray, PhysicalDeviceMesh

PyTree = Any


class _MsgpackExtType(enum.IntEnum):
    """Messagepack custom type ids."""
    ndarray = 1
    native_complex = 2
    npscalar = 3
    distarray = 4
    replicated_distarray = 5


def _msgpack_ext_pack_wrapper(ckpt_dir):

    def _msgpack_ext_pack(x):
        """Messagepack encoders for custom types."""
        if isinstance(x, (np.ndarray, jax.xla.DeviceArray)):
            return msgpack.ExtType(_MsgpackExtType.ndarray,
                                   _ndarray_to_bytes(x))
        if np.issctype(type(x)):
            # pack scalar as ndarray
            return msgpack.ExtType(_MsgpackExtType.npscalar,
                                   _ndarray_to_bytes(np.asarray(x)))
        elif isinstance(x, complex):
            return msgpack.ExtType(_MsgpackExtType.native_complex,
                                   msgpack.packb((x.real, x.imag)))
        elif isinstance(x, DistributedArray):
            save_dir = os.path.join(ckpt_dir, uuid.uuid4().hex)
            x.save(save_dir)
            return msgpack.ExtType(_MsgpackExtType.distarray,
                                   msgpack.packb(save_dir))
        elif isinstance(x, ReplicatedDistributedArray):
            save_dir = os.path.join(ckpt_dir, uuid.uuid4().hex)
            x.replica.save(save_dir)
            return msgpack.ExtType(_MsgpackExtType.replicated_distarray,
                                   msgpack.packb(save_dir))
        return x

    return _msgpack_ext_pack


def _msgpack_ext_unpack(code, data):
    """Messagepack decoders for custom types."""
    if code == _MsgpackExtType.ndarray:
        return _ndarray_from_bytes(data)
    elif code == _MsgpackExtType.native_complex:
        complex_tuple = msgpack.unpackb(data)
        return complex(complex_tuple[0], complex_tuple[1])
    elif code == _MsgpackExtType.npscalar:
        ar = _ndarray_from_bytes(data)
        return ar[()]  # unpack ndarray to scalar
    elif code == _MsgpackExtType.distarray:
        return msgpack.unpackb(data)
    elif code == _MsgpackExtType.replicated_distarray:
        return msgpack.unpackb(data)
    return msgpack.ExtType(code, data)


def save_checkpoint(ckpt_dir: Union[str, os.PathLike], target: PyTree,
                    step: int):
    """Save a checkpoint of the `target` to `path`. 

        Similar to flax.training.checkpoints.save_checkpoint, but support DistributedArrays 
        and ReplicatedDistributedArray in alpa.
        # TODO: copy all the safe-saving stuff from 
        https://flax.readthedocs.io/en/latest/_modules/flax/training/checkpoints.html#save_checkpoint

        Args:
           ckpt_dir: str or pathlib-like path to store checkpoint directories in.
           target: serializable flax object, usually a trainState
           step: training step number or other metric number
    """
    state_dict = to_state_dict(target)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step}")
    with open(ckpt_path, 'wb') as fp:
        fp.write(
            msgpack.packb(state_dict,
                          default=_msgpack_ext_pack_wrapper(ckpt_dir),
                          strict_types=True))


class LoadInfo:
    """
    A wrapper for the loading information.
    """
    def __init__(self, 
                 avals: Sequence[ShapedArray], 
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


def restore_checkpoint(ckpt_dir: Union[str, os.PathLike], step: int, target: PyTree, load_info: PyTree):
    """Restore the specified checkpoint from `path`. 

        Similar to flax.training.checkpoints.load_checkpoint, 
        but support DistributedArrays and ReplicatedDistributedArray in alpa.
        # TODO: copy all the safe-loading stuff from 
        https://flax.readthedocs.io/en/latest/_modules/flax/training/checkpoints.html#restore_checkpoint

        Args:
            ckpt_dir: directory of checkpoints to restore from.
            step: step number to load.
            target: matching object to rebuild via deserialized state-dict.
            load_info: shardingSpec and deviceMesh allocation info for loading.
    """
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step}")
    with open(ckpt_path, 'rb') as fp:
        ckpt_contents = fp.read()
    state_dict_content = msgpack.unpackb(ckpt_contents,
                                         ext_hook=_msgpack_ext_unpack,
                                         raw=False)
    state_paths, state_tree = tree_flatten(from_state_dict(target, state_dict_content))
    flat_info = tree_leaves(load_info)
    flat_load_state = []
    for path, info in zip(state_paths, flat_info):
        if info.is_replicated():
            meshes, arrays = [], []
            for aval, mesh, spec in info.get_info():
                meshes.append(mesh)
                arrays.append(DistributedArray.load(path, aval, mesh, spec))
            flat_load_state.append(ReplicatedDistributedArray(meshes, arrays)) 
        else:
            aval, mesh, spec = info.get_info()
            flat_load_state.append(DistributedArray.load(path, aval, mesh, spec))
    return tree_unflatten(state_tree, flat_load_state)
