"""Serialization utilities for Alpa.
Adapted from https://flax.readthedocs.io/en/latest/_modules/flax/serialization.html, 
add support for DistributedArray serialization in Alpa.
"""

import enum
import os
from typing import Callable, Optional, Sequence, Union, Any
import uuid

from flax.serialization import to_state_dict, from_state_dict
import jax
import msgpack
import numpy as np
from tensorflow.io import gfile

from alpa import DistributedArray, ReplicatedDistributedArray

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
        # if isinstance(x, (np.ndarray, jax.xla.DeviceArray)):
        #     return msgpack.ExtType(_MsgpackExtType.ndarray, _ndarray_to_bytes(x))
        # if np.issctype(type(x)):
        #     # pack scalar as ndarray
        #     return msgpack.ExtType(_MsgpackExtType.npscalar,
        #                            _ndarray_to_bytes(np.asarray(x)))
        # elif isinstance(x, complex):
        #     return msgpack.ExtType(_MsgpackExtType.native_complex,
        #                            msgpack.packb((x.real, x.imag)))
        if isinstance(x, DistributedArray):
            save_dir = os.path.join(ckpt_dir, uuid.uuid4().hex)
            x.save(save_dir)
            return msgpack.ExtType(_MsgpackExtType.distarray, msgpack.packb(save_dir))
        elif isinstance(x, ReplicatedDistributedArray):
            save_dir = os.path.join(ckpt_dir, uuid.uuid4().hex)
            x.replica.save(save_dir)
            return msgpack.ExtType(_MsgpackExtType.replicated_distarray, msgpack.packb(save_dir))
        return x
    return _msgpack_ext_pack

def _msgpack_ext_unpack(code, data):
    """Messagepack decoders for custom types."""
    # if code == _MsgpackExtType.ndarray:
    #     return _ndarray_from_bytes(data)
    # elif code == _MsgpackExtType.native_complex:
    #     complex_tuple = msgpack.unpackb(data)
    #     return complex(complex_tuple[0], complex_tuple[1])
    # elif code == _MsgpackExtType.npscalar:
    #     ar = _ndarray_from_bytes(data)
    #     return ar[()]  # unpack ndarray to scalar
    if code == _MsgpackExtType.distarray:
        return msgpack.unpackb(data)
    elif code == _MsgpackExtType.replicated_distarray:
        return msgpack.unpackb(data)
    return msgpack.ExtType(code, data)


def save_checkpoint(ckpt_dir: Union[str, os.PathLike], 
                    target: PyTree,
                    step: int):
    """The same as flax.training.checkpoints.save_checkpoint. Save a checkpoint of the model, but support DistributedArrays in alpa."""
    # TODO: copy all the safe-saving stuff from https://flax.readthedocs.io/en/latest/_modules/flax/training/checkpoints.html#save_checkpoint
    state_dict = to_state_dict(target)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_tmp_path = os.path.join(ckpt_dir, f"checkpoint_{step}")
    with gfile.GFile(ckpt_tmp_path, 'wb') as fp:
        fp.write(msgpack.packb(state_dict, default=_msgpack_ext_pack_wrapper(ckpt_dir), strict_types=True))

def restore_checkpoint(ckpt_dir: Union[str, os.PathLike], 
                       target: PyTree,
                       step: int):
    """The same as flax.training.checkpoints.load_checkpoint. Restore last/best checkpoint from checkpoints in path, but support DistributedArrays in alpa."""
    # TODO: copy all the safe-loading stuff from https://flax.readthedocs.io/en/latest/_modules/flax/training/checkpoints.html#restore_checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step}")
    with gfile.GFile(ckpt_path, 'rb') as fp:
        ckpt_contents = fp.read()
    state_dict = msgpack.unpackb(ckpt_contents, ext_hook=_msgpack_ext_unpack, raw=False)
    return from_state_dict(target, state_dict)
