"""Tensor-related utility functions.
"""
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

import alpa
import alpa.torch as atorch

# Copied from torch/testing/_internal/common_utils.py#L349
# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
numpy_to_torch_dtype_dict = {
    np.dtype(np.bool): torch.bool,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.complex64): torch.complex64,
    np.dtype(np.complex128): torch.complex128,
}

# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}


def make_shaped_array_from_pt_tensor(pt_tensors):

    def transform(pt_tensor):
        shape = list(pt_tensor.shape)
        np_dtype = torch_to_numpy_dtype_dict[pt_tensor.dtype]
        return jax.abstract_arrays.ShapedArray(shape, np_dtype)

    return jax.tree_map(transform, pt_tensors)


def initialize_with_zeros(*args):
    if atorch.mode() == "local":
        return jax.tree_map(lambda x: torch.zeros(*x.shape, dtype=x.dtype),
                            args)
    else:
        return jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), args)


def to_format(target_format: str, inp: Any):
    """Converts inputs to the format specified by `target_format`.
    Supported formats are "local" and "dist".
    """
    assert target_format in ["local", "dist"]
    ret = None
    if isinstance(inp, tuple):
        ret = tuple(to_format(target_format, x) for x in inp)
    elif isinstance(inp, list):
        ret = [to_format(target_format, x) for x in inp]
    elif isinstance(inp, dict):
        ret = dict(
            zip(inp.keys(),
                [to_format(target_format, x) for x in inp.values()]))
    elif isinstance(inp, torch.Tensor):
        if target_format == "dist":
            if str(inp.device) == "meta":
                ret = make_shaped_array_from_pt_tensor(inp)
            elif str(inp.device) == "cpu":
                ret = inp.numpy()
            else:
                # TODO: add support for CUDA input tensor
                raise NotImplementedError(
                    f"PyTorch tensor of device {type(inp.device)} "
                    "is not supported yet.")
        elif target_format == "local":
            ret = inp
    elif isinstance(inp, alpa.device_mesh.DistributedArray):
        if target_format == "local":
            ret = torch.from_numpy(np.array(inp))
        elif target_format == "dist":
            ret = inp
    if ret is not None:
        return ret
    else:
        raise NotImplementedError(
            f"Value of type {type(inp)} is not supported yet.")


def assert_format(target_format: str, *inputs):
    """Asserts inputs are in the format specified by `target_format`.
    Supported formats are "local" and "dist".
    """
    assert target_format in ["local", "dist"]
    for inp in inputs:
        if isinstance(inp, (tuple, list)):
            assert_format(target_format, *inp)
        elif isinstance(inp, dict):
            assert_format(target_format, *inp.values())
        else:
            assert (
                isinstance(inp, torch.Tensor) and target_format == "local"
            ) or (
                isinstance(inp,
                           (alpa.device_mesh.DistributedArray,
                            alpa.device_mesh.ReplicatedDistributedArray)) and
                target_format == "dist"
            ), f"This input is not of {target_format} format: {inp}, " + \
            "of type {type(inp)}"
