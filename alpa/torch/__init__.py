"""Miscellaneous functions available in `alpa.torch.*` namespace."""

try:
    import torch
except ImportError as e:
    print("""
        Attempted to use Alpa-PyTorch frontend, but PyTorch is not installed.

        Please follow instructions at 
        https://alpa-projects.github.io/install.html#pytorch-frontend-experimental
        to install PyTorch and related dependencies.""")
    raise e

from typing import Any, Callable, Union, Tuple
from functools import partial, wraps

import numpy as np

import alpa
from alpa.device_mesh import DistributedArray
from alpa.torch.nn import functionalize, meta_init
from alpa.torch.ops.mapping import enable_dist_for_func
from alpa.torch.tensor_utils import (make_shaped_array_from_pt_tensor,
                                     initialize_with_zeros, to_format,
                                     assert_format)
from alpa.torch import trainer

# If True, prints verbose log for debugging.
debug = False


def set_mode(new_mode: str):
    """This sets the current alpa.torch mode. Supports one of following:

    "local":
    - Pure PT eager mode on a single CPU/GPU
    - Allows print in middle of graph
    - No dist training

    "dist":
    - Graph mode by lowering PT programs to JAX and then run them with Alpa
    - Doesn't allow print in middle of graph
    - Supports dist training
    """
    assert new_mode in ["local", "dist"]
    if new_mode == "dist":
        torch.local_mode = False
    elif new_mode == "local":
        torch.local_mode = True


def mode():
    if torch.local_mode:
        return "local"
    else:
        return "dist"


def functorch_value_and_grad(func: Callable,
                             argnums: Union[int, Tuple[int, ...]] = 0,
                             has_aux: bool = False) -> Callable:
    """The same implementation as functorch.grad_and_value,
    but puts value first and grad second in output.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from functorch._C import (_grad_increment_nesting,
                                  _grad_decrement_nesting)
        from functorch._src.eager_transforms import (
            _wrap_all_tensors, _slice_argnums, _create_differentiable,
            _as_tuple, _autograd_grad, _undo_create_differentiable)
        from functorch._src.pytree_hacks import tree_map_
        from torch.utils._pytree import tree_flatten, tree_unflatten
        level = _grad_increment_nesting()
        try:
            output, aux, grad_input = None, None, None
            # See NOTE [grad and vjp interaction with no_grad]
            with torch.enable_grad():
                args = _wrap_all_tensors(args, level)
                kwargs = _wrap_all_tensors(kwargs, level)
                diff_args = _slice_argnums(args, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level),
                          diff_args)

                output = func(*args, **kwargs)
                if has_aux:
                    if not (isinstance(output, tuple) and len(output) == 2):
                        raise RuntimeError(
                            "value_and_grad(f)(*args): output of function f "
                            "should be a tuple: (output, aux) "
                            "if has_aux is True")
                    output, aux = output

                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(
                        "value_and_grad(f)(*args): Expected f(*args) "
                        f"to return a Tensor, got {type(output)}")
                if output.dim() != 0:
                    raise RuntimeError(
                        "value_and_grad(f)(*args): Expected f(*args) "
                        "to return a scalar Tensor, got tensor with "
                        f"{output.dim()} dims. Maybe you wanted to "
                        "use the vjp or jacrev APIs instead?")

                flat_diff_args, spec = tree_flatten(diff_args)

                # NB: need create_graph so that backward pass isn't run
                # in no_grad mode
                flat_outputs = _as_tuple(output)
                flat_grad_input = _autograd_grad(flat_outputs,
                                                 flat_diff_args,
                                                 create_graph=True)
                grad_input = tree_unflatten(flat_grad_input, spec)

                grad_input = _undo_create_differentiable(grad_input, level)
                output = _undo_create_differentiable(output, level)
                if aux is not None:
                    aux = _undo_create_differentiable(aux, level)

            if has_aux:
                return (output, aux), grad_input
            return output, grad_input
        finally:
            _grad_decrement_nesting()

    return wrapper


def value_and_grad(func, argnums=0, has_aux=False):
    if mode() == "local":
        return functorch_value_and_grad(func, argnums=argnums, has_aux=has_aux)
    else:
        return alpa.value_and_grad(func, argnums=argnums, has_aux=has_aux)
