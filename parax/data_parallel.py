# flake8: noqa
"""Data parallel based on pmap or gshard."""
from collections import OrderedDict

import flax
import jax
from jax import linear_util as lu
from jax.api_util import (
    argnums_partial,
    flatten_axes,
    flatten_fun_nokwargs,
)
from jax.core import ShapedArray
from jax.experimental.maps import mesh
from jax.interpreters import xla, partial_eval as pe
from jax.interpreters.pxla import parallel_callable, mesh_callable, Mesh
from jax.lib import xla_bridge as xb, xla_client as xc
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax._src.util import (
    extend_name_stack,
    safe_map,
    safe_zip,
    partial,
    wraps,
    wrap_name,
    HashableFunction,
)
import numpy as np

from parax import util, testing


unsafe_map, map = map, safe_map  # type: ignore


def annotate_gradient(gradients):
    """Annotate gradient for pmap based data parallel"""
    from jax.core import thread_local_state

    axis_env = thread_local_state.trace_state.axis_env
    in_data_parallel = False
    for x in axis_env:
        if x.name == "data_parallel_batch":
            in_data_parallel = True

    if in_data_parallel:
        return jax.lax.pmean(gradients, "data_parallel_batch")
    else:
        return gradients


def data_parallel_split(x, axis_size):
    assert x.shape[0] % axis_size == 0, x.shape
    if isinstance(x, ShapedArray):
        new_shape = (axis_size, x.shape[0] // axis_size) + x.shape[1:]
        return x.update(shape=new_shape)
    else:
        return x.reshape((axis_size, x.shape[0] // axis_size) + x.shape[1:])


def should_replicate_map(x):
    """Detect whether we should replicate an argument for data parallel"""
    if isinstance(x, flax.optim.base.Optimizer):
        return True

    if util.compute_bytes(x) < 128:
        return True
    else:
        return False


def should_replicate_is_leaf(x):
    if isinstance(x, flax.optim.base.Optimizer):
        return True
    return False


@lu.cache
def pmap_data_parallel_callable(
    fun: lu.WrappedFun,
    in_tree,
    out_tree_thunk,
    devices,
    donated_invars,
    *avals
):
    fun_name = fun.__name__
    devices = devices or tuple(jax.devices())
    axis_name = "data_parallel_batch"
    axis_size = len(devices)

    # Detect weight tensors and mark them as "should_replicate"
    dyn_args = tree_unflatten(in_tree, avals)
    should_replicate = tree_map(
        should_replicate_map, dyn_args, is_leaf=should_replicate_is_leaf
    )
    should_replicate = tuple(
        flatten_axes("pmap_data_parallel_callable should_replicate", in_tree, should_replicate)
    )

    # Create in_axes paritition spec
    flatten_in_axes = tuple(unsafe_map(lambda x: None if x else 0, should_replicate))

    # Get jaxpr and XLA hlo
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(fun, avals)

    # Create out_axes paritition spec
    unflatten_out_avals = tree_unflatten(out_tree_thunk(), out_avals)
    out_should_replicate = tree_map(
        should_replicate_map, unflatten_out_avals, is_leaf=should_replicate_is_leaf
    )
    out_should_replicate = flatten_axes(
        "pmap_data_parallel_callable out_should_replicate",
        out_tree_thunk(),
        out_should_replicate,
    )
    flatten_out_axes = tuple(
        unsafe_map(lambda x: None if x else 0, out_should_replicate)
    )

    # Compile parallel_callable
    backend = None
    global_axis_size = None
    name = fun.__name__
    out_axes_thunk = HashableFunction(
        lambda: flatten_out_axes, closure=flatten_out_axes
    )
    global_arg_shapes = (None,) * len(avals)

    # Clean stores for the next call
    for store in fun.stores:
        store and store.reset()

    # Split avals and lower to parallel_callable
    split_avals = (
        avals[i] if should_replicate[i] else data_parallel_split(avals[i], axis_size)
        for i in range(len(avals))
    )
    compiled_fun = parallel_callable(
        fun,
        backend,
        axis_name,
        axis_size,
        global_axis_size,
        devices,
        name,
        flatten_in_axes,
        out_axes_thunk,
        donated_invars,
        global_arg_shapes,
        *split_avals
    )
    testing.last_compiled_executable = compiled_fun.args[0]

    def ret_func(*args):
        split_args = (
            args[i] if should_replicate[i] else data_parallel_split(args[i], axis_size)
            for i in range(len(args))
        )
        return compiled_fun(*split_args)

    return ret_func

unsafe_map, map = map, safe_map  # type: ignore


def shard_first_dim(x):
    if util.compute_bytes(x) < 128:
        return OrderedDict()
    return OrderedDict([('mesh_x', 0)])


@lu.cache
def shard_data_parallel_callable(
    fun: lu.WrappedFun,
    in_tree,
    out_tree_thunk,
    devices,
    donated_invars,
    *avals
):
    fun_name = fun.__name__
    devices = devices or np.array(jax.devices())

    # Get jaxpr and XLA hlo
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(fun, avals)

    # Detect weight tensors and mark them as "should_replicate"
    dyn_args = tree_unflatten(in_tree, avals)
    should_replicate = tree_map(
        should_replicate_map, dyn_args, is_leaf=should_replicate_is_leaf
    )
    should_replicate = tuple(
        flatten_axes("shard_parallel_callable should_replicate", in_tree, should_replicate)
    )

    # Create in_axes paritition spec
    in_axes = tuple(OrderedDict() if should_replicate[i] else shard_first_dim(avals[i])
                    for i in range(len(avals)))

    # Create out_axes paritition spec
    unflatten_out_avals = tree_unflatten(out_tree_thunk(), out_avals)
    out_should_replicate = tree_map(
        should_replicate_map, unflatten_out_avals, is_leaf=should_replicate_is_leaf
    )
    out_should_replicate = flatten_axes(
        "shard_parallel_callable out_should_replicate",
        out_tree_thunk(),
        out_should_replicate,
    )
    out_axes = tuple(OrderedDict() if out_should_replicate[i] else shard_first_dim(out_avals[i])
                    for i in range(len(out_avals)))

    devices = np.array(devices)
    mesh = Mesh(devices, ('mesh_x',))
    out_axes_thunk = lambda: out_axes

    # Clean stores for the next call
    for store in fun.stores:
        store and store.reset()

    # Lower to mesh_callable
    compiled_func = mesh_callable(fun, fun_name, None, mesh,
                                  in_axes, out_axes_thunk, donated_invars,
                                  True, *avals, tile_by_mesh_axes=False,
                                  do_resource_typecheck=None)
    testing.last_compiled_executable = compiled_func.args[0]
    return compiled_func

