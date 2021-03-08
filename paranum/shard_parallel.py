from functools import wraps, partial

import numpy as np

import jax
from jax import linear_util as lu
from jax.api_util import (
    shaped_abstractify,
    flatten_fun,
    flatten_axes,
    flatten_fun_nokwargs,
    argnums_partial,
)
from jax.config import flags, config, bool_env
from jax.core import ShapedArray
from jax.experimental.maps import mesh
from jax.experimental.pjit import pjit
from jax.interpreters import xla, partial_eval as pe
from jax.interpreters.pxla import parallel_callable
from jax.interpreters.sharded_jit import PartitionSpec
from jax.lib import xla_bridge as xb, xla_client as xc
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax._src.util import (
    unzip2,
    curry,
    partial,
    safe_map,
    safe_zip,
    prod,
    split_list,
    extend_name_stack,
    wrap_name,
    cache,
    wraps,
    HashableFunction,
)

from paranum import util

unsafe_map, map = map, safe_map  # type: ignore


def jaxpr_to_xla_computation(jaxpr, in_avals, consts, fun_name="", backend=None):
    c = xb.make_computation_builder(f"xla_computation_{fun_name}")
    xla_consts = map(partial(xb.constant, c), consts)
    should_tuple = len(in_avals) > 100
    xla_args, donated_invars = xla._xla_callable_args(c, in_avals, should_tuple)
    axis_env = xla.AxisEnv(1, (), ())
    out_nodes = xla.jaxpr_subcomp(
        c,
        jaxpr,
        backend,
        axis_env,
        xla_consts,
        extend_name_stack(wrap_name(fun_name, "xla_computation")),
        *xla_args,
    )
    build_out_tuple = partial(xc.ops.Tuple, c, out_nodes)
    out_tuple = build_out_tuple()
    built = c.build(out_tuple)
    return built


def decorate_data_parallel_pjit(fun, args, static_argnums, out_avals):
    dyn_args = [args[i] for i in range(len(args)) if i not in static_argnums]

    def make_partition_spec(x):
        if isinstance(x, flax.optim.base.Optimizer):
            return None

        if len(x.shape) == 0:
            return None
        else:
            spec = [None] * len(x.shape)
            if util.compute_bytes(x) > 1024:
                # partition the first dimension for large tensors
                spec[0] = "data_parallel_batch"
            return PartitionSpec(*spec)

    def is_leaf(x):
        if isinstance(x, flax.optim.base.Optimizer):
            return True
        return False

    # automatically detect weight or data
    in_axis_resources = jax.tree_util.tree_map(
        make_partition_spec, dyn_args, is_leaf=is_leaf
    )
    out_axis_resources = jax.tree_util.tree_map(
        make_partition_spec, out_avals, is_leaf=is_leaf
    )

    out_axis_resources = util.freeze_dict(out_axis_resources)

    shard_fun = pjit(
        fun,
        in_axis_resources=in_axis_resources,
        out_axis_resources=out_axis_resources,
        static_argnums=static_argnums,
    )

    with mesh(np.array(jax.devices()), ("data_parallel_batch",)):
        return shard_fun(*args)

@lu.cache
def shard_parallel_callable(
    fun: lu.WrappedFun,
    in_tree,
    out_tree,
    devices,
    *avals
):
    fun_name = fun.__name__

    # Get jaxpr and XLA hlo
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(fun, avals)
    #c = jaxpr_to_xla_computation(jaxpr, avals, consts, fun_name)
    print(jaxpr)

    strategy = 'partition_all'

    if strategy == 'partition_all':
        pass

    exit()

#    def make_partition_spec(x):
#        if len(x.shape) == 0:
#            return None
#        else:
#            spec = [None] * len(x.shape)
#            if util.compute_bytes(x) > 1024:
#                # partition the first dimension for large tensors
#                spec[0] = "x"
#            return PartitionSpec(*spec)
#
#    in_axis_resources = tree_map(make_partition_spec, dyn_args)
#    out_axis_resources = tree_map(make_partition_spec, out_avals)
#
#    shard_fun = pjit(
#        fun,
#        in_axis_resources=in_axis_resources,
#        out_axis_resources=out_axis_resources,
#        static_argnums=static_argnums,
#    )
#
#    with mesh(np.array(jax.devices()), ("x",)):
#        return shard_fun(*args)


