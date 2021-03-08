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
import flax

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


def decorate_partition_all(fun, args, static_argnums, out_avals):
    dyn_args = [args[i] for i in range(len(args)) if i not in static_argnums]

    def make_partition_spec(x):
        if len(x.shape) == 0:
            return None
        else:
            spec = [None] * len(x.shape)
            if util.compute_bytes(x) > 1024:
                # partition the first dimension for large tensors
                spec[0] = "x"
            return PartitionSpec(*spec)

    in_axis_resources = tree_map(make_partition_spec, dyn_args)
    out_axis_resources = tree_map(make_partition_spec, out_avals)

    shard_fun = pjit(
        fun,
        in_axis_resources=in_axis_resources,
        out_axis_resources=out_axis_resources,
        static_argnums=static_argnums,
    )

    with mesh(np.array(jax.devices()), ("x",)):
        return shard_fun(*args)


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


def parallelize(fun, static_argnums="auto"):
    n_devices = len(jax.devices())
    fun_name = getattr(fun, "__name__", "unknown")

    @wraps(fun)
    def ret_fun(*args, **kwargs):
        raw_args = args
        if kwargs:
            raise NotImplementedError("parallelize over kwargs not yet supported")
        # Automatic static argnums
        nonlocal static_argnums
        if static_argnums == "auto":
            static_argnums = [i for i in range(len(args)) if is_static_arg(args[i])]

        return decorate_data_parallel_pmap(fun, raw_args, static_argnums)

        wrapped = lu.wrap_init(fun)

        # Abstractify arguments
        if static_argnums:
            dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
            wrapped, args = argnums_partial(wrapped, dyn_argnums, args)
        jax_args, in_tree = tree_flatten(args)
        flat_fun, out_tree = flatten_fun_nokwargs(wrapped, in_tree)
        in_avals = map(shaped_abstractify, jax_args)

        assert config.omnistaging_enabled

        # Get jaxpr and XLA hlo
        jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
        c = jaxpr_to_xla_computation(jaxpr, in_avals, consts, fun_name)

        # Choose parallel strategy
        strategy = None
        if "threefry" in c.as_hlo_text():  # weight initialization
            strategy = "partition_all"
        else:
            strategy = "data_parallel"

        # Apply parallel strategy
        if strategy == "partition_all":
            return decorate_partition_all(
                fun, raw_args, static_argnums, tree_unflatten(out_tree(), out_avals)
            )
        elif strategy == "data_parallel":
            return decorate_data_parallel(
                fun, raw_args, static_argnums, tree_unflatten(out_tree(), out_avals)
            )
        else:
            raise ValueError("Invalid parallel strategy")

    return ret_fun
