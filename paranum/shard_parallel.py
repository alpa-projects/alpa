from collections import OrderedDict
from functools import wraps, partial
import itertools
import re
import threading

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
from jax.interpreters.pxla import parallel_callable, mesh_callable, Mesh
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
from paranum.data_parallel import should_replicate_map, should_replicate_is_leaf

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


def shard_first_dim(x):
    if util.compute_bytes(x) < 128:
        return OrderedDict()
    return OrderedDict([('mesh_x', 0)])


def shard_last_dim(x):
    if util.compute_bytes(x) < 128:
        return OrderedDict()
    return OrderedDict([('mesh_x', len(x.shape) - 1)])


@lu.cache
def shard_parallel_callable(
    fun: lu.WrappedFun,
    in_tree,
    out_tree_thunk,
    devices,
    *avals
):
    fun_name = fun.__name__
    devices = devices or np.array(jax.devices())

    # Get jaxpr and XLA hlo
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(fun, avals)

    strategy = 'data_parallel'

    if strategy == 'partition_all':
        mesh = Mesh(devices, ('mesh_x',))
        in_axes = tuple(unsafe_map(shard_first_dim, avals))
        out_axes = tuple(unsafe_map(shard_first_dim, out_avals))
        out_axes_thunk = lambda: out_axes
        donated_invars = (False,) * len(avals)
    elif strategy == 'data_parallel':
        # Detect weight tensors and mark them as "should_replicate"
        dyn_args = tree_unflatten(in_tree, avals)
        should_replicate = tree_map(
            should_replicate_map, dyn_args, should_replicate_is_leaf
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
            should_replicate_map, unflatten_out_avals, should_replicate_is_leaf
        )
        out_should_replicate = flatten_axes(
            "shard_parallel_callable out_should_replicate",
            out_tree_thunk(),
            out_should_replicate,
        )
        out_axes = tuple(OrderedDict() if out_should_replicate[i] else shard_first_dim(out_avals[i])
                        for i in range(len(out_avals)))

        mesh = Mesh(devices, ('mesh_x',))
        out_axes_thunk = lambda: out_axes
        donated_invars = (False,) * len(avals)
    elif strategy == 'model_parallel':
        # Detect weight tensors and mark them as "should_replicate"
        dyn_args = tree_unflatten(in_tree, avals)
        should_replicate = tree_map(
            should_replicate_map, dyn_args, should_replicate_is_leaf
        )
        should_replicate = tuple(
            flatten_axes("shard_parallel_callable should_replicate", in_tree, should_replicate)
        )

        # Create in_axes paritition spec
        in_axes = tuple(shard_last_dim(avals[i]) if should_replicate[i] else OrderedDict()
                        for i in range(len(avals)))

        # Create out_axes paritition spec
        unflatten_out_avals = tree_unflatten(out_tree_thunk(), out_avals)
        out_should_replicate = tree_map(
            should_replicate_map, unflatten_out_avals, should_replicate_is_leaf
        )
        out_should_replicate = flatten_axes(
            "shard_parallel_callable out_should_replicate",
            out_tree_thunk(),
            out_should_replicate,
        )
        out_axes = tuple(shard_last_dim(out_avals[i]) if out_should_replicate[i] else OrderedDict()
                         for i in range(len(out_avals)))

        mesh = Mesh(devices, ('mesh_x',))
        out_axes_thunk = lambda: out_axes
        donated_invars = (False,) * len(avals)
    elif strategy == 'naive_search':
        # Generate search space
        subspaces = []
        for aval in avals:
            subspace = []
            subspace.append(OrderedDict())
            if len(aval.shape) == 1:
                subspace.append([shard_first_dim(aval)])
            elif len(aval.shape) >= 2:
                subspace.extend([shard_first_dim(aval), shard_last_dim(aval)])
            subspaces.append(subspace)

        search_space = tuple(itertools.product(*subspaces))

        # Grid search
        best_idx = -1
        best_cost = float("inf")
        best_in_axes = None
        donated_invars = (False,) * len(avals)
        mesh = Mesh(devices, ('mesh_x',))
        perm = np.random.permutation(len(search_space))

        for i in range(len(search_space)):
        #for i in [56]:
            #in_axes = (OrderedDict(), OrderedDict([('mesh_x', 1)]), OrderedDict([('mesh_x', 0)]), OrderedDict(), OrderedDict())
            in_axes = search_space[i]
            out_axes = in_axes[:len(out_avals)]
            out_axes_thunk = lambda: out_axes
            cost = compute_mesh_callable_cost(
                fun, fun_name, None, mesh,
                in_axes, out_axes_thunk, donated_invars,
                True, *avals, tile_by_mesh_axes=False)

            if cost <= best_cost:
                best_cost = cost
                best_idx = i
                best_in_axes = in_axes

            print(f"idx: {i}/{len(search_space)}, cost : {cost:.2f}")

        print(f"Best idx: {best_idx}")
        print(f"Best cost: {best_cost}")
        print(f"Best in_axes: {best_in_axes}")

        in_axes = best_in_axes
        out_axes = in_axes[:len(out_avals)]
        out_axes_thunk = lambda: out_axes
    else:
        raise ValueError("Invalid strategy: " + strategy)

    # Clean stores for the next call
    for store in fun.stores:
        store and store.reset()

    # Lower to mesh_callable
    compiled_func = mesh_callable(fun, fun_name, None, mesh,
                                  in_axes, out_axes_thunk, donated_invars,
                                  True, *avals, tile_by_mesh_axes=False)
    return compiled_func

def compile_to_hlo_string(fun, fun_name, backend, mesh,
                          in_axes, out_axes_thunk, donated_invars,
                          spmd_lowering, avals, tile_by_mesh_axes):
    # Clean stores for the next call
    for store in fun.stores:
        store and store.reset()

    compiled_func = mesh_callable(fun, fun_name, backend, mesh,
                                  in_axes, out_axes_thunk, donated_invars,
                                  spmd_lowering, *avals, tile_by_mesh_axes=tile_by_mesh_axes)
    compiled = compiled_func.args[0]
    hlo_string = compiled.hlo_modules()[0].to_string()
    return hlo_string


def compute_mesh_callable_cost(fun: lu.WrappedFun,
                               fun_name,
                               backend,
                               mesh,
                               in_axes,
                               out_axes_thunk,
                               donated_invars,
                               spmd_lowering,
                               *avals,
                               tile_by_mesh_axes):
    hlo_string = compile_to_hlo_string(fun, fun_name, backend, mesh,
                                       in_axes, out_axes_thunk, donated_invars,
                                       spmd_lowering, avals, tile_by_mesh_axes)
    compute_cost = compute_compute_cost(hlo_string)
    communication_cost = compute_commuinication_cost(hlo_string)
    #print(hlo_string)
    #print(f"compute: {compute_cost}, communication: {communication_cost}")
    cost = 0.001 * compute_cost + communication_cost
    return cost


def compute_compute_cost(hlo_string):
    cost = 0
    for match in re.finditer("custom-call.*f32\[(\d+),(\d+)\].*f32\[(\d+),(\d+)\].*f32\[(\d+),(\d+)\].*cublas", hlo_string):
        cost += np.sqrt(np.prod([float(match.group(i)) for i in range(1, 7)]))

    return cost


def compute_commuinication_cost(hlo_string):
    # collective-permute is not implemented with channel id 
    if "collective-permute" in hlo_string:
        return float("inf")

    num_comm = 0
    cost = 0
    num_devices = 4  # number of devices

    # all-reduce
    for match in re.finditer("all-reduce\(f32\[(\d+),(\d+)\]", hlo_string):
        cost += 2 * np.prod([float(match.group(i)) for i in range(1, 3)])
        num_comm += 1

    # all-gather
    for match in re.finditer("all-gather\(f32\[(\d+),(\d+),(\d+)\]", hlo_string):
        cost += num_devices * np.prod([float(match.group(i)) for i in range(1, 4)])
        num_comm += 1

    # all-to-all
    for match in re.finditer("all-to-all\(f32\[(\d+),(\d+),(\d+)\]", hlo_string):
        cost += np.prod([float(match.group(i)) for i in range(1, 4)])
        num_comm += 1

    # make sure that we count all communication
    for match in re.finditer("channel_id", hlo_string):
        num_comm -= 1
    assert num_comm == 0

    return cost

