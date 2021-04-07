"""Top-level user API"""
from functools import wraps

import numpy as np

import jax
from jax import linear_util as lu
from jax.api import _check_callable
from jax.api_util import (flatten_fun_nokwargs, argnums_partial,
    donation_vector, rebase_donate_argnums)
from jax.interpreters import xla, partial_eval as pe
from jax.experimental.maps import FrozenDict
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax._src.util import safe_map, wraps, HashableFunction

from paranum import util
from paranum.pmap_data_parallel import pmap_data_parallel_callable
from paranum.shard_parallel import shard_parallel_callable

unsafe_map, map = map, safe_map  # type: ignore


def parallelize(fun=None, donate_argnums="auto", static_argnums="auto", devices=None):
    def decorate_fun(fun):
        @wraps(fun)
        def ret_func(*args, **kwargs):
            assert not kwargs, "kwargs not supported"

            f = lu.wrap_init(fun)

            # Deal with static arguments
            nonlocal static_argnums
            if static_argnums == "auto":
                static_argnums = util.auto_static_argnums(args)

            if static_argnums:
                dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
                # freeze static dict to make it hashable
                frozen_args = []
                for i in range(len(args)):
                    if i in static_argnums and isinstance(args[i], dict):
                        frozen_args.append(FrozenDict(args[i]))
                    else:
                        frozen_args.append(args[i])
                f, dyn_args = argnums_partial(f, dyn_argnums, frozen_args)
            else:
                dyn_args = args

            # Deal with donate argnums
            nonlocal donate_argnums
            if donate_argnums == "auto":
                donate_argnums = util.auto_donate_argnums(args)

            donate_tuple = rebase_donate_argnums(donate_argnums, static_argnums)
            if donate_tuple:
                donated_invars = donation_vector(donate_tuple, dyn_args, kwargs)
            else:
                donated_invars = (False,) * len(dyn_args)

            # Flatten pytree arguments
            args_flat, in_tree = tree_flatten(dyn_args)
            f, out_tree = flatten_fun_nokwargs(f, in_tree)
            out_tree_hashable = HashableFunction(lambda: out_tree(), closure=None)

            # JIT compile and call the compiled func
            abstract_args = unsafe_map(xla.abstractify, args_flat)
            compiled_func = auto_parallel_callable(
                f, in_tree, out_tree_hashable, devices, donated_invars, *abstract_args
            )
            out = compiled_func(*args_flat)

            return tree_unflatten(out_tree(), out)

        return ret_func

    if fun is None:
        return decorate_fun
    else:
        _check_callable(fun)
        return decorate_fun(fun)


@lu.cache
def auto_parallel_callable(
    fun: lu.WrappedFun,
    in_tree,
    out_tree_thunk,
    devices,
    donated_invars,
    *avals
):
    fun_name = fun.__name__

    # Get jaxpr and XLA hlo
    #jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(fun, avals)
    #c = jaxpr_to_xla_computation(jaxpr, avals, consts, fun_name)

    # Choose parallel strategy
    strategy = 'shard_parallel'

    # Clean stores for the next call
    for store in fun.stores:
        store and store.reset()

    # Apply parallel strategy
    if strategy == "shard_parallel":
        return shard_parallel_callable(
            fun, in_tree, out_tree_thunk, devices, donated_invars, *avals
        )
    elif strategy == "pmap_data_parallel":
        return pmap_data_parallel_callable(
            fun, in_tree, out_tree_thunk, devices, donated_invars, *avals
        )
    else:
        raise ValueError("Invalid parallel strategy")

