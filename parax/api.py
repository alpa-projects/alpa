"""Top-level user API."""
from functools import wraps

from jax import linear_util as lu
from jax.api import _check_callable
from jax.api_util import flatten_fun_nokwargs, argnums_partial, \
    donation_vector, rebase_donate_argnums
from jax.interpreters import xla
from jax.experimental.maps import FrozenDict
from jax.tree_util import tree_flatten, tree_unflatten
from jax._src.util import safe_map, HashableFunction

from parax import util
from parax.pmap_data_parallel import pmap_data_parallel_callable
from parax.shard_parallel import shard_parallel_callable
from parax.pipeline_parallel import pipeline_parallel_callable, \
    distributed_pipeline_parallel_callable
from parax.three_d_parallel import three_d_parallel_callable


# pylint: disable=redefined-builtin
unsafe_map, map = map, safe_map  # type: ignore


# pylint: disable=too-many-arguments
def parallelize(fun=None,
                donate_argnums="auto",
                static_argnums="auto",
                devices=None,
                memory_budget_per_device=None,
                strategy="shard_parallel"):
    """
    Automatically parallelize a jax function.

    Args:
        donate_argnums: The same as the donated_argnums in jax.jit. If is "auto",
          parax uses heuristic rules to infer this.
        static_argnums: The same as the donated_argnums in jax.jit. If is "auto",
          parax uses heuristic rules to infer this.
    """

    def decorate_fun(fun):
        @wraps(fun)
        def ret_func(*args, **kwargs):
            # pylint: disable=too-many-locals
            shard_args_only_mode = kwargs.pop("__shard_args_only_mode", False)
            assert not kwargs, "kwargs is not supported"

            f = lu.wrap_init(fun)

            # Deal with static arguments and extract dynamic arguments
            nonlocal static_argnums
            if static_argnums == "auto":
                static_argnums = util.auto_static_argnums(args)

            if static_argnums:
                dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
                # freeze static dict to make it hashable
                frozen_args = []
                for i, arg in enumerate(args):
                    if i in static_argnums and isinstance(arg, dict):
                        frozen_args.append(FrozenDict(arg))
                    else:
                        frozen_args.append(arg)
                f, dyn_args = argnums_partial(f, dyn_argnums, frozen_args)
            else:
                dyn_args = args

            # Flatten pytree arguments
            args_flat, in_tree = tree_flatten(dyn_args)
            f, out_tree = flatten_fun_nokwargs(f, in_tree)
            # pylint: disable=unnecessary-lambda
            out_tree_hashable = HashableFunction(lambda: out_tree(), closure=None)

            # Deal with donate argnums
            nonlocal donate_argnums
            if donate_argnums == "auto":
                donate_argnums = util.auto_donate_argnums(args)

            donate_tuple = rebase_donate_argnums(donate_argnums, static_argnums)
            if donate_tuple:
                donated_invars = donation_vector(donate_tuple, dyn_args, kwargs)
            else:
                donated_invars = (False,) * len(args_flat)

            # JIT compile and call the compiled func
            abstract_args = unsafe_map(xla.abstractify, args_flat)
            compiled_func = auto_parallel_callable(
                f, in_tree, out_tree_hashable, devices, donated_invars,
                memory_budget_per_device, strategy, *abstract_args
            )

            if shard_args_only_mode:
                # In this mode, this function returns sharded arguments without executing
                # the computation. This is used to prepare sharded arguments
                # for benchmark purposes, so we can exclude the time for sharding arguments.
                sharded_args = compiled_func.shard_args_only(*args_flat)
                return tree_unflatten(in_tree, sharded_args)
            else:
                # Execute the compiled func and return results
                out = compiled_func(*args_flat)
                return tree_unflatten(out_tree(), out)

        @wraps(fun)
        def preshard_dynamic_args(*args, **kwargs):
            kwargs['__shard_args_only_mode'] = True
            return ret_func(*args, **kwargs)

        ret_func.preshard_dynamic_args = preshard_dynamic_args
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
    memory_budget_per_device,
    strategy,
    *avals,
):
    """Auto parallel callable."""
    # Clean stores for the next call
    for store in fun.stores:
        if store:
            store.reset()

    # Apply parallel strategy
    if strategy == "shard_parallel":
        return shard_parallel_callable(
            fun, in_tree, out_tree_thunk, devices, donated_invars,
            memory_budget_per_device, *avals
        )
    elif strategy == "pmap_data_parallel":
        return pmap_data_parallel_callable(
            fun, in_tree, out_tree_thunk, devices, donated_invars, *avals
        )
    elif strategy == "pipeline_parallel":
        return pipeline_parallel_callable(fun, *avals)
    elif strategy == "distributed_pipeline_parallel":
        return distributed_pipeline_parallel_callable(fun, *avals)
    elif strategy == "3d_parallel":
        return three_d_parallel_callable(
            fun, in_tree, out_tree_thunk, devices, donated_invars,
            memory_budget_per_device, *avals
        )
    else:
        raise ValueError("Invalid parallel strategy: " + strategy)
