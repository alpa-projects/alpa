"""Top-level user API."""
from functools import wraps

from jax import linear_util as lu
from jax._src.util import safe_map, HashableFunction
from jax.api import _check_callable
from jax.api_util import (argnums_partial, donation_vector,
                          flatten_fun_nokwargs, rebase_donate_argnums)
from jax.experimental.maps import FrozenDict
from jax.interpreters import xla
from jax.lib import xla_bridge as xb
from jax.tree_util import tree_flatten, tree_unflatten

from parax.device_mesh import DeviceCluster, LogicalDeviceMesh, PhysicalDeviceMesh
from parax.global_env import global_config
from parax.pipeline_parallel.local_pipeline_parallel import local_pipeline_parallel_callable
from parax.pipeline_parallel.three_d_parallel import three_d_parallel_callable
from parax.shard_parallel.data_parallel import pmap_data_parallel_callable, shard_data_parallel_callable
from parax.shard_parallel.auto_sharding import shard_parallel_callable
from parax.util import auto_donate_argnums, auto_static_argnums

# pylint: disable=redefined-builtin
unsafe_map, map = map, safe_map  # type: ignore


def parallelize(fun=None, donate_argnums="auto", static_argnums="auto"):
    """
    Automatically parallelize a jax function.

    Args:
        fun: The function to be parallelized.
        donate_argnums: The same as the donated_argnums argument of jax.jit.
          If is "auto", parax uses heuristic rules to infer this.
        static_argnums: The same as the static_argnums argument of jax.jit.
          If is "auto", parax uses heuristic rules to infer this.
    """

    def decorate_fun(fun):

        @wraps(fun)
        def ret_func(*args, **kwargs):
            return_value_mode = kwargs.pop("__return_value_mode", "normal")
            assert not kwargs, "kwargs is not supported"

            f = lu.wrap_init(fun)

            # Deal with static arguments and extract dynamic arguments
            nonlocal static_argnums
            if static_argnums == "auto":
                static_argnums = auto_static_argnums(args)

            if static_argnums:
                dyn_argnums = [
                    i for i in range(len(args)) if i not in static_argnums
                ]
                # Freeze static dict to make it hashable
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
            out_tree_hashable = HashableFunction(lambda: out_tree(),
                                                 closure=None)

            # Deal with donate argnums
            nonlocal donate_argnums
            if donate_argnums == "auto":
                donate_argnums = auto_donate_argnums(args)

            donate_tuple = rebase_donate_argnums(donate_argnums, static_argnums)
            if donate_tuple:
                donated_invars = donation_vector(donate_tuple, dyn_args, kwargs)
            else:
                donated_invars = (False,) * len(args_flat)

            # JIT compile and call the compiled func
            abstract_args = unsafe_map(xla.abstractify, args_flat)
            devices = global_config.devices
            if isinstance(devices, list):
                devices = tuple(devices)
            compiled_func = parallelize_callable(
                f, in_tree, out_tree_hashable, donated_invars, devices,
                global_config.strategy, global_config.memory_budget_per_device,
                *abstract_args)

            if return_value_mode == "normal":
                # Execute the compiled func and return results
                out = compiled_func(*args_flat)
                return tree_unflatten(out_tree(), out)
            elif return_value_mode == "preshard_dynamic_args":
                # In this mode, this function returns sharded arguments without executing
                # the computation. This is used to prepare sharded arguments
                # for benchmark purposes, so we can exclude the time for sharding arguments.
                sharded_args = compiled_func.shard_args_only(*args_flat)
                return tree_unflatten(in_tree, sharded_args)
            elif return_value_mode == "get_executable":
                # Return the compiled executable
                return compiled_func.args[0]
            else:
                raise ValueError(
                    f"Invalid return_value_mode: {return_value_mode}")

        @wraps(fun)
        def preshard_dynamic_args(*args, **kwargs):
            """Prepare sharded arguments for benchmark purposes,
            so we can exclude the time for sharding arguments."""
            kwargs['__return_value_mode'] = "preshard_dynamic_args"
            return ret_func(*args, **kwargs)

        @wraps(fun)
        def get_executable(*args, **kwargs):
            """Return the compiled executable."""
            kwargs['__return_value_mode'] = "get_executable"
            return ret_func(*args, **kwargs)

        ret_func.preshard_dynamic_args = preshard_dynamic_args
        ret_func.get_executable = get_executable
        return ret_func

    if fun is None:
        return decorate_fun
    else:
        _check_callable(fun)
        return decorate_fun(fun)


@lu.cache
def parallelize_callable(
    fun: lu.WrappedFun,
    in_tree,
    out_tree_thunk,
    donated_invars,
    devices,
    strategy,
    memory_budget_per_device,
    *avals,
):
    """Auto parallel callable."""
    # Clean stores for the next call
    for store in fun.stores:
        if store:
            store.reset()

    # Choose parallel strategy
    if strategy == "shard_parallel":
        return shard_parallel_callable(fun, in_tree, out_tree_thunk,
                                       donated_invars, devices,
                                       memory_budget_per_device, *avals)
    elif strategy == "shard_data_parallel":
        return shard_data_parallel_callable(fun, in_tree, out_tree_thunk,
                                            donated_invars, devices, *avals)
    elif strategy == "pmap_data_parallel":
        return pmap_data_parallel_callable(fun, in_tree, out_tree_thunk,
                                           donated_invars, devices, *avals)
    elif strategy == "local_pipeline_parallel":
        return local_pipeline_parallel_callable(fun, devices, *avals)
    elif strategy == "3d_parallel":
        # TODO (zhuohan): Support search_logical_mesh_shape for 3d parallel
        assert not global_config.search_logical_mesh_shape
        return three_d_parallel_callable(fun, in_tree, out_tree_thunk,
                                         donated_invars, devices,
                                         memory_budget_per_device, *avals)
    else:
        raise ValueError("Invalid parallel strategy: " + strategy)


def clear_callable_cache():
    """Clear all cached auto_parallel_callable."""
    parallelize_callable.cache_clear()
