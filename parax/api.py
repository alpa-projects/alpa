"""Top-level user API."""
from functools import wraps
import hashlib
import inspect

from jax import linear_util as lu
from jax.api import _check_callable
from jax.api_util import (argnums_partial, donation_vector,
                          flatten_fun_nokwargs, rebase_donate_argnums)
from jax.experimental.maps import FrozenDict
from jax.interpreters import xla
from jax.lib import xla_bridge as xb
from jax.tree_util import tree_flatten, tree_unflatten
from jax._src.util import safe_map, HashableFunction

from parax import util
from parax.auto_sharding import auto_sharding_callable
from parax.device_mesh import DeviceCluster, LogicalDeviceMesh, PhysicalDeviceMesh
from parax.data_parallel import pmap_data_parallel_callable, shard_data_parallel_callable
from parax.global_env import global_config
from parax.measure_record import SearchTask, load_best_record
from parax.pipeline_parallel.callable import pipeline_parallel_callable
from parax.three_d_parallel import three_d_parallel_callable


# pylint: disable=redefined-builtin
unsafe_map, map = map, safe_map  # type: ignore


def parallelize(fun=None,
                donate_argnums="auto",
                static_argnums="auto"):
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
                static_argnums = util.auto_static_argnums(args)

            if static_argnums:
                dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
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
            devices = global_config.devices
            if isinstance(devices, list):
                devices = tuple(devices)
            compiled_func = auto_parallel_callable(
                f, in_tree, out_tree_hashable, donated_invars,
                devices, global_config.strategy,
                global_config.memory_budget_per_device, *abstract_args
            )

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
                raise ValueError(f"Invalid return_value_mode: {return_value_mode}")

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
def auto_parallel_callable(
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
    if strategy == "auto_sharding_parallel":
        # Get physical mesh and logical mesh.
        if devices is None:
            devices = PhysicalDeviceMesh(devices=xb.devices())
        elif isinstance(devices, (list, tuple)):
            devices = PhysicalDeviceMesh(devices=devices)
        elif isinstance(devices, DeviceCluster):
            devices = devices.get_physical_mesh()

        search_task = None
        record_file = None
        strategy_config = None
        if isinstance(devices, PhysicalDeviceMesh):
            physical_mesh = devices

            if global_config.search_logical_mesh_shape:
                # Check cached strategy folder
                compute_key = get_compute_key(fun, in_tree, donated_invars, *avals)
                device_key = physical_mesh.get_signature()
                search_task = SearchTask(compute_key, device_key)
                record_file = global_config.mesh_shape_search_log_file

                if record_file:
                    inp, _ = load_best_record(search_task, filename=record_file)
                else:
                    inp = None

                if inp is None:
                    # Generate a search space that contains all possible mesh shapes.
                    logical_mesh_choices = []
                    total_devices = physical_mesh.total_devices
                    for i in range(1, total_devices):
                        if total_devices % i == 0:
                            logical_mesh_shape = (total_devices // i, i)
                            logical_mesh_choices.append(physical_mesh.get_logical_mesh(
                                mesh_shape=logical_mesh_shape,
                                # TODO(lmzheng): export this as an arugment in
                                # set_parallelize_options or physical_mesh.
                                #mesh_alpha=[1,1],
                                #mesh_beta=[1,1]))
                                mesh_topology="tree",
                                inter_host_bandwidth=1,
                                intra_host_bandwidth=30))
                else:
                    logical_mesh_choices = []
                    strategy_config = inp.config
            else:
                logical_mesh_choices = [physical_mesh.get_default_logical_mesh()]
        elif isinstance(devices, LogicalDeviceMesh):
            physical_mesh = devices.physical_mesh
            logical_mesh_choices = [devices]
        else:
            raise ValueError("Invalid value of devices")

        return auto_sharding_callable(
            fun, in_tree, out_tree_thunk, donated_invars,
            physical_mesh, logical_mesh_choices,
            global_config.mesh_shape_search_mode,
            memory_budget_per_device,
            search_task, record_file, strategy_config, *avals
        )
    elif strategy == "shard_data_parallel":
        return shard_data_parallel_callable(
            fun, in_tree, out_tree_thunk, devices, donated_invars, *avals
        )
    elif strategy == "pmap_data_parallel":
        return pmap_data_parallel_callable(
            fun, in_tree, out_tree_thunk, devices, donated_invars, *avals
        )
    elif strategy == "pipeline_parallel":
        return pipeline_parallel_callable(fun, devices, *avals)
    elif strategy == "3d_parallel":
        # TODO (zhuohan): Support search_logical_mesh_shape for 3d parallel
        assert not global_config.search_logical_mesh_shape
        return three_d_parallel_callable(
            fun, in_tree, out_tree_thunk, devices, donated_invars,
            memory_budget_per_device, *avals
        )
    else:
        raise ValueError("Invalid parallel strategy: " + strategy)


def clear_callable_cache():
    """Clear all cached auto_parallel_callable."""
    auto_parallel_callable.cache_clear()


def get_compute_key(fun, in_tree, donated_invars, *aval):
    """Return a unique string as the query key of a computation definition."""
    # Algorithm:
    # Concatenate the definition location, source code,
    # input arguments specification to a string.
    # Then compute a hash value of this string.
    #
    # TODO(lmzheng): use jaxpr or hlo instead of source code?

    location = fun.f.__str__().split("at")[0]
    source_code = inspect.getsource(fun.f)
    donated_invars = str(donated_invars)
    aval = "".join(x.str_short() for x in aval)

    string = location + source_code + donated_invars + aval
    hash_key = hashlib.md5(string.encode(encoding="utf-8")).hexdigest()
    return hash_key
