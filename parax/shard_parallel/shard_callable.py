import hashlib
import inspect
import time
from warnings import warn

from jax import linear_util as lu
from jax._src.util import (partial, extend_name_stack, wrap_name)
from jax.core import ClosedJaxpr, gensym
from jax.interpreters import xla, partial_eval as pe
from jax.lib import xla_bridge as xb, xla_client as xc

from parax.device_mesh import LogicalDeviceMesh, PhysicalDeviceMesh, DeviceCluster
from parax.global_env import global_config
from parax.measure_record import SearchTask, load_best_record
from parax.shard_parallel.auto_sharding import (compile_with_search, compile_with_given_strategy,
                                                get_input_output_sharding_specs, HloProtoStatus)


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


def shard_parallel_callable(
    fun: lu.WrappedFun,
    in_tree,
    out_tree_thunk,
    donated_invars,
    devices,
    memory_budget_per_device,
    *avals,
):
    """
    Compile a callable with auto-sharding pass.
    """
    # This function resolves the polymorphism in arguments and global configurations
    # and calls actual compilation in shard_parallel_internal.

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
                        logical_mesh_choices.append(
                            physical_mesh.get_logical_mesh(
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

    if global_config.num_micro_batches is not None:
        return shard_parallel_internal_gradient_accumulation(
            fun, in_tree, out_tree_thunk, donated_invars,
            physical_mesh, logical_mesh_choices,
            global_config.mesh_shape_search_mode,
            memory_budget_per_device, search_task,
            record_file, strategy_config, *avals)

    return shard_parallel_internal(fun, in_tree, out_tree_thunk, donated_invars,
                                   physical_mesh, logical_mesh_choices,
                                   global_config.mesh_shape_search_mode,
                                   memory_budget_per_device, search_task,
                                   record_file, strategy_config, *avals)


def shard_parallel_internal(fun: lu.WrappedFun, in_tree, out_tree_thunk,
                            donated_invars, physical_mesh, logical_mesh_choices,
                            logical_mesh_search_mode, memory_budget_per_device,
                            search_task, record_file, strategy_config, *avals):
    """
    Compile a callable with auto-sharding pass.

    Args:
      func (lu.WrappedFun): The wrapped jax function to be compiled.
      in_tree (PyTree): The pytree of input arguments.
      out_tree_thunk (Callable[()->PyTree]): The thunk to produce output pytree.
      donated_invars (List[bool]): Whether to donate input parameters.
      physical_mesh (PhysicalDeviceMesh): The physical device mesh.
      logical_mesh_choices (List[Tuple[int]]): The candidates of logical mesh shape.
        If there is only one choice, use the given one. If there are multple choices,
        we will try all of them and pick the best.
      logical_mesh_search_mode (str): The choices are {"measurement", "cost_model"}.
        If is "measurement", use real profiling to pick the best logical mesh shape.
        If is "cost_model", use cost estimation in HLO IR to pick the best one.
        This is ignored if len(logical_mesh_choices) == 1.
      memory_budget_per_device (Optional[float]): The memory budget per device in bytes.
      search_task (Optional[SearchTask]): Only used when doing logical mesh shape search.
        Used when dumping measurement records to the file.
      record_file (Optional[str]): If is not None, dump measurement records into
        this file.
      strategy_config (Optional[StrategyConfig]): If is not None, do compilation
        according to this configuration.
    """
    tic = time.time()
    # Trace to get jaxpr
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, avals)

    # Convert jaxpr to XLA HLO
    c = xb.make_computation_builder(f"auto_shard_{fun.__name__}")
    xla_consts = map(partial(xb.constant, c), consts)
    tuple_args = False
    xla_args, donated_invars = xla._xla_callable_args(
        c, avals, tuple_args, donated_invars=donated_invars)
    backend_name = 'gpu'
    axis_env = xla.AxisEnv(nreps=1, names=(),
                           sizes=())  # All named axes have been vmapped
    transformed_name = fun.__name__
    out_nodes = xla.jaxpr_subcomp(
        c, jaxpr, backend_name, axis_env, xla_consts,
        extend_name_stack(wrap_name(transformed_name, 'auto_sharding')),
        *xla_args)
    out_tuple = xc.ops.Tuple(c, out_nodes)

    # Set up aliases (donating invars)
    backend = xb.get_backend(backend_name)
    if backend.platform in ("gpu", "tpu"):
        donation_results = xla.set_up_aliases(c, xla_args, out_tuple,
                                              donated_invars, tuple_args)
    if any(donation_results):
        # TODO(tomhennigan): At call time we should mark these buffers as deleted.
        unused_donations = [
            str(c.GetShape(a)) for a, d in zip(xla_args, donation_results) if d
        ]
        warn("Some donated buffers were not usable: {}".format(
            ", ".join(unused_donations)))

    # Compile and optimize HLO to an executable
    built = c.Build(out_tuple)
    #print(built.as_hlo_text())
    if strategy_config is None:
        compiled, strategy_config = compile_with_search(
            backend,
            built,
            physical_mesh,
            logical_mesh_choices,
            logical_mesh_search_mode,
            memory_budget_per_device,
            search_task,
            record_file,
            multiple_stages=False)
    else:
        compiled = compile_with_given_strategy(backend, built, strategy_config,
                                               physical_mesh.total_devices,
                                               physical_mesh.is_distributed,
                                               HloProtoStatus.UNOPTIMIZED)
    hlo_module = compiled.hlo_modules()[0]
    logical_mesh_shape = strategy_config.logical_mesh_shape

    if global_config.print_xla_compilation_time:
        print(f" - XLA Compilation time: {time.time() - tic:.2f} s")

    # Send the code and strategy to remote workers
    if physical_mesh.is_distributed:
        compiled = physical_mesh.compile_remote_executable(
            hlo_module.as_serialized_hlo_module_proto(), strategy_config,
            HloProtoStatus.FULLY_OPTIMIZED)

    # Read HloSharding from HloModule and convert them to ShardingSpec
    # Return the final callable
    input_sharding_specs, output_sharding_specs = get_input_output_sharding_specs(
        hlo_module, physical_mesh.total_devices, avals, out_avals,
        logical_mesh_shape)
    return physical_mesh.get_callable_with_arg_handler(compiled, avals,
                                                       out_avals,
                                                       input_sharding_specs,
                                                       output_sharding_specs,
                                                       donated_invars)

def shard_parallel_internal_gradient_accumulation(
    fun: lu.WrappedFun, in_tree, out_tree_thunk,
    donated_invars, physical_mesh, logical_mesh_choices,
    logical_mesh_search_mode, memory_budget_per_device,
    search_task, record_file, strategy_config, *avals
):
    from parax.pipeline_parallel.stage import slice_closed_jaxpr_by_pipeline_marks

    # Slice the jaxpr into pipeline stages
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    gensym_func = gensym([closed_jaxpr.jaxpr])
    jax_pipeline_stages = slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr)
    print(jax_pipeline_stages)
    exit()


