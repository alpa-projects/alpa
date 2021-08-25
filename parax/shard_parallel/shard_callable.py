import hashlib
import inspect
import time
from warnings import warn

from jax import linear_util as lu, disable_jit
from jax._src.util import (partial, extend_name_stack, wrap_name)
from jax.core import (Atom, Var, JaxprEqn, Jaxpr, ClosedJaxpr, DropVar, Literal,
                      jaxpr_as_fun, new_jaxpr_eqn, gensym)
from jax.interpreters import xla, partial_eval as pe, pxla
from jax.lax import add_p, div_p
from jax.lib import xla_bridge as xb, xla_client as xc
import jax.numpy as jnp

from parax.device_mesh import LogicalDeviceMesh, PhysicalDeviceMesh, DeviceCluster
from parax.global_env import global_config
from parax.measure_record import SearchTask, load_best_record
from parax.shard_parallel.auto_sharding import (compile_with_search,
                                                compile_with_given_strategy,
                                                get_input_output_sharding_specs,
                                                HloProtoStatus)
from parax.util import jaxpr_to_hlo_computation


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
    batch_invars,
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
            fun, in_tree, out_tree_thunk, donated_invars, batch_invars,
            physical_mesh, logical_mesh_choices,
            global_config.mesh_shape_search_mode, memory_budget_per_device,
            search_task, record_file, strategy_config, *avals)

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
    name = f"{fun.__name__}_shard_parallel"
    backend = xb.get_backend("gpu")
    built = jaxpr_to_hlo_computation(name, ClosedJaxpr(jaxpr, consts),
                                     donated_invars, backend)
    #print(built.as_hlo_text())

    # Compile an executable
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
        fun: lu.WrappedFun, in_tree, out_tree_thunk, donated_invars,
        batch_invars, physical_mesh, logical_mesh_choices,
        logical_mesh_search_mode, memory_budget_per_device, search_task,
        record_file, strategy_config, *raw_avals):
    # Split the batch dimension
    num_micro_batches = global_config.num_micro_batches
    avals = []
    for aval, is_batch_var in zip(raw_avals, batch_invars):
        if is_batch_var:
            assert aval.shape[0] % num_micro_batches == 0,\
                "The batch dimension must be divisable by num_micro_batches."
            shape = (aval.shape[0] // num_micro_batches,) + aval.shape[1:]
            avals.append(aval.update(shape=shape))
        else:
            avals.append(aval)

    # Get the jaxpr and add gradient accumulation logics to it
    with disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)

    closed_jaxpr, accumulate_grad_invar_ids, apply_grad_invar_ids, num_grads =\
        add_gradient_accumulation(closed_jaxpr, num_micro_batches)
    global_in_avals = [x.aval for x in closed_jaxpr.jaxpr.invars]
    global_out_avals = [x.aval for x in closed_jaxpr.jaxpr.outvars]
    grad_avals = global_in_avals[-num_grads:]
    accumulate_grad_in_avals = [
        global_in_avals[i] for i in accumulate_grad_invar_ids
    ]
    apply_grad_in_avals = [global_in_avals[i] for i in apply_grad_invar_ids
                          ] + grad_avals

    # Run auto-sharding and slice the combined HLO into two HLO: accumulate_grad and apply_grad
    backend = xb.get_backend("gpu")
    donated_invars = donated_invars + (False,) * num_grads
    name = f"{fun.__name__}_shard_parallel"
    built = jaxpr_to_hlo_computation(name, closed_jaxpr, donated_invars,
                                     backend)
    hlo_protos, strategy_config = compile_with_search(backend,
                                                      built,
                                                      physical_mesh,
                                                      logical_mesh_choices,
                                                      logical_mesh_search_mode,
                                                      memory_budget_per_device,
                                                      search_task,
                                                      record_file,
                                                      multiple_stages=True)
    assert len(hlo_protos) == 2

    # Compile these two HLOs separately to get two executable
    accumulate_grad = xc.XlaComputation(hlo_protos[0])
    apply_grad = xc.XlaComputation(hlo_protos[1])
    logical_mesh_shape = strategy_config.logical_mesh_shape

    accumulate_grad = compile_with_given_strategy(
        backend, accumulate_grad, strategy_config, physical_mesh.total_devices,
        False, HloProtoStatus.SHARDING_ANNOTATED)
    apply_grad = compile_with_given_strategy(backend, apply_grad,
                                             strategy_config,
                                             physical_mesh.total_devices, False,
                                             HloProtoStatus.SHARDING_ANNOTATED)

    assert not physical_mesh.is_distributed

    # Read sharding specs
    accumulate_grad_input_sharding_specs, grad_sharding_specs =\
        get_input_output_sharding_specs(
        accumulate_grad.hlo_modules()[0], physical_mesh.total_devices,
        accumulate_grad_in_avals, grad_avals, logical_mesh_shape)
    apply_grad_input_sharding_specs, output_sharding_specs =\
        get_input_output_sharding_specs(
        apply_grad.hlo_modules()[0], physical_mesh.total_devices,
        apply_grad_in_avals, global_out_avals, logical_mesh_shape)

    input_sharding_specs = [None] * len(global_in_avals)
    for i, idx in enumerate(accumulate_grad_invar_ids):
        input_sharding_specs[idx] = accumulate_grad_input_sharding_specs[i]
    for i, idx in enumerate(apply_grad_invar_ids):
        input_sharding_specs[idx] = apply_grad_input_sharding_specs[i]
    assert input_sharding_specs[-num_grads:] == grad_sharding_specs

    # Cache indices for the final callable
    global_in_indices = [
        pxla.spec_to_indices(aval.shape, spec)
        for aval, spec in zip(global_in_avals, input_sharding_specs)
    ]
    global_is_batch_arg = batch_invars + (False,) * num_grads
    devices = physical_mesh.devices
    outs_handler = pxla.avals_to_results_handler(1, len(devices),
                                                 output_sharding_specs,
                                                 global_out_avals)

    def final_callable(*args):
        global_args = list(args)

        # Prepare gradient buffers
        for i in range(num_grads):
            global_args.append(
                jnp.zeros(grad_avals[i].shape, grad_avals[i].dtype))

        # Shard args
        global_buffers = []
        for i, arg in enumerate(global_args):
            if global_is_batch_arg[i]:
                new_shape = (num_micro_batches,
                             arg.shape[0] // num_micro_batches) + arg.shape[1:]
                reshaped = arg.reshape(new_shape)
                micro_batches = jnp.split(reshaped, num_micro_batches)
                micro_batches = [x.squeeze(0) for x in micro_batches]
                micro_batches = pxla.shard_args(
                    devices, (global_in_indices[i],) * len(micro_batches),
                    micro_batches)
                global_buffers.append(micro_batches)
            else:
                global_buffers.append(
                    pxla.shard_args(devices, [global_in_indices[i]], [arg])[0])

        # Call accumulate_grad multiple times
        for j in range(num_micro_batches):
            input_buffers = []
            for i in accumulate_grad_invar_ids:
                if global_is_batch_arg[i]:
                    input_buffers.append(global_buffers[i][j])
                else:
                    input_buffers.append(global_buffers[i])

            output_bufs = accumulate_grad.execute_sharded_on_local_devices(
                input_buffers)
            global_buffers[-num_grads:] = output_bufs

        # Call apply_grad
        input_buffers = []
        for i in apply_grad_invar_ids:
            if global_is_batch_arg[i]:
                assert False, "cannot use batch data in apply_grad"
            else:
                input_buffers.append(global_buffers[i])
        input_buffers.extend(global_buffers[-num_grads:])
        output_bufs = apply_grad.execute_sharded_on_local_devices(input_buffers)

        # Wrap output buffers as ShardedArray
        output = outs_handler(output_bufs)
        return output

    return final_callable


def filter_used_vars(all_vars, eqns):
    """Return the vars in all_vars that are used by eqns.
    The returned vars preserve their original order in all_vars.
    """
    used_vars = set()
    for eqn in eqns:
        used_vars.update(x for x in eqn.invars if not isinstance(x, Literal))
    return [var for var in all_vars if var in used_vars]


def clone_vars(var_list, gensym_func):
    """Clone variables"""
    return [gensym_func(x.aval) for x in var_list]


def add_gradient_accumulation(raw_jaxpr, num_micro_batches):
    """Add gradient accumulation logics into the raw jaxpr.

    Signatures of functions:
        raw_jaxpr(opt_state, param, batch) -> [new_opt_state, new_param]

        The original_jaxpr can be split into:
        'compute_grad(param, batch) -> out_grad'
        'apply_grad(opt_state, param, in_grad) -> [new_opt_state, new_param]'

        We then derive accumulate_grad from compute_grad:
        'accumulate_grad(old_grad, param, batch) -> new_grad'

        The returned jaxpr is composed by [
            pipeline_marker_start
            accumulate_grad
            pipeline_marker_end

            pipeline_marker_start
            apply_grad
            pipeline_marker_end
        ].
    """
    from parax.pipeline_parallel.primitive_def import pipeline_p

    global_invars = set(raw_jaxpr.jaxpr.invars)
    global_outvars = set(
        var for var in raw_jaxpr.jaxpr.outvars if isinstance(var, Var))
    global_consts_dir = dict(zip(raw_jaxpr.jaxpr.constvars, raw_jaxpr.consts))
    gensym_func = gensym([raw_jaxpr.jaxpr])

    # Find the gradient separator marker.
    # This separator partitions orginal_jaxpr into two part:
    # compute_grad and apply_grad
    marker_eqn = None
    for marker_pos, eqn in enumerate(raw_jaxpr.jaxpr.eqns):
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'grad':
            marker_eqn = eqn
            break
    assert marker_eqn is not None, "Must have exactly one gradient marker"
    compute_grad_eqns = raw_jaxpr.jaxpr.eqns[:marker_pos]
    apply_grad_eqns = raw_jaxpr.jaxpr.eqns[marker_pos + 1:]

    # Build the new jaxpr with gradient accumulation and pipeline marker
    global_invar_substitute = {}
    combined_eqns = []

    # Create vars for gradient accumulation
    out_grad_vars = marker_eqn.invars
    old_grad_vars = clone_vars(out_grad_vars, gensym_func)
    new_grad_vars = clone_vars(out_grad_vars, gensym_func)
    num_grads = len(out_grad_vars)

    # Wrap all invars of accumulate_grad
    old_invars = filter_used_vars(raw_jaxpr.jaxpr.invars,
                                  compute_grad_eqns) + old_grad_vars
    new_invars = clone_vars(old_invars, gensym_func)
    combined_eqns.append(
        new_jaxpr_eqn(new_invars, old_invars, pipeline_p,
                      {"mark_type": "start"}, None))
    global_invar_substitute.update(
        {x: y for x, y in zip(old_invars, new_invars)})
    accumulate_grad_invars = new_invars

    # Append eqns of compute_grad
    combined_eqns.extend(raw_jaxpr.jaxpr.eqns[:marker_pos])

    # Append eqns of gradient accumulation
    for i in range(len(out_grad_vars)):
        combined_eqns.append(
            new_jaxpr_eqn([old_grad_vars[i], out_grad_vars[i]],
                          [new_grad_vars[i]], add_p, {}, None))

    # Wrap all outvars of accumulate_grad
    inter_grad_vars = [gensym_func(x.aval) for x in out_grad_vars]
    combined_eqns.append(
        new_jaxpr_eqn(new_grad_vars, inter_grad_vars, pipeline_p,
                      {"mark_type": "end"}, None))
    accumulate_grad_outvars = new_grad_vars

    # Wrap all invars of apply_grad
    in_grad_vars = marker_eqn.outvars
    old_invars = filter_used_vars(raw_jaxpr.jaxpr.invars,
                                  apply_grad_eqns) + in_grad_vars
    new_invars = []
    for var in old_invars:
        if var in global_invars:
            if var in global_invar_substitute:
                new_invars.append(global_invar_substitute[var])
            else:
                new_var = gensym_func(var.aval)
                global_invar_substitute[var] = new_var
                new_invars.append(new_var)
        else:
            new_invars.append(inter_grad_vars[in_grad_vars.index(var)])
    apply_grad_invars = new_invars
    combined_eqns.append(
        new_jaxpr_eqn(new_invars, old_invars, pipeline_p,
                      {"mark_type": "start"}, None))

    # Append eqns for gradient reduction
    for i in range(num_grads):
        combined_eqns.append(
            new_jaxpr_eqn(
                [old_invars[-(i + 1)],
                 Literal(float(num_micro_batches))], [old_invars[-(i + 1)]],
                div_p, {}, None))
    # TODO(lmzheng): This breaks the SSA form of the combined_eqns
    # But I find jax can convert this non-SSA jaxpr to HLO correctly,
    # so I leave this issue as todo. To fix this, we should substitute
    # all grad vars in these equations with new vars.

    # Append eqns of apply_grad
    combined_eqns.extend(apply_grad_eqns)
    # TODO(lmzheng): The param vars are used in both compute_grad and apply_grad,
    # so there will be some duplicated intermediate vars in compute_grad_eqns
    # and apply_grad_eqns. This breaks the SSA form of the combined_eqns.
    # But I find jax can convert this non-SSA jaxpr to HLO correctly,
    # so I leave this issue as todo. To fix this, we should substitute
    # all param vars in these equations with new vars.

    # Wrap all outvars of apply_grad
    old_outvars = raw_jaxpr.jaxpr.outvars
    new_outvars = [gensym_func(x.aval) for x in old_outvars]
    combined_eqns.append(
        new_jaxpr_eqn(old_outvars, new_outvars, pipeline_p,
                      {"mark_type": "end"}, None))
    apply_grad_outvars = new_outvars

    # Make the new jaxpr
    combined_jaxpr = ClosedJaxpr(
        Jaxpr(raw_jaxpr.jaxpr.constvars, [
            global_invar_substitute.get(x, x)
            for x in (raw_jaxpr.jaxpr.invars + old_grad_vars)
        ], new_outvars, combined_eqns), raw_jaxpr.consts)

    # The indices of the arguments in global arguments.
    accumulate_grad_invar_ids = [
        combined_jaxpr.jaxpr.invars.index(var) for var in accumulate_grad_invars
    ]
    apply_grad_invar_ids = [
        combined_jaxpr.jaxpr.invars.index(var)
        for var in apply_grad_invars[:-num_grads]
    ]
    return (combined_jaxpr, accumulate_grad_invar_ids, apply_grad_invar_ids,
            num_grads)
