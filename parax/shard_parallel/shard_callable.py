"""Generate callables for shard_parallel."""
import hashlib
import inspect
import time

import numpy as np

from jax import linear_util as lu, disable_jit
from jax.core import (Jaxpr, ClosedJaxpr, Literal, new_jaxpr_eqn, gensym)
from jax.interpreters import partial_eval as pe
from jax.lax import add_p, div_p
from jax.lib import xla_bridge as xb, xla_client as xc, xla_extension

from parax.util import OrderedSet
from parax.device_mesh import LogicalDeviceMesh, PhysicalDeviceMesh, DeviceCluster
from parax.global_env import global_config
from parax.measure_record import SearchTask, load_best_record
from parax.mesh_executable import NormalMeshDriverExecutable, GradAccMeshDriverExecutable
from parax.shard_parallel.auto_sharding import (compile_with_search,
                                                compile_with_given_strategy,
                                                HloProtoStatus)
from parax.util import jaxpr_to_hlo_computation, trace_jaxpr_with_micro_batch, setup_computation_alias


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
    """Compile a callable with auto-sharding pass."""
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
      fun (lu.WrappedFun): The wrapped jax function to be compiled.
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
    flop_count = xla_extension.hlo_module_count_flop_dot_conv_only(
        built.as_hlo_module())

    # Compile a XLA executable
    if strategy_config is None:
        compiled, strategy_config = compile_with_search(
            backend,
            built,
            avals,
            out_avals,
            donated_invars,
            physical_mesh,
            logical_mesh_choices,
            logical_mesh_search_mode,
            memory_budget_per_device,
            search_task,
            record_file,
            multiple_stages=False,
            grad_acc_num_micro_batches=None,
            bypass_device_assignment_check=physical_mesh.is_distributed)
    else:
        compiled = compile_with_given_strategy(backend, built, strategy_config,
                                               physical_mesh.total_devices,
                                               physical_mesh.is_distributed,
                                               HloProtoStatus.UNOPTIMIZED)

    if global_config.print_xla_compilation_time:
        print(f" - XLA Compilation time: {time.time() - tic:.2f} s")

    # Compile a mesh executable
    compiled = NormalMeshDriverExecutable(physical_mesh,
                                          compiled,
                                          strategy_config,
                                          avals,
                                          out_avals,
                                          donated_invars,
                                          flop_count=flop_count)
    return compiled.get_driver_callable()


def shard_parallel_internal_gradient_accumulation(
        fun: lu.WrappedFun, in_tree, out_tree_thunk, donated_invars,
        batch_invars, physical_mesh, logical_mesh_choices,
        logical_mesh_search_mode, memory_budget_per_device, search_task,
        record_file, strategy_config, *raw_avals):
    """Compile a gradient accumulation callable with auto-sharding pass."""
    # Split the batch dimension
    num_micro_batches = global_config.num_micro_batches
    closed_jaxpr, avals, batch_size = trace_jaxpr_with_micro_batch(
        fun, batch_invars, num_micro_batches, raw_avals)

    closed_jaxpr, accumulate_grad_invar_indices, apply_grad_invar_indices, num_grads =\
        add_gradient_accumulation(closed_jaxpr, num_micro_batches)
    in_avals = [x.aval for x in closed_jaxpr.jaxpr.invars[:-num_grads]]
    out_avals = [x.aval for x in closed_jaxpr.jaxpr.outvars]
    grad_avals = [x.aval for x in closed_jaxpr.jaxpr.invars[-num_grads:]]

    # Run auto-sharding and slice the combined HLO into two HLO: accumulate_grad and apply_grad
    backend = xb.get_backend("gpu")
    donated_invars = donated_invars + (False,) * num_grads
    name = f"{fun.__name__}_shard_parallel"
    built = jaxpr_to_hlo_computation(name, closed_jaxpr, donated_invars,
                                     backend)
    flop_count = xla_extension.hlo_module_count_flop_dot_conv_only(
        built.as_hlo_module())
    flop_count *= num_micro_batches

    hlo_protos, strategy_config = compile_with_search(
        backend,
        built,
        avals,
        out_avals,
        donated_invars,
        physical_mesh,
        logical_mesh_choices,
        logical_mesh_search_mode,
        memory_budget_per_device,
        search_task,
        record_file,
        multiple_stages=True,
        grad_acc_num_micro_batches=num_micro_batches,
        bypass_device_assignment_check=physical_mesh.is_distributed)
    assert len(hlo_protos) == 2

    # Compile these two HLOs separately to get two XLA executables
    accumulate_grad = xc.XlaComputation(hlo_protos[0])
    apply_grad = xc.XlaComputation(hlo_protos[1])

    ## donate old_grad to make the gradient accumulation in-place
    tmp_donate_invars = (False,) * len(accumulate_grad_invar_indices) + (
        True,) * num_grads
    setup_computation_alias(accumulate_grad, tmp_donate_invars)

    ## donate old opt_state and params to make the weight update in-place
    tmp_donate_invars = tuple(donated_invars[i] for i in apply_grad_invar_indices) +\
        (False,) * num_grads
    setup_computation_alias(apply_grad, tmp_donate_invars)

    bypass_device_assignment_check = physical_mesh.is_distributed
    accumulate_grad = compile_with_given_strategy(
        backend,
        accumulate_grad,
        strategy_config,
        physical_mesh.total_devices,
        bypass_device_assignment_check,
        HloProtoStatus.SHARDING_ANNOTATED,
        rewrite_for_grad_acc=True)
    apply_grad = compile_with_given_strategy(backend, apply_grad,
                                             strategy_config,
                                             physical_mesh.total_devices,
                                             bypass_device_assignment_check,
                                             HloProtoStatus.SHARDING_ANNOTATED)

    # Compile them to a single mesh executable
    mesh_executable = GradAccMeshDriverExecutable(physical_mesh,
                                                  accumulate_grad,
                                                  apply_grad,
                                                  strategy_config,
                                                  in_avals,
                                                  out_avals,
                                                  grad_avals,
                                                  donated_invars,
                                                  batch_invars,
                                                  accumulate_grad_invar_indices,
                                                  apply_grad_invar_indices,
                                                  num_micro_batches,
                                                  flop_count=flop_count)
    return mesh_executable.get_driver_callable()


def filter_used_vars(all_vars, eqns):
    """Return the vars in all_vars that are used by eqns.

    The returned vars preserve their original order in all_vars.
    """
    used_vars = OrderedSet()
    for eqn in eqns:
        used_vars.update(x for x in eqn.invars if not isinstance(x, Literal))
    return [var for var in all_vars if var in used_vars]


def clone_vars(var_list, gensym_func):
    """Clone variables."""
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
    # pylint: disable=import-outside-toplevel
    from parax.pipeline_parallel.primitive_def import pipeline_p

    global_invars = OrderedSet(raw_jaxpr.jaxpr.invars)
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
    global_invar_substitute.update(zip(old_invars, new_invars))
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
        new_jaxpr_eqn(new_grad_vars, inter_grad_vars, pipeline_p, {
            "mark_type": "end",
            "name": "grad_acc_boundary"
        }, None))

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
        tmp_var = old_invars[-(i + 1)]
        combined_eqns.append(
            new_jaxpr_eqn([
                tmp_var,
                Literal(np.array(num_micro_batches, tmp_var.aval.dtype))
            ], [tmp_var], div_p, {}, None))
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

    # Make the new jaxpr
    combined_jaxpr = ClosedJaxpr(
        Jaxpr(raw_jaxpr.jaxpr.constvars, [
            global_invar_substitute.get(x, x)
            for x in (raw_jaxpr.jaxpr.invars + old_grad_vars)
        ], new_outvars, combined_eqns), raw_jaxpr.consts)

    # The indices of the arguments in global arguments.
    # TODO(lmzheng): this step is O(n^2)
    accumulate_grad_invar_indices = [
        combined_jaxpr.jaxpr.invars.index(var)
        for var in accumulate_grad_invars[:-num_grads]
    ]
    apply_grad_invar_indices = [
        combined_jaxpr.jaxpr.invars.index(var)
        for var in apply_grad_invars[:-num_grads]
    ]
    return (combined_jaxpr, accumulate_grad_invar_indices,
            apply_grad_invar_indices, num_grads)
