import hashlib
import inspect
import time
from warnings import warn

from jax import linear_util as lu, disable_jit
from jax._src.util import (partial, extend_name_stack, wrap_name)
from jax.core import (Atom, Var, JaxprEqn, Jaxpr, ClosedJaxpr, DropVar, Literal,
                      jaxpr_as_fun, new_jaxpr_eqn, gensym)
from jax.interpreters import xla, partial_eval as pe
from jax.lax import add_p
from jax.lib import xla_bridge as xb, xla_client as xc

from parax.device_mesh import LogicalDeviceMesh, PhysicalDeviceMesh, DeviceCluster
from parax.global_env import global_config
from parax.measure_record import SearchTask, load_best_record
from parax.shard_parallel.auto_sharding import (compile_with_search, compile_with_given_strategy,
                                                get_input_output_sharding_specs, HloProtoStatus)
from parax.util import jaxpr_to_hlo_computation, OrderedSet


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
    name = f"{fun.__name__}_auto_shard"
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

def list_minus(a, b):
    b = set(b)
    return [x for x in a if x not in b]


def list_intersect(a, b):
    b = set(b)
    return [x for x in a if x in b]


def copy_closed_jaxpr(src):
    jaxpr = Jaxpr(src.jaxpr.constvars, src.jaxpr.invars, src.jaxpr.outvars, src.jaxpr.eqns)
    return ClosedJaxpr(jaxpr, src.consts)


class GradAccumulationInfo:
    def __init__(self, original_jaxpr):
        # Slice the jaxpr into two functions: compute_grad and apply_grad
        sliced_jaxprs = slice_closed_jaxpr_by_gradient_mark(original_jaxpr)
        assert len(sliced_jaxprs) == 2, "Must have exactly one gradient marker" 
        compute_grad_jaxpr, apply_grad_jaxpr = sliced_jaxprs

        # Signatures of the functions:
        #
        # original_jaxpr(opt_state, param, batch) -> [new_opt_state, new_param]
        # compute_grad(param, batch) -> out_grad
        # accumulate_grad(old_grad, param, batch) -> new_grad
        # apply_grad(opt_state, param, in_grad) -> [new_opt_state, new_param]
        #
        # TODO(lmzheng): handle other auxiliary variables (e.g., loss)

        # Derive batch, grad, param and opt_state variables
        param_vars = list_intersect(apply_grad_jaxpr.jaxpr.invars, compute_grad_jaxpr.jaxpr.invars)
        batch_vars = list_minus(original_jaxpr.jaxpr.invars, apply_grad_jaxpr.jaxpr.invars)
        out_grad_vars = compute_grad_jaxpr.jaxpr.outvars
        in_grad_vars = apply_grad_jaxpr.jaxpr.invars[:len(out_grad_vars)]
        opt_state_vars = list_minus(original_jaxpr.jaxpr.invars, compute_grad_jaxpr.jaxpr.invars)

        # TODO(lmzheng): add some assertion to veirfy the results.

        # Build accumulate_grad jaxpr from compute_grad jaxpr
        gensym_func = gensym([original_jaxpr.jaxpr])
        old_grad_vars = [gensym_func(x.aval) for x in out_grad_vars]
        new_grad_vars = [gensym_func(x.aval) for x in out_grad_vars]
        accumulate_grad_jaxpr = copy_closed_jaxpr(compute_grad_jaxpr)
        accumulate_grad_jaxpr.jaxpr.invars = \
            old_grad_vars + accumulate_grad_jaxpr.jaxpr.invars
        for i in range(len(out_grad_vars)):
            accumulate_grad_jaxpr.eqns.append(new_jaxpr_eqn(
                [old_grad_vars[i], out_grad_vars[i]], [new_grad_vars[i]],
                add_p, {}, None))
        accumulate_grad_jaxpr.jaxpr.outvars = new_grad_vars

        # Reorganize the order of invars
        apply_grad_jaxpr.jaxpr.invars = apply_grad_jaxpr.jaxpr.invars[len(in_grad_vars):] +\
                                        apply_grad_jaxpr.jaxpr.invars[:len(in_grad_vars)]

        # Store useful information
        self.original_jaxpr = original_jaxpr
        self.compute_grad_jaxpr = compute_grad_jaxpr
        self.apply_grad_jaxpr = apply_grad_jaxpr
        self.accumulate_grad_jaxpr = accumulate_grad_jaxpr

        self.param_vars = param_vars
        self.batch_vars = batch_vars
        self.old_grad_vars = old_grad_vars
        self.new_grad_vars = new_grad_vars
        self.in_grad_vars = in_grad_vars
        self.opt_state_vars = opt_state_vars

    def __str__(self):
        ret = ""
        ret += f"param vars: {self.param_vars}\n"
        ret += f"batch_vars: {self.batch_vars}\n"
        ret += f"old_grad_vars: {self.old_grad_vars}\n"
        ret += f"new_grad_vars: {self.new_grad_vars}\n"
        ret += f"in_grad_vars: {self.in_grad_vars}\n"
        ret += f"opt_state_vars: {self.opt_state_vars}\n\n"

        ret += f"accumulate_grad: {self.accumulate_grad_jaxpr}\n"
        ret += f"apply_grad: {self.apply_grad_jaxpr}\n"
        return ret


def shard_parallel_internal_gradient_accumulation(
    fun: lu.WrappedFun, in_tree, out_tree_thunk,
    donated_invars, batch_invars, physical_mesh, logical_mesh_choices,
    logical_mesh_search_mode, memory_budget_per_device,
    search_task, record_file, strategy_config, *raw_avals
):
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

    # Trace the function and split the jaxpr for gradient accumulation.
    with disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    grad_acc_info = GradAccumulationInfo(closed_jaxpr)

    # Compile accumulate_grad
    backend = xb.get_backend("gpu")
    accumulate_grad_jaxpr = grad_acc_info.accumulate_grad_jaxpr
    n_all_invars = len(accumulate_grad_jaxpr.jaxpr.invars)
    n_old_grad_vars = len(grad_acc_info.old_grad_vars)
    donated_invars = (True,) * n_old_grad_vars + (False,) * (n_all_invars - n_old_grad_vars)
    accumulate_grad_hlo = jaxpr_to_hlo_computation(
        f"{fun.__name__}_accumulate_grad",
        grad_acc_info.accumulate_grad_jaxpr,
        donated_invars, backend)

    compiled, strategy_config = compile_with_search(
        backend,
        accumulate_grad_hlo,
        physical_mesh,
        logical_mesh_choices,
        logical_mesh_search_mode,
        memory_budget_per_device,
        search_task,
        record_file,
        multiple_stages=False)

    # Compile apply_grad


def slice_closed_jaxpr_by_gradient_mark(closed_jaxpr):
    from parax.pipeline_parallel.primitive_def import pipeline_p

    global_invars = set(closed_jaxpr.jaxpr.invars)
    global_outvars = set(
        var for var in closed_jaxpr.jaxpr.outvars if isinstance(var, Var))
    global_consts_dir = dict(
        zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))
    result_jaxprs = []
    var2jaxpr_id = {}

    cur_constvars = cur_invars = cur_outvars = cur_eqns = cur_intermediate_vars = None

    def begin_jaxpr():
        nonlocal cur_constvars, cur_invars, cur_outvars, cur_eqns, cur_intermediate_vars
        cur_constvars, cur_invars = OrderedSet(), OrderedSet()
        cur_outvars, cur_eqns = OrderedSet(), list()
        cur_intermediate_vars = set()

    def end_jaxpr():
        jaxpr = Jaxpr(cur_constvars, cur_invars, cur_outvars, cur_eqns)
        consts = [global_consts_dir[x] for x in jaxpr.constvars]
        result_jaxprs.append(ClosedJaxpr(jaxpr, consts))

    begin_jaxpr()
    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'grad':
            cur_outvars.update(eqn.invars)
            end_jaxpr()
            begin_jaxpr()
            cur_invars.update(eqn.outvars)
        else:
            cur_eqns.append(eqn)

            for var in eqn.invars:
                if isinstance(var, Literal) or var in cur_intermediate_vars:
                    continue
                if var in global_consts_dir:
                    cur_constvars.add(var)
                elif var in global_invars:
                    cur_invars.add(var)

            for var in eqn.outvars:
                if not isinstance(var, DropVar):
                    cur_intermediate_vars.add(var)
                    var2jaxpr_id[var] = len(result_jaxprs)
                    if var in global_outvars:
                        cur_outvars.add(var)
    end_jaxpr()

    return result_jaxprs
