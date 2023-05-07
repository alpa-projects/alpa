"""Compile executables for shard parallelism."""
import hashlib
import inspect
from typing import Callable, Sequence, Optional, Union

import numpy as np
from jax import linear_util as lu
from jax._src import traceback_util
from jax._src.lib import xla_extension as xe
from jax.core import (Jaxpr, ClosedJaxpr, Literal, gensym, get_aval,
                      raise_to_shaped, AbstractValue)
from jax.lax import add_p, div_p
from jax.tree_util import PyTreeDef

from alpa.device_mesh import LogicalDeviceMesh, PhysicalDeviceMesh
from alpa.global_env import global_config
from alpa.mesh_executable import (NormalMeshDriverExecutable,
                                  GradAccMeshDriverExecutable)
from alpa.pipeline_parallel.apply_grad import APPLY_GRAD_MARKER_SUFFIX
from alpa.shard_parallel.auto_sharding import (run_auto_sharding_pass,
                                               run_spmd_partitioner_pass,
                                               AutoShardingOption)
from alpa.shard_parallel.manual_sharding import (ManualShardingOption,
                                                 get_manual_sharding_spec)
from alpa.util import (jaxpr_to_hlo, new_jaxpr_eqn, setup_computation_alias,
                       trace_jaxpr_with_micro_batch,
                       undefined_sharding_spec_proto, OrderedSet)

traceback_util.register_exclusion(__file__)


def get_compute_key(fun: lu.WrappedFun, in_tree: PyTreeDef,
                    donated_invars: Sequence[bool],
                    *aval: Sequence[AbstractValue]):
    """Return a unique string as the query key of a computation definition."""
    # pylint: disable=unused-argument
    # Algorithm:
    # Concatenate the definition location, source code,
    # input arguments specification to a string.
    # Then compute a hash value of this string.
    #
    # TODO(lmzheng): use jaxpr or hlo instead of source code?

    location = str(fun.f).split("at", maxsplit=1)[0]
    source_code = inspect.getsource(fun.f)
    donated_invars = str(donated_invars)
    aval = "".join(x.str_short() for x in aval)

    string = location + source_code + donated_invars + aval
    hash_key = hashlib.md5(string.encode(encoding="utf-8")).hexdigest()
    return hash_key


def compile_shard_executable(
    fun: lu.WrappedFun,
    in_tree: PyTreeDef,
    out_tree_thunk: Callable,
    static_argnums: Sequence[int],
    donated_invars: Sequence[bool],
    batch_invars: Sequence[bool],
    device_mesh: Union[PhysicalDeviceMesh, LogicalDeviceMesh],
    num_micro_batches: Optional[int],
    as_option: AutoShardingOption,
    ms_option: ManualShardingOption,
    *avals: Sequence[AbstractValue],
):
    """Compile an executable with auto-sharding pass."""
    if isinstance(device_mesh, PhysicalDeviceMesh):
        physical_mesh = device_mesh
        logical_mesh_choices = [physical_mesh.get_logical_mesh()]
    elif isinstance(device_mesh, LogicalDeviceMesh):
        physical_mesh = device_mesh.physical_mesh
        logical_mesh_choices = [device_mesh]
    else:
        raise ValueError("Invalid value of devices")

    if num_micro_batches is None:
        return shard_parallel_internal(fun, in_tree, out_tree_thunk,
                                       static_argnums, donated_invars,
                                       physical_mesh, logical_mesh_choices,
                                       as_option, ms_option, *avals)
    else:
        if global_config.backend == "tpu":
            raise NotImplementedError(
                "Gradient accumulation for tpu is not supported")
        return shard_parallel_internal_gradient_accumulation(
            fun, in_tree, out_tree_thunk, static_argnums, donated_invars,
            batch_invars, physical_mesh, logical_mesh_choices,
            num_micro_batches, as_option, ms_option, *avals)


def shard_parallel_internal(
        fun: lu.WrappedFun, in_tree: PyTreeDef, out_tree_thunk: Callable,
        static_argnums: Sequence[int], donated_invars: Sequence[bool],
        physical_mesh: PhysicalDeviceMesh,
        logical_mesh_choices: Sequence[LogicalDeviceMesh],
        as_option: AutoShardingOption, ms_option: ManualShardingOption,
        *avals: Sequence[AbstractValue]):
    """
    Compile an executable with auto-sharding pass.

    Args:
      fun: The wrapped jax function to be compiled.
      in_tree: The pytree of input arguments.
      out_tree_thunk: The thunk to produce output pytree.
      donated_invars: Whether to donate input parameters.
      physical_mesh: The physical device mesh.
      logical_mesh_choices: The candidates of logical mesh shape.
        If there is only one choice, use the given one. If there are multiple
        choices, we will try all of them and pick the best.
      as_option: The options of auto-sharding solver.
      avals: The input abstract values.
    """
    # pylint: disable=unused-argument
    # Trace to get jaxpr
    closed_jaxpr, _ = trace_jaxpr_with_micro_batch(fun, [False] * len(avals), 1,
                                                   avals)
    out_avals = [v.aval for v in closed_jaxpr.jaxpr.outvars]

    # Convert jaxpr to XLA HLO
    name = f"{fun.__name__}_shard_parallel"
    hlo = jaxpr_to_hlo(name, closed_jaxpr, donated_invars)
    # Set user specified sharding specs.
    if ms_option:
        if as_option.enable_auto_sharding:
            raise NotImplementedError("hybrid auto sharding is unsupported")
        in_sharding_proto, out_sharding_proto = get_manual_sharding_spec(
            ms_option, logical_mesh_choices[0].shape, in_tree, out_tree_thunk(),
            avals, out_avals)
        if in_sharding_proto is not None:
            hlo.set_input_shardings(in_sharding_proto)
            hlo.is_manually_annotated = True
        if out_sharding_proto is not None:
            hlo.set_output_shardings(out_sharding_proto)
            hlo.is_manually_annotated = True
    flop_count = xe.hlo_module_count_flop_dot_conv_only(hlo.get_module())

    # Compile a XLA executable
    hlo, stage_plan = run_auto_sharding_pass(hlo, logical_mesh_choices[0],
                                             "single", 1, as_option)
    # This is a walkaround because XLA GpuCompiler has some issue
    # FIXME: further test if this can be thoroughly removed.
    # if global_config.backend == "gpu":
    #     hlo = run_spmd_partitioner_pass(hlo,
    #                                     np.prod(logical_mesh_choices[0].shape))

    # Compile a mesh executable
    return NormalMeshDriverExecutable(physical_mesh,
                                      hlo,
                                      stage_plan,
                                      avals,
                                      out_avals,
                                      donated_invars,
                                      static_argnums=static_argnums,
                                      in_tree=in_tree,
                                      out_tree=out_tree_thunk(),
                                      flop_count=flop_count)


def shard_parallel_internal_gradient_accumulation(
        fun: lu.WrappedFun, in_tree: PyTreeDef, out_tree_thunk: Callable,
        static_argnums: Sequence[int], donated_invars: Sequence[bool],
        batch_invars: Sequence[bool], physical_mesh: PhysicalDeviceMesh,
        logical_mesh_choices: Sequence[LogicalDeviceMesh],
        num_micro_batches: int, as_option: AutoShardingOption,
        ms_option: ManualShardingOption, *raw_avals: Sequence[AbstractValue]):
    """Compile a gradient accumulation executable with auto-sharding pass."""
    # pylint: disable=unused-argument
    # Split the batch dimension
    closed_jaxpr, _ = trace_jaxpr_with_micro_batch(fun, batch_invars,
                                                   num_micro_batches, raw_avals)

    (closed_jaxpr, accumulate_grad_invar_indices, apply_grad_invar_indices,
     num_grads) = (add_gradient_accumulation(closed_jaxpr, num_micro_batches))
    in_avals = [x.aval for x in closed_jaxpr.jaxpr.invars[:-num_grads]]
    out_avals = [x.aval for x in closed_jaxpr.jaxpr.outvars]
    grad_avals = [x.aval for x in closed_jaxpr.jaxpr.invars[-num_grads:]]

    # Run auto-sharding and slice the combined HLO into two HLO: accumulate_grad
    # and apply_grad
    donated_invars = donated_invars + (False,) * num_grads
    name = f"{fun.__name__}_shard_parallel"
    hlo = jaxpr_to_hlo(name, closed_jaxpr, donated_invars)
    flop_count = xe.hlo_module_count_flop_dot_conv_only(hlo.get_module())
    flop_count *= num_micro_batches

    # Set user specified sharding specs.
    if ms_option:
        if as_option.enable_auto_sharding:
            raise NotImplementedError("hybrid auto sharding is unsupported")
        in_sharding_proto, out_sharding_proto = get_manual_sharding_spec(
            ms_option, logical_mesh_choices[0].shape, in_tree, out_tree_thunk(),
            in_avals, out_avals)
        grad_sharding_proto = [undefined_sharding_spec_proto()] * num_grads
        if in_sharding_proto is not None:
            in_sharding_proto += tuple(grad_sharding_proto)
            hlo.set_input_shardings(in_sharding_proto)
            hlo.is_manually_annotated = True
        if out_sharding_proto is not None:
            hlo.set_output_shardings(out_sharding_proto)
            hlo.is_manually_annotated = True

    # pylint: disable=unbalanced-tuple-unpacking
    hlo_stage_names, hlo_stages, stage_plan = run_auto_sharding_pass(
        hlo, logical_mesh_choices[0], "stages", num_micro_batches, as_option)
    assert len(hlo_stages) == 2

    if hlo_stage_names[0].endswith(APPLY_GRAD_MARKER_SUFFIX):
        hlo_stage_names[0], hlo_stages[0], hlo_stage_names[1], hlo_stages[1] = (
            hlo_stage_names[1], hlo_stages[1], hlo_stage_names[0],
            hlo_stages[0])
    assert hlo_stage_names[1].endswith(APPLY_GRAD_MARKER_SUFFIX)

    # Compile these two HLOs separately to get two XLA executables
    accumulate_grad, apply_grad = hlo_stages

    ## donate old_grad to make the gradient accumulation in-place
    tmp_donate_invars = ((False,) * len(accumulate_grad_invar_indices) +
                         (True,) * num_grads)
    setup_computation_alias(accumulate_grad, tmp_donate_invars)

    ## donate old opt_state and params to make the weight update in-place
    tmp_donate_invars = (
        tuple(donated_invars[i] for i in apply_grad_invar_indices) +
        (False,) * num_grads)
    setup_computation_alias(apply_grad, tmp_donate_invars)

    # Compile them to a single mesh executable
    return GradAccMeshDriverExecutable(physical_mesh,
                                       accumulate_grad,
                                       apply_grad,
                                       stage_plan,
                                       in_avals,
                                       out_avals,
                                       grad_avals,
                                       donated_invars,
                                       batch_invars,
                                       accumulate_grad_invar_indices,
                                       apply_grad_invar_indices,
                                       num_micro_batches,
                                       in_tree=in_tree,
                                       out_tree=out_tree_thunk(),
                                       flop_count=flop_count)


def filter_used_vars(all_vars, eqns):
    """Return the vars in all_vars that are used by eqns.

    The returned vars preserve their original order in all_vars.
    """
    used_vars = OrderedSet()
    for eqn in eqns:
        used_vars.update(x for x in eqn.invars if not isinstance(x, Literal))
    return [var for var in all_vars if var in used_vars]


def filter_pass_through_vars(in_vars, out_vars):
    in_vars_set = set(x for x in in_vars if not isinstance(x, Literal))
    return [x for x in out_vars if x in in_vars_set]


def clone_vars(var_list, gensym_func: Callable):
    """Clone variables."""
    return [gensym_func(x.aval) for x in var_list]


def add_gradient_accumulation(raw_jaxpr, num_micro_batches):
    """Add gradient accumulation logics into the raw jaxpr.

    Signatures of functions:
        raw_jaxpr(param, opt_state, batch) -> [new_param, new_opt_state]

        The original_jaxpr can be split into:
        "compute_grad(param, batch) -> out_grad"
        "apply_grad(param, opt_state, in_grad) -> [new_param, new_opt_state]"

        We then derive accumulate_grad from compute_grad:
        "accumulate_grad(param, batch, old_grad) -> new_grad"

        The returned jaxpr is composed by [
            pipeline_marker_start
            accumulate_grad
            pipeline_marker_end

            pipeline_marker_start
            apply_grad
            pipeline_marker_end
        ], with the signature
        "new_jaxpr(param, opt_state, batch, grad) -> [new_param, new_opt_state]"
    """
    # pylint: disable=import-outside-toplevel
    from alpa.pipeline_parallel.primitive_def import pipeline_p

    global_invars = OrderedSet(raw_jaxpr.jaxpr.invars)
    gensym_func = gensym([raw_jaxpr.jaxpr])

    # Find the gradient separator marker.
    # This separator partitions orginal_jaxpr into two part:
    # compute_grad and apply_grad
    marker_eqn = None
    marker_pos = 0
    for pos, eqn in enumerate(raw_jaxpr.jaxpr.eqns):
        if eqn.primitive is pipeline_p and eqn.params["mark_type"] == "grad":
            marker_eqn = eqn
            marker_pos = pos
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
        new_jaxpr_eqn(new_invars, old_invars, pipeline_p, {
            "mark_type": "start",
            "name": "accumulate_grad"
        }))
    global_invar_substitute.update(zip(old_invars, new_invars))
    accumulate_grad_invars = new_invars

    # Append eqns of compute_grad
    combined_eqns.extend(raw_jaxpr.jaxpr.eqns[:marker_pos])

    # Append eqns of gradient accumulation
    for i in range(len(out_grad_vars)):
        combined_eqns.append(
            new_jaxpr_eqn([old_grad_vars[i], out_grad_vars[i]],
                          [new_grad_vars[i]], add_p, {}))

    # Wrap all outvars of accumulate_grad
    inter_grad_vars = [gensym_func(x.aval) for x in out_grad_vars]
    combined_eqns.append(
        new_jaxpr_eqn(new_grad_vars, inter_grad_vars, pipeline_p, {
            "mark_type": "end",
            "name": "accumulate_grad"
        }))

    # Wrap all invars of apply_grad
    in_grad_vars = marker_eqn.outvars
    old_invars = (filter_used_vars(raw_jaxpr.jaxpr.invars, apply_grad_eqns) +
                  filter_pass_through_vars(raw_jaxpr.jaxpr.invars,
                                           raw_jaxpr.jaxpr.outvars) +
                  in_grad_vars)
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
        new_jaxpr_eqn(new_invars, old_invars, pipeline_p, {
            "mark_type": "start",
            "name": APPLY_GRAD_MARKER_SUFFIX
        }))

    # Append eqns for gradient reduction
    for i in range(num_grads):
        tmp_var = old_invars[-(i + 1)]
        literal_val = np.array(num_micro_batches, tmp_var.aval.dtype)
        combined_eqns.append(
            new_jaxpr_eqn([
                tmp_var,
                Literal(literal_val, raise_to_shaped(get_aval(literal_val))),
            ], [tmp_var], div_p, {}))
    # TODO(lmzheng): This breaks the SSA form of the combined_eqns
    # But I find jax can convert this non-SSA jaxpr to HLO correctly,
    # so I leave this issue as todo. To fix this, we should substitute
    # all grad vars in these equations with new vars.

    # Append eqns of apply_grad
    combined_eqns.extend(apply_grad_eqns)
    # TODO(lmzheng): The param vars are used in both compute_grad and
    #   apply_grad, so there will be some duplicated intermediate vars in
    #   compute_grad_eqns and apply_grad_eqns. This breaks the SSA form of the
    #   combined_eqns. But I find jax can convert this non-SSA jaxpr to HLO
    #   correctly, so I leave this issue as todo. To fix this, we should
    #   substitute all param vars in these equations with new vars.

    # Wrap all outvars of apply_grad
    old_outvars = raw_jaxpr.jaxpr.outvars
    new_outvars = [gensym_func(x.aval) for x in old_outvars]
    combined_eqns.append(
        new_jaxpr_eqn(old_outvars, new_outvars, pipeline_p, {
            "mark_type": "end",
            "name": APPLY_GRAD_MARKER_SUFFIX
        }))

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
