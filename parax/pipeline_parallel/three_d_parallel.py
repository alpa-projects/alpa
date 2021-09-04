"""3D parallel on a Ray cluster."""
import logging
from typing import Sequence

import jax
from jax import linear_util as lu
from jax.core import ClosedJaxpr, gensym
from jax.interpreters import partial_eval as pe

from parax.device_mesh import VirtualMesh
from parax.global_env import global_config
from parax.pipeline_parallel.runtime import (GpipeSchedule, Jax3DPipeline,
                                             gen_linear_pipeline_dependency)
from parax.pipeline_parallel.stage import (
    compute_to_acc_pipe, generate_sharded_xla_stages, mark_global_and_local_vars,
    slice_closed_jaxpr_by_manual_pipeline_marks,
    slice_closed_jaxpr_by_full_pipeline_marks,
    mark_missing_vars_in_pipeline_marks, compute_to_acc, slice_apply_gradient,
    mark_grad_mesh, rewrite_apply_grad)
from parax.util import get_micro_batch, slices_to_jaxpr

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def split_compute_and_apply(closed_jaxpr: ClosedJaxpr):
    from parax.pipeline_parallel.primitive_def import pipeline_p
    split_eqn = None
    for idx, eqn in enumerate(closed_jaxpr.jaxpr.eqns):
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'grad':
            split_eqn = eqn
            split_idx = idx
    assert split_eqn is not None, 'missing barrier between compute and apply'
    sliced_eqns = [
        closed_jaxpr.eqns[:split_idx], [split_eqn], closed_jaxpr.eqns[split_idx + 1:]
    ]
    compute, _, apply = slices_to_jaxpr(closed_jaxpr, sliced_eqns)
    return compute, apply, split_eqn


def split_donate_invars(donated_invars: Sequence[bool], global_invars,
                        sliced_jaxprs: Sequence[ClosedJaxpr], pattern):
    """
    Split donated invars for sliced jaxprs. The pattern is in form of:
    1. parallel. jaxprs are in different meshes.
    2. serial. jaxprs are in the same mesh and executes in serial.
    3. some mesh in a stage, some in another. 
    Args:
        donated_invars: whether a global invar is donated.
        global_invars: global invars for the whole jaxpr.
        sliced_jaxprs: slices in topology order of execution.
        pattern: The outter list is for parallel, and innter for serial.
    """
    not_donated = [
        global_invars[i]
        for i in range(len(donated_invars))
        if not donated_invars[i]
    ]
    not_donated = set(not_donated)
    ans = [None for _ in range(len(sliced_jaxprs))]
    for serial_group in pattern:
        serial_group = reversed(sorted(serial_group))
        use_later = set()
        for idx in serial_group:
            invars = sliced_jaxprs[idx].jaxpr.invars
            # donate an invar if it is neither a not donated global invar nor an input of later slices
            donate_status = [
                not (var in not_donated or var in use_later) for var in invars
            ]
            use_later.update(invars)
            ans[idx] = donate_status
    return ans


@lu.cache
def three_d_parallel_callable(fun: lu.WrappedFun, in_tree, out_tree_thunk,
                              donated_invars, batch_invars, devices,
                              memory_budget_per_device, pipeline_marker_type,
                              *avals):
    """End-to-end 3d parallel combining pipelining and sharding."""
    if not (isinstance(devices, VirtualMesh) or
            global_config.search_logical_mesh_shape):
        raise RuntimeError("Unrecognized type of `devices`, got: {}, "
                           "expected type: {}.".format(type(devices),
                                                       "VirtualMesh"))

    # Slice the jaxpr into pipeline stages
    virtual_mesh = devices
    num_micro_batches = global_config.num_micro_batches
    assert num_micro_batches
    microbatch_avals = get_micro_batch(batch_invars, num_micro_batches, *avals)
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, microbatch_avals)
    # the sliced_meshes is set when tracing into forward decorator
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    compute_grad_jaxpr, apply_grad_jaxpr, barrier = split_compute_and_apply(closed_jaxpr)
    # compute grad to accumulate grad
    acc_grad_jaxpr, acc_grad_dict = compute_to_acc_pipe(compute_grad_jaxpr)
    # change invars of apply grad to output of accumulate grad
    apply_grad_jaxpr = rewrite_apply_grad(apply_grad_jaxpr, acc_grad_dict, barrier)
    # slice accumulate grad
    global_invars = acc_grad_jaxpr.jaxpr.invars
    global_outvars = acc_grad_jaxpr.jaxpr.outvars
    if pipeline_marker_type == "manual":
        gensym_func = gensym([closed_jaxpr.jaxpr])
        jax_pipeline_stages = slice_closed_jaxpr_by_manual_pipeline_marks(
            acc_grad_jaxpr)
        jax_pipeline_stages = [
            mark_global_and_local_vars(stage, gensym_func)
            for stage in jax_pipeline_stages
        ]
    elif pipeline_marker_type == "full":
        jax_pipeline_stages = slice_closed_jaxpr_by_full_pipeline_marks(
            acc_grad_jaxpr)
        jax_pipeline_stages = mark_missing_vars_in_pipeline_marks(
            jax_pipeline_stages, global_invars, global_outvars)
    else:
        raise ValueError("Invalid pipeline marker type", pipeline_marker_type)
    # TODO(yonghao): move auto mesh slicing until here and get stage_to_mesh
    # delete the two lines below in auto mesh version
    stage_num = len(jax_pipeline_stages)
    stage_to_mesh = {
        i: (i if i < stage_num / 2 else stage_num - i - 1) 
        for i, _ in enumerate(jax_pipeline_stages)
    }
    # slice apply-grad stages
    grad_mesh = mark_grad_mesh(apply_grad_jaxpr.jaxpr.invars, jax_pipeline_stages, stage_to_mesh)
    sliced_apply_grad, invars_alloc = slice_apply_gradient(apply_grad_jaxpr, grad_mesh)
    # TODO(yonghao): split donate invar with mesh info

    # Generate schedule and placement
    num_batch = 1
    n_stages = len(jax_pipeline_stages)
    dependency = gen_linear_pipeline_dependency(n_stages)
    schedule = GpipeSchedule(dependency=dependency,
                             mesh=virtual_mesh,
                             num_batch=num_batch)
    physical_meshes = []
    n_meshes = len(schedule.meshes)
    # TODO(Hao): delay the creation of physical mesh here
    for i, mesh in enumerate(schedule.meshes):
        logger.debug("Launch the {}th mesh...".format(i))
        physical_meshes.append(mesh.get_physical_mesh())

    stage_dict = [[] for _ in range(n_meshes)]
    stage_id_dict = [[] for _ in range(n_meshes)]
    for i, stage in enumerate(jax_pipeline_stages):
        mesh_indices = list(schedule.stage_placement(i))
        assert len(mesh_indices) == 1
        mesh_idx = mesh_indices[0]
        stage_id_dict[mesh_idx].append(i)
        stage_dict[mesh_idx].append(stage)

    # Call auto-sharding pass to shard each stage
    xla_stages = [None] * n_stages
    for mesh_idx in range(n_meshes):
        # TODO (zhuohan): Support search logical device shape for 3d parallel
        physical_mesh = physical_meshes[mesh_idx]
        logical_mesh_choices = [physical_mesh.get_default_logical_mesh()]
        logical_mesh_search_mode = "cost_model"
        search_task = None
        record_file = None
        sharded_xla_stages = generate_sharded_xla_stages(
            str(mesh_idx), stage_dict[mesh_idx], physical_mesh,
            logical_mesh_choices, logical_mesh_search_mode,
            memory_budget_per_device, search_task, record_file)
        for i, xla_stage in zip(stage_id_dict[mesh_idx], sharded_xla_stages):
            xla_stages[i] = xla_stage

    jp = Jax3DPipeline(pipeline_stages=xla_stages,
                       global_invars=global_invars,
                       global_outvars=global_outvars,
                       physical_meshes=physical_meshes,
                       dependency=dependency,
                       schedule=schedule,
                       num_batch=num_batch)

    return lambda *args, **kwargs: jp.run(*args, **kwargs)  # pylint: disable=unnecessary-lambda
