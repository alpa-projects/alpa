"""3D parallel on a Ray cluster."""
import logging
from typing import Sequence

import jax
from jax import linear_util as lu
from jax.core import ClosedJaxpr, DropVar, Jaxpr, gensym
from jax.interpreters import partial_eval as pe

from parax.device_mesh import VirtualMesh
from parax.global_env import global_config
from parax.pipeline_parallel.primitive_def import mark_pipeline_jaxpreqn
from parax.pipeline_parallel.runtime import (
    GpipeSchedule, Jax3DPipeline, gen_linear_pipeline_dependency,
    gen_linear_pipeline_dependency_with_apply)
from parax.pipeline_parallel.stage import (
    JaxPipelineStage, apply_grad_add_marker, apply_grad_get_mean,
    compute_to_acc_pipe, generate_sharded_xla_stages, get_var_mapping,
    mark_grad_mesh, mark_missing_vars_in_pipeline_marks, pipeline_dce,
    rearrange_vars, slice_apply_gradient,
    slice_closed_jaxpr_by_full_pipeline_marks)
from parax.util import get_micro_batch, slices_to_jaxpr

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def split_compute_and_apply(closed_jaxpr: ClosedJaxpr):
    from parax.pipeline_parallel.primitive_def import pipeline_p
    split_eqn = None
    for idx, eqn in enumerate(closed_jaxpr.eqns):
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'grad':
            split_eqn = eqn
            split_idx = idx
    if split_eqn is None:
        logger.warning(
            'missing barrier between compute and apply, hint: replace jax.grad by parax.grad'
        )
        return closed_jaxpr, ClosedJaxpr(Jaxpr([], [], [], []), []), None
    sliced_eqns = [
        closed_jaxpr.eqns[:split_idx], [split_eqn],
        closed_jaxpr.eqns[split_idx + 1:]
    ]
    compute, _, apply = slices_to_jaxpr(closed_jaxpr, sliced_eqns)
    if len(apply.eqns) == 0:
        logger.warning('the apply gradient part is None. hint: apply() after parax.grad')
    return compute, apply, split_eqn


def has_corresponding_outvar(var, stage):
    pure_invar = None
    for invar, invar_sym in zip(stage.eqns[0].invars, stage.eqns[0].outvars):
        if invar == var:
            pure_invar = invar_sym
            break
    assert pure_invar
    for outvar_sym, outvar in zip(stage.eqns[-1].invars,
                                  stage.eqns[-1].outvars):
        if outvar_sym == pure_invar:
            return outvar
    return None


def split_donate_invars(global_invars, donation_mapping,
                        stages: Sequence[JaxPipelineStage], pattern, gensym_fn):
    """
    Split donated invars for sliced jaxprs. The pattern is in form of:
    1. parallel. jaxprs are in different meshes.
    2. serial. jaxprs are in the same mesh and executes in serial.
    3. Inside a mesh is serial, between a mesh is parallel.
    In the third pattern, we should consider main buffer and copy buffer.
    A main buffer should not be donated unless no other mesh requires its result.
    Currently, we only donate:
    1. global invars that can be donated(set by users);
    2. buffers for accumulated gradients.
    But if auto-sharding supports, we can add:
    1. local invars not used later in this mesh, not main copy
    2. local invars not used later in all meshes, main copy
    Args:
        donation_mapping (Dict[Var, Var]): known mapping of donations, including 
            global invar-outvar and accumulate gradients
        stages: slices in topology order of execution.
        pattern: The outter list is for parallel, and inner for serial.
    Returns:
        donate_invars_dict:List[Sequence[bool]]: donate_invars for each stage
    """
    reversed_donation_mapping = {v: k for k, v in donation_mapping.items()}
    # global last use to consider if the main copy can be discarded
    global_last_use = dict()
    for stage_idx, stage in enumerate(stages):
        for invar in stage.invars:
            global_last_use[invar] = stage_idx

    ans = [None for _ in range(len(stages))]
    donate_mappings = [None for _ in range(len(stages))]

    for serial_group in pattern:
        serial_group = sorted(serial_group)
        main_copy_vars = set()
        for stage_idx in serial_group:
            stage = stages[stage_idx]
            main_copy_vars.update(
                [v for v in stage.invars if v in global_invars])
            main_copy_vars.update(stage.outvars)
        serial_group = reversed(serial_group)
        use_later = set()
        for stage_idx in serial_group:
            stage = stages[stage_idx]
            invars = set(stage.invars)
            donate_mapping = dict()
            appended_invars = set()
            for var in stage.outvars:
                if var not in reversed_donation_mapping:
                    continue
                invar = reversed_donation_mapping[var]
                assert global_last_use[
                    invar] <= stage_idx, "invar is donated before last use"
                assert invar in main_copy_vars, f"donate invar {invar} not at the mesh"
                assert invar.aval.shape == var.aval.shape
                donate_mapping[invar] = var
                if invar in invars:
                    continue
                appended_invars.add(invar)
            use_later.update(invars)
            donate_mappings[stage_idx] = donate_mapping
            # append dummy invars:
            if not appended_invars:
                continue
            appended_invars = list(appended_invars)
            logger.warning(f'append into stage {stage_idx} for donation:{appended_invars}')
            stage.invars = stage.invars + appended_invars
            pipe_start = stage.eqns[0]
            stage.eqns[0] = mark_pipeline_jaxpreqn(
                pipe_start.invars + appended_invars, pipe_start.outvars +
                list(map(lambda v: gensym_fn(v.aval), appended_invars)),
                pipe_start.params['name'], pipe_start.params['mark_type'])

    # rearrange to keep donated invars and outvars have same index
    for stage_idx, stage in enumerate(stages):
        donate_mapping = donate_mappings[stage_idx]
        new_invars, new_pipe_start = rearrange_vars(
            stage.invars, list(donate_mapping.keys()), stage.eqns[0], True)
        new_outvars, new_pipe_end = rearrange_vars(
            stage.outvars, list(donate_mapping.values()), stage.eqns[-1], False)
        stage.invars = new_invars
        stage.outvars = new_outvars
        stage.eqns[0] = new_pipe_start
        stage.eqns[-1] = new_pipe_end
        donated_num = len(donate_mapping)
        ans[stage_idx] = (True,) * donated_num + (False,) * (len(new_invars) -
                                                             donated_num)

    return ans


@lu.cache
def three_d_parallel_callable(fun: lu.WrappedFun, in_tree, out_tree_thunk,
                              donated_invars, batch_invars, devices,
                              memory_budget_per_device, *avals):
    """End-to-end 3d parallel combining pipelining and sharding."""
    if not (isinstance(devices, VirtualMesh) or
            global_config.search_logical_mesh_shape):
        raise RuntimeError("Unrecognized type of `devices`, got: {}, "
                           "expected type: {}.".format(type(devices),
                                                       "VirtualMesh"))

    # Slice the jaxpr into pipeline stages
    virtual_mesh = devices
    num_micro_batches = global_config.num_micro_batches
    if num_micro_batches is None:
        logger.warning('num microbatch is unset, automatically use 1')
        num_micro_batches = 1
    microbatch_avals = get_micro_batch(batch_invars, num_micro_batches, *avals)
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, microbatch_avals)
    # the sliced_meshes is set when tracing into forward decorator
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    gensym_func = gensym([closed_jaxpr.jaxpr])
    compute_grad_jaxpr, apply_grad_jaxpr, barrier = split_compute_and_apply(
        closed_jaxpr)
    # TODO(yonghao): The case that barrier is None should be deprecated
    if barrier is None:
        acc_grad_jaxpr = compute_grad_jaxpr
    else:
        # compute grad to accumulate grad
        acc_grad_jaxpr, acc_grad_dict, grad_in_to_out = compute_to_acc_pipe(
            compute_grad_jaxpr, gensym_func)
    # slice accumulate grad
    acc_grad_invars = acc_grad_jaxpr.jaxpr.invars
    acc_grad_outvars = acc_grad_jaxpr.jaxpr.outvars

    jax_pipeline_stages = slice_closed_jaxpr_by_full_pipeline_marks(
        acc_grad_jaxpr)
    jax_pipeline_stages = mark_missing_vars_in_pipeline_marks(
        jax_pipeline_stages, acc_grad_invars, acc_grad_outvars)
    jax_pipeline_stages = pipeline_dce(jax_pipeline_stages, acc_grad_outvars)
    # TODO(yonghao): move auto mesh slicing until here and get stage_to_mesh
    # delete the 4 lines below in auto mesh version
    stage_num = len(jax_pipeline_stages)
    stage_to_mesh = {
        i: (i if i < stage_num / 2 else stage_num - i - 1)
        for i, _ in enumerate(jax_pipeline_stages)
    }
    assert stage_num % 2 == 0
    mesh_num = stage_num // 2

    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    if barrier is not None:
        # Process apply gradient:
        # 1. change invars of apply grad to output of accumulate grad
        gradients = [g for g in barrier.outvars if not isinstance(g, DropVar)]
        mask = {
            outv: acc_grad_dict[inv]
            for outv, inv in zip(gradients, barrier.invars)
        }
        # 2. Add compute mean and slice apply-grad stages
        grad_mesh = mark_grad_mesh(gradients, jax_pipeline_stages,
                                   stage_to_mesh, mask)
        apply_grad_jaxpr, global_outvars = apply_grad_get_mean(
            apply_grad_jaxpr, gradients, gensym_func, num_micro_batches,
            global_outvars)
        sliced_apply_grad, info = slice_apply_gradient(apply_grad_jaxpr,
                                                       grad_mesh, mesh_num)
        apply_deps, apply_grad_schedule, _ = info
        sliced_apply_grad, out_map = apply_grad_add_marker(sliced_apply_grad,
                                                           mask,
                                                           gensym_func,
                                                           stage=True)
        global_outvars = list(
            map(lambda x: get_var_mapping(out_map, x), global_outvars))
        n_stages = len(jax_pipeline_stages) + len(sliced_apply_grad)
        dependency = gen_linear_pipeline_dependency_with_apply(
            n_stages, mesh_num, apply_deps)
        jax_all_stages = jax_pipeline_stages + sliced_apply_grad
    else:
        jax_all_stages = jax_pipeline_stages
        n_stages = len(jax_pipeline_stages)
        dependency = gen_linear_pipeline_dependency(n_stages)
        apply_grad_schedule = {}
        grad_in_to_out = {}
    # Generate schedule and placement
    num_batch = num_micro_batches
    schedule = GpipeSchedule(dependency=dependency,
                             mesh=virtual_mesh,
                             num_pipeline_worker=mesh_num,
                             apply_grad_schedule=apply_grad_schedule,
                             num_batch=num_batch)
    physical_meshes = []
    n_meshes = len(schedule.meshes)
    # TODO(Hao): delay the creation of physical mesh here
    for i, mesh in enumerate(schedule.meshes):
        logger.debug("Launch the {}th mesh...".format(i))
        physical_meshes.append(mesh.get_physical_mesh())

    stage_dict = [[] for _ in range(n_meshes)]
    stage_id_dict = [[] for _ in range(n_meshes)]
    for i, stage in enumerate(jax_all_stages):
        mesh_indices = list(schedule.stage_placement(i))
        assert len(mesh_indices) == 1
        mesh_idx = mesh_indices[0]
        stage_id_dict[mesh_idx].append(i)
        stage_dict[mesh_idx].append(stage)
    # split donate invar with mesh info
    if barrier is not None:
        grad_invars = list(grad_in_to_out.keys())
        all_invars = closed_jaxpr.jaxpr.invars + grad_invars
        donation_mapping = dict(grad_in_to_out)
    else:
        all_invars = closed_jaxpr.jaxpr.invars
        donation_mapping = dict()
    # infer donation of global invar-outvars
    donated_outvars = set()

    for donate, invar in zip(donated_invars, global_invars):
        if not donate:
            continue
        for outvar in global_outvars:
            if outvar in donated_outvars:
                continue
            if invar.aval.shape != outvar.aval.shape:
                continue
            donated_outvars.add(outvar)
            donation_mapping[invar] = outvar
            break
        if invar not in donation_mapping:
            logger.warning(f"{invar} is marked as donated but actually no match outvar")
    pattern = [stage_id_dict[mesh_idx] for mesh_idx in range(mesh_num)]
    donate_invars_dict = split_donate_invars(all_invars, donation_mapping,
                                             jax_all_stages, pattern,
                                             gensym_func)

    # Call auto-sharding pass to shard each stage
    xla_stages = [None] * n_stages
    for mesh_idx in range(n_meshes):
        # TODO (zhuohan): Support search logical device shape for 3d parallel
        physical_mesh = physical_meshes[mesh_idx]
        logical_mesh_choices = [physical_mesh.get_default_logical_mesh()]
        logical_mesh_search_mode = "cost_model"
        stage_donate_invars = [
            donate_invars_dict[stage_idx]
            for stage_idx in stage_id_dict[mesh_idx]
        ]
        search_task = None
        record_file = None
        sharded_xla_stages = generate_sharded_xla_stages(
            str(mesh_idx), stage_dict[mesh_idx], stage_donate_invars,
            physical_mesh, logical_mesh_choices, logical_mesh_search_mode,
            memory_budget_per_device, acc_grad_outvars, search_task,
            record_file)
        for i, xla_stage in zip(stage_id_dict[mesh_idx], sharded_xla_stages):
            xla_stages[i] = xla_stage

    grad_in_to_out = {k:repr(v) for k, v in grad_in_to_out.items()}
    jp = Jax3DPipeline(pipeline_stages=xla_stages,
                       global_invars=global_invars,
                       grad_dummy_invars=grad_in_to_out,
                       global_outvars=global_outvars,
                       physical_meshes=physical_meshes,
                       dependency=dependency,
                       schedule=schedule,
                       is_batch=batch_invars,
                       num_batch=num_batch)

    def ret_func(*args, **kwargs):
        return jp.run(*args, **kwargs)

    ret_func.get_executable = lambda: jp

    return ret_func  # pylint: disable=unnecessary-lambda
