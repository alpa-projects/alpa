"""3D parallel on a Ray cluster."""
import logging
from typing import Sequence, List

import numpy as np
import jax
from jax import linear_util as lu
from jax.core import ClosedJaxpr, DropVar, Jaxpr, gensym
from jax.interpreters import partial_eval as pe

from parax.device_mesh import VirtualMesh
from parax.global_env import global_config
from parax.pipeline_parallel.decentralized_distributed_runtime import DecentralizedDistributedRuntime
from parax.pipeline_parallel.primitive_def import (mark_pipeline_jaxpreqn,
                                                   pipeline_p)
from parax.pipeline_parallel.centralized_distributerd_runtime import (
    CentralizedDistributedRuntime)
from parax.pipeline_parallel.schedules import (GpipeSchedule,
                                               gen_linear_pipeline_dependency,
                                               gen_dependency_with_stages)
from parax.pipeline_parallel.stage import (
    JaxPipelineStage, apply_grad_add_marker, apply_grad_get_mean,
    compute_grad_to_accumulate_grad, generate_sharded_xla_stages,
    get_var_mapping, mark_gradvar_to_mesh, mark_missing_vars_in_pipeline_marks,
    pipeline_dce, rearrange_vars, slice_apply_gradient, merge_stage_jaxprs,
    slice_closed_jaxpr_by_full_pipeline_marks)
from parax.util import get_micro_batch, slices_to_jaxpr
from parax.pipeline_parallel.stage_construction import get_stage_and_mesh_assignments, get_stage_outvars

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def split_compute_grad_and_apply_grad(closed_jaxpr: ClosedJaxpr):
    split_eqn = None
    for idx, eqn in enumerate(closed_jaxpr.eqns):
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'grad':
            split_eqn = eqn
            split_idx = idx
    if split_eqn is None:
        logger.warning(
            'Missing barrier between compute and apply. Assume there is no '
            'apply gradient step. Hint: replace jax.grad by parax.grad.')
        return closed_jaxpr, ClosedJaxpr(Jaxpr([], [], [], []), []), None
    sliced_eqns = [
        closed_jaxpr.eqns[:split_idx], [split_eqn],
        closed_jaxpr.eqns[split_idx + 1:]
    ]
    compute, _, apply = slices_to_jaxpr(closed_jaxpr, sliced_eqns)
    if len(apply.eqns) == 0:
        logger.warning(
            'the apply gradient part is None. hint: apply() after parax.grad')
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


def create_donation_mapping(initial_mapping, donated_invars, invars, outvars):
    # infer donation of global invar-outvars
    donation_mapping = initial_mapping
    donated_outvars = set()

    for donate, invar in zip(donated_invars, invars):
        if not donate:
            continue
        for outvar in outvars:
            if outvar in donated_outvars:
                continue
            if invar.aval.shape != outvar.aval.shape:
                continue
            donated_outvars.add(outvar)
            donation_mapping[invar] = outvar
            break
        if invar not in donation_mapping:
            logger.warning(
                f"{invar} is marked as donated but actually no match outvar")
    return donation_mapping


def get_donation_mapping_and_modify(stage, reversed_donation_mapping,
                                    gensym_fn):
    invars = set(stage.invars)
    donation_mapping = dict()
    appended_invars = set()
    for var in stage.outvars:
        if var not in reversed_donation_mapping:
            continue
        invar = reversed_donation_mapping[var]
        assert invar.aval.shape == var.aval.shape
        donation_mapping[invar] = var
        if invar not in invars:
            appended_invars.add(invar)
    if not donation_mapping:
        return donation_mapping, stage
    # append invars for donation
    new_invars = list(stage.invars)
    new_outvars = list(stage.outvars)
    new_eqns = list(stage.eqns)
    appended_invars = list(appended_invars)
    if appended_invars:
        new_invars = new_invars + appended_invars
        pipe_start = new_eqns[0]
        new_eqns[0] = mark_pipeline_jaxpreqn(
            pipe_start.invars + appended_invars, pipe_start.outvars +
            list(map(lambda v: gensym_fn(v.aval), appended_invars)),
            pipe_start.params['name'], pipe_start.params['mark_type'])
    # rearrange to keep donated invars and outvars have same index
    new_invars, new_pipe_start = rearrange_vars(new_invars,
                                                list(donation_mapping.keys()),
                                                new_eqns[0], True)
    new_outvars, new_pipe_end = rearrange_vars(new_outvars,
                                               list(donation_mapping.values()),
                                               new_eqns[-1], False)
    new_eqns[0] = new_pipe_start
    new_eqns[-1] = new_pipe_end
    new_stage = JaxPipelineStage(stage.name, new_invars, new_outvars, new_eqns,
                                 stage.consts_dir)
    return donation_mapping, new_stage


def split_donate_invars(donation_mapping, stages: Sequence[JaxPipelineStage]):
    """
    Split donated invars for sliced jaxprs, then rewrite stages.
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
    Returns:
        donate_invars_dict:List[Sequence[bool]]: donate_invars for each stage
    """
    reversed_donation_mapping = {v: k for k, v in donation_mapping.items()}
    gensym_fn = gensym([stage.closed_jaxpr().jaxpr for stage in stages])
    # global last use to consider if the main copy can be discarded

    ans = [None for _ in range(len(stages))]
    new_stages = []

    for stage_idx, stage in enumerate(stages):
        # find donation mapping of the stage
        donation_mapping, new_stage = get_donation_mapping_and_modify(
            stage, reversed_donation_mapping, gensym_fn)
        donated_num = len(donation_mapping)
        ans[stage_idx] = (True,) * donated_num + (False,) * (
            len(new_stage.invars) - donated_num)
        new_stages.append(new_stage)

    return ans, new_stages


@lu.cache
def three_d_parallel_callable(fun: lu.WrappedFun, in_tree, out_tree_thunk,
                              donated_invars, batch_invars, devices,
                              memory_budget_per_device, *avals):
    """End-to-end 3d parallel combining pipelining and sharding."""
    donated_invars = list(donated_invars)
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

    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    gensym_func = gensym([closed_jaxpr.jaxpr])
    compute_grad_jaxpr, apply_grad_jaxpr, barrier = (
        split_compute_grad_and_apply_grad(closed_jaxpr))
    have_apply_grad = barrier is not None

    if have_apply_grad:
        acc_grad_jaxpr, acc_grad_dict, grad_in_to_out = compute_grad_to_accumulate_grad(
            compute_grad_jaxpr, gensym_func)
    else:
        acc_grad_jaxpr = compute_grad_jaxpr
        acc_grad_dict = {}
        grad_in_to_out = {}

    # slice accumulate grad
    acc_grad_invars = acc_grad_jaxpr.jaxpr.invars
    acc_grad_outvars = acc_grad_jaxpr.jaxpr.outvars

    jax_pipeline_layers = slice_closed_jaxpr_by_full_pipeline_marks(
        acc_grad_jaxpr)
    jax_pipeline_layers = mark_missing_vars_in_pipeline_marks(
        jax_pipeline_layers, acc_grad_invars, acc_grad_outvars)
    jax_pipeline_layers = pipeline_dce(jax_pipeline_layers, acc_grad_outvars)
    # initialize donation map
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    if barrier is not None:
        donation_mapping = dict(grad_in_to_out)
    else:
        donation_mapping = dict()

    # For mesh-slicing's profiling, we can use the create_donation_mapping
    # to get a sketchy donation_mapping: only accumulate grad, no applygrad
    if global_config.pipeline_stage_mode == "auto_gpipe":
        # Assume each forward layer corresponds to a backward layer
        assert len(jax_pipeline_layers) % 2 == 0
        num_layers = len(jax_pipeline_layers) // 2
        compute_cost = None
        if global_config.cache_compute_cost is not None:
            compute_cost = np.load(global_config.cache_compute_cost)
        forward_stage_layer_ids, sliced_meshes = get_stage_and_mesh_assignments(
            virtual_mesh,
            jax_pipeline_layers,
            donation_mapping,
            acc_grad_outvars,
            num_micro_batches,
            compute_cost=compute_cost)
        num_forward_stages = len(forward_stage_layer_ids)
        backward_stage_layer_ids = [[
            2 * num_layers - 1 - i for i in reversed(layer_ids)
        ] for layer_ids in reversed(forward_stage_layer_ids)]
        stage_layer_ids = forward_stage_layer_ids + backward_stage_layer_ids
        stage_to_mesh = list(range(num_forward_stages)) + list(
            reversed(range(num_forward_stages)))
        stage_outvars = get_stage_outvars(jax_pipeline_layers, stage_layer_ids,
                                          acc_grad_outvars)
        merged_stages = []
        for stage_id, layer_ids in enumerate(stage_layer_ids):
            stage_layer_jaxprs = [
                jax_pipeline_layers[i].closed_jaxpr() for i in layer_ids
            ]
            stage_name = str(stage_id)
            merged_stage_jaxpr = merge_stage_jaxprs(stage_layer_jaxprs,
                                                    stage_outvars[stage_id],
                                                    stage_name,
                                                    donation_mapping)
            merged_stage = JaxPipelineStage.from_closed_jaxpr(
                stage_name, merged_stage_jaxpr)
            merged_stages.append(merged_stage)
        num_meshes = num_forward_stages
        jax_pipeline_stages = merged_stages
    elif global_config.pipeline_stage_mode == "uniform_layer_gpipe":
        num_acc_grad_stages = len(jax_pipeline_layers)
        stage_to_mesh = {
            i:
            (i if i < num_acc_grad_stages / 2 else num_acc_grad_stages - i - 1)
            for i, _ in enumerate(jax_pipeline_layers)
        }
        assert num_acc_grad_stages % 2 == 0
        num_meshes = num_acc_grad_stages // 2
        jax_pipeline_stages = jax_pipeline_layers
        sliced_meshes = None
    else:
        raise ValueError("Unknown pipeline_stage_mode",
                         global_config.pipeline_stage_mode)

    if have_apply_grad:
        # Process apply gradient:
        # 1. change invars of apply grad to output of accumulate grad
        gradients = [g for g in barrier.outvars if not isinstance(g, DropVar)]
        mask = {
            outv: acc_grad_dict[inv]
            for outv, inv in zip(gradients, barrier.invars)
        }
        # 2. Add compute mean and slice apply-grad stages
        gradvar_to_mesh = mark_gradvar_to_mesh(gradients, jax_pipeline_stages,
                                               stage_to_mesh, mask)
        # FIXME (zhuohan): get_mean only works when we use jax.mean to
        #                  calculate loss. It will fail if we use sum.
        apply_grad_jaxpr, global_outvars = apply_grad_get_mean(
            apply_grad_jaxpr, gradients, gensym_func, num_micro_batches,
            global_outvars)
        sliced_apply_grad, info = slice_apply_gradient(apply_grad_jaxpr,
                                                       gradvar_to_mesh,
                                                       num_meshes)
        apply_deps, apply_grad_schedule, _ = info
        sliced_apply_grad, out_map = apply_grad_add_marker(sliced_apply_grad,
                                                           mask,
                                                           gensym_func,
                                                           stage=True)
        global_outvars = list(
            map(lambda x: get_var_mapping(out_map, x), global_outvars))
        n_stages = len(jax_pipeline_stages) + len(sliced_apply_grad)
        dependency = gen_dependency_with_stages(jax_pipeline_stages,
                                                len(sliced_apply_grad),
                                                apply_deps)
        jax_all_stages = jax_pipeline_stages + sliced_apply_grad

        used_simultaneously = set()
        used = set()
        for stage in sliced_apply_grad:
            used_simultaneously.update(used.intersection(stage.invars))
            used.update(stage.invars)
        for idx, invar in enumerate(global_invars):
            if invar in used_simultaneously and donated_invars[idx]:
                logger.warning(f"cannot donate {invar}")
                donated_invars[idx] = False
    else:
        jax_all_stages = jax_pipeline_stages
        n_stages = len(jax_pipeline_stages)
        dependency = gen_linear_pipeline_dependency(n_stages)
        apply_grad_schedule = {}
    # Generate schedule and placement
    schedule = GpipeSchedule(dependency=dependency,
                             mesh=virtual_mesh,
                             num_pipeline_worker=num_meshes,
                             apply_grad_schedule=apply_grad_schedule,
                             num_batch=num_micro_batches,
                             sliced_meshes=sliced_meshes)
    physical_meshes = []
    n_meshes = len(schedule.meshes)
    for i, mesh in enumerate(schedule.meshes):
        logger.debug("Launch the {}th mesh...".format(i))
        physical_meshes.append(mesh.get_physical_mesh())

    stage_dict = [[] for _ in range(n_meshes)]
    stage_id_dict = [[] for _ in range(n_meshes)]
    # create donation mapping again, as apply gradients are considered
    donation_mapping = create_donation_mapping(donation_mapping, donated_invars,
                                               global_invars, global_outvars)
    donate_invars_dict, jax_all_stages = split_donate_invars(
        donation_mapping, jax_all_stages)
    for i, stage in enumerate(jax_all_stages):
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

    grad_in_to_out = {k: repr(v) for k, v in grad_in_to_out.items()}
    jp = DecentralizedDistributedRuntime(pipeline_stages=xla_stages,
                                         global_invars=global_invars,
                                         grad_dummy_invars=grad_in_to_out,
                                         global_outvars=global_outvars,
                                         physical_meshes=physical_meshes,
                                         dependency=dependency,
                                         schedule=schedule,
                                         is_batch=batch_invars,
                                         num_batch=num_micro_batches)

    def ret_func(*args, **kwargs):
        return jp.run(*args, **kwargs)

    ret_func.get_executable = lambda: jp

    return ret_func  # pylint: disable=unnecessary-lambda
