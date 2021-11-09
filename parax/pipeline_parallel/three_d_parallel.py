"""Generate callables for 3d parallel that combines pipelining and 2d sharding."""
import logging
from typing import Sequence

import jax
from jax import linear_util as lu
from jax.core import ClosedJaxpr, DropVar, Jaxpr, gensym
from jax.interpreters import partial_eval as pe
import numpy as np

from parax.device_mesh import VirtualMesh
from parax.global_env import global_config
from parax.pipeline_parallel.decentralized_distributed_runtime import DecentralizedDistributedRuntime
from parax.pipeline_parallel.centralized_distributerd_runtime import (
    CentralizedDistributedRuntime)
from parax.pipeline_parallel.schedules import (GpipeSchedule,
                                               gen_dependency_with_stages)
from parax.pipeline_parallel.computation import (
    create_donation_mapping, generate_sharded_xla_computations,
    mark_missing_vars_in_pipeline_marks, pipeline_dce,
    slice_closed_jaxpr_by_full_pipeline_marks, split_donate_invars)
from parax.pipeline_parallel.apply_grad import (
    compute_grad_to_accumulate_grad, process_apply_gradient,
    split_compute_grad_and_apply_grad)
from parax.pipeline_parallel.stage_construction import cluster_layers_and_slice_mesh
from parax.util import get_micro_batch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@lu.cache
def three_d_parallel_callable(fun: lu.WrappedFun, in_tree, out_tree_thunk,
                              donated_invars, batch_invars, devices,
                              memory_budget_per_device, *avals):
    """3d parallel combining pipelining and 2d sharding."""

    if not isinstance(devices, VirtualMesh):
        raise RuntimeError("Unrecognized type of `devices`, got: {}, "
                           "expected type: {}.".format(type(devices),
                                                       "VirtualMesh"))

    # Trace the function to get the jaxpr
    virtual_mesh = devices
    num_micro_batches = global_config.num_micro_batches
    if num_micro_batches is None:
        logger.warning("num microbatch is unset. Use 1 by default.")
        num_micro_batches = 1
    microbatch_avals = get_micro_batch(batch_invars, num_micro_batches, *avals)
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, microbatch_avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)

    # Split the jaxpr into compute_grad and apply_grad
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

    # Slice the jaxpr into layers
    acc_grad_invars = acc_grad_jaxpr.jaxpr.invars
    acc_grad_outvars = acc_grad_jaxpr.jaxpr.outvars

    jax_pipeline_layers = slice_closed_jaxpr_by_full_pipeline_marks(
        acc_grad_jaxpr)
    jax_pipeline_layers = mark_missing_vars_in_pipeline_marks(
        jax_pipeline_layers, acc_grad_invars, acc_grad_outvars)
    jax_pipeline_layers = pipeline_dce(jax_pipeline_layers, acc_grad_outvars)

    # Initialize donation map
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    if have_apply_grad:
        donation_mapping = dict(grad_in_to_out)
    else:
        donation_mapping = dict()

    # Construct pipeline stages by merging layers
    jax_pipeline_stages, stage_to_mesh, sliced_meshes = (
        cluster_layers_and_slice_mesh(
            jax_pipeline_layers,
            virtual_mesh,
            donation_mapping,
            acc_grad_outvars,
            num_micro_batches,
            pipeline_stage_mode=global_config.pipeline_stage_mode,
            cache_compute_cost=global_config.cache_compute_cost,
            forward_stage_layer_ids=global_config.forward_stage_layer_ids,
            submesh_shapes=global_config.sub_physical_mesh_shapes))
    num_meshes = len(sliced_meshes)

    # Process apply_gradient and donation
    if have_apply_grad:
        jax_all_stages, n_stages, dependency, apply_grad_placement, global_outvars, donated_invars =\
            process_apply_gradient(apply_grad_jaxpr,
                barrier, acc_grad_dict, jax_pipeline_stages, stage_to_mesh,
                gensym_func, num_micro_batches, num_meshes,
                global_invars, global_outvars, donated_invars)
    else:
        jax_all_stages = jax_pipeline_stages
        n_stages = len(jax_pipeline_stages)
        dependency = gen_dependency_with_stages(jax_pipeline_stages)
        apply_grad_placement = {}

    donation_mapping = create_donation_mapping(donation_mapping, donated_invars,
                                               global_invars, global_outvars)
    donate_invars_dict, jax_all_stages = split_donate_invars(
        donation_mapping, jax_all_stages)

    # Generate pipeline schedule and placement
    schedule = GpipeSchedule(dependency=dependency,
                             sliced_meshes=sliced_meshes,
                             apply_grad_placement=apply_grad_placement,
                             num_batch=num_micro_batches)
    physical_meshes = []
    for i, mesh in enumerate(schedule.meshes):
        logger.debug("Launch the {}th mesh...".format(i))
        physical_meshes.append(mesh.get_physical_mesh())

    stage_dict = [[] for _ in range(num_meshes)]
    stage_id_dict = [[] for _ in range(num_meshes)]
    for i, stage in enumerate(jax_all_stages):
        mesh_indices = list(schedule.stage_placement(i))
        assert len(mesh_indices) == 1
        mesh_idx = mesh_indices[0]
        stage_id_dict[mesh_idx].append(i)
        stage_dict[mesh_idx].append(stage)

    # address logical mesh requirement by users
    slms = global_config.sub_logical_mesh_shapes
    if slms != None:
        assert len(slms) == len(global_config.sub_physical_mesh_shapes)
        assert all(np.prod(slms[i]) == np.prod(global_config.sub_physical_mesh_shapes[i])
                   for i in range(num_meshes))

    # Call auto-sharding pass to shard each stage
    xla_stages = [None] * n_stages
    for mesh_idx in range(num_meshes):
        physical_mesh = physical_meshes[mesh_idx]
        if global_config.sub_logical_mesh_shapes[mesh_idx]:
            # set to a user-required logical mesh shape
            # e.g. [1, 4] physical mesh could produce a [2, 2] logical mesh
            logical_mesh_choices = [physical_mesh.get_logical_mesh(slms[mesh_idx])]
        else:
            # logical mesh shape == physical mesh shape
            logical_mesh_choices = [physical_mesh.get_default_logical_mesh()]
        logical_mesh_search_mode = "cost_model"
        stage_donate_invars = [
            donate_invars_dict[stage_idx]
            for stage_idx in stage_id_dict[mesh_idx]
        ]
        search_task = None
        record_file = None
        sharded_xla_stages = generate_sharded_xla_computations(
            str(mesh_idx), stage_dict[mesh_idx], stage_donate_invars,
            physical_mesh, logical_mesh_choices, logical_mesh_search_mode,
            memory_budget_per_device, acc_grad_outvars, search_task,
            record_file)
        for i, xla_stage in zip(stage_id_dict[mesh_idx], sharded_xla_stages):
            xla_stages[i] = xla_stage

    # Wrap all things into a distributed runtime
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
