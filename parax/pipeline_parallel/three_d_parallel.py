"""3D parallel on a Ray cluster."""
import logging

import jax
from jax import linear_util as lu
from jax.core import ClosedJaxpr, gensym
from jax.interpreters import partial_eval as pe

from parax.device_mesh import VirtualMesh
from parax.pipeline_parallel.runtime import (GpipeSchedule, Jax3DPipeline,
                                             gen_linear_pipeline_dependency)
from parax.pipeline_parallel.stage import (
    generate_sharded_xla_stages, mark_global_and_local_vars,
    slice_closed_jaxpr_by_manual_pipeline_marks,
    slice_closed_jaxpr_by_full_pipeline_marks,
    mark_missing_vars_in_pipeline_marks)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@lu.cache
def three_d_parallel_callable(fun: lu.WrappedFun, in_tree, out_tree_thunk,
                              donated_invars, devices, memory_budget_per_device,
                              pipeline_marker_type, *avals):
    """End-to-end 3d parallel combining pipelining and sharding."""
    if not isinstance(devices, VirtualMesh):
        raise RuntimeError("Unrecognized type of `devices`, got: {}, "
                           "expected type: {}.".format(type(devices),
                                                       "VirtualMesh"))

    # Slice the jaxpr into pipeline stages
    virtual_mesh = devices
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    if pipeline_marker_type == "manual":
        gensym_func = gensym([closed_jaxpr.jaxpr])
        jax_pipeline_stages = slice_closed_jaxpr_by_manual_pipeline_marks(
            closed_jaxpr)
        jax_pipeline_stages = [
            mark_global_and_local_vars(stage, gensym_func)
            for stage in jax_pipeline_stages
        ]
    elif pipeline_marker_type == "full":
        jax_pipeline_stages = slice_closed_jaxpr_by_full_pipeline_marks(
            closed_jaxpr)
        jax_pipeline_stages = mark_missing_vars_in_pipeline_marks(
            jax_pipeline_stages, global_invars, global_outvars)
    else:
        raise ValueError("Invalid pipeline marker type", pipeline_marker_type)

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
                       num_batch=num_batch,
                       profile=False)

    return lambda *args, **kwargs: jp.run(*args, **kwargs)  # pylint: disable=unnecessary-lambda
