"""3D parallel on a Ray cluster."""

import jax
from jax import linear_util as lu
from jax.core import ClosedJaxpr, gensym
from jax.interpreters import partial_eval as pe

from parax.device_mesh import VirtualMesh
from parax.pipe import Jax3DPipeline, GpipeSchedule, gen_linear_dependency
from parax.pipeline_stage import PipelineStage, JaxPipelineStage, XlaPipelineStage, generate_sharded_xla_stages, mark_global_and_local_vars, slice_closed_jaxpr_by_pipeline_marks


@lu.cache
def three_d_parallel_callable(
        fun: lu.WrappedFun,
        in_tree,
        out_tree_thunk,
        devices,
        donated_invars,
        memory_budget_per_device,
        *avals
):
    """End-to-end 3d parallel combining pipelining and sharding."""
    if not isinstance(devices, VirtualMesh):
        raise RuntimeError("Unrecognized type of `devices`, got: {}, "
                           "expected type: {}.".format(type(devices), "VirtualMesh"))
    virtual_mesh = devices
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    gensym_func = gensym([closed_jaxpr.jaxpr])
    jax_pipeline_stages = slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr)
    jax_pipeline_stages = [mark_global_and_local_vars(stage, gensym_func) for stage in jax_pipeline_stages]

    num_batch = 1
    n_stages = len(jax_pipeline_stages)
    dependency = gen_linear_dependency(n_stages)
    schedule = GpipeSchedule(mesh=virtual_mesh, num_batch=num_batch)
    physical_meshes = []
    n_meshes = len(schedule.meshes)
    for i, mesh in enumerate(schedule.meshes):
        physical_meshes.append(mesh.get_physical_mesh())

    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    stage_dict = [[] for _ in range(n_meshes)]
    stage_id_dict = [[] for _ in range(n_meshes)]
    for i, stage in enumerate(jax_pipeline_stages):
        mesh_indices = list(schedule.stage_placement(i))
        assert len(mesh_indices) == 1
        mesh_idx = mesh_indices[0]
        stage_id_dict[mesh_idx].append(i)
        stage_dict[mesh_idx].append(stage)

    xla_stages = [None] * n_stages
    for mesh_idx in range(n_meshes):
        sharded_xla_stages = generate_sharded_xla_stages(
            str(mesh_idx), stage_dict[mesh_idx],
            physical_meshes[i].get_default_logical_mesh(), physical_meshes[i],
            memory_budget_per_device=memory_budget_per_device)
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


def mock_slicing_algo(fun, avals, mesh):
    """Slice and generate the stages."""
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    jax_pipeline_stages = slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr)
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    return jax_pipeline_stages, global_invars, global_outvars
