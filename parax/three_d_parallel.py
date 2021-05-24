"""3D parallel on a Ray cluster."""

import jax
from jax import linear_util as lu
from jax.core import ClosedJaxpr
from jax.interpreters import partial_eval as pe
from parax import PhysicalDeviceMesh, LogicalDeviceMesh

from parax.pipe import JaxPipeline
from parax.pipeline_parallel import slice_closed_jaxpr_by_pipeline_marks
from parax.pipeline_stage import XlaShardedPipelineStage
from parax.pipe import GpipeSchedule, _gen_linear_dependency


@lu.cache
def three_d_parallel_callable(
        fun: lu.WrappedFun,
        out_tree_thunk,
        devices,
        donated_invars,
        memory_budget_per_device,
        *avals
):
    """End-to-end 3d parallel combining pipelining and sharding."""
    # parse the device mesh (copied code from auto-sharding)
    physical_mesh = None
    logical_mesh = None
    if isinstance(devices, (list, tuple)):
        physical_mesh = PhysicalDeviceMesh(devices==devices)
        logical_mesh = physical_mesh.get_default_logical_mesh()
    elif isinstance(devices, PhysicalDeviceMesh):
        physical_mesh = devices
        logical_mesh = physical_mesh.get_default_logical_mesh()
    elif isinstance(devices, LogicalDeviceMesh):
        logical_mesh = devices
        physical_mesh = logical_mesh.physical_mesh
    else:
        raise RuntimeError("Unrecognized type of `devices`, got: {}".
                           format(type(devices)))
    # Note(Hao): For now we manually slice the model to get stages.
    # Later we shall run a scheduling algorithm (signature below) to get the stages.
    jax_pipeline_stages, global_invars, global_outvars = \
        mock_slicing_algo(fun, avals, physical_mesh, logical_mesh)

    # Slice mesh based on the stages
    meshes = slice_mesh(physical_mesh, len(jax_pipeline_stages))

    # convert JaxPipelineStage to XLAshardedStage:
    xla_sharded_pipeline_stages = \
        [XlaShardedPipelineStage.from_jax_pipeline_stage(stage, meshes[i], donated_invars, memory_budget_per_device)
         for i, stage in enumerate(jax_pipeline_stages)]

    jp = JaxPipeline(pipeline_stages=xla_sharded_pipeline_stages,
                     global_invars=global_invars,
                     global_outvars=global_outvars)
    return lambda *args, **kwargs: jp.run(*args, **kwargs)  # pylint: disable=unnecessary-lambda


def mock_slicing_algo(fun, avals, physical_mesh, logical_mesh):
    """Slice and generate the stages."""
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    jax_pipeline_stages = slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr)
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    return jax_pipeline_stages, global_invars, global_outvars

def slice_mesh(physical_mesh, num_stage):
    """TODO"""
    sliced_meshes = []
    return sliced_meshes