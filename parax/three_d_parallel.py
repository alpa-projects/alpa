"""3D parallel on a Ray cluster."""

import jax
from jax import linear_util as lu
from jax.core import ClosedJaxpr
from jax.interpreters import partial_eval as pe
from jax.lib import xla_bridge as xb
from parax.device_mesh import PhysicalDeviceMesh, LogicalDeviceMesh, VirtualMesh

from parax.pipe import JaxPipeline, Jax3DPipeline
from parax.pipeline_parallel import slice_closed_jaxpr_by_pipeline_marks
from parax.pipeline_stage import XlaShardedPipelineStage
from parax.pipe import GpipeSchedule, _gen_linear_dependency


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
    # parse the device mesh (copied code from auto-sharding)
    # physical_mesh = None
    # logical_mesh = None
    # if devices is None:
    #     # physical_mesh = SingleHostDeviceMesh(xb.devices())
    #     physical_mesh = PhysicalDeviceMesh(devices=xb.devices())
    #     logical_mesh = physical_mesh.get_default_logical_mesh()
    # if isinstance(devices, (list, tuple)):
    #     physical_mesh = PhysicalDeviceMesh(devices==devices)
    #     logical_mesh = physical_mesh.get_default_logical_mesh()
    # elif isinstance(devices, PhysicalDeviceMesh):
    #     physical_mesh = devices
    #     logical_mesh = physical_mesh.get_default_logical_mesh()
    # elif isinstance(devices, LogicalDeviceMesh):
    #     logical_mesh = devices
    #     physical_mesh = logical_mesh.physical_mesh

    if not isinstance(devices, VirtualMesh):
        raise RuntimeError("Unrecognized type of `devices`, got: {}, expected: {}.".
                           format(type(devices), "VirtualMesh"))
    virtual_mesh = devices
    # Note(Hao): For now we manually slice the model to get stages.
    # Later we shall run a scheduling algorithm (signature below) to get the stages.
    jax_pipeline_stages, global_invars, global_outvars = \
        mock_slicing_algo(fun, avals, virtual_mesh)

    # some temporary params
    dependency = _gen_linear_dependency(len(jax_pipeline_stages))
    # Gpipe will slice the mesh.
    gpipe_schedule = GpipeSchedule(dependency=dependency,
                                   mesh=virtual_mesh)
    meshes = gpipe_schedule.meshes

    # convert JaxPipelineStage to XLAshardedStage:
    xla_sharded_pipeline_stages = \
        [XlaShardedPipelineStage.from_jax_pipeline_stage(stage, meshes[i], donated_invars, memory_budget_per_device)
         for i, stage in enumerate(jax_pipeline_stages)]
    jp = Jax3DPipeline(pipeline_stages=xla_sharded_pipeline_stages,
                       global_invars=global_invars,
                       global_outvars=global_outvars,
                       mesh=virtual_mesh,
                       dependency=dependency,
                       schedule=gpipe_schedule)

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
