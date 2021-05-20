"""3D parallel on a Ray cluster."""

import jax
from jax import linear_util as lu
from jax.core import ClosedJaxpr
from jax.interpreters import partial_eval as pe
from parax import SingleHostDeviceMesh, MultiHostDeviceMesh, LogicalDeviceMesh

from parax.pipe import JaxPipeline
from parax.pipeline_parallel import slice_closed_jaxpr_by_pipeline_marks
from parax.pipeline_stage import XlaPipelineStage
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
        physical_mesh = SingleHostDeviceMesh(devices)
        logical_mesh = physical_mesh.get_default_logical_mesh()
    elif isinstance(devices, MultiHostDeviceMesh):
        physical_mesh = devices
        logical_mesh = physical_mesh.get_default_logical_mesh()
    elif isinstance(devices, LogicalDeviceMesh):
        logical_mesh = devices
        physical_mesh = logical_mesh.physical_mesh
    if isinstance(physical_mesh, MultiHostDeviceMesh):
        distributed_compilation_head = True

    # Note(Hao): For now we manually slice the model to get stages.
    # Later we shall run a scheduling algorithm (signature below) to get the stages.
    xla_pipeline_stages, global_invars, global_outvars = \
        mock_slicing_algo(fun, avals, physical_mesh, logical_mesh)

    auto_sharding_callable_args = {
        "fun": fun,
        "out_tree_thunk": out_tree_thunk,
        "devices": devices,
        "donated_invars": donated_invars,
        "memory_budget_per_device": memory_budget_per_device,
        **avals
    }

    jp = JaxPipeline(pipeline_stages=xla_pipeline_stages,
                     global_invars=global_invars,
                     global_outvars=global_outvars,
                     physical_mesh=physical_mesh,
                     logical_mesh=logical_mesh,
                     auto_sharding_callable_args=auto_sharding_callable_args)
    return lambda *args, **kwargs: jp.run(*args, **kwargs)  # pylint: disable=unnecessary-lambda


def mock_slicing_algo(fun, avals, physical_mesh, logical_mesh):
    """Slice and generate the stages."""
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    jax_pipeline_stages = slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr)
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    xla_pipeline_stages = [XlaPipelineStage.from_jax_pipeline_stage(stage)
                           for stage in jax_pipeline_stages]
    return xla_pipeline_stages, global_invars, global_outvars
