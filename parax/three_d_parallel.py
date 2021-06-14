"""3D parallel on a Ray cluster."""

import jax
from jax import linear_util as lu
from jax.core import ClosedJaxpr
from jax.interpreters import partial_eval as pe

from parax.device_mesh import VirtualMesh
from parax.pipe import Jax3DPipeline
from parax.pipeline_parallel import slice_closed_jaxpr_by_pipeline_marks


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
    # pylint: disable=too-many-arguments
    if not isinstance(devices, VirtualMesh):
        raise RuntimeError("Unrecognized type of `devices`, got: {}, "
                           "expected type: {}.".format(type(devices), "VirtualMesh"))
    virtual_mesh = devices
    jax_pipeline_stages, global_invars, global_outvars = \
        mock_slicing_algo(fun, avals, virtual_mesh)

    sharding_compilation_kwargs = {
        "donated_invars": donated_invars,
        "memory_budget_per_device": memory_budget_per_device
    }
    jp = Jax3DPipeline(pipeline_stages=jax_pipeline_stages,
                       global_invars=global_invars,
                       global_outvars=global_outvars,
                       mesh=virtual_mesh,
                       sharding_compilation_kwargs=sharding_compilation_kwargs)

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
