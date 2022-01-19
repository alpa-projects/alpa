"""Pipeline parallel on a single device."""
from typing import Sequence, Mapping, Any, Dict, List

import jax
from jax import linear_util as lu
from jax._src.util import safe_map
from jax.core import Var, ClosedJaxpr, Literal, gensym
from jax.interpreters import partial_eval as pe
from jax.interpreters.xla import DeviceArray

from parax.device_mesh import PhysicalDeviceMesh
from parax.pipeline_parallel.base_runtime import BaseRuntime
from parax.pipeline_parallel.computation import (
    PipelineComputation, XlaPipelineComputation, XlaShardedPipelineComputation,
    slice_closed_jaxpr_by_full_pipeline_marks,
    mark_missing_vars_in_backward_computation_pipeline_marks)

# pylint: disable=redefined-builtin
unsafe_map, map = map, safe_map  # type: ignore


class LocalPipelineRunner:
    """
    Single-node local pipeline runner.

    Args:
        name (str): pipeline runner name.
        global_invals (List[DeviceArrays]): input device arrays to the stage.

    Returns:
        None
    """

    def __init__(self, name: str, global_invals: List[DeviceArray]):
        self.name = name
        self.env = {}
        self.global_invals = global_invals

    def run_stage(self, stage: PipelineComputation, invals: Dict[Var, Any]):
        """
        Run a pipeline stage.

        Args:
            stage (PipelineComputation): The pipeline stage to run.
            invals (Dict[Var, Any], optional): Input value dict.
        """
        runnable = stage.get_runnable()
        invals_list = []
        for var in stage.invars:
            invals_list.append(invals[var])
        outvals_list = runnable(*invals_list)
        outvals = dict(zip(stage.outvars, outvals_list))
        self.env.update(outvals)

    def get_val(self, var):
        """Get the value of a variable from the env."""
        return self.env[var]

    def del_var(self, var):
        """Delete a variable from the env."""
        del self.env[var]


class LocalRuntime(BaseRuntime):
    """A local pipeline parallel runtime running on a single GPU."""

    def __init__(self,
                 *,
                 pipeline_stages: Sequence[PipelineComputation],
                 global_invars: Sequence[Var],
                 global_outvars: Sequence[Var],
                 physical_meshes: Sequence[PhysicalDeviceMesh] = None,
                 get_hlo_texts=False):
        """
        Return a runtime that runs all pipeline stages on a single local device.

        Args:
            pipeline_stages (Sequence[PipelineComputation]): the pipeline stages to be
                executed.
            global_invars (Sequence[Var]): Global input variables.
            global_outvars (Sequence[Var]): Global output variables.
            physical_meshes (Sequence[PhysicalDeviceMesh]): physical meshes for the runtime.
            get_hlo_texts (bool): Whether to record hlo_texts. If True, input stages
                should be XlaShardedComputation.
        """
        super().__init__(pipeline_stages=pipeline_stages,
                         global_invars=global_invars,
                         global_outvars=global_outvars,
                         physical_meshes=physical_meshes)

        if get_hlo_texts:
            self.hlo_texts_after_spmd_partitioner = []
            for stage in self.stages:
                assert isinstance(stage, XlaShardedPipelineComputation)
                compiled = stage.get_compiled(is_distributed=True)
                hlo_module = compiled.hlo_modules()[0]
                self.hlo_texts_after_spmd_partitioner.append(
                    hlo_module.to_string())

    def run(self, *args, **kwargs):
        """Run function."""
        assert not kwargs, "kwargs not supported"
        global_invals = dict(zip(self.global_invars, args))
        runners = {}

        var_stage_mapping = {}
        var_reference_count = {}

        # Create variable dependency mapping.
        for stage in self.stages:
            for var in stage.invars:
                if var not in global_invals:
                    assert var in var_stage_mapping, f"referred to an unknown var {var}"
                    var_reference_count[var] = var_reference_count.get(var,
                                                                       0) + 1
            for var in stage.outvars:
                var_stage_mapping[var] = stage.name

        for var in self.global_outvars:
            if not isinstance(var, Literal):
                assert var in var_stage_mapping, f"referred to an unknown var {var}"
                var_reference_count[var] = var_reference_count.get(var, 0) + 1

        for stage in self.stages:
            stage_invals = {}
            for var in stage.invars:
                if var in global_invals:
                    stage_invals[var] = global_invals[var]
                else:
                    assert var in var_stage_mapping, f"referred to an unknown var {var}"
                    sender_runner = runners[var_stage_mapping[var]]
                    stage_invals[var] = sender_runner.get_val(var)
                    var_reference_count[var] -= 1
                    if var_reference_count[var] == 0:
                        sender_runner.del_var(var)

            if stage.name not in runners:
                runners[stage.name] = LocalPipelineRunner(
                    stage.name, global_invals)
            runners[stage.name].run_stage(stage, stage_invals)

        global_outvals_list = []
        for var in self.global_outvars:
            if isinstance(var, Literal):
                global_outvals_list.append(var.val)
            else:
                assert var in var_stage_mapping, f"referred to an unknown var {var}"
                sender_runner = runners[var_stage_mapping[var]]
                global_outvals_list.append(sender_runner.get_val(var))
                var_reference_count[var] -= 1
                if var_reference_count[var] == 0:
                    sender_runner.del_var(var)
        return global_outvals_list

    def shutdown(self):
        """Shutdown the pipeline runtime."""
        return

    def get_hlo_text(self, after_spmd_partitioner=True):
        """Return the HLO text for all stages."""
        if after_spmd_partitioner:
            return self.hlo_texts_after_spmd_partitioner
        else:
            ret = []
            for stage in self.stages:
                assert isinstance(stage, XlaShardedPipelineComputation)
                ret.append(stage.get_hlo_text())
            return ret

    def get_executable(self):
        """Get the self as an executable."""
        return self


@lu.cache
def local_pipeline_parallel_callable(fun: lu.WrappedFun,
                                     devices: Mapping[str, Any], *avals):
    """Pipeline parallel callable."""
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    gensym_func = gensym([closed_jaxpr.jaxpr])
    jax_pipeline_stages = slice_closed_jaxpr_by_full_pipeline_marks(
        closed_jaxpr)
    jax_pipeline_stages = mark_missing_vars_in_backward_computation_pipeline_marks(
        jax_pipeline_stages, global_invars, global_outvars, gensym_func)
    xla_pipeline_stages = [
        XlaPipelineComputation.from_jax_pipeline_computation(stage)
        for stage in jax_pipeline_stages
    ]

    local_runtime = LocalRuntime(pipeline_stages=xla_pipeline_stages,
                                 global_invars=global_invars,
                                 global_outvars=global_outvars)

    def ret_func(*args, **kwargs):
        return local_runtime.run(*args, **kwargs)

    ret_func.get_executable = lambda: local_runtime

    return ret_func
