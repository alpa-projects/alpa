"""Pipeline parallel on a single device. This is only used for debugging."""
from typing import Sequence, Any, Dict

import jax
from jax import linear_util as lu
from jax.core import Var, ClosedJaxpr, Literal, gensym
from jax.interpreters import partial_eval as pe
from jax.interpreters.xla import DeviceArray

from alpa.pipeline_parallel.computation import (
    PipelineComputation, XlaPipelineComputation,
    slice_closed_jaxpr_by_full_pipeline_marks,
    mark_missing_vars_in_backward_computation_pipeline_marks)


class LocalPipelineRunner:
    """Single-device local pipeline runner."""

    def __init__(self, name: str, global_invals: Sequence[DeviceArray]):
        self.name = name
        self.env = {}
        self.global_invals = global_invals

    def run_stage(self, stage: PipelineComputation, invals: Dict[Var, Any], swap: bool):
        """
        Run a pipeline stage.

        Args:
            stage (PipelineComputation): The pipeline stage to run.
            invals (Dict[Var, Any], optional): Input value dict.
        """
        runnable = stage.get_runnable()
        if swap:
            cpu = jax.devices(backend="cpu")[0]
            gpu = jax.devices(backend="gpu")[0]
            invals_list = []
            for var in stage.invars:
                cpu_val = invals[var]
                gpu_val = jax.device_put(cpu_val, device=gpu)
                invals_list.append(gpu_val)
            outvals_list = runnable(*invals_list)
            gpu_outvals = dict(zip(stage.outvars, outvals_list))
            outvals = {}
            for var, val in gpu_outvals.items():
                outvals[var] = jax.device_put(val, device=cpu)
        else:
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


class LocalPipelineExecutable:
    """A pipeline parallel executable running on a single local device.

    Args:
        stages (Sequence[PipelineComputation]): the pipeline stages to be
            executed.
        global_invars (Sequence[Var]): Global input variables.
        global_outvars (Sequence[Var]): Global output variables.
    """

    def __init__(self, *, stages: Sequence[PipelineComputation],
                 global_invars: Sequence[Var], global_outvars: Sequence[Var],
                 swap: bool):
        self.stages = stages
        self.global_invars = global_invars
        self.global_outvars = global_outvars
        self.swap = swap

    def launch_on_driver(self, *args):
        """Run function."""
        if self.swap:
            # Move all tensors to CPU memory.
            cpu = jax.devices(backend="cpu")[0]
            cpu_args = []
            for a in args:
                if isinstance(a, jax.numpy.DeviceArray) and a.device_buffer.device() != cpu:
                    cpu_args.append(a)
                else:
                    cpu_args.append(jax.device_put(a, device=cpu))
            del args
            global_invals = dict(zip(self.global_invars, cpu_args))
        else:
            global_invals = dict(zip(self.global_invars, args))
        runners = {}

        var_stage_mapping = {}
        var_reference_count = {}

        # Create variable dependency mapping.
        for stage in self.stages:
            for var in stage.invars:
                if var not in global_invals:
                    assert var in var_stage_mapping, (
                        f"referred to an unknown var {var}")
                    var_reference_count[var] = var_reference_count.get(var,
                                                                       0) + 1
            for var in stage.outvars:
                var_stage_mapping[var] = stage.name

        for var in self.global_outvars:
            if not isinstance(var, Literal):
                assert var in var_stage_mapping, (
                    f"referred to an unknown var {var}")
                var_reference_count[var] = var_reference_count.get(var, 0) + 1

        for stage in self.stages:
            stage_invals = {}
            for var in stage.invars:
                if var in global_invals:
                    stage_invals[var] = global_invals[var]
                else:
                    assert var in var_stage_mapping, (
                        f"referred to an unknown var {var}")
                    sender_runner = runners[var_stage_mapping[var]]
                    stage_invals[var] = sender_runner.get_val(var)
                    var_reference_count[var] -= 1
                    if var_reference_count[var] == 0:
                        sender_runner.del_var(var)

            if stage.name not in runners:
                runners[stage.name] = LocalPipelineRunner(
                    stage.name, global_invals)
            runners[stage.name].run_stage(stage, stage_invals, self.swap)

        global_outvals_list = []
        for var in self.global_outvars:
            if isinstance(var, Literal):
                global_outvals_list.append(var.val)
            else:
                assert var in var_stage_mapping, (
                    f"referred to an unknown var {var}")
                sender_runner = runners[var_stage_mapping[var]]
                global_outvals_list.append(sender_runner.get_val(var))
                var_reference_count[var] -= 1
                if var_reference_count[var] == 0:
                    sender_runner.del_var(var)
        
        # mesh = get_global_physical_mesh(create_if_not_exist=True)
        # print(mesh.get_memory_allocated())
        # print(mesh.get_max_memory_allocated())
        # print(mesh.get_available_memory())

        # jax.profiler.save_device_memory_profile(
        #     f"/home/dlzou/projects/alpa-experiments/swap_{self.swap}.prof")
        return global_outvals_list


def compile_local_pipeline_executable(fun: lu.WrappedFun, swap: bool, *avals):
    """Compile a local pipeline executable that only runs on a single device."""
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    gensym_func = gensym([closed_jaxpr.jaxpr])
    jax_pipeline_stages = slice_closed_jaxpr_by_full_pipeline_marks(
        closed_jaxpr)
    jax_pipeline_stages = (
        mark_missing_vars_in_backward_computation_pipeline_marks(
            jax_pipeline_stages, global_invars, global_outvars, gensym_func))
    xla_pipeline_stages = [
        XlaPipelineComputation.from_jax_pipeline_computation(stage)
        for stage in jax_pipeline_stages
    ]

    return LocalPipelineExecutable(stages=xla_pipeline_stages,
                                   global_invars=global_invars,
                                   global_outvars=global_outvars,
                                   swap=swap)
