"""Pipeline parallel on a single device. This is only used for debugging."""
from abc import ABC, abstractmethod
from collections import OrderedDict
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

    def __init__(self, name: str, global_invals: Dict[Var, DeviceArray]):
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

    def place_var(self, var, device):
        # var may have been freed, in which case we do nothing.
        if var in self.env:
            val = self.env[var]
            self.env[var] = jax.device_put(val, device=device)
            return val.nbytes
        return 0

    def del_var(self, var):
        """Delete a variable from the env."""
        del self.env[var]


class LocalSwapManager(ABC):
    
    def __init__(
        self,
        global_invals: Dict[Var, DeviceArray],
        stages: Sequence[PipelineComputation],
        var_stage_mapping: Dict[Var, PipelineComputation],
        runners: Dict[PipelineComputation, LocalPipelineRunner],
        logging,
    ):
        self.global_invals = global_invals
        self.var_stage_mapping = var_stage_mapping
        self.runners = runners
        self.log = None
        if logging:
            self.log = []

        self.host = jax.local_devices(backend="cpu")[0]
        self.device = jax.local_devices(backend="gpu")[0]
        self.swap_out = {}
        self.swap_in = {}
        self._compute_swap_strategy(stages)
   
    @abstractmethod
    def _compute_swap_strategy(self, stages: Sequence[PipelineComputation]):
        raise NotImplementedError()

    def swap(self, stage_name: str):
        for var in self.swap_out[stage_name]:
            if var in self.global_invals:
                val = self.global_invals[var]
                self.global_invals[var] = jax.device_put(val, device=self.host)
                bytes = val.nbytes
            else:
                assert var in self.var_stage_mapping, (
                    f"cannot swap out unknown var {var}")
                runner = self.runners[self.var_stage_mapping[var]]
                bytes = runner.place_var(var, self.host)
            if self.log is not None:
                self.log.append(("out", var, bytes))
        for var in self.swap_in[stage_name]:
            if var in self.global_invals:
                val = self.global_invals[var]
                self.global_invals[var] = jax.device_put(val, device=self.host)
                bytes = val.nbytes
            else:
                assert var in self.var_stage_mapping, (
                    f"cannot swap in unknown var {var}")
                runner = self.runners[self.var_stage_mapping[var]]
                bytes = runner.place_var(var, self.device)
            if self.log is not None:
                self.log.append(("in", var, bytes))
    
    def view_log(self):
        s = ""
        for entry in self.log:
            s += "\t".join([str(e) for e in entry]) + "\n"
        return s


class LocalSimpleSwapManager(LocalSwapManager):
    """
    Swap in all tensors needed to run a stage. Swap all tensors out once the stage ends.
    """

    def __init__(
        self,
        global_invals: Dict[Var, DeviceArray],
        stages: Sequence[PipelineComputation],
        var_stage_mapping: Dict[Var, PipelineComputation],
        runners: Dict[PipelineComputation, LocalPipelineRunner],
        logging=False,
    ):
        super().__init__(global_invals, stages, var_stage_mapping, runners, logging)

        # Ensure all inputs start in host memory.
        for var, val in global_invals.copy().items():
            if isinstance(val, jax.numpy.DeviceArray) \
                and val.device_buffer.device() != self.host:
                global_invals[var] = jax.device_put(val, self.host)
        
        # Swap in tensors needed in first stage.
        for var in stages[0].invars:
            assert var in global_invals, (
                f"{var} not in global invars"
            )
            val = global_invals[var]
            global_invals[var] = jax.device_put(val, device=self.host)
            if self.log is not None:
                self.log.append(("in", var, val.nbytes))
    
    def _compute_swap_strategy(self, stages):
        """
        Order: execute stage -> swap out this stage -> swap in next stage
        """
        for i in range(len(stages) - 1):
            self.swap_out[stages[i].name] = stages[i].invars + stages[i].outvars
            self.swap_in[stages[i].name] = stages[i + 1].invars
        last_stage = stages[len(stages) - 1]
        self.swap_out[last_stage.name] = last_stage.invars + last_stage.outvars
        self.swap_in[last_stage.name] = []


class SwapValue:
    def __init__(self, val: Any, priority: int):
        self.val = val
        self.priority = priority
    
    def __lt__(self, other):
        return self.priority < other.priority


class LocalLRUSwapManager(LocalSwapManager):

    def __init__(
        self,
        global_invals: Dict[Var, DeviceArray],
        stages: Sequence[PipelineComputation],
        var_stage_mapping: Dict[Var, PipelineComputation],
        runners: Dict[PipelineComputation, LocalPipelineRunner],
    ):
        super().__init__(global_invals, stages, var_stage_mapping, runners)

        # Somehow analyze stages to decide allocation pool
        self.pool = OrderedDict() # size_class: queue_of_vals
        self.pool[1000] = []
        self.val_size_map = {}

        # Swap in tensors needed by first stage
    
    # def swap_in(self, val: DeviceArray, priority: int):
    #     for sc in self.pool:
    #         if sc >= val.bytes:
    #             break
    #     sval = SwapValue(val, priority)
    #     heapq.heappush(self.pool[sc], sval)
    #     if self.log is not None:
    #         self.log.append((f"in", val.bytes))
    #     return jax.device_put(val, self.gpu)
    
    # def swap_out(self, val: DeviceArray):
    #     pass
    

class LocalMinSwapManager(LocalSwapManager):

    def __init__(self, logging=False):
        pass
    
    def _compute_swap_strategy(self, stages):
        """
        Ideas:
            1. Belady's strategy: swap out tensors not needed for the longest time.
            2. A var can only be swapped after the stage where it's created.
            3. Avoid swapping in too early because it might be swapped out again.
            4. Account for reference counting.
        """
        pass
    

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
        global_invals = dict(zip(self.global_invars, args))
        runners = {}

        var_stage_mapping = {}
        var_reference_count = {}

        swap_manager = None
        if self.swap:
            swap_manager = LocalSimpleSwapManager(
                global_invals,
                self.stages,
                var_stage_mapping,
                runners,
                logging=True,
            )

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
            runners[stage.name].run_stage(stage, stage_invals)
            if swap_manager:
                swap_manager.swap(stage.name)
        
        # Examine swap patterns.
        print(swap_manager.view_log())

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
