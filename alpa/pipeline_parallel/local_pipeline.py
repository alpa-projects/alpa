"""Pipeline parallel on a single device. This is only used for debugging."""
from math import inf
from typing import Sequence, Any, Dict, Optional
import itertools

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

    def set_val(self, var, val):
        """Set the value of a variable in the env. Only use for swapping."""
        self.env[var] = val

    def del_var(self, var):
        """Delete a variable from the env."""
        del self.env[var]


class SwapElement:
    def __init__(self, var: Optional[Var], priority=0, in_stage=-1):
        self.var = var
        self.priority = priority
        self.in_stage = in_stage
    
    def __lt__(self, other):
        return self.priority < other.priority

    def __eq__(self, other):
        return self.var == other.var


class SwapPool:
    """A queue for simulating a pool of variables on device."""

    def __init__(self, max_size: int):
        self.queue = []
        assert max_size > 0, "max_size must be greater than 0"
        self.max_size = max_size

    def alloc(self, var: Var, priority: int, in_stage: int, max_out_stage: int):
        """Simulating swapping in a variable for a stage.
        Args:
            var: the variable to swap in.
            priority: lower means earlier to swap out.
            in_stage: stage for whch var is swapped in.
            max_prev_stage: don't swap out if variable in_stage is greater.
        Returns:
            swap_in: whether var actually is swapped in.
            swap_out_var: variable that is swapped out.
        """
        swap_in = True
        swap_out_var = None
        for i in range(len(self.queue)):
            if self.queue[i].var == var:
                # Re-insert var into queue with new stats.
                self.queue.pop(i)
                swap_in = False
                break
        if swap_in and len(self.queue) == self.max_size:
            # Pick a variable to swap out.
            i = 0
            out_element = None
            while out_element is None:
                out_element = self.queue[i]
                if out_element.in_stage > max_out_stage:
                    out_element = None
                    i += 1
                    assert i == len(self.queue), f"SwapPool can't allocate for {var}"
            self.queue.remove(out_element)
            swap_out_var = out_element.var

        # Insert new element in order of priority.
        in_element = SwapElement(var, priority, in_stage)
        i = 0
        while i < len(self.queue):
            if in_element < self.queue[i]:
                self.queue.insert(i, in_element)
                break
            i += 1
        if i == len(self.queue):
            self.queue.append(in_element)
        
        return swap_in, swap_out_var

    def free(self, var: Var):
        for i in range(len(self.queue)):
            if self.queue[i].var == var:
                # Re-insert var into queue with new stats.
                self.queue.pop(i)
                break
            

def view_swap_log(log):
        s = ""
        for entry in log:
            s += "\t".join([str(e) for e in entry]) + "\n"
        return s


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
                 swap: Optional[str]=None, swap_args={}):
        self.stages = stages
        self.global_invars = global_invars
        self.global_outvars = global_outvars
        if swap is None or swap == "none":
            self.launch_fn = self._no_swap_launch_on_driver
        elif swap == "basic":
            self.launch_fn = self._basic_swap_launch_on_driver
        elif swap == "preempt":
            self.launch_fn = self._preempt_swap_launch_on_driver
        elif swap == "min":
            self.max_size = swap_args.get("max_size", 12)
            self.launch_fn = self._min_swap_launch_on_driver
        else:
            raise ValueError(f"swap parameter does not support {swap}")

    def launch_on_driver(self, *args):
        return self.launch_fn(*args)

    def _no_swap_launch_on_driver(self, *args):
        """Run function."""
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
            runners[stage.name].run_stage(stage, stage_invals)

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

    def _basic_swap_launch_on_driver(self, *args):
        """Run function."""
        # Move all tensors to CPU memory.
        host = jax.devices(backend="cpu")[0]
        device = jax.devices(backend="gpu")[0]
        host_args = []
        for a in args:
            if isinstance(a, DeviceArray) and a.device_buffer.device() == host:
                host_args.append(a)
            else:
                host_args.append(jax.device_put(a, device=host))
        del args
        global_invals = dict(zip(self.global_invars, host_args))
        runners = {}

        var_stage_mapping = {}
        var_reference_count = {}
        swap_log = []

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

            # Swap in all arrays inputted in this stage.
            stage_invals_swapped = {}
            for var, val in stage_invals.items():
                stage_invals_swapped[var] = jax.device_put(val, device=device)
                swap_log.append(("in", var, val.nbytes))

            # Execute stage.
            if stage.name not in runners:
                runners[stage.name] = LocalPipelineRunner(
                    stage.name, global_invals)
            runners[stage.name].run_stage(stage, stage_invals_swapped)
            
            # Swap out all arrays outputted from this stage.
            for var in stage_invals_swapped:
                swap_log.append(("out", var, stage_invals_swapped[var].nbytes))
            del stage_invals_swapped

            runner = runners[stage.name]
            for var in runner.env:
                swap_out_val = runner.get_val(var)
                runner.set_val(var, jax.device_put(swap_out_val, device=host))
                swap_log.append(("out", var, swap_out_val.nbytes))
        
        # Examine swap patterns.
        # print(view_swap_log(swap_log))

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

    def _preempt_swap_launch_on_driver(self, *args):
        """Run function."""
        host = jax.devices(backend="cpu")[0]
        device = jax.devices(backend="gpu")[0]
        global_invals = dict(zip(self.global_invars, args))
        runners = {}

        var_stage_mapping = {}
        var_reference_count = {}
        swap_log = []
        
        print(self.global_invars)
        print(self.global_outvars)

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
        
        def on_device(val: Any, device: Any):
            return isinstance(val, DeviceArray) and val.device_buffer.device() == device

        def swap_var(var: Var, stage_index: int, direction: str, device: Any):
            if var in global_invals:
                val = global_invals[var]
                if on_device(val, device):
                    return
                global_invals[var] = jax.device_put(val, device=device)
                swap_log.append((f"{stage_index}:{direction}", var, val.nbytes))
            elif var in var_stage_mapping:
                sender_runner = runners[var_stage_mapping[var]]
                val = sender_runner.get_val(var)
                if on_device(val, device):
                    return
                sender_runner.set_val(var, jax.device_put(val, device=device))
                swap_log.append((f"{stage_index}:{direction}", var, val.nbytes))
            elif isinstance(var, Literal):
                if on_device(val, device):
                    return
                var.val = jax.device_put(var.val, device=device)
                swap_log.append((f"{stage_index}:{direction}", var, var.val.nbytes))
            else:
                raise ValueError(f"refereed to an anknown var {var}") 

        # Swap out all global_invals not used in first stage.
        for var in global_invals:
            if var not in self.stages[0].invars:
                swap_var(var, -1, "out", device)

        # Swap in the first stage's invars.
        for var in self.stages[0].invars:
            swap_var(var, -1, "in", device)

        for i, stage in enumerate(self.stages):
            # Preemptive swapping for next stage.
            if i < len(self.stages) - 1:
                for var in self.stages[i + 1].invars:
                    if var in stage.outvars:
                        continue
                    swap_var(var, i, "in", device)

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
            
            # Swap out variables that are not used next stage.
            if i < len(self.stages) - 1:
                for var in stage.outvars:
                    if var not in self.stages[i + 1].invars:
                        swap_var(var, i, "out", host)

        global_outvals_list = []
        for var in self.global_outvars:
            # Swap all outvals into device.
            swap_var(var, len(self.stages), "in", device)
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

        # Examine swap patterns.
        print(view_swap_log(swap_log))

        return global_outvals_list

    def _min_swap_launch_on_driver(self, *args):
        """Run function.
        
        DISCLAIMER 1: premise may be flawed because intermediate vars not accounted.
        DISCLAIMER 2: implementation may be buggy.
        """
        host = jax.devices(backend="cpu")[0]
        device = jax.devices(backend="gpu")[0]
        global_invals = dict(zip(self.global_invars, args))
        runners = {}

        var_stage_mapping = {}
        var_reference_count = {}

        # Data structures used for OPT swapping strategy.
        pool = SwapPool(self.max_size)
        var_access_map = {} # List of stage indices where var is accessed.
        var_access_index = {} # Indexes most recent access in var_access_map.
        swap_log = []

        # Create variable mappings.
        for i, stage in enumerate(self.stages):
            for var in stage.invars:
                if var not in global_invals:
                    assert var in var_stage_mapping, (
                        f"referred to an unknown var {var}")
                    var_reference_count[var] = var_reference_count.get(var,
                                                                       0) + 1
                if var in var_access_map:
                    var_access_map[var].append(i)
                else:
                    var_access_map[var] = [i]
                    var_access_index[var] = 0

            for var in stage.outvars:
                var_stage_mapping[var] = stage.name
                
        for var in var_access_map: #
            print(f"{var}: {var_access_map[var]}")

        # Fill pool based on current array placement.
        for var, val in global_invals.items():
            if isinstance(val, DeviceArray) and val.device_buffer.device() == device:
                # Use number of stages until first access to compute priority.
                pool.alloc(var, -var_access_map[var][0], -1, -1)

        for var in self.global_outvars:
            if not isinstance(var, Literal):
                assert var in var_stage_mapping, (
                    f"referred to an unknown var {var}")
                var_reference_count[var] = var_reference_count.get(var, 0) + 1

        def swap_var(var: Var, stage: int, direction: str, device: Any):
            if var in global_invals:
                val = global_invals[var]
                global_invals[var] = jax.device_put(val, device=device)
                swap_log.append((f"{stage}:{direction}", var, val.nbytes))
            else:
                assert var in var_stage_mapping, (
                    f"referred to an unknown var {var}"
                )
                sender_runner = runners[var_stage_mapping[var]]
                val = sender_runner.get_val(var)
                sender_runner.set_val(var, jax.device_put(val, device=device))
                swap_log.append((f"{stage}:{direction}", var, val.nbytes))

        def update_pool(var: Var, stage_index: int, computed=True):
            # Use number of stages until next access to compute priority.
            if var not in var_access_index:
                return None, None
            next_index = var_access_index[var] + 1
            if next_index < len(var_access_map[var]):
                swap_in, swap_out_var = pool.alloc(
                    var, (stage_index + 1) - var_access_map[var][next_index],
                    in_stage=stage_index+1, max_out_stage=stage_index-1
                )
                var_access_index[var] = next_index
            else:
                swap_in, swap_out_var = pool.alloc(
                    var, -inf, in_stage=stage_index+1, max_out_stage=stage_index-1
                )
            swap_in_var = var if swap_in and computed else None
            return swap_in_var, swap_out_var
        
        def perform_swap(swap_in_var: Optional[Var], swap_out_var: Optional[Var], stage_index: int):
            if swap_out_var is not None:
                swap_var(swap_out_var, stage_index, "out", host)
            if swap_in_var is not None:
                swap_var(swap_in_var, stage_index, "in", device)

        # Immediately swap in the first stage's invars.
        for var in self.stages[0].invars:
            perform_swap(*update_pool(var, -1), -1)

        for i, stage in enumerate(self.stages):
            # print([(e.var, e.priority, e.in_stage) for e in pool.queue]) #
            
            # Make space for the outvars of this stage.
            for var in stage.outvars:
                if var not in stage.invars:
                    perform_swap(*update_pool(var, i, computed=False), i)

            # Preemptive swapping for next stage.
            if i < len(self.stages) - 1:
                for var in self.stages[i + 1].invars:
                    if var in stage.outvars:
                        # var doesn't exist yet, but will at the end of this stage.
                        continue
                    perform_swap(*update_pool(var, i), i)

            stage_invals = {}
            for var in stage.invars:
                if var in global_invals:
                    stage_invals[var] = global_invals[var]
                else:
                    assert var in var_stage_mapping, (
                        f"referred to an unknown var {var}"
                    )
                    sender_runner = runners[var_stage_mapping[var]]
                    stage_invals[var] = sender_runner.get_val(var)
                    var_reference_count[var] -= 1
                    if var_reference_count[var] == 0:
                        sender_runner.del_var(var)
                        pool.free(var)

            for var in stage_invals:
                assert stage_invals[var].device_buffer.device() == device, "not swapped"
            
            if stage.name not in runners:
                runners[stage.name] = LocalPipelineRunner(
                    stage.name, global_invals)
            runners[stage.name].run_stage(stage, stage_invals)
        
        # Examine swap patterns.
        print(view_swap_log(swap_log))

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
