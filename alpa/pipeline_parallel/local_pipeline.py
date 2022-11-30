"""Pipeline parallel on a single device. This is only used for debugging."""
from abc import ABC, abstractmethod
from collections import OrderedDict
from math import inf
from typing import Sequence, Any, Dict, Optional
import heapq

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
    def __init__(self, var: Optional[Var], priority: int):
        self.var = var
        self.priority = priority
        self.valid = True
    
    def __lt__(self, other):
        return self.priority < other.priority

    def __eq__(self, other):
        return self.var == other.var


class SwapPool:
    """A heap queue for simulating a pool of variables on device.

    This implementation lazily invalidates elements upon removal.
    """

    def __init__(self, max_size: int):
        self.heap = []
        self.map = {}
        self.size = 0
        assert max_size > 0, "max_size must be greater than 0"
        self.max_size = max_size

    def alloc(self, var: Var, priority: int):
        """Simulating swapping in a variable for a stage.
        Args:
            var: the variable to swap in.
            priority: lesser means earlier to swap out. 
        Returns:
            A list of variables to swap out.
        """
        swap_in = True
        swap_out_var = None
        if var in self.map:
            self.map[var].valid = False
            swap_in = False
        elif self.size < self.max_size:
            self.size += 1
        else:
            popped = heapq.heappop(self.heap)
            self.map.pop(popped.var)
            while not popped.valid:
                popped = heapq.heappop(self.heap)
                self.map.pop(popped.var)
            swap_out_var = popped.var
        new_element = SwapElement(var, priority)
        self.map[var] = new_element
        heapq.heappush(self.heap, new_element)
        return swap_in, swap_out_var

    def free(self, var: Var):
        assert var in self.map, f"{var} is not in pool"
        self.map[var].valid = False
        self.size -= 1


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
                 swap: Optional[str]=None):
        self.stages = stages
        self.global_invars = global_invars
        self.global_outvars = global_outvars
        if swap is None or swap == "none":
            self.launch_fn = self._no_swap_launch_on_driver
        elif swap == "basic":
            self.launch_fn = self._basic_swap_launch_on_driver
        elif swap == "optimal":
            self.launch_fn = self._opt_swap_launch_on_driver
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

    def _opt_swap_launch_on_driver(self, *args):
        """Run function."""
        host = jax.devices(backend="cpu")[0]
        device = jax.devices(backend="gpu")[0]
        global_invals = dict(zip(self.global_invars, args))
        runners = {}

        var_stage_mapping = {}
        var_reference_count = {}

        # Data structures used for OPT swapping strategy.
        pool = SwapPool(max_size=32)
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

        # Fill pool based on current array placement.
        for var, val in global_invals.items():
            if isinstance(val, DeviceArray) and val.device_buffer.device() == device:
                # Use number of stages until first access to compute priority.
                swap_in, swap_out_var = pool.alloc(var, -var_access_map[var][0])

        for var in self.global_outvars:
            if not isinstance(var, Literal):
                assert var in var_stage_mapping, (
                    f"referred to an unknown var {var}")
                var_reference_count[var] = var_reference_count.get(var, 0) + 1

        for i, stage in enumerate(self.stages):
            stage_invals = {}
            for var in stage.invars:
                # Use number of stages until next access to compute priority.
                next_index = var_access_index[var] + 1
                if next_index < len(var_access_map[var]):
                    swap_in, swap_out_var = pool.alloc(var, i - var_access_map[var][next_index])
                    var_access_index[var] = next_index
                else:
                    swap_in, swap_out_var = pool.alloc(var, -inf)

                # Swap out an array if needed.
                if swap_out_var is not None:
                    if swap_out_var in global_invals:
                        swap_out_val = global_invals[swap_out_var]
                        global_invals[var] = jax.device_put(swap_out_val, device=host)
                    else:
                        assert swap_out_var in var_stage_mapping, (
                            f"referred to an unknown var {swap_out_var}"
                        )
                        holder_runner = runners[var_stage_mapping[swap_out_var]]
                        swap_out_val = holder_runner.get_val(swap_out_var)
                        holder_runner.set_val(var, jax.device_put(swap_out_val, device=host))
                    swap_log.append(("out", swap_out_var, swap_out_val.nbytes))
                    del swap_out_val

                # Get stage input arrays, swapping in if needed.
                if var in global_invals:
                    if swap_in:
                        swap_in_val = global_invals[var]
                        global_invals[var] = jax.device_put(swap_in_val, device=device)
                        swap_log.append(("in", var, swap_in_val.nbytes))
                    stage_invals[var] = global_invals[var]
                else:
                    assert var in var_stage_mapping, (
                        f"referred to an unknown var {var}"
                    )
                    sender_runner = runners[var_stage_mapping[var]]
                    if swap_in:
                        swap_in_val = sender_runner.get_val(var)
                        sender_runner.set_val(var, jax.device_put(swap_in_val, device=device))
                        swap_log.append(("in", var, swap_in_val.nbytes))
                    stage_invals[var] = sender_runner.get_val(var)
                    var_reference_count[var] -= 1
                    if var_reference_count[var] == 0:
                        sender_runner.del_var(var)
                        pool.free(var)

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
