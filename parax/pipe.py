import logging
from collections import OrderedDict

import jax
from jax import jit
from jax.core import jaxpr_as_fun

from parax.pipeline_stage import PicklableStage
from parax.pipeline_primitive_def import *

import ray

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ref_to_array(array_ref):
    """Ray does not understand DeviceArray."""
    numpy_array = ray.get(array_ref)
    device_array = jax.numpy.asarray(numpy_array)
    return device_array


class RunnerV2:
    def __init__(self,
                 *,
                 name,
                 forward_runnable=None,
                 backward_runnable=None,
                 forward_closed_jaxpr=None,
                 backward_closed_jaxpr=None):
        self.name = name
        self.forward_runnable = forward_runnable
        self.backward_runnable = backward_runnable
        self.forward_closed_jaxpr = forward_closed_jaxpr
        self.backward_closed_jaxpr = backward_closed_jaxpr
        self.env = dict()

    def compute(self, input_ref, is_forward=True):
        runnable = self.forward_runnable if is_forward else self.backward_runnable
        closed_jaxpr = self.forward_closed_jaxpr if is_forward else self.backward_closed_jaxpr
        # sanity check
        inputs = []
        for var in closed_jaxpr.jaxpr.invars:
            key = repr(var)
            val_ref = input_ref[key]
            if val_ref:
                inputs.append(ray.get(val_ref))
            else:
                inputs.append(self.env[var])
        print(inputs)
        outputs = runnable(*inputs)
        return outputs


class Runner:
    def __init__(self,
                 *,
                 name,
                 forward_stage=None,
                 backward_stage=None):
        self.name = name
        # Note 0: TypeError: cannot pickle 'jaxlib.xla_extension.Traceback' object
        # Note 1: We don't have to distinguish forward and backward program.
        self.forward_stage = forward_stage
        self.backward_stage = backward_stage
        self.forward_closed_jaxpr = self.forward_stage.closed_jaxpr()
        self.backward_closed_jaxpr = self.backward_stage.closed_jaxpr()
        self.forward_runnable = None
        self.backward_runnable = None
        self.compile()
        self.env = dict()

    def compile(self):
        # gen forward runnable
        self.forward_runnable = jit(jaxpr_as_fun(self.forward_closed_jaxpr))
        self.backward_runnable = jit(jaxpr_as_fun(self.backward_closed_jaxpr))

    def compute(self, input_ref, is_forward=True):
        """
        Args:
            input_ref (OrderedDict): with key being `repr(var)` and value being its reference.
        """
        # check gpu devices
        program = self.forward_stage if is_forward else self.backward_stage
        runnable = self.forward_runnable if is_forward else self.backward_runnable
        closed_jaxpr = self.forward_closed_jaxpr if is_forward else self.backward_closed_jaxpr
        # sanity check
        inputs = []
        for var in closed_jaxpr.jaxpr.invars:
            key = repr(var)
            val_ref = input_ref[key]
            if val_ref:
                inputs.append(ref_to_array(val_ref))
            else:
                inputs.append(self.env[var])

        outputs = runnable(*inputs)
        outputs_dict = {var: val for var, val in zip(closed_jaxpr.jaxpr.outvars, outputs)}
        # now split the outputs_dict
        pipeline_outputs_dict = dict()
        global_outputs_dict = dict()
        print("all outputs: ", outputs_dict.keys())
        print("local_outvars: ",  program.local_outvars)
        print("pipeline_outvars: ", program.pipeline_outvars)
        print("global outvars: ", program.global_outvars)
        for var, val in outputs_dict.items():
            if var in program.local_outvars:
                self.env.update({var: val})
            if var in program.pipeline_outvars:
                pipeline_outputs_dict[repr(var)] = ray.put(val)
            if var in program.global_outvars:
                global_outputs_dict[repr(var)] = ray.put(val)
        print("pipeline_outputs_dict: ", pipeline_outputs_dict)
        print("global_outputs_dict: ", global_outputs_dict)
        return pipeline_outputs_dict, global_outputs_dict


class JaxPipeline:
    def __init__(self,
                 *,
                 pipeline_stages,
                 global_invars,
                 global_outvars,
                 num_batches=1,
                 schedules=None):
        assert (len(pipeline_stages) % 2 == 0)
        self.stage_names = []
        self.stages = pipeline_stages
        self.stages_closed_jaxpr = []

        self.forward_stages = dict()
        self.backward_stages = dict()
        for stage in pipeline_stages:
            # assuming first scan through all fwd stages, then backward.
            self.stages_closed_jaxpr.append(stage.closed_jaxpr())
            if not stage.name in self.stage_names:
                self.stage_names.append(stage.name)
                self.forward_stages[stage.name] = stage
                continue
            if stage.name in self.stage_names:
                self.backward_stages[stage.name] = stage
        self.num_stages = len(self.stage_names)

        # invars and outvars symbols
        self.global_invars = global_invars
        self.global_outvars = global_outvars

        self.num_batches = num_batches

        self.schedules = schedules
        # tuple(List, List)
        if not self.schedules:
            self.schedules = gen_gpipe_schedule(self.num_batches, self.num_stages)

        # init actors
        self.workers = dict()
        self._create_workers()
        # self._create_workers_v2()

        # inputs and outputs
        self.stage_inputs = self._init_stage_inputs()
        # processed at self.run() call.
        self.microbatches = None

    def run(self, *args, **kwargs):
        assert not kwargs, "kwargs not supported"
        self.microbatches = self._make_microbatches(*args)

        global_output_refs = {}
        # forward
        for clock, sched in enumerate(self.schedules[0]):
            # submit work in parallel
            print("At clock {}, working on {} at forward phase.".format(clock, sched))
            for i, j in sched:
                # i is micro-batch idx
                # j is stage idx
                inputs = self._identify_stage_inputs(j, i, clock)
                # TODO(Hao): this mapping is too simple, make it more complex
                stage_name = str(j + 1)
                results_ref = self.workers[stage_name].compute.remote(inputs)
                # put results in the stage_inputs
                pipeline_outputs_dict, stage_global_outputs_dict = ray.get(results_ref)
                print(pipeline_outputs_dict)
                print(stage_global_outputs_dict)
                global_output_refs.update(stage_global_outputs_dict)
                next_stage = j + 1
                next_clock = clock + 1
                if next_stage < self.num_stages:
                    self.stage_inputs[next_clock][next_stage].update(pipeline_outputs_dict)

        # backward
        for clock, sched in enumerate(self.schedules[1]):
            print("At clock {}, working on {} at backward phase.".format(clock, sched))
            for i, j in sched:
                inputs = self._identify_stage_inputs(j, i, clock, is_forward=False)
                stage_name = str(j + 1)
                results_ref = self.workers[stage_name].compute.remote(inputs, is_forward=False)
                pipeline_outputs_dict, stage_global_outputs_dict = ray.get(results_ref)
                print(pipeline_outputs_dict)
                print(stage_global_outputs_dict)
                global_output_refs.update(stage_global_outputs_dict)
                next_stage = j - 1
                next_clock = clock + 1
                if next_stage >= 0:
                    self.stage_inputs[next_clock][next_stage].update(pipeline_outputs_dict)

    def _init_stage_inputs(self):
        """Construct the batch_ref matrix,

        Batch_ref store ray refs to data, it implicitly captures the
        input-output dependency of pipelining.
        """
        S = self.num_stages
        C = len(self.schedules)  # num_clock
        # batch_ref is C by S
        # batch_refs[i][j]: dict() is the input dict for stage j at clock i
        batch_refs = [[dict() for _ in range(S)] for _ in range(C)]
        return batch_refs

    def _make_microbatches(self, *inputs, batch_dim=0, batch_size=128):
        assert (len(inputs) == len(self.global_invars))
        microbatches = [dict() for _ in range(self.num_batches)]
        for i, var in enumerate(self.global_invars):
            array = inputs[i]
            if not array.shape or array.shape[batch_dim] != batch_size:
                # empty shape means it is not the input batch
                # no need to split
                ref = ray.put(inputs[i])
                for b in range(self.num_batches):
                    microbatches[b][var] = ref
            else:
                splits = jax.numpy.split(array, self.num_batches, axis=batch_dim)
                for b, split in enumerate(splits):
                    microbatches[b][var] = ray.put(split)
        return microbatches

    def _create_workers(self):
        remote_runner = ray.remote(num_cpus=1, num_gpus=1)(Runner)
        for stage_name in self.stage_names:
            # f_stage = self.forward_stages[stage_name]
            # b_stage = self.backward_stages[stage_name]
            f_stage = PicklableStage.from_pipeline_stage(self.forward_stages[stage_name])
            b_stage = PicklableStage.from_pipeline_stage(self.backward_stages[stage_name])
            # print("f_stage order at actor creation: {}".format(f_stage.closed_jaxpr().jaxpr.invars))
            worker = remote_runner.remote(name=stage_name,
                                          forward_stage=f_stage,
                                          backward_stage=b_stage)
            self.workers[stage_name] = worker

    def _create_workers_v2(self):
        """This does not work!"""
        remote_runner = ray.remote(num_cpus=1, num_gpus=1)(RunnerV2)
        for stage_name in self.stage_names:
            forward_stage = self.forward_stages[stage_name]
            backward_stage = self.backward_stages[stage_name]
            forward_closed_jaxpr = forward_stage.closed_jaxpr()
            backward_closed_jaxpr = backward_stage.closed_jaxpr()
            forward_runnable = jit(jaxpr_as_fun(forward_closed_jaxpr))
            backward_runnable = jit(jaxpr_as_fun(backward_closed_jaxpr))
            worker = remote_runner.remote(name=stage_name,
                                          forward_runnable=forward_runnable,
                                          backward_runnable=backward_runnable,
                                          forward_closed_jaxpr=forward_closed_jaxpr,
                                          backward_closed_jaxpr=backward_closed_jaxpr)
            self.workers[stage_name] = worker

    def _identify_stage_inputs(self, stage_idx, batch_idx, clock, is_forward=True):
        # stage_input is a dict: var_name -> array ref
        stage_inputs = OrderedDict()
        stage_name = str(stage_idx + 1)
        if is_forward:
            stage = self.forward_stages[stage_name]
        else:
            stage = self.backward_stages[stage_name]
        closed_jaxpr = stage.closed_jaxpr()

        # print("f_stage order at identify_stage_inputs: {}".format(closed_jaxpr.jaxpr.invars))
        for var in closed_jaxpr.jaxpr.invars:
            key = repr(var)
            if var in stage.pipeline_invars:
                if stage_idx == 0:
                    stage_inputs[key] = self.microbatches[batch_idx][var]
                else:
                    stage_inputs[key] = self.stage_inputs[clock][stage_idx][key]  # the key is string because of the pickling signature issue..
            elif var in self.global_invars:
                stage_inputs[key] = self.microbatches[batch_idx][var]
            else:
                assert var in stage.local_invars
                stage_inputs[key] = None
        return stage_inputs


def gen_gpipe_schedule(m, n):
    # m: number of micro-batches
    # n: number of partitions
    # i: index of micro-batch
    # j: index of partition/device
    # k: clock number
    #
    # k (i,j) (i,j) (i,j)
    # - ----- ----- -----
    # 0 (0,0)
    # 1 (1,0) (0,1)
    # 2 (2,0) (1,1) (0,2)
    # 3       (2,1) (1,2)
    # 4             (2,2)
    # 5 reverse...
    forward_schedules = []
    for k in range(m + n - 1):
        forward_schedules.append([(k - j, j) for j in range(max(1 + k - m, 0), min(1 + k, n))])
    backward_schedules = forward_schedules[::-1]
    return forward_schedules, backward_schedules
