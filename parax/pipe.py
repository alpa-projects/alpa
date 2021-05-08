import logging
from collections import OrderedDict

import functools
import jax
import ray

from parax.pipeline_primitive_def import *
from parax.pipeline_stage import StrVarPipelineStage

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ref_to_array(array_ref):
    """Ray does not understand DeviceArray."""
    numpy_array = ray.get(array_ref)
    device_array = jax.numpy.asarray(numpy_array)
    return device_array


class Runner:
    """Pipeline worker.

    Args:
        name (str): The name of this runner
        stages (dict): serializable stages assigned to this runner.
    """
    def __init__(self,
                 *,
                 name,
                 stages):
        self.name = name
        self._raw_stages = stages
        self.stages = dict()
        for stage_idx, stage in self._raw_stages.items():
            self.stages[stage_idx] = StrVarPipelineStage.from_pipeline_stage(stage)

        self.runnables = None
        # TODO: we could defer this compile to later.
        self.compile()
        self.env = dict()

    def compile(self):
        self.runnables = dict()
        for stage_idx, stage in self._raw_stages.items():
            self.runnables[stage_idx] = stage.get_runnable()

    def compute(self, input_refs, stage_idx):
        """Compute on a given stage.

        Args:
            input_refs (OrderedDict): with key being `repr(var)` and value being its reference.
            stage_idx (int): the stage to run.
        """
        runnable = self.runnables[stage_idx]
        stage = self.stages[stage_idx]

        # sanity check
        inputs = []
        for var in stage.invars:
            val_ref = input_refs[var]
            if val_ref:
                inputs.append(ref_to_array(val_ref))
            else:
                assert(var in self.env)
                inputs.append(self.env[var])

        outputs = runnable(*inputs)
        outvals = {var: val for var, val in zip(stage.outvars, outputs)}

        # now split the outputs_dict
        pipeline_outvals = dict()
        global_outvals = dict()
        logger.debug("all outputs: ", outvals.keys())
        logger.debug("local_outvars: ",  program.local_outvars)
        logger.debug("pipeline_outvars: ", program.pipeline_outvars)
        logger.debug("global outvars: ", program.global_outvars)
        for var, val in outvals.items():
            if var in stage.local_outvars:
                self.env.update({var: val})
            if var in stage.pipeline_outvars:
                pipeline_outvals[var] = ray.put(val)
            if var in stage.global_outvars:
                global_outvals[var] = ray.put(val)
        logger.debug("pipeline outvals: ", pipeline_outvals)
        print("global outvals: ", global_outvals)
        return pipeline_outvals, global_outvals


class JaxPipeline:
    def __init__(self,
                 *,
                 pipeline_stages,
                 global_invars,
                 global_outvars,
                 dependency=None,
                 num_batch=1,
                 num_pipeline_worker=2,
                 schedule=None):
        self.stages = pipeline_stages
        self.global_invars = global_invars
        self.global_outvars = global_outvars
        self.num_stage = len(self.stages)

        # TODO(Hao): analyze stages/dependencies and generate placement and schedule.
        self.num_batch = num_batch
        self.num_pipeline_worker = num_pipeline_worker
        self.dependency = dependency
        if not self.dependency:
            self.dependency = _gen_linear_dependency(self.num_stage)
        self.schedule = schedule
        # schedules here include placement information.
        if not self.schedule:
            self.schedule = GpipeSchedule(dependency=self.dependency,
                                          num_batch=self.num_batch,
                                          num_pipeline_worker=self.num_pipeline_worker)

        self.workers = []
        self._create_workers()
        self.stage_outputs = None
        self.microbatches = None
        # Note(Hao): all dicts constructed in this class are
        # string-keyed instead of Object-keyed.
        self.stage_outputs = self._init_stage_outputs()

    def run(self, *args, **kwargs):
        assert not kwargs, "kwargs not supported"
        self.microbatches = self._make_microbatches(*args)

        global_output_refs = {}
        for clock, sched in enumerate(self.schedule.schedules):
            # submit work in parallel
            logger.debug("At clock {}, working on tasks {}.".format(clock, sched))
            for device, task in enumerate(sched):
                # i is micro-batch idx
                # j is stage idx
                if not task:
                    continue
                batch_idx, stage_idx = task
                inputs = self._identify_stage_inputs(clock, stage_idx, batch_idx)
                results_ref = self.workers[device].compute.remote(inputs)
                # put result refs in the stage_outputs
                pipeline_outputs_dict, stage_global_outputs_dict = ray.get(results_ref)
                print(pipeline_outputs_dict)
                print(stage_global_outputs_dict)
                global_output_refs.update(stage_global_outputs_dict)
                self.stage_outputs[clock][stage_idx].update(pipeline_outputs_dict)

    def _init_stage_outputs(self):
        """Construct the output matrix.

        it is a C by S matrix where C is the #clocks and S is #stages.
        """
        stage_outputs = [[dict() for _ in range(self.num_stage)]
                         for _ in range(len(self.schedule))]
        return stage_outputs

    def _make_microbatches(self, *inputs, batch_dim=0, batch_size=128):
        assert (len(inputs) == len(self.global_invars))
        microbatches = [dict() for _ in range(self.num_batch)]
        for i, var in enumerate(self.global_invars):
            key = repr(var)
            array = inputs[i]
            if not array.shape or array.shape[batch_dim] != batch_size:
                # empty shape means it is not the input batch
                # no need to split
                ref = ray.put(inputs[i])
                for b in range(self.num_batch):
                    microbatches[b][key] = ref
            else:
                splits = jax.numpy.split(array, self.num_batch, axis=batch_dim)
                for b, split in enumerate(splits):
                    microbatches[b][key] = ray.put(split)
        return microbatches

    def _create_workers(self):
        remote_runner = ray.remote(num_cpus=1, num_gpus=1)(Runner)
        # Distribute stages based on schedule.
        for worker_idx in range(self.num_pipeline_worker):
            stages = dict()
            stage_assignments = self.schedule.worker_placement(worker_idx)
            for stage_idx in stage_assignments:
                stages[stage_idx] = self.stages[stage_idx]
            stage_name = ",".join(str(stage_assignments))
            logger.debug("Launching worker {} with stages {}.".format(
                worker_idx, stage_name))
            worker = remote_runner.remote(name=stage_name, stages=stages)
        self.workers.append(worker)

    def _identify_stage_inputs(self, clock, stage_idx, batch_idx):
        """Find the input refs based on dependency.

        Args:
            clock (int):
            stage_idx (int):
            batch_idx (int):

        Returns:
            stage_inputs (dict[str, Any]):
        """
        stage_inputs = OrderedDict()
        stage = self.stages[stage_idx]
        # find stages that depend on it
        ancestors = np.where(self.dependency[stage_idx] == 1)
        for var in stage.invars:
            key = repr(var)
            if var in stage.pipeline_invars:
                if stage_idx == 0:
                    stage_inputs[key] = self.microbatches[batch_idx][key]
                else:
                    for ans in ancestors:
                        if key in self.stage_outputs[clock][ans]:
                            stage_inputs[key] = self.stage_outputs[clock][ans][key]
            elif var in stage.global_invars:
                assert var in self.global_invars
                stage_inputs[key] = self.microbatches[batch_idx][key]
            elif var in stage.local_invars:
                # set it as None.
                stage_inputs[key] = None
            else:
                raise RuntimeError("Var `{}` not in any of global, pipeline or local "
                                   "var sets.".format(repr(var)))
        return stage_inputs

def _gen_linear_dependency(num_stage):
    """Generate a linear dependency matrix."""
    d = np.zeros([num_stage, num_stage], dtype=np.int)
    for i in range(num_stage - 1):
        d[i+1][i] = 1
    return d


def cached_property(fn, *args, **kwargs):
    """
    Decorator to make a function a "cached property".
    This means that it is a property whose return value is cached after the
    first time it is called.
    Args:
        fn: The function to be made a cached property
        *args: Any args for the function
        **kwargs: Any kwargs for the function
    Returns:
        function
    """
    return property(functools.lru_cache()(fn, *args, **kwargs))


class GpipeSchedule:
    def __init__(self,
                 *,
                 dependency,
                 num_batch=1,
                 num_pipeline_worker=1,
                 costs=None):

        self.dependency = dependency
        self.num_batch = num_batch
        self.num_pipeline_worker = num_pipeline_worker
        self.costs = costs
        self.num_stage = dependency.shape[0]
        self._schedules = self._generate_schedule()

    def _generate_schedule(self):
        """This function DOES NOT WORK!!!"""
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
        m = self.num_batch
        n = self.num_pipeline_worker
        schedules = []
        # num_clock = m + n - 1
        # for clock in range(num_clock):
        #     schedules
        for k in range(m + n - 1):
            schedules.append([(k - j, j) for j in range(max(1 + k - m, 0), min(1 + k, n))])
        # backward_schedules = forward_schedules[::-1]
        return schedules

    @cached_property
    def stage_worker_mapping(self):
        placements = dict()
        for clock, tasks in enumerate(self._schedules):
            for worker_idx, task in enumerate(tasks):
                if task:
                    batch_idx, stage_idx = task
                    if stage_idx not in placements:
                        placements[stage_idx] = set()
                    if worker_idx not in placements[stage_idx]:
                        placements[stage_idx].add(worker_idx)
        return placements

    @cached_property
    def worker_stage_mapping(self):
        ownership = dict()
        for clock, tasks in enumerate(self._schedules):
            for worker_idx, task in enumerate(tasks):
                if task:
                    batch_idx, stage_idx = task
                    if worker_idx not in ownership:
                        ownership[worker_idx] = set()
                    if stage_idx not in ownership[worker_idx]:
                        ownership[worker_idx].add(stage_idx)
        return ownership

    def stage_placement(self, stage_idx):
        return self.stage_worker_mapping[stage_idx]

    def worker_placement(self, worker_idx):
        return self.worker_stage_mapping[worker_idx]

    @property
    def schedule(self):
        return self._schedules

    def __len__(self):
        return len(self.schedule)

    @property
    def num_clock(self):
        return len(self.schedule)
