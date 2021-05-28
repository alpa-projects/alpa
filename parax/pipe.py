"""Distributed JAX pipeline parallelism."""
import logging
from collections import OrderedDict
import functools
import numpy as np

import jax
from jax.core import Literal
import ray

from parax.pipeline_stage import StrVarPipelineStage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def ref_to_array(array_ref):
    """Ray does not understand DeviceArray."""
    numpy_array = ray.get(array_ref)
    device_array = jax.numpy.asarray(numpy_array)
    return device_array


class Runner:
    """
    Pipe runner class the coordinates sharding compilation and execution.

    This class should not be instantiated as a ray remote actors.
    Instead it invokes the mesh's functionality on creating and
    launching remote processes.

    Args:

    """

    def __init__(self,
                 *,
                 name,
                 stages,
                 mesh,
                 auto_sharding_args):
        self.name = name
        self.stages = stages
        self.mesh = mesh
        self.auto_sharding_args = auto_sharding_args
        self.sharding_compile_outputs = []

    def sharding_compile(self):
        for i, stage in enumerate(self.stages):
            # process stages to get the necessary arguments
            x, y, z = _auto_sharding_compile(stage, **self.auto_sharding_args)
            self.sharding_compile_outputs[i] = (x, y ,z)

    def compute(self, stage_idx):
        return


    # def compute(self, input_refs, stage_idx):
    #     """
    #     Compute on a given stage.
    #
    #     Args:
    #         input_refs (OrderedDict): with key being `repr(var)` and value being its reference.
    #         stage_idx (int): the stage to run.
    #     """
    #     runnable = self.runnables[stage_idx]
    #     stage = self.stages[stage_idx]
    #
    #     logger.debug("stage invars: {}".format(stage.invars))
    #     logger.debug("input refs: {}".format(input_refs.keys()))
    #
    #     # sanity check
    #     inputs = []
    #     for var in stage.invars:
    #         val_ref = input_refs[var]
    #         if val_ref:
    #             inputs.append(ref_to_array(val_ref))
    #         else:
    #             assert var in self.env
    #             inputs.append(self.env[var])
    #
    #     outputs = runnable(*inputs)
    #     outvals = dict(zip(stage.outvars, outputs))
    #
    #     # now split the outputs_dict
    #     pipeline_outvals = dict()
    #     global_outvals = dict()
    #     logger.debug("all outputs: {}".format(list(outvals.keys())))
    #     logger.debug("local_outvars: {}".format(list(stage.local_outvars)))
    #     logger.debug("pipeline_outvars: {}".format(list(stage.pipeline_outvars)))
    #     logger.debug("global outvars: {}".format(list(stage.global_outvars)))
    #     for var, val in outvals.items():
    #         if var in stage.local_outvars:
    #             self.env.update({var: val})
    #         if var in stage.pipeline_outvars:
    #             pipeline_outvals[var] = ray.put(val)
    #         if var in stage.global_outvars:
    #             global_outvals[var] = ray.put(val)
    #     logger.debug("pipeline outvals: {}".format(pipeline_outvals.keys()))
    #     logger.debug("global outvals: {}".format(global_outvals.keys()))
    #     logger.debug("worker {} env : {}".format(self.name, self.env.keys()))
    #     return pipeline_outvals, global_outvals


class RemoteRunner:
    """
    Distributed pipeline parallelism worker.

    NOTE: This class will always be instantiated as a Ray remote actor.

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
        # TODO: we could defer this compile to later.+
        self.compile()
        self.env = {}

    def compile(self):
        """Compile the HLO and get the runnable."""
        self.runnables = dict()
        for stage_idx, stage in self._raw_stages.items():
            self.runnables[stage_idx] = stage.get_runnable()

    def compute(self, input_refs, stage_idx):
        """
        Compute on a given stage.

        Args:
            input_refs (OrderedDict): with key being `repr(var)` and value being its reference.
            stage_idx (int): the stage to run.
        """
        runnable = self.runnables[stage_idx]
        stage = self.stages[stage_idx]

        logger.debug("stage invars: {}".format(stage.invars))
        logger.debug("input refs: {}".format(input_refs.keys()))

        # sanity check
        inputs = []
        for var in stage.invars:
            val_ref = input_refs[var]
            if val_ref:
                inputs.append(ref_to_array(val_ref))
            else:
                assert var in self.env
                inputs.append(self.env[var])

        outputs = runnable(*inputs)
        outvals = dict(zip(stage.outvars, outputs))

        # now split the outputs_dict
        pipeline_outvals = dict()
        global_outvals = dict()
        logger.debug("all outputs: {}".format(list(outvals.keys())))
        logger.debug("local_outvars: {}".format(list(stage.local_outvars)))
        logger.debug("pipeline_outvars: {}".format(list(stage.pipeline_outvars)))
        logger.debug("global outvars: {}".format(list(stage.global_outvars)))
        for var, val in outvals.items():
            if var in stage.local_outvars:
                self.env.update({var: val})
            if var in stage.pipeline_outvars:
                pipeline_outvals[var] = ray.put(val)
            if var in stage.global_outvars:
                global_outvals[var] = ray.put(val)
        logger.debug("pipeline outvals: {}".format(pipeline_outvals.keys()))
        logger.debug("global outvals: {}".format(global_outvals.keys()))
        logger.debug("worker {} env : {}".format(self.name, self.env.keys()))
        return pipeline_outvals, global_outvals


class JaxPipeline:         # pylint: disable=too-many-instance-attributes
    """
    JAX distributed pipeline.

    Args:
        pipeline_stages (List[PipelineStage]): list of pipleline stage programs.
        global_invars (List[Var]): input variables.
        global_outvars (List[Var]): output variables.
        dependency (np.array): dependency between stages as an adjacency matrix.
        num_batch (int): number of microbatches.
        schedule (): schedule to execute the pipeline.

    Returns:
        None
    """

    def __init__(self,
                 *,
                 pipeline_stages,
                 global_invars,
                 global_outvars,
                 mesh,
                 dependency=None,
                 num_batch=1,
                 schedule=None):
        self.stages = pipeline_stages
        self.global_invars = global_invars
        self.global_outvars = global_outvars
        self.num_stage = len(self.stages)

        self.mesh = mesh
        self.logical_mesh = self.mesh.get_default_logical_mesh()
        # TODO(Hao): analyze stages/dependencies and generate placement and schedule.
        self.num_batch = num_batch
        self.dependency = dependency
        # the code below will be removed
        if not self.dependency:
            self.dependency = _gen_linear_dependency(self.num_stage)

        self.schedule = schedule
        if not self.schedule:
            self.schedule = GpipeSchedule(dependency=self.dependency,
                                          num_batch=self.num_batch,
                                          mesh=self.mesh,
                                          sliced_meshes=self.sliced_meshes)
            logger.debug(self.schedule.pprint_schedule())

        if self.schedule:
            self.sliced_meshes = self.schedule.meshes

        # Below are runtime-related
        self.workers = []
        self._create_workers()
        self.stage_outputs = None
        self.microbatches = None
        # Note(Hao): all dicts constructed in this class are
        # string-keyed instead of Object-keyed.
        self.stage_outputs = self._init_stage_outputs()

    def run(self, *args, **kwargs):
        """Run the training with pipelining."""
        # pylint: disable=too-many-locals
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
                results_ref = self.workers[device].compute.remote(inputs, stage_idx)
                # put result refs in the stage_outputs
                pipeline_outputs_dict, stage_global_outputs_dict = ray.get(results_ref)
                if stage_global_outputs_dict:
                    global_output_refs.update(stage_global_outputs_dict)
                if pipeline_outputs_dict:
                    self.stage_outputs[clock][stage_idx].update(pipeline_outputs_dict)
            logger.info("All pipelining jobs done!")

        global_outvals_list = []
        for var in self.global_outvars:
            if isinstance(var, Literal):
                global_outvals_list.append(var.val)
            else:
                key = repr(var)
                assert key in global_output_refs
                val = ref_to_array(global_output_refs[key])
                global_outvals_list.append(val)
        return global_outvals_list

    def _init_stage_outputs(self):
        """
        Construct the output matrix.

        it is a C by S matrix where C is the #clocks and S is #stages.
        """
        stage_outputs = [[dict() for _ in range(self.num_stage)]
                         for _ in range(len(self.schedule))]
        return stage_outputs

    def _make_microbatches(self, *inputs, batch_dim=0, batch_size=128):
        assert len(inputs) == len(self.global_invars)
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
        remote_runner_cls = ray.remote(num_cpus=1, num_gpus=1)(RemoteRunner)
        # Distribute stages based on schedule.
        for worker_idx in range(self.num_pipeline_worker):
            stages = dict()
            stage_assignments = self.schedule.worker_placement(worker_idx)
            for stage_idx in stage_assignments:
                stages[stage_idx] = self.stages[stage_idx]
            stage_name = ",".join([str(s) for s in stage_assignments])
            logger.debug("Launching worker {} with stages {}.".format(
                worker_idx, stage_name))
            worker = remote_runner_cls.remote(name=stage_name, stages=stages)
            self.workers.append(worker)

    def _identify_stage_inputs(self, clock, stage_idx, batch_idx):
        """
        Find the input refs based on dependency.

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
        ancestors = list(np.squeeze(
            np.argwhere(self.dependency[stage_idx] == 1), axis=1))
        for var in stage.invars:
            key = repr(var)
            if var in stage.pipeline_invars:
                if stage_idx == 0:
                    stage_inputs[key] = self.microbatches[batch_idx][key]
                else:
                    for ans in ancestors:
                        if key in self.stage_outputs[clock - 1][ans]:
                            stage_inputs[key] = self.stage_outputs[clock - 1][ans][key]
            elif var in stage.global_invars:
                assert var in self.global_invars
                stage_inputs[key] = self.microbatches[batch_idx][key]
            elif var in stage.local_invars:
                # set it as None.
                stage_inputs[key] = None
            else:
                raise RuntimeError("Var `{}` not in any of global, pipeline or local "
                                   "var sets.".format(repr(var)))
        if len(stage_inputs) != len(stage.invars):
            raise RuntimeError("Failed to find stage inputs.")
        return stage_inputs


class Jax3DPipeline:         # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 *,
                 pipeline_stages,
                 global_invars,
                 global_outvars,
                 mesh,
                 dependency=None,
                 num_batch=1,
                 schedule=None):
        self.stages = pipeline_stages
        self.global_invars = global_invars
        self.global_outvars = global_outvars
        self.num_stage = len(self.stages)
        self.mesh = mesh
        self.logical_mesh = self.mesh.get_default_logical_mesh()

        self.num_batch = num_batch
        self.dependency = dependency
        # the code below will be removed
        if self.dependency is not None:
            self.dependency = _gen_linear_dependency(self.num_stage)

        self.schedule = schedule
        if not self.schedule:
            self.schedule = GpipeSchedule(dependency=self.dependency,
                                          num_batch=self.num_batch,
                                          mesh=self.mesh,
                                          sliced_meshes=self.sliced_meshes)
            logger.debug(self.schedule.pprint_schedule())

        if self.schedule:
            self.sliced_meshes = self.schedule.meshes

        # Below are runtime-related
        self._start_physical_meshes()

        self.stage_outputs = None
        self.microbatches = None
        # Note(Hao): all dicts constructed in this class are
        # string-keyed instead of Object-keyed.
        self.stage_outputs = self._init_stage_outputs()

    def run(self, *args, **kwargs):
        """Run the training with pipelining."""
        # pylint: disable=too-many-locals
        assert not kwargs, "kwargs not supported"
        self.microbatches = self._make_microbatches(*args)

        global_outputs = {}
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
                # results_ref = self.workers[device].compute.remote(inputs, stage_idx)
                # unroll
                inputs_list = []
                for var in self.stages[stage_idx].invars:
                    val = inputs[repr(var)]
                    inputs_list.append(val)
                outputs = self.runnables[stage_idx](*inputs_list)
                # put result refs in the stage_outputs
                # pipeline_outputs_dict, stage_global_outputs_dict = ray.get(results_ref)
                pipeline_outvals, global_outvals, local_outvals = self._post_process_stage_outputs(stage_idx, outputs)
                if pipeline_outvals:
                    self.stage_outputs[clock][stage_idx].update(pipeline_outvals)
                if global_outvals:
                    global_outputs.update(global_outvals)
                # if stage_global_outputs_dict:
                #     global_output_refs.update(stage_global_outputs_dict)
                # if pipeline_outputs_dict:
                #     self.stage_outputs[clock][stage_idx].update(pipeline_outputs_dict)
            logger.info("All pipelining jobs done!")

        global_outvals_list = []
        for var in self.global_outvars:
            if isinstance(var, Literal):
                global_outvals_list.append(var.val)
            else:
                key = repr(var)
                assert key in global_outputs
                val = ref_to_array(global_outputs[key])
                global_outvals_list.append(val)
        return global_outvals_list

    def _init_stage_outputs(self):
        """
        Construct the output matrix.

        it is a C by S matrix where C is the #clocks and S is #stages.
        """
        stage_outputs = [[dict() for _ in range(self.num_stage)]
                         for _ in range(len(self.schedule))]
        return stage_outputs

    def _make_microbatches(self, *inputs, batch_dim=0, batch_size=128):
        assert len(inputs) == len(self.global_invars)
        microbatches = [dict() for _ in range(self.num_batch)]
        for i, var in enumerate(self.global_invars):
            key = repr(var)
            array = inputs[i]
            if not array.shape or array.shape[batch_dim] != batch_size:
                # empty shape means it is not the input batch
                # no need to split
                # ref = ray.put(inputs[i])
                for b in range(self.num_batch):
                    microbatches[b][key] = inputs[i]
            else:
                splits = jax.numpy.split(array, self.num_batch, axis=batch_dim)
                for b, split in enumerate(splits):
                    microbatches[b][key] = split
        return microbatches

    def _start_physical_meshes(self):
        self.runnables = []
        for stage in self.stages:
            # This will request resources from Ray
            self.runnables.append(stage.get_runnable())

    def _identify_stage_inputs(self, clock, stage_idx, batch_idx):
        """
        Find the input refs based on dependency.

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
        ancestors = list(np.squeeze(
            np.argwhere(self.dependency[stage_idx] == 1), axis=1))
        for var in stage.invars:
            key = repr(var)
            if var in stage.pipeline_invars:
                if stage_idx == 0:
                    stage_inputs[key] = self.microbatches[batch_idx][key]
                else:
                    for ans in ancestors:
                        if key in self.stage_outputs[clock - 1][ans]:
                            stage_inputs[key] = ray.get(self.stage_outputs[clock - 1][ans][key])
            elif var in stage.global_invars:
                assert var in self.global_invars
                stage_inputs[key] = self.microbatches[batch_idx][key]
            elif var in stage.local_invars:
                # set it as None.
                stage_inputs[key] = None
            else:
                raise RuntimeError("Var `{}` not in any of global, pipeline or local "
                                   "var sets.".format(repr(var)))
        if len(stage_inputs) != len(stage.invars):
            raise RuntimeError("Failed to find stage inputs.")
        return stage_inputs

    def _post_process_stage_outputs(self, stage_idx, outputs):
        stage = self.stages[stage_idx]
        outvals = dict(zip(stage.outvars, outputs))
            # now split the outputs_dict
        pipeline_outvals = dict()
        global_outvals = dict()
        local_outvals = dict()
        logger.debug("all outputs: {}".format(list(outvals.keys())))
        logger.debug("local_outvars: {}".format(list(stage.local_outvars)))
        logger.debug("pipeline_outvars: {}".format(list(stage.pipeline_outvars)))
        logger.debug("global outvars: {}".format(list(stage.global_outvars)))
        for var, val in outvals.items():
            key = repr(var)
            if var in stage.local_outvars:
                # self.env.update({var: val})
                local_outvals[key] = val
            if var in stage.pipeline_outvars:
                pipeline_outvals[key] = ray.put(val)
            if var in stage.global_outvars:
                global_outvals[key] = ray.put(val)
        logger.debug("pipeline outvals: {}".format(pipeline_outvals.keys()))
        logger.debug("global outvals: {}".format(global_outvals.keys()))
        logger.debug("local outvals: {}".format(local_outvals.keys()))
        return pipeline_outvals, global_outvals, local_outvals



def _gen_linear_dependency(num_stage):
    """Generate a linear dependency matrix."""
    d = np.zeros([num_stage, num_stage], dtype=np.int)
    for i in range(num_stage - 1):
        d[i + 1][i] = 1
    return d


class GpipeSchedule:
    """
    Construct a Gpipe-like schedule.

    Args:
        dependency (np.array): dependency adjacency matrix.
        num_batch (int): number of microbatches.
        num_pipeline_worker (int): number of pipelining workers.
        costs (List[int]): running costs of each stage

    Returns:
        None
    """

    def __init__(self,
                 *,
                 dependency,
                 mesh,
                 sliced_meshes=None,
                 num_batch=1,
                 costs=None):

        self.dependency = dependency
        self.original_mesh = mesh
        self.meshes = sliced_meshes
        self.num_batch = num_batch
        self.costs = costs
        self.num_stage = dependency.shape[0]

        if not self.meshes:
            self.meshes = self.slice_mesh(self.original_mesh)
        self.num_pipeline_worker = len(self.meshes)
        # if self.num_stage != 2 * self.num_pipeline_worker:
        #     raise RuntimeError("Gpipe schedule requires #stages being twice of #workers.")
        self._schedules = self._generate_schedule()

    def _generate_schedule(self):
        """
        Generate a Gpipe-like schedule.

        Note that here we always assume num_pipeline_workers = num_stage / 2.

        The schedule will look like below:
        i: index of micro-batch
        j: index of partition/device
        k: clock number

        k (i,j) (i,j) (i,j)
        - ----- ----- -----
        0 (0,0)
        1 (1,0) (0,1)
        2 (2,0) (1,1) (0,2)
        3       (2,1) (1,2)
        4             (2,2)
        5 reverse...
        """
        m = self.num_batch
        n = self.num_pipeline_worker
        num_clock = m + n - 1
        schedules = []
        for k in range(num_clock):
            scheds = [None] * n
            for d in range(max(1 + k - m, 0), min(1 + k, n)):
                scheds[d] = (k - d, d)
            schedules.append(scheds)

        def reverse(scheds):
            rev = []
            for task in scheds:
                if not task:
                    rev.append(None)
                else:
                    rev.append((task[0], 2 * n - 1 - task[1]))
            return rev

        # backward schedules
        for k in range(num_clock):
            mapped_scheds = schedules[num_clock - k - 1]
            schedules.append(reverse(mapped_scheds))
        return schedules

    def pprint_schedule(self):
        """Pretty print the schedule."""
        printout = "\n"
        device_str = " ".join(["{:<8}".format("d" + str(d))
                               for d in range(self.num_pipeline_worker)])
        printout = printout + "Clock {:<2}: {} \n".format("k", device_str)
        for clock, scheds in enumerate(self.schedules):
            sched_str = " ".join(["{:<8}".format(str(sched)) for sched in scheds])
            printout = printout + "Clock {:<2}: {} \n".format(clock, sched_str)
        return printout

    @cached_property
    def stage_worker_mapping(self):
        """Generate a stage-worker mapping according to the schedule."""
        placements = dict()
        for tasks in self._schedules:
            for worker_idx, task in enumerate(tasks):
                if task:
                    _, stage_idx = task
                    if stage_idx not in placements:
                        placements[stage_idx] = set()
                    if worker_idx not in placements[stage_idx]:
                        placements[stage_idx].add(worker_idx)
        return placements

    @cached_property
    def worker_stage_mapping(self):
        """Generate a worker-stage mapping according to the schedule."""
        ownership = dict()
        for tasks in self._schedules:
            for worker_idx, task in enumerate(tasks):
                if task:
                    _, stage_idx = task
                    if worker_idx not in ownership:
                        ownership[worker_idx] = set()
                    if stage_idx not in ownership[worker_idx]:
                        ownership[worker_idx].add(stage_idx)
        return ownership

    def stage_placement(self, stage_idx):
        """Query the placement of a stage given its stage index."""
        return self.stage_worker_mapping[stage_idx]

    def worker_placement(self, worker_idx):
        """Query the responsible stages of a worker given a worker index."""
        return self.worker_stage_mapping[worker_idx]

    @property
    def schedules(self):
        """Return the schedules as a matrix."""
        return self._schedules

    def __len__(self):
        return len(self._schedules)

    @property
    def num_clock(self):
        """Return the number of clocks in the schedule."""
        return len(self._schedules)

    def slice_mesh(self, mesh):
        """
        Slice the mesh for each remote runner.

        In this impl, we guarantee the slicing follows:
        - len(sliced_meshes) == num_stages / 2 (place forward/backward in a mesh);
        - higher priority to slice over the node dimension rather than gpu dimension.

        Args:
            mesh: a device mesh.

        Returns:
            sliced_meshes (List[Mesh]):
        """
        sliced_meshes = []
        # TODO (should be self.num_stage / 2 after the merging of HLO is done..)
        num_mesh = self.num_stage
        if mesh.is_distributed and False: # Hao: disable this branch now
            num_host = mesh.num_hosts
            if num_host < num_mesh:
                raise RuntimeError("The number of available hosts in the mesh is insufficient.")
            indices = list(range(num_host))
            num_host_per_mesh = num_host / num_mesh
            for i in range(num_host):
                ins = indices[num_host_per_mesh:i:num_host_per_mesh*(i+1)]
                sliced_meshes.append(mesh.slice(0, ins))
        else:
            if mesh.total_devices < num_mesh:
                raise RuntimeError("The number of available devices in the mesh is insufficient.")
            indices = list(range(mesh.total_devices))
            num_device_per_mesh = int(mesh.total_devices / num_mesh)
            for i in range(num_mesh):
                ins = indices[num_device_per_mesh*i:num_device_per_mesh*(i+1)]
                sliced_meshes.append(mesh.slice(1, ins))
        return sliced_meshes
