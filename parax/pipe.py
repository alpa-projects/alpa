"""Distributed JAX pipeline parallelism."""
import logging
import math
from collections import OrderedDict

import functools
import numpy as np
import ray
from jax.core import Literal
import jax.numpy as jnp

from parax.device_mesh import DistributedArray
from parax.pipeline_stage import StrVarPipelineStage, \
    XlaShardedPipelineStage

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
    device_array = jnp.asarray(numpy_array)
    return device_array


class RemoteRunner:
    """
    Distributed pipeline parallelism worker.

    NOTE: This class shall always be instantiated as a Ray remote actor.

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
        mesh (VirtualMesh): the cluster mesh to pipeline on.
        dependency (np.array): dependency between stages as an adjacency matrix.
        num_batch (int): number of microbatches.
        schedule (GpipeSchedule): schedule to follow to execute the pipeline.
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
        # TODO(Hao): analyze stages/dependencies and generate placement and schedule.
        self.num_batch = num_batch
        self.dependency = dependency
        # the code below will be removed
        if not self.dependency:
            self.dependency = _gen_linear_dependency(self.num_stage)

        # Generate schedule
        self.schedule = schedule
        if not self.schedule:
            self.schedule = GpipeSchedule(dependency=self.dependency,
                                          num_batch=self.num_batch,
                                          mesh=self.mesh)
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
                splits = jnp.split(array, self.num_batch, axis=batch_dim)
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


class Jax3DPipeline:  # pylint: disable=too-many-instance-attributes
    """
    A class to coordinate 3D parallelism.

    Args:
        pipeline_stages (List[PipelineStage]): list of pipeline stage programs.
        global_invars (List[Var]): input variables.
        global_outvars (List[Var]): output variables.
        mesh (VirtualMesh): the cluster mesh to pipeline on.
        sharding_compilation_kwargs (dict): a dict of keyword arguments as the sharding
            compilation parameters.
        dependency (np.array): dependency between stages as an adjacency matrix.
        num_batch (int): number of microbatches.
        schedule (): schedule to follow to execute the pipeline.
    """

    def __init__(self,
                 *,
                 pipeline_stages,
                 global_invars,
                 global_outvars,
                 mesh,
                 sharding_compilation_kwargs=None,
                 dependency=None,
                 num_batch=1,
                 schedule=None):
        self.stages = pipeline_stages
        self.global_invars = global_invars
        self.global_outvars = global_outvars
        self.num_stage = len(self.stages)
        self.mesh = mesh
        self.num_batch = num_batch
        self.dependency = dependency
        # the code below will be removed
        if self.dependency is None:
            self.dependency = _gen_linear_dependency(self.num_stage)

        self.schedule = schedule
        if not self.schedule:
            self.schedule = GpipeSchedule(dependency=self.dependency,
                                          mesh=self.mesh,
                                          num_batch=self.num_batch)
            logger.debug(self.schedule.pprint_schedule())

        self.sliced_meshes = self.schedule.meshes
        self._sharding_compilation_kwargs = sharding_compilation_kwargs

        # private attributes
        self._sharded_stages = []
        self._physical_meshes = []
        self._runnables = []
        self._stage_outputs = []
        self._microbatches = []

        self._prepare()

    def _prepare(self):
        # TODO(Hao): up to change, incorporate HLO merging logic
        # For each stage, compile it and get it sharding strategy
        for stage_idx, raw_stage in enumerate(self.stages):
            meshes = [self.sliced_meshes[mesh_idx]
                      for mesh_idx in self.schedule.stage_placement(stage_idx)]
            assert len(meshes) == 1
            mesh = meshes[0]
            sharded_stage = XlaShardedPipelineStage.from_jax_pipeline_stage(
                jax_pipeline_stage=raw_stage,
                mesh=mesh,
                **self._sharding_compilation_kwargs)
            self._sharded_stages.append(sharded_stage)

        # start physical mesh (launch xla runtime)
        # Ray cluster resources are allocated from this point.
        for i, mesh in enumerate(self.sliced_meshes):
            logger.debug("Launch the {}th mesh...".format(i))
            self._physical_meshes.append(mesh.get_physical_mesh())

        # Let each physical mesh to re-compile the sharded stage
        self._runnables = []
        for stage_idx, stage in enumerate(self._sharded_stages):
            mesh_indices = list(self.schedule.stage_placement(stage_idx))
            assert len(mesh_indices) == 1
            mesh_idx = mesh_indices[0]
            self._runnables.append(stage.get_runnable(self._physical_meshes[mesh_idx]))

        # prepare inputs/outputs buffers and communication between stages.
        self._stage_outputs = self._init_stage_outputs()
        self._microbatches = None

    def run(self, *args, **kwargs):
        """Run the training with pipelining."""
        # pylint: disable=too-many-locals
        assert not kwargs, "kwargs not supported"
        self._microbatches = self._make_microbatches(*args)

        global_outputs = {}
        for clock, sched in enumerate(self.schedule.schedules):
            # submit work in parallel
            logger.info(">>> At clock {}, working on tasks {}.".format(clock, sched))
            for _, task in enumerate(sched):
                # i is micro-batch idx
                # j is stage idx
                if not task:
                    continue
                batch_idx, stage_idx = task
                inputs = self._identify_stage_inputs(clock, stage_idx, batch_idx)
                # check DistributedArray colocation.
                inputs_list = self._process_stage_inputs(stage_idx, inputs)
                outputs = self._runnables[stage_idx](*inputs_list)
                pipeline_outvals, global_outvals, local_outvals = \
                    self._process_stage_outputs(stage_idx, outputs)
                if pipeline_outvals:
                    self._stage_outputs[batch_idx][stage_idx].update(pipeline_outvals)
                if local_outvals:
                    self._stage_outputs[batch_idx][stage_idx].update(local_outvals)
                if global_outvals:
                    global_outputs.update(global_outvals)
            logger.info(">>> At clock {}, pipelining jobs finished!".format(clock))

        global_outvals_list = []
        for var in self.global_outvars:
            if isinstance(var, Literal):
                global_outvals_list.append(var.val)
            else:
                key = repr(var)
                assert key in global_outputs
                val = global_outputs[key]
                global_outvals_list.append(val._value)
        logger.info(">>> All pipeline jobs done.")
        return global_outvals_list

    def _init_stage_outputs(self):
        """
        Construct the output matrix.

        it is a C by S matrix where C is the #clocks and S is #stages.
        """
        stage_outputs = [[dict() for _ in range(self.num_stage)]
                         for _ in range(self.num_batch)]
        return stage_outputs

    def _make_microbatches(self,
                           *inputs,
                           batch_dim=0,
                           batch_size=128):
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
                splits = jnp.split(array, self.num_batch, axis=batch_dim)
                for b, split in enumerate(splits):
                    microbatches[b][key] = split
        return microbatches

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
                    stage_inputs[key] = self._microbatches[batch_idx][key]
                else:
                    for ans in ancestors:
                        if key in self._stage_outputs[batch_idx][ans]:
                            stage_inputs[key] = self._stage_outputs[batch_idx][ans][key]
            elif var in stage.local_invars:
                counter_stage_idx = self.num_stage - stage_idx - 1
                stage_inputs[key] = self._stage_outputs[batch_idx][counter_stage_idx][key]
            elif var in stage.global_invars:
                stage_inputs[key] = self._microbatches[batch_idx][key]
            else:
                raise RuntimeError("Var `{}` not in any of global, pipeline or local "
                                   "var sets.".format(repr(var)))
        if len(stage_inputs) != len(stage.invars):
            raise RuntimeError("Failed to find stage inputs. "
                               "`stage_inputs` got {}, but expect {}.".format(len(stage_inputs), len(stage.invars)))
        return stage_inputs

    def _process_stage_inputs(self, stage_idx, inputs):
        """Check distributed arrays locality, and convert the input as a list."""
        stage = self.stages[stage_idx]
        inputs_list = []
        for var in stage.invars:
            key = repr(var)
            val = inputs[key]
            if isinstance(val, DistributedArray):
                mesh_idx = list(self.schedule.stage_placement(stage_idx))[0]
                if val.device_mesh == self._physical_meshes[mesh_idx]:
                    inputs_list.append(val)
                else:
                    # TODO(Hao): change to NCCL here.
                    # fetched_val = jnp.asarray(val._value)
                    fetched_val = val._value
                    inputs_list.append(fetched_val)
            else:
                inputs_list.append(val)
        return inputs_list

    def _process_stage_outputs(self, stage_idx, outputs):
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
                pipeline_outvals[key] = val
            if var in stage.global_outvars:
                global_outvals[key] = val
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
        mesh (VirtualMesh): a virtual mesh representing the entire cluster.
        sliced_mesh (List[VirtualMesh]): a list of pre-sliced virtual meshes
            to assign workers on.
        num_batch (int): number of microbatches.
        costs (List[int]): running costs of each stage.
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

        if self.num_stage % 2 != 0:
            raise RuntimeError("Gpipe schedule require an even number of stages.")
        self.num_pipeline_worker = self.num_stage // 2
        if not self.meshes:
            self.meshes = self.slice_mesh(self.original_mesh)
        if len(self.meshes) != self.num_pipeline_worker:
            raise RuntimeError("Gpipe schedule requires #meshes = #workers.")
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

    def slice_mesh(self, original_mesh):
        """
        Slice the mesh for each remote runner.

        In this impl, we guarantee the slicing follows:
        - len(sliced_meshes) == num_stages / 2 (place forward/backward in a mesh);
        - higher priority to slice over the node dimension rather than gpu dimension.

        Args:
            original_mesh: a virtual device mesh.

        Returns:
            sliced_meshes (List[Mesh]): List of meshes to spawn worker on.
        """
        output_meshes = []
        num_mesh_expected = self.num_pipeline_worker
        if not original_mesh.is_distributed:
            raise RuntimeError("SingleDeviceMesh is not supported.")
        if original_mesh.total_devices < num_mesh_expected:
            raise RuntimeError("#device < #workers.")

        num_device_per_mesh = int(original_mesh.total_devices / num_mesh_expected)
        num_device_per_host = original_mesh.num_devices_per_host
        num_host = original_mesh.num_hosts
        if num_device_per_host >= num_device_per_mesh:
            num_mesh_per_host = num_device_per_host // num_device_per_mesh
            for i in range(num_mesh_expected):
                host_idx = i // num_mesh_per_host
                mesh_idx = i % num_mesh_per_host
                ind = list(range(num_device_per_host))
                mesh = original_mesh.slice(0, [host_idx])\
                    .slice(1, ind[mesh_idx * num_device_per_mesh:(mesh_idx + 1) * num_device_per_mesh])
                output_meshes.append(mesh)
        else:
            num_host_per_mesh = math.ceil(num_device_per_mesh / num_device_per_host)
            ind = list(range(num_host))
            for i in range(num_mesh_expected):
                output_meshes.append((
                    original_mesh.slice(0, ind[num_host_per_mesh * i:num_host_per_mesh * (i + 1)])))
        return output_meshes
