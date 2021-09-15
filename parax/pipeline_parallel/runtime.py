"""Distributed JAX pipeline parallelism."""
import time
from collections import OrderedDict
import functools
import logging
import math

from parax.mesh_executable import AllocZeroBufferDriverExecutable

import numpy as np
import ray
from jax.core import Literal
import jax.numpy as jnp

from parax.device_mesh import DistributedArray
from parax.pipeline_parallel.cross_mesh_resharding import CrossMeshCommunicator, CollectiveGroup, ReshardingTask
from parax.pipeline_parallel.stage import StrVarPipelineStage

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


class Jax3DPipeline:  # pylint: disable=too-many-instance-attributes
    """
    A class to coordinate 3D parallelism.

    This class implements pipeline parallelism and sharding.

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
                 grad_dummy_invars,
                 global_outvars,
                 physical_meshes,
                 dependency,
                 schedule,
                 is_batch,
                 num_batch=1,
                 profile=False):
        self.stages = pipeline_stages
        self.global_invars = global_invars
        self.grad_dummy_invars = grad_dummy_invars
        self.global_outvars = global_outvars
        self.global_outvars_repr_set = set()
        for var in self.global_outvars:
            if not isinstance(var, Literal):
                self.global_outvars_repr_set.add(repr(var))
        self.num_stage = len(self.stages)
        self.is_batch = is_batch
        self.num_batch = num_batch
        self.dependency = dependency
        self.schedule = schedule
        self.physical_meshes = physical_meshes

        # private attributes
        self._profile = profile
        self._runnables = []
        self._stage_outputs = []
        self._microbatches = []

        # for resharding
        self._communicator = None
        self._collective_groups = [
            [None for _ in range(self.num_mesh)] for _ in range(self.num_mesh)
        ]
        self._prepare()

    @property
    def num_mesh(self):
        """Return the number of meshes in the pipeline job."""
        return len(self.physical_meshes)

    def _prepare(self):
        # Let each physical mesh to re-compile the sharded stage
        self._runnables = []
        for stage_idx, stage in enumerate(self.stages):
            mesh_indices = list(self.schedule.stage_placement(stage_idx))
            assert len(mesh_indices) == 1
            mesh_idx = mesh_indices[0]
            self._runnables.append(
                stage.get_runnable(self.physical_meshes[mesh_idx]))

        # Based on dependency and sharding specs, infer communication spec (cross-mesh).
        self._communicator = CrossMeshCommunicator(self.stages, self.schedule)

        # Now we establish NCCL collective groups and communicators
        # because we need physical meshes we have to do this out of the CrossMeshCommunicator class.
        self._establish_nccl_groups()

        # prepare inputs/outputs buffers and communication between stages.
        self._stage_outputs = self._init_stage_outputs()
        self._microbatches = None

        # Allocate buffers for accumulated gradients
        mesh_num = len(self.physical_meshes)
        mesh_grad_vars = [dict() for _ in range(mesh_num)]
        # collect buffers to allocate in each mesh
        for stage_idx, stage in enumerate(self.stages):
            mesh_indices = list(self.schedule.stage_placement(stage_idx))
            assert len(mesh_indices) == 1
            mesh_idx = mesh_indices[0]
            grad_var_specs = mesh_grad_vars[mesh_idx]
            input_specs = stage.input_sharding_specs
            for var_idx, invar in enumerate(stage.invars):
                if invar in self.grad_dummy_invars:
                    if invar in grad_var_specs:
                        raise NotImplemented(
                            f'accumulate {invar} in a mesh but multiple stages')
                    grad_var_specs[invar] = input_specs[var_idx]
        # create executable for each mesh
        self.allocate_zero_buffers = []
        for mesh_idx in range(mesh_num):
            grad_vars_specs = mesh_grad_vars[mesh_idx]
            grad_vars = list(grad_vars_specs.keys())
            self.allocate_zero_buffers.append((
                AllocZeroBufferDriverExecutable(
                    physical_mesh=self.physical_meshes[mesh_idx],
                    grad_vars=grad_vars,
                    grad_vars_specs=mesh_grad_vars[mesh_idx]).
                get_driver_callable(), grad_vars))

    def _establish_nccl_groups(self):
        """
        Identify and create NCCL groups based on specs.

        We establish one collective group between two physical meshes, covering all the devices in
        these two meshes that require NCCL communication.

        Returns:
            device_str_groups (List[List[set]]): a num_mesh x num_mesh matrix. Only entries at
                device_str_groups[i][j] (i < j) are filled, entries with i > j are None, because
                (spec[i][j], spec[j][i]) will share collective groups.
        """
        device_str_groups = [
            [set() for _ in range(self.num_mesh)] for _ in range(self.num_mesh)
        ]
        # Merge (i, j) and (j, i)
        for i, j, var_spec_map in self._communicator.task_spec_iter():
            participants = set()
            for _, spec in var_spec_map.items():  # for each var
                participants = participants | spec.get_participant_device_strs()
            if i <= j:
                device_str_groups[i][j] = device_str_groups[i][j] | participants
            else:
                device_str_groups[j][i] = device_str_groups[j][i] | participants

        # construct groups
        for i in range(self.num_mesh):
            for j in range(self.num_mesh):
                if i >= j:
                    assert not device_str_groups[i][j]
                    continue
                cg = CollectiveGroup(device_str_groups[i][j],
                                     self.physical_meshes[i],
                                     self.physical_meshes[j])
                cg.instantiate()
                self._collective_groups[i][j] = cg
                self._collective_groups[j][i] = cg

    def run(self, *args, **kwargs):
        """Run the training with pipelining."""
        # pylint: disable=too-many-locals
        assert not kwargs, "kwargs not supported"

        if self._profile:
            self.overall_time = 0.0
            self.stage_compute_time = 0.0
            self.driver_stitching_time = 0.0
            self.process_stage_inputs_time = 0.0
            self.indentify_stage_inputs_time = 0.0
            self.process_stage_outputs_time = 0.0
            self.make_microbatch_time = 0.0
            self.unknown_time = 0.0

            overall_time_tic = 0.0
            stage_compute_time_tic = 0.0
            process_stage_inputs_tic = 0.0
            identify_stage_inputs_tic = 0.0
            process_stage_outputs_tic = 0.0
            make_microbatch_tic = 0.0
            unknown_tic = 0.0

        if self._profile:
            overall_time_tic = time.time()
            make_microbatch_tic = time.time()
        self._microbatches = self._make_microbatches(*args)

        if self._profile:
            self.make_microbatch_time = time.time() - make_microbatch_tic

        global_outputs = {}

        for clock, sched in enumerate(self.schedule.schedules):
            # submit work in parallel
            logger.debug(">>> At clock {}, working on tasks {}.".format(
                clock, sched))
            for _, task in enumerate(sched):
                # i is micro-batch idx
                # j is stage idx
                if not task:
                    continue
                batch_idx, stage_idx = task
                if self._profile:
                    identify_stage_inputs_tic = time.time()

                inputs = self._identify_stage_inputs(clock, stage_idx,
                                                     batch_idx)

                if self._profile:
                    self.indentify_stage_inputs_time += time.time(
                    ) - identify_stage_inputs_tic
                    process_stage_inputs_tic = time.time()
                # check DistributedArray colocation and do resharding if necessary
                inputs_list = self._process_stage_inputs(stage_idx, inputs)
                if self._profile:
                    self.process_stage_inputs_time += time.time(
                    ) - process_stage_inputs_tic
                    stage_compute_time_tic = time.time()
                outputs = self._runnables[stage_idx](*inputs_list)
                if self._profile:
                    self.stage_compute_time += time.time(
                    ) - stage_compute_time_tic
                    process_stage_outputs_tic = time.time()
                outvals = self._process_stage_outputs(stage_idx, outputs)

                if self._profile:
                    self.process_stage_outputs_time += time.time(
                    ) - process_stage_outputs_tic
                # TODO: Add reference counting here to reduce memory usage
                self._stage_outputs[batch_idx][stage_idx].update(outvals)
                for key, val in outvals.items():
                    if key in self.global_outvars_repr_set:
                        global_outputs[key] = val
            logger.debug(
                ">>> At clock {}, pipelining jobs finished!".format(clock))

        if self._profile:
            unknown_tic = time.time()
        global_outvals_list = []
        for var in self.global_outvars:
            if isinstance(var, Literal):
                global_outvals_list.append(var.val)
            else:
                key = repr(var)
                assert key in global_outputs
                val = global_outputs[key]
                # global_outvals_list.append(val._value)
                global_outvals_list.append(val)
        logger.debug(">>> All pipeline jobs done.")

        if self._profile:
            self.unknown_time = time.time() - unknown_tic
            self.overall_time = time.time() - overall_time_tic
            self.report_profiling_results()
        return global_outvals_list

    def _init_stage_outputs(self):
        """
        Construct the output matrix.

        it is a C by S matrix where C is the #clocks and S is #stages.
        """
        stage_outputs = [[dict()
                          for _ in range(self.num_stage)]
                         for _ in range(self.num_batch)]
        return stage_outputs

    def _make_microbatches(self, *inputs, batch_dim=0):
        assert len(inputs) == len(self.global_invars)
        microbatches = [dict() for _ in range(self.num_batch)]
        for i, var in enumerate(self.global_invars):
            key = repr(var)
            array = inputs[i]
            if not self.is_batch[i]:
                # empty shape means it is not the input batch
                # no need to split
                # ref = ray.put(inputs[i])
                for b in range(self.num_batch):
                    microbatches[b][key] = inputs[i]
            else:
                splits = jnp.split(array, self.num_batch, axis=batch_dim)
                for b, split in enumerate(splits):
                    microbatches[b][key] = split
        for allocate_info in self.allocate_zero_buffers:
            allocate_callable, allocate_vars = allocate_info
            allocate_vals = allocate_callable()
            for val, var in zip(allocate_vals, allocate_vars):
                key = repr(var)
                microbatches[self.num_batch - 1][key] = val
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
        ancestors = list(
            np.squeeze(np.argwhere(self.dependency[stage_idx] == 1), axis=1))
        for var in stage.invars:
            key = repr(var)
            # TODO(yonghao): record where to obtain variables at compile time
            if var in self.global_invars:
                stage_inputs[key] = self._microbatches[batch_idx][key]
            elif var in self.grad_dummy_invars:
                # TODO(yonghao): only work for GPipeSchedule
                if batch_idx == self.num_batch - 1:
                    stage_inputs[key] = self._microbatches[batch_idx][key]
                else:
                    _key = self.grad_dummy_invars[var]
                    stage_inputs[key] = self._stage_outputs[batch_idx +
                                                            1][stage_idx][_key]
            else:
                for ans in ancestors:
                    if key in self._stage_outputs[batch_idx][ans]:
                        stage_inputs[key] = self._stage_outputs[batch_idx][ans][
                            key]
        if len(stage_inputs) != len(stage.invars):
            raise RuntimeError("Failed to find stage inputs. "
                               "`stage_inputs` got {}, but expect {}.".format(
                                   len(stage_inputs), len(stage.invars)))
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
                if val.device_mesh == self.physical_meshes[mesh_idx]:
                    inputs_list.append(val)
                else:
                    # find the corresponded resharding task
                    src_mesh_idx = self.physical_meshes.index(val.device_mesh)
                    task_spec = self._communicator.resharding_specs[
                        src_mesh_idx][mesh_idx][key]
                    assert task_spec
                    task = ReshardingTask(
                        task_spec,
                        self._collective_groups[src_mesh_idx][mesh_idx], val)
                    resharded_val = task.do()
                    inputs_list.append(resharded_val)
            else:
                inputs_list.append(val)
        return inputs_list

    def _process_stage_outputs(self, stage_idx, outputs):
        stage = self.stages[stage_idx]
        outvals = {repr(var): val for var, val in zip(stage.outvars, outputs)}
        return outvals

    def report_profiling_results(self):
        stitching_time = self.overall_time - self.stage_compute_time
        logger.info(
            "Overall time: {}, compute time: {}, ray stitching time: {}.".
            format(self.overall_time, self.stage_compute_time, stitching_time))
        logger.info("Identify stage input time: {}.".format(
            self.indentify_stage_inputs_time))
        logger.info("Process stage input time: {}.".format(
            self.process_stage_inputs_time))
        logger.info("Process stage ouptput time: {}.".format(
            self.process_stage_outputs_time))
        logger.info("Make microbatch time: {}.".format(
            self.make_microbatch_time))
        logger.info("Unknown time: {}.".format(self.unknown_time))


def gen_linear_dependency(num_stage):
    """Generate a linear dependency matrix."""
    d = np.zeros([num_stage, num_stage], dtype=np.int)
    for i in range(num_stage - 1):
        d[i + 1][i] = 1
    return d


def gen_linear_pipeline_dependency(num_stage):
    """
    Generate a dependency matrix that marks the neighbors and forward/backward
    stage pairs as neighbors.
    """
    assert num_stage % 2 == 0
    d = np.zeros([num_stage, num_stage], dtype=np.int)
    for i in range(num_stage - 1):
        d[i + 1][i] = 1
    for i in range(num_stage // 2):
        d[num_stage - 1 - i][i] = 1
    return d


def gen_linear_pipeline_dependency_with_apply(num_stage, mesh_num, apply_deps):
    """
    Generate dependency matrix marks compute grad and apply grad
    """
    d = np.zeros([num_stage, num_stage], dtype=np.int)
    for i in range(mesh_num * 2 - 1):
        d[i + 1][i] = 1
    for i in range(mesh_num):
        d[mesh_num * 2 - 1 - i][i] = 1
    for pair in apply_deps:
        d[pair[0]][pair[1]] = 1
    return d


class GpipeSchedule:
    """
    Construct a Gpipe-like schedule.

    Args:
        dependency (np.array): dependency adjacency matrix.
        mesh (VirtualMesh): a virtual mesh representing the entire cluster.
        sliced_mesh (List[VirtualMesh]): a list of pre-sliced virtual meshes
            to assign workers on.
        num_pipeline_worker (int): 
        apply_grad_schedule (Dict[int, int]): A map from apply grad's stage idx
            to the worker it is assigned
        num_batch (int): number of microbatches.
        costs (List[int]): running costs of each stage.
    """

    def __init__(self,
                 *,
                 dependency,
                 mesh,
                 num_pipeline_worker,
                 apply_grad_schedule,
                 sliced_meshes=None,
                 num_batch=1,
                 costs=None):
        self.dependency = dependency
        self.original_mesh = mesh
        self.meshes = sliced_meshes
        self.apply_grad_schedule = apply_grad_schedule
        self.num_batch = num_batch
        self.costs = costs
        self.num_stage = dependency.shape[0]

        self.num_pipeline_worker = num_pipeline_worker
        # TODO (zhuohan): Seperate device placement and runtime scheduling
        if not self.meshes:
            # These are virtual meshes
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
        # apply grad schedules
        scheds = [None] * n
        for stage_idx, worker in self.apply_grad_schedule.items():
            scheds[worker] = (0, stage_idx)
        schedules.append(scheds)
        return schedules

    def pprint_schedule(self):
        """Pretty print the schedule."""
        printout = "\n"
        device_str = " ".join([
            "{:<8}".format("d" + str(d))
            for d in range(self.num_pipeline_worker)
        ])
        printout = printout + "Clock {:<2}: {} \n".format("k", device_str)
        for clock, scheds in enumerate(self.schedules):
            sched_str = " ".join(
                ["{:<8}".format(str(sched)) for sched in scheds])
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

    @property
    def num_worker(self):
        """Return the number of workers (physical meshes)."""
        return self.num_pipeline_worker

    @property
    def num_mesh(self):
        """Return the number of meshes in the schedule."""
        return self.num_pipeline_worker

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

        num_device_per_mesh = int(original_mesh.total_devices /
                                  num_mesh_expected)
        num_device_per_host = original_mesh.num_devices_per_host
        num_host = original_mesh.num_hosts
        if num_device_per_host >= num_device_per_mesh:
            num_mesh_per_host = num_device_per_host // num_device_per_mesh
            for i in range(num_mesh_expected):
                host_idx = i // num_mesh_per_host
                mesh_idx = i % num_mesh_per_host
                ind = list(range(num_device_per_host))
                mesh = original_mesh.slice(0, [host_idx])\
                    .slice(1, [ind[mesh_idx * num_device_per_mesh:(mesh_idx + 1) * num_device_per_mesh]])
                output_meshes.append(mesh)
        else:
            num_host_per_mesh = math.ceil(num_device_per_mesh /
                                          num_device_per_host)
            ind = list(range(num_host))
            for i in range(num_mesh_expected):
                output_meshes.append((original_mesh.slice(
                    0, ind[num_host_per_mesh * i:num_host_per_mesh * (i + 1)])))
        return output_meshes
