"""A driver-centralized pipeline-parallel runtime."""
import copy
import logging

from jax.core import Literal
import jax.numpy as jnp
import ray

from parax.device_mesh import DistributedArray, ReplicatedDistributedArray
from parax.mesh_executable import AllocZeroBufferDriverExecutable
from parax.pipeline_parallel.cross_mesh_resharding import CrossMeshCommunicator, CollectiveGroup, ReshardingTask
from parax.global_env import global_config
from parax.timer import timers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

timer_names = {
    "overall": "average",
    "compute": "sum",
    "resharding": "sum",
}


def reset_pipeline_runtime_benchmark_timers():
    """Reset all related timers."""
    logger.debug(">>> Reset all timers.")
    for t in timer_names:
        timers(t).reset()


# TODO (abstract out some methods to base_runtime.py)
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
                 num_batch=1):
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
        self._runnables = None
        self._env = None
        self._initial_var_reference_count = None

        # for resharding
        self._communicator = None
        self._collective_groups = [
            [None for _ in range(self.num_mesh)] for _ in range(self.num_mesh)
        ]
        self._resharding_tasks = [[dict() for _ in range(self.num_mesh)]
                                  for _ in range(self.num_mesh)]

        # Init and warm-up
        self._prepare_runables()
        # use virtual mesh and VDAs to generate sharding task spec and strategies
        self._prepare_communicator()
        # create all tasks and put buffers
        self._initialize_resharding_tasks()
        self._prepare_gradient_accumulation()
        self._prepare_reference_count()

    @property
    def num_mesh(self):
        """Return the number of meshes in the pipeline job."""
        return len(self.physical_meshes)

    def _prepare_runables(self):
        # Let each physical mesh to re-compile the sharded stage
        self._runnables = []
        for stage_idx, stage in enumerate(self.stages):
            mesh_indices = list(self.schedule.stage_placement(stage_idx))
            assert len(mesh_indices) == 1
            mesh_idx = mesh_indices[0]
            self._runnables.append(
                stage.get_runnable(self.physical_meshes[mesh_idx]))

    def _prepare_communicator(self):
        # Based on dependency and sharding specs, infer communication spec (cross-mesh).
        self._communicator = CrossMeshCommunicator(self.stages, self.schedule)

        # Now we establish NCCL collective groups and communicators
        # because we need physical meshes we have to do this out of the CrossMeshCommunicator class.
        self._establish_nccl_groups()

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

    def _initialize_resharding_tasks(self):
        """
        Launch all resharding tasks and do all necessary work.

        In this function, we do the following:
        1. we create a resharding task for each resharding spec
        2. for each resharding task, we put task-related info (rank, group) in the
           corresponded remote workers *in advance*.
        At runtime, we use the source distributed array to index the task and perform
        cross-mesh communication; in this way, we avoid creating/transferring task
        info at runtime loop.
        """
        # Create resharding tasks for each var
        for src_mesh_idx, dst_mesh_idx, var_spec_map \
                in self._communicator.task_spec_iter():
            for key, spec in var_spec_map.items():
                cg = self._collective_groups[src_mesh_idx][dst_mesh_idx]
                src_mesh = self.physical_meshes[src_mesh_idx]
                dst_mesh = self.physical_meshes[dst_mesh_idx]
                t = ReshardingTask(spec, cg, src_mesh, dst_mesh)
                # TODO(Hao): double check/optimize this function
                t.prepare_send_recv_tasks()
                self._resharding_tasks[src_mesh_idx][dst_mesh_idx][key] = t

    def _prepare_gradient_accumulation(self):
        # Allocate buffers for accumulated gradients
        mesh_num = len(self.physical_meshes)
        mesh_grad_vars = [dict() for _ in range(mesh_num)]
        # collect buffers to allocate in each mesh
        for stage_idx, stage in enumerate(self.stages):
            mesh_indices = list(self.schedule.stage_placement(stage_idx))
            assert len(mesh_indices) == 1
            mesh_idx = mesh_indices[0]
            grad_var_spec_dict = mesh_grad_vars[mesh_idx]
            input_specs = stage.input_sharding_specs
            for var_idx, invar in enumerate(stage.invars):
                if invar in self.grad_dummy_invars:
                    if invar in grad_var_spec_dict:
                        raise NotImplemented(
                            f'accumulate {invar} in a mesh but multiple stages')
                    grad_var_spec_dict[invar] = input_specs[var_idx]
        # create executable for each mesh
        self.allocate_zero_buffers = []
        if len(grad_var_spec_dict):
            for mesh_idx in range(mesh_num):
                grad_var_spec_dict = mesh_grad_vars[mesh_idx]
                grad_vars, grad_sharding_specs = list(
                    zip(*grad_var_spec_dict.items()))
                self.allocate_zero_buffers.append(
                    (AllocZeroBufferDriverExecutable(
                        physical_mesh=self.physical_meshes[mesh_idx],
                        grad_vars=grad_vars,
                        grad_sharding_specs=grad_sharding_specs).
                     get_driver_callable(), grad_vars))

    def _prepare_reference_count(self):
        self._initial_var_reference_count = {}
        for i, var in enumerate(self.global_invars):
            for b in range(self.num_batch):
                self._initial_var_reference_count[(b, repr(var))] = 0

        for _, allocate_vars in self.allocate_zero_buffers:
            for var in allocate_vars:
                self._initial_var_reference_count[(self.num_batch - 1,
                                                   repr(var))] = 0

        for clock, sched in enumerate(self.schedule.schedules):
            for _, task in enumerate(sched):
                if not task:
                    continue
                batch_idx, stage_idx = task
                stage = self.stages[stage_idx]

                for var in stage.invars:
                    key = (batch_idx, repr(var))
                    # TODO: only work for GPipeSchedule
                    if var in self.grad_dummy_invars and batch_idx < self.num_batch - 1:
                        key = (batch_idx + 1, self.grad_dummy_invars[var])
                    self._initial_var_reference_count[key] += 1

                for var in stage.outvars:
                    key = (batch_idx, repr(var))
                    self._initial_var_reference_count[key] = 0

        for var in self.global_outvars:
            if not isinstance(var, Literal):
                key = (0, repr(var))
                self._initial_var_reference_count[key] += 1

    def run(self, *args, **kwargs):
        """Run the training with pipelining."""
        # pylint: disable=too-many-locals
        assert not kwargs, "kwargs not supported"

        timers("overall").start(sync_func=self.sync)

        # timers("overall").start()
        timers("initialize_microbatch").start()
        self._prepare_env(*args)
        timers("initialize_microbatch").stop()

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

                timers("identify_input").start()
                inputs = self._identify_stage_inputs(clock, stage_idx,
                                                     batch_idx)
                timers("identify_input").suspend()

                timers("resharding").start()
                # check DistributedArray colocation and do resharding if necessary
                inputs_list = self._process_stage_inputs(stage_idx, inputs)
                timers("resharding").suspend()

                timers("compute").start()
                skip_grad_sync = True
                # TODO(yonghao): only works for GPipeSchedule
                if stage_idx >= self.num_mesh and batch_idx == 0:
                    skip_grad_sync = False
                outputs = self._runnables[stage_idx](
                    *inputs_list, skip_grad_sync=skip_grad_sync)
                timers("compute").suspend()

                self._process_stage_outputs(batch_idx, stage_idx, outputs)
            logger.debug(
                ">>> At clock {}, pipelining jobs finished!".format(clock))

        # stop loop timers
        for timer_name in ["compute", "resharding", "identify_input"]:
            timers(timer_name).stop()

        global_outvals_list = []
        for var in self.global_outvars:
            if isinstance(var, Literal):
                global_outvals_list.append(var.val)
            else:
                key = (0, repr(var))
                val = self._pop_var(key)
                global_outvals_list.append(val)
        logger.debug(">>> All pipeline jobs done.")
        timers("overall").stop(sync_func=self.sync)
        # timers("overall").stop()

        # Make sure reference counting is correct
        assert len(self._env) == 0, (self._env, self._env_reference_count)

        return global_outvals_list

    def get_execution_time_costs(self, warmup=2, timer_name="overall"):
        if timer_name not in timer_names:
            raise RuntimeError(
                "Unrecognized timer name for pipeline parallel runtime. "
                "Query timer name from the following: {}.".format(
                    timer_names.keys()))
        return timers(timer_name).costs[warmup:]

    def _prepare_env(self, *inputs, batch_dim=0):
        assert self._initial_var_reference_count is not None
        self._env = {}
        self._env_reference_count = copy.deepcopy(
            self._initial_var_reference_count)
        assert len(inputs) == len(self.global_invars)
        for i, var in enumerate(self.global_invars):
            if not self.is_batch[i]:
                splits = [inputs[i]] * self.num_batch
            else:
                splits = jnp.split(inputs[i], self.num_batch, axis=batch_dim)
            for b, split in enumerate(splits):
                key = (b, repr(var))
                assert key in self._env_reference_count
                # Do not include unused inputs
                if self._env_reference_count[key] == 0:
                    del self._env_reference_count[key]
                else:
                    self._env[key] = split

        for allocate_callable, allocate_vars in self.allocate_zero_buffers:
            allocate_vals = allocate_callable()
            for val, var in zip(allocate_vals, allocate_vars):
                self._env[(self.num_batch - 1, repr(var))] = val

    def _pop_var(self, key):
        assert key in self._env and key in self._env_reference_count
        result = self._env[key]
        self._env_reference_count[key] -= 1
        if self._env_reference_count[key] == 0:
            del self._env_reference_count[key]
            del self._env[key]
        return result

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
        stage_inputs = {}
        stage = self.stages[stage_idx]
        for var in stage.invars:
            key = (batch_idx, repr(var))
            # TODO: only work for GPipeSchedule
            if var in self.grad_dummy_invars and batch_idx < self.num_batch - 1:
                key = (batch_idx + 1, self.grad_dummy_invars[var])
            stage_inputs[repr(var)] = self._pop_var(key)
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
                    if global_config.precompile_resharding_tasks:
                        # The tasks have been prepared.
                        if key not in self._resharding_tasks[src_mesh_idx][mesh_idx]:
                            raise RuntimeError("Cannot find a ready resharding task.")
                        resharding_task = self._resharding_tasks[src_mesh_idx][mesh_idx][key]
                        resharded_val = resharding_task.do_prepared(val)
                    else:
                        task_spec = self._communicator.resharding_specs[
                            src_mesh_idx][mesh_idx][key]
                        assert task_spec
                        task = ReshardingTask(task_spec,
                                              self._collective_groups[src_mesh_idx][mesh_idx],
                                              self.physical_meshes[src_mesh_idx],
                                              self.physical_meshes[mesh_idx])
                        resharded_val = task.do(val)
                    inputs_list.append(resharded_val)
            elif isinstance(val, ReplicatedDistributedArray):
                mesh_idx = list(self.schedule.stage_placement(stage_idx))[0]
                # find the local copy of val
                local_replica = val.get_replica_on_mesh(self.physical_meshes[mesh_idx])
                inputs_list.append(local_replica)
            else:
                # it is a DeviceArray
                inputs_list.append(val)
        return inputs_list

    def _process_stage_outputs(self, batch_idx, stage_idx, outputs):
        stage = self.stages[stage_idx]
        for var, val in zip(stage.outvars, outputs):
            key = (batch_idx, repr(var))
            if key not in self._env:
                self._env[key] = val
            else:
                if isinstance(self._env[key], DistributedArray):
                    # construct the ReplicatedDA, and put it back to env
                    rda = ReplicatedDistributedArray([self._env[key].device_mesh], [self._env[key]])
                    rda.add_replica(val.device_mesh, val)
                    self._env[key] = rda
                elif isinstance(self._env[key], ReplicatedDistributedArray):
                    self._env[key].add_replica(val.device_mesh, val)
                else:
                    raise RuntimeError("Unrecognized type.")

    def sync(self):
        all_workers = [w for mesh in self.physical_meshes for w in mesh.workers]
        ray.get([w.sync.remote() for w in all_workers])

    def shutdown(self):
        """Shutdown the pipeline runtime."""
        # Recycle the groups an Ray resources
        for i in range(self.num_mesh):
            for j in range(self.num_mesh):
                if i < j:
                    self._collective_groups[i][j].destroy()

        # Recycle the recompiled runnables
        del self._runnables

        # Destroy the Ray workers
        if not self.physical_meshes:
            raise RuntimeError("No physical meshes spawned yet in "
                               "the runtime before shutting down.")
        for mesh in self.physical_meshes:
            mesh.shutdown()

        # reset all timers
        reset_pipeline_runtime_benchmark_timers()
