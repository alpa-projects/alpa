"""Abstract runtime classes and methods."""
import ray
from abc import ABCMeta
from abc import abstractmethod
from typing import Sequence, List, Any

from jax.core import Var

from parax.device_mesh import PhysicalDeviceMesh
from parax.pipeline_parallel.cross_mesh_resharding import CrossMeshCommunicator, \
    CollectiveGroup, ReshardingTask
from parax.pipeline_parallel.stage import PipelineStage, XlaShardedPipelineStage


class BaseRuntime(metaclass=ABCMeta):

    def __init__(self, *, pipeline_stages, global_invars, global_outvars,
                 physical_meshes, **kwargs):
        """
        An abstract class for pipeline-parallel runtime.

        Args:
            pipeline_stages (Sequence[PipelineStage, XlaShardedPipelineStage]):  pipeline stages.
            global_invars (Sequence[Var]): input variables.
            global_outvars (input variables.): output varialbes.
        """
        self.global_invars = global_invars
        self.global_outvars = global_outvars
        self.stages = pipeline_stages
        self.physical_meshes = physical_meshes

    @property
    def num_mesh(self):
        """Return the number of meshes involved in the pipeline-parallel runtime."""
        if self.physical_meshes == None:
            return 0
        return len(self.physical_meshes)

    @property
    def num_stage(self):
        """Return the number of stages."""
        return len(self.stages)

    @abstractmethod
    def shutdown(self):
        """Shutdown the pipeline runtime."""
        raise NotImplementedError()

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run function."""
        raise NotImplementedError()


class BaseDistributedRuntime(BaseRuntime):

    def __init__(self,
                 *,
                 pipeline_stages: List[XlaShardedPipelineStage],
                 global_invars: List[Var],
                 grad_dummy_invars,
                 global_outvars: List[Var],
                 physical_meshes: List[PhysicalDeviceMesh],
                 dependency,
                 schedule,
                 is_batch,
                 num_batch=1,
                 **kwargs):
        """
        A base class of distributed pipeline-parallel runtime.

        This class abstract out some communication-related methods that will be shared
        across different distributed runtime implementations.

        Args:
            pipeline_stages (List[XlaShardedPipelineStage]): list of pipeline stage programs.
            global_invars (List[Var]): input variables.
            global_outvars (List[Var]): output variables.
            physical_meshes (List[PhysicalDeviceMesh]): the cluster meshes to pipeline over.
            sharding_compilation_kwargs (dict): a dict of keyword arguments as the sharding
                compilation parameters.
            dependency (np.array): dependency between stages as an adjacency matrix.
            num_batch (int): number of microbatches.
            schedule (GpipeSchedule): schedule to follow to execute the pipeline.
        """
        super(BaseDistributedRuntime,
              self).__init__(pipeline_stages=pipeline_stages,
                             global_invars=global_invars,
                             global_outvars=global_outvars,
                             physical_meshes=physical_meshes)
        self.grad_dummy_invars = grad_dummy_invars
        self.is_batch = is_batch
        self.dependency = dependency
        self.schedule = schedule
        self.num_batch = num_batch

        # Before we can setup the communicator, we need generate the sharding spec for each stage.
        self._precompile_sharding_specs()

        # Communication-related setup
        # Based on dependency and sharding specs, infer communication spec (cross-mesh)。
        self._communicator = CrossMeshCommunicator(self.stages, self.schedule)
        self._collective_groups: List[List[Any]] = [
            [None for _ in range(self.num_mesh)] for _ in range(self.num_mesh)
        ]
        self._resharding_tasks = [
            [dict() for _ in range(self.num_mesh)] for _ in range(self.num_mesh)
        ]
        # pre-setup
        self._establish_nccl_groups()
        self._create_resharding_and_get_send_recv_tasks()
        self._put_resharding_tasks()

    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def sync(self):
        """Sync all workers' GPU activities."""
        all_workers = [w for mesh in self.physical_meshes for w in mesh.workers]
        ray.get([w.sync.remote() for w in all_workers])

    def shutdown(self):
        raise NotImplementedError()

    def _precompile_sharding_specs(self):
        """Run get_compiled() for each stage to get sharding specs."""
        for stage_idx, stage in enumerate(self.stages):
            mesh_indices = list(self.schedule.stage_placement(stage_idx))
            assert len(mesh_indices) == 1
            mesh_idx = mesh_indices[0]
            stage.get_compiled(self.physical_meshes[mesh_idx])

    def _establish_nccl_groups(self):
        """
        Identify and create NCCL groups based on resharding specs.

        We establish one collective group between two physical meshes, covering all the devices in
        these two meshes that require NCCL communication.
        TODO(Hao): we might need to improve this part later for adding scatter-gather optimization.

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
                if not device_str_groups[i][j]:
                    continue
                cg = CollectiveGroup(list(device_str_groups[i][j]),
                                     self.physical_meshes[i],
                                     self.physical_meshes[j])
                cg.instantiate()
                self._collective_groups[i][j] = cg
                self._collective_groups[j][i] = cg

    def _create_resharding_and_get_send_recv_tasks(self):
        """
        Initialize all resharding (send/recv) tasks.

        In this function, we do the following:
        1. we create a resharding task for each resharding spec
        2. for each resharding task, we generate all related send/recv tasks.
        """

        # Create resharding tasks for each var
        for src_mesh_idx, dst_mesh_idx, var_spec_map \
                in self._communicator.task_spec_iter():
            for key, spec in var_spec_map.items():
                cg = self._collective_groups[src_mesh_idx][dst_mesh_idx]
                src_mesh = self.physical_meshes[src_mesh_idx]
                dst_mesh = self.physical_meshes[dst_mesh_idx]
                t = ReshardingTask(spec, cg, src_mesh, dst_mesh)
                t.get_send_recv_tasks()
                self._resharding_tasks[src_mesh_idx][dst_mesh_idx][key] = t

    def _put_resharding_tasks(self):
        """
        Setup all send/recv tasks on the remote workers.

        for each resharding task, we put task-related info (rank, group) in
        the corresponded remote workers *in advance*. At runtime, we use the
        source distributed array to index the task and perform cross-mesh
        communication; in this way, we avoid creating/transferring task info
        at runtime loop.
        """
        for i in range(self.num_mesh):
            for j in range(self.num_mesh):
                for _, task in self._resharding_tasks[i][j].items():
                    task.put_send_recv_tasks()

    def _destroy_collective_groups(self):
        for i in range(self.num_mesh):
            for j in range(self.num_mesh):
                if i < j and self._collective_groups[i][j]:
                    self._collective_groups[i][j].destroy()
