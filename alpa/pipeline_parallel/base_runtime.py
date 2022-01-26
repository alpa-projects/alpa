"""Abstract runtime classes and methods."""
import logging
import time

from abc import ABCMeta, abstractmethod
from typing import List, Any

from jax.core import Var
import ray

from alpa.util import OrderedSet
from alpa.device_mesh import PhysicalDeviceMesh
from alpa.pipeline_parallel.cross_mesh_resharding import (CrossMeshCommunicator,
                                                          CollectiveGroup,
                                                          SymbolicReshardingTask
                                                         )
from alpa.pipeline_parallel.computation import XlaShardedPipelineComputation
from alpa.global_env import global_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseRuntime(metaclass=ABCMeta):
    """Abstract class for pipeline runtime."""

    def __init__(self, *, pipeline_stages, global_invars, global_outvars,
                 physical_meshes, **kwargs):
        """
        An abstract class for pipeline-parallel runtime.

        Args:
            pipeline_stages (Sequence[PipelineComputation, XlaShardedPipelineComputation]): pipeline stages.
            global_invars (Sequence[Var]): input variables.
            global_outvars (Sequence[Var]): output varialbes.
            physical_meshes (Sequence[PhysicalDeviceMesh]): input physical meshes.
        """
        self.global_invars = global_invars
        self.global_outvars = global_outvars
        self.stages = pipeline_stages
        self.physical_meshes = physical_meshes

    @property
    def num_mesh(self):
        """Return the number of meshes involved in the pipeline-parallel runtime."""
        if self.physical_meshes is None:
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
    """Abstract class for distributed pipepline runtime."""

    def __init__(self,
                 *,
                 pipeline_stages: List[XlaShardedPipelineComputation],
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
            pipeline_stages (List[XlaShardedPipelineComputation]): list of pipeline stage programs.
            global_invars (List[Var]): input variables.
            grad_dummy_invars (List[Var]): vars for gradient accumulation.
            global_outvars (List[Var]): output variables.
            physical_meshes (List[PhysicalDeviceMesh]): the cluster meshes to pipeline over.
            dependency (np.array): dependency between stages as an adjacency matrix.
            schedule (GpipeSchedule): schedule to follow to execute the pipeline.
            is_batch (): indicator of the batch dimension.
            num_batch (int): number of microbatches.
            kwargs (dict): a dict of keyword arguments as the sharding
                compilation parameters.
        """
        super().__init__(pipeline_stages=pipeline_stages,
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
            [{} for _ in range(self.num_mesh)] for _ in range(self.num_mesh)
        ]

        # TODO(Hao): this establish_nccl_groups needs to be improved to cover allgather.
        self._establish_nccl_groups()

        start_time = time.time()
        self._compile_resharding_tasks()
        end_time = time.time()
        logger.debug(
            f"Compile resharding tasks takes {end_time - start_time:.2f}")

    def run(self, *args, **kwargs):
        """The runtime invocation interface."""
        raise NotImplementedError()

    def get_memory_allocated(self):
        """Get the current size of allocated memory."""
        calls = []
        for mesh in self.physical_meshes:
            for worker in mesh.workers:
                calls.append(worker.get_memory_allocated.remote())
        return max(ray.get(calls))

    def get_max_memory_allocated(self):
        """Get the maximal size of memory allocated so far."""
        calls = []
        for mesh in self.physical_meshes:
            for worker in mesh.workers:
                calls.append(worker.get_max_memory_allocated.remote())
        return max(ray.get(calls))

    def sync(self):
        """Sync all workers' GPU activities."""
        all_workers = [w for mesh in self.physical_meshes for w in mesh.workers]
        ray.get([w.sync.remote() for w in all_workers])

    def shutdown(self):
        """Abstract method for shutting down the runtime."""
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
        device_str_groups = [[OrderedSet()
                              for _ in range(self.num_mesh)]
                             for _ in range(self.num_mesh)]
        # Merge (i, j) and (j, i)
        for i, j, var_spec_map in self._communicator.task_spec_iter():
            participants = OrderedSet()
            for _, spec in var_spec_map.items():  # for each var
                participants = participants | spec.get_participant_device_strs()
            if i <= j:
                device_str_groups[i][j] = device_str_groups[i][j] | participants
            else:
                device_str_groups[j][i] = device_str_groups[j][i] | participants

        # construct groups
        start_time = time.time()
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
                if global_config.eagerly_create_communicators:
                    cg.instantiate_now()
                else:
                    cg.instantiate()
                self._collective_groups[i][j] = cg
                self._collective_groups[j][i] = cg
        end_time = time.time()
        logger.debug(
            f"Initialize collective group takes {end_time - start_time:.2f}")

    def _compile_resharding_tasks(self):
        """Create and compile all resharding (send/recv/allgather) tasks."""
        for src_mesh_idx, dst_mesh_idx, var_spec_map \
                in self._communicator.task_spec_iter():
            for key, spec in var_spec_map.items():
                cg = self._collective_groups[src_mesh_idx][dst_mesh_idx]
                src_mesh = self.physical_meshes[src_mesh_idx]
                dst_mesh = self.physical_meshes[dst_mesh_idx]
                self._resharding_tasks[src_mesh_idx][dst_mesh_idx][key] = \
                    SymbolicReshardingTask(spec, cg, src_mesh, dst_mesh)

    def _destroy_collective_groups(self):
        for i in range(self.num_mesh):
            for j in range(self.num_mesh):
                if i < j and self._collective_groups[i][j]:
                    self._collective_groups[i][j].destroy()

    def print_resharding_tasks(self):
        """Pretty print all compiled resharding tasks."""
        ret = ""
        for src_idx in range(len(self._resharding_tasks)):
            for dst_idx in range(len(self._resharding_tasks)):
                for var, task in self._resharding_tasks[src_idx][dst_idx].items(
                ):
                    ret += f"{var}: Mesh {src_idx}->{dst_idx}, {task}\n\n"
        return ret
