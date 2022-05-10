"""Abstract runtime classes and methods."""
import logging
import time

from abc import ABCMeta, abstractmethod
from typing import Sequence, Union

from jax.core import Var
import numpy as np
import ray
from alpa.global_env import global_config
from alpa.util import OrderedSet
from alpa.pipeline_parallel.cross_mesh_resharding import (
    CrossMeshCommunicator, SymbolicReshardingTask, SymbolicBroadcastReshardingTask)
from alpa.pipeline_parallel.schedules import PipelineSchedule
from alpa.pipeline_parallel.device_mesh_group import DistributedPhysicalDeviceMeshGroup
from alpa.pipeline_parallel.computation import (PipelineComputation,
                                                XlaShardedPipelineComputation)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseRuntime(metaclass=ABCMeta):
    """Abstract class for pipeline runtime."""

    def __init__(self,
                 *,
                 pipeline_stages: Sequence[Union[PipelineComputation,
                                                 XlaShardedPipelineComputation]],
                 global_invars: Sequence[Var],
                 global_outvars: Sequence[Var],
                 physical_meshes: Sequence[Var]):
        """
        An abstract class for pipeline-parallel runtime.

        Args:
            pipeline_stages: pipeline stages.
            global_invars: input variables.
            global_outvars: output varialbes.
            physical_meshes: input physical meshes.
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
    def run(self, *args):
        """Run function."""
        raise NotImplementedError()


class BaseDistributedRuntime(BaseRuntime):
    """Abstract class for distributed pipepline runtime."""

    def __init__(self,
                 *,
                 pipeline_stages: Sequence[XlaShardedPipelineComputation],
                 global_invars: Sequence[Var],
                 grad_dummy_invars: Sequence[Var],
                 global_outvars: Sequence[Var],
                 physical_meshes: DistributedPhysicalDeviceMeshGroup,
                 dependency: np.ndarray,
                 schedule: PipelineSchedule,
                 is_batch: Sequence[bool],
                 num_batch: int):
        """
        A base class of distributed pipeline-parallel runtime.

        This class abstract out some communication-related methods that will be shared
        across different distributed runtime implementations.

        Args:
            pipeline_stages: list of pipeline stage programs.
            global_invars: input variables.
            grad_dummy_invars: vars for gradient accumulation.
            global_outvars: output variables.
            physical_meshes: the cluster meshes to pipeline over.
            dependency: dependency between stages as an adjacency matrix.
            schedule: schedule to follow to execute the pipeline.
            is_batch: indicators of the batch variables.
            num_batch: number of microbatches.
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
        self._resharding_tasks = [
            [{} for _ in range(self.num_mesh)] for _ in range(self.num_mesh)
        ]

        self._establish_nccl_groups()

        start_time = time.time()
        self._compile_resharding_tasks()
        end_time = time.time()
        logger.debug(
            f"Compile resharding tasks takes {end_time - start_time:.2f}")

    def run(self, *args):
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
        """Run spmd partitioner pass for each stage to get sharding specs."""
        for stage_idx, stage in enumerate(self.stages):
            mesh_indices = list(self.schedule.stage_placement(stage_idx))
            assert len(mesh_indices) == 1
            stage.get_spmd_partitioned()

    def _establish_nccl_groups(self):
        """
        Identify and create NCCL groups based on resharding specs.

        We establish one collective group between two physical meshes, covering all the devices in
        these two meshes that require NCCL communication.

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
                self.physical_meshes.establish_nccl_group(i, j)
        end_time = time.time()
        logger.debug(
            f"Initialize collective group takes {end_time - start_time:.2f}")

    def _compile_resharding_tasks(self):
        """Create and compile all resharding (send/recv/allgather) tasks."""
        for (src_mesh_idx, dst_mesh_idx,
             var_spec_map) in self._communicator.task_spec_iter():
            for key, spec in var_spec_map.items():
                cg = self.physical_meshes.collective_groups[src_mesh_idx][
                    dst_mesh_idx]
                src_mesh = self.physical_meshes[src_mesh_idx]
                dst_mesh = self.physical_meshes[dst_mesh_idx]
                if global_config.resharding_mode == "send_recv":
                    self._resharding_tasks[src_mesh_idx][dst_mesh_idx][
                        key] = SymbolicReshardingTask(spec, cg, src_mesh, dst_mesh)
                else:
                    self._resharding_tasks[src_mesh_idx][dst_mesh_idx][
                        key] = SymbolicBroadcastReshardingTask(spec, cg, src_mesh, dst_mesh)

    def print_resharding_tasks(self):
        """Pretty print all compiled resharding tasks."""
        ret = ""
        for src_idx in range(len(self._resharding_tasks)):
            for dst_idx in range(len(self._resharding_tasks)):
                for var, task in self._resharding_tasks[src_idx][dst_idx].items(
                ):
                    ret += f"{var}: Mesh {src_idx}->{dst_idx}, {task}\n\n"
        return ret
