"""Methods for parallelzing a function.

Alpa classifies common parallel techniques into two categories:
1. Shard parallelism or intra-operator parallelism. This includes data
   parallelism, operator parallelism (or tensor model parallelism), expert
   parallelism, zero optimizer and their combinations.
2. Pipeline parallelism or inter-operator parallleism.
Please refer to the Alpa paper (https://arxiv.org/abs/2201.12023) for more
details.

Based on this, alpa provides two base parallel methods:
- ShardParallel: which only uses shard parallelsim.
- PipeshardParallel: which combines pipeline parallelism and shard parallelism.
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Union, Any

from jax import linear_util as lu
from jax.core import AbstractValue
from jax.tree_util import PyTreeDef
import numpy as np

from alpa.create_state_parallel import compile_create_state_executable
from alpa.device_mesh import (PhysicalDeviceMesh, VirtualPhysicalMesh,
                              LocalPhysicalDeviceMesh, get_global_physical_mesh,
                              get_global_virtual_physical_mesh)
from alpa.pipeline_parallel.compile_executable import compile_pipeshard_executable
from alpa.pipeline_parallel.local_pipeline import compile_local_pipeline_executable
from alpa.pipeline_parallel.stage_construction import (AutoStageOption,
                                                       ManualStageOption,
                                                       UniformStageOption)
from alpa.shard_parallel.auto_sharding import AutoShardingOption, LogicalDeviceMesh
from alpa.shard_parallel.compile_executable import compile_shard_executable


class ParallelMethod(ABC):
    """Methods for parallelzing a function."""

    @abstractmethod
    def compile_executable(
        self,
        fun: lu.WrappedFun,
        in_tree: PyTreeDef,
        out_tree_thunk: Callable[[], PyTreeDef],
        static_argnums: Sequence[int],
        donated_invars: Sequence[bool],
        batch_invars: Sequence[bool],
        *avals: Sequence[AbstractValue],
    ):
        """Compile an executable."""
        raise NotImplementedError()


class ShardParallel(ParallelMethod):
    """Use shard parallelism to parallelize a function.

    Args:
        devices: Specify the devices to use. If it is None, use all devices
          in the cluster.
        num_micro_batches: The number of micro batches for gradient
          accumulation.
        auto_sharding_option: The options of the auto-sharding solver.
    """

    def __init__(self,
                 devices: Optional[Union[LogicalDeviceMesh,
                                         PhysicalDeviceMesh]] = None,
                 num_micro_batches: Optional[int] = None,
                 auto_sharding_option: Optional[AutoShardingOption] = None):
        self.devices = devices
        self.num_micro_batches = num_micro_batches
        self.as_option = auto_sharding_option or AutoShardingOption()

    def compile_executable(
        self,
        fun: lu.WrappedFun,
        in_tree: PyTreeDef,
        out_tree_thunk: Callable[[], PyTreeDef],
        static_argnums: Sequence[int],
        donated_invars: Sequence[bool],
        batch_invars: Sequence[bool],
        *avals: Sequence[AbstractValue],
    ):
        # Resolve the polymorphism in arguments
        if self.devices is None:
            mesh = get_global_physical_mesh(create_if_not_exist=True)
        elif isinstance(self.devices, (list, tuple)):
            mesh = LocalPhysicalDeviceMesh(self.devices)
        else:
            mesh = self.devices

        assert isinstance(mesh, (PhysicalDeviceMesh, LogicalDeviceMesh))

        return compile_shard_executable(fun, in_tree, out_tree_thunk,
                                        static_argnums, donated_invars,
                                        batch_invars, mesh,
                                        self.num_micro_batches, self.as_option,
                                        *avals)


class PipeshardParallel(ParallelMethod):
    """Use pipeshard parallelism which combines pipeline parallelism and shard
    parallelism.

    Args:
        devices: Specify the devices to use. If it is None, use all the devices
          in the cluster.
        num_micro_batches: The number of micro batches for gradient
          accumulation.
        default_auto_sharding_option: The default options of the auto-sharding
          solver.
        pipeline_schedule: The pipieline schedules.
          Possible choices: {"1f1b", "gpipe", "inference"}
        stage_mode: How to construct stages.
          Possible choices: {"uniform", "auto"}
        submesh_physical_shape_space: The search space of the physical submesh
          shapes.
          Possible choices: {"power_of_two", "small_power_of_two", "all"}.
        submesh_logical_shape_space: The search space of the logical mesh
          shapes.
          Possible choices: {"default", "single_node_model_parallel", "all"}.
        auto_stage_imbalance_tolerance: The tolerance of imbalance
          in the auto-stage construction.
        use_hlo_cost_model: Whether to use the Hlo instruction cost model for
          pipeline profiling.
        profiling_database_filename: The filename of profiling result database.
        cached_compute_cost: The file name of the cached compute cost.
    """

    def __init__(
            self,
            devices: Optional[VirtualPhysicalMesh] = None,
            num_micro_batches: int = 1,
            default_auto_sharding_option: Optional[AutoShardingOption] = None,
            pipeline_schedule: str = "1f1b",
            stage_mode: str = "uniform",
            submesh_physical_shape_space: str = "power_of_two",
            submesh_logical_shape_space: str = "default",
            auto_stage_imbalance_tolerance: float = np.inf,
            use_hlo_cost_model: bool = False,
            profiling_database_filename: Optional[str] = None,
            cached_compute_cost: Optional[str] = None):
        self.devices = devices
        self.num_micro_batches = num_micro_batches
        self.as_option = default_auto_sharding_option or AutoShardingOption()
        self.pipeline_schedule = pipeline_schedule
        if stage_mode == "auto":
            self.stage_option = AutoStageOption(submesh_physical_shape_space,
                                                submesh_logical_shape_space,
                                                auto_stage_imbalance_tolerance,
                                                use_hlo_cost_model,
                                                profiling_database_filename,
                                                cached_compute_cost)
        elif stage_mode == "uniform":
            self.stage_option = UniformStageOption()
        else:
            raise ValueError(f"Invalid stage mode: {stage_mode}")

    def compile_executable(
        self,
        fun: lu.WrappedFun,
        in_tree: PyTreeDef,
        out_tree_thunk: Callable[[], PyTreeDef],
        static_argnums: Sequence[int],
        donated_invars: Sequence[bool],
        batch_invars: Sequence[bool],
        *avals: Sequence[AbstractValue],
    ):
        # Resolve the polymorphism in arguments
        if self.devices is None:
            mesh = get_global_virtual_physical_mesh()
            assert mesh is not None, (
                "Please run `alpa.init()` to initialize alpa.")
        else:
            mesh = self.devices

        assert isinstance(mesh, VirtualPhysicalMesh)

        return compile_pipeshard_executable(
            fun, in_tree, out_tree_thunk, static_argnums, donated_invars,
            batch_invars, mesh, self.num_micro_batches, self.pipeline_schedule,
            self.as_option, self.stage_option, *avals)


class ManualPipeshardParallel(PipeshardParallel):
    """Use pipeshard parallelism with manual assignment.

    This method can be used to load the solution found by auto
    PipeshardParallel.

    Args:
        forward_stage_layer_ids: Layer IDs of each forward stage.
        submesh_physical_shapes: The physical shapes of submeshes of each stage.
        submesh_logical_shapes: The logical shapes of submeshes of each stage.
        submesh_autosharding_option_dicts: The auto-sharding options of each
          stage.
        devices: Specify the devices to use. If it is None, use all the devices
          in the cluster.
        num_micro_batches: The number of micro batches for gradient
          accumulation.
        default_auto_sharding_option: The default options of the auto-sharding
          solver.
        pipeline_schedule: The pipieline schedules.
          Possible choices: {"1f1b", "gpipe", "inference"}
    """

    def __init__(
            self,
            forward_stage_layer_ids: Sequence[Sequence[int]],
            submesh_physical_shapes: Sequence[Sequence[int]],
            submesh_logical_shapes: Sequence[Sequence[int]],
            submesh_autosharding_option_dicts: Sequence[dict],
            devices: Optional[VirtualPhysicalMesh] = None,
            num_micro_batches: int = 1,
            default_auto_sharding_option: Optional[AutoShardingOption] = None,
            pipeline_schedule: str = "1f1b"):
        # pylint: disable=super-init-not-called
        self.devices = devices
        self.num_micro_batches = num_micro_batches
        self.as_option = default_auto_sharding_option or AutoShardingOption()
        self.pipeline_schedule = pipeline_schedule
        self.stage_option = ManualStageOption(
            forward_stage_layer_ids,
            submesh_physical_shapes,
            submesh_logical_shapes,
            submesh_autosharding_option_dicts,
        )


class LocalPipelineParallel(ParallelMethod):
    """
    Run pipeline parallel on a single device.
    This is only used for debugging.
    """

    def compile_executable(
        self,
        fun: lu.WrappedFun,
        in_tree: PyTreeDef,
        out_tree_thunk: Callable[[], PyTreeDef],
        static_argnums: Sequence[int],
        donated_invars: Sequence[bool],
        batch_invars: Sequence[bool],
        *avals: Sequence[AbstractValue],
    ):
        return compile_local_pipeline_executable(fun, *avals)


class CreateStateParallel(ParallelMethod):
    """
    Follow a train_step function to create the initial states distributedly.

    Args:
        train_step: The training step function.
          See notes below for requirements.
        other_args: Other arguments for calling the train_step function.

    Notes:
        To use thie parallel method, the function being parallelized should
        return a single output `state`. Then train_step should take `state`
        as the first argument and `other_args` as successive arguments.
        See tests/test_create_state.py for example usages.
    """

    def __init__(self, train_step: "ParallelizedFunc",
                 other_args: Sequence[Any]):
        # pylint: disable=import-outside-toplevel
        from alpa.api import ParallelizedFunc
        assert isinstance(train_step, ParallelizedFunc)

        self.train_step = train_step
        self.other_args = other_args

        # TODO(lmzheng): support more flexible signatures.
        # For example, the state does not have to be the first argument.

    def compile_executable(
        self,
        fun: lu.WrappedFun,
        in_tree: PyTreeDef,
        out_tree_thunk: Callable[[], PyTreeDef],
        static_argnums: Sequence[int],
        donated_invars: Sequence[bool],
        batch_invars: Sequence[bool],
        *avals: Sequence[AbstractValue],
    ):
        return compile_create_state_executable(fun, in_tree, out_tree_thunk,
                                               static_argnums, donated_invars,
                                               self.train_step, self.other_args,
                                               *avals)
