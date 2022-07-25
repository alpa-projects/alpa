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
from jax.interpreters import pxla
from jax.tree_util import PyTreeDef
import numpy as np

from alpa.create_state_parallel import compile_create_state_executable
from alpa.device_mesh import (PhysicalDeviceMesh, VirtualPhysicalMesh,
                              LocalPhysicalDeviceMesh, get_global_physical_mesh,
                              get_global_virtual_physical_mesh)
from alpa.pipeline_parallel.compile_executable import compile_pipeshard_executable
from alpa.pipeline_parallel.local_pipeline import compile_local_pipeline_executable
from alpa.pipeline_parallel.layer_construction import (LayerOption,
                                                       AutoLayerOption,
                                                       ManualLayerOption)
from alpa.pipeline_parallel.stage_construction import (StageOption,
                                                       AutoStageOption,
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


class DataParallel(ShardParallel):
    """
    Use vanilla data parallelism.
    This method syncs gradient by using all-reduce.
    """

    def __init__(self,
                 devices: Optional[Union[LogicalDeviceMesh,
                                         PhysicalDeviceMesh]] = None,
                 num_micro_batches: Optional[int] = None):
        as_option = AutoShardingOption(force_data_parallel=True,
                                       prefer_reduce_scatter=False)
        super().__init__(devices, num_micro_batches, as_option)


class Zero2Parallel(ShardParallel):
    """
    Use zero-2 based data parallelism. This method
    1. replaces all-reduce by reduce-scatter and all-gather.
    2. partitions more tensors such as optimizer states.
    """

    def __init__(self,
                 devices: Optional[Union[LogicalDeviceMesh,
                                         PhysicalDeviceMesh]] = None,
                 num_micro_batches: Optional[int] = None):
        as_option = AutoShardingOption(force_data_parallel=True,
                                       prefer_reduce_scatter=True)
        super().__init__(devices, num_micro_batches, as_option)


class Zero3Parallel(ShardParallel):
    """
    Use zero-3 based data parallelism.
    Note that this method is experimental and not fully tested.
    """

    def __init__(self,
                 devices: Optional[Union[LogicalDeviceMesh,
                                         PhysicalDeviceMesh]] = None,
                 num_micro_batches: Optional[int] = None):
        as_option = AutoShardingOption(force_zero_stage_3=True)
        super().__init__(devices, num_micro_batches, as_option)


class PipeshardParallel(ParallelMethod):
    """
    Use pipeshard parallelism which combines pipeline parallelism and
    shard parallelism.

    Args:
        devices: Specify the devices to use. If it is None, use all the devices
          in the cluster.
        num_micro_batches: The number of micro batches for gradient
          accumulation.
        default_auto_sharding_option: The default options of the auto-sharding
          solver.
        pipeline_schedule: The pipieline schedules.
          Possible choices: {"1f1b", "gpipe", "inference"}
        layer_option: Options of grouping basic operators to layers.
          Possible choices are {"manual", alpa.AutoLayerOption,
                                 alpa.ManualLayerOption}
        stage_option: Options of grouping layers into pipeline stages.
          Possible choices are {"uniform", "auto", alpa.AutoStageOption,
                                 alpa.ManualStageOption}
        stage_input_shardings: Options of input sharding specs for each stage.
          Shape: [num_pipeline_stages, num_input_vars_in_hlo_module].
    """

    def __init__(
        self,
        devices: Optional[VirtualPhysicalMesh] = None,
        num_micro_batches: int = 1,
        default_auto_sharding_option: Optional[AutoShardingOption] = None,
        pipeline_schedule: str = "1f1b",
        layer_option: Optional[Union[LayerOption, str]] = None,
        stage_option: Optional[Union[StageOption, str]] = None,
        stage_input_shardings: Optional[Sequence[Sequence[
            pxla.ShardingSpec]]] = None):
        self.devices = devices
        self.num_micro_batches = num_micro_batches
        self.as_option = (default_auto_sharding_option or
                          AutoShardingOption(prefer_reduce_scatter=True))
        self.pipeline_schedule = pipeline_schedule
        if layer_option == "manual":
            layer_option = ManualLayerOption()
        self.layer_option = layer_option or AutoLayerOption(layer_num=2)
        if stage_option == "auto":
            stage_option = AutoStageOption(
                submesh_physical_shape_space="power_of_two",
                submesh_logical_shape_space="single_node_model_parallel",
                stage_imbalance_tolerance=np.inf,
                use_hlo_cost_model=False,
                profiling_database_filename=None,
                cached_compute_cost=None,
            )
        elif stage_option == "uniform":
            stage_option = UniformStageOption()
        self.stage_option = stage_option or UniformStageOption()
        self.stage_input_shardings = stage_input_shardings

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
            self.as_option, self.layer_option, self.stage_option,
            self.stage_input_shardings, *avals)


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
