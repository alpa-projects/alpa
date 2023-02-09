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
from jax._src import traceback_util
from jax.core import AbstractValue
from jax.interpreters import pxla
from jax.tree_util import PyTreeDef
import numpy as np

from alpa.create_state_parallel import compile_create_state_executable
from alpa.follow_parallel import compile_follow_parallel_executable
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
                                                       ManualStageOption,
                                                       UniformStageOption)
from alpa.shard_parallel.auto_sharding import AutoShardingOption, LogicalDeviceMesh
from alpa.shard_parallel.compile_executable import compile_shard_executable
from alpa.shard_parallel.manual_sharding import ManualShardingOption

traceback_util.register_exclusion(__file__)


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
                 auto_sharding_option: Optional[AutoShardingOption] = None,
                 manual_sharding_option: Optional[ManualShardingOption] = None):
        self.devices = devices
        self.num_micro_batches = num_micro_batches
        self.as_option = auto_sharding_option or AutoShardingOption()
        self.ms_option = manual_sharding_option

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
            # Use 1d mesh by default
            mesh = mesh.get_logical_mesh().flatten()
        elif isinstance(self.devices, (list, tuple)):
            mesh = LocalPhysicalDeviceMesh(self.devices)
        else:
            mesh = self.devices

        assert isinstance(mesh, (PhysicalDeviceMesh, LogicalDeviceMesh))

        return compile_shard_executable(fun, in_tree, out_tree_thunk,
                                        static_argnums, donated_invars,
                                        batch_invars, mesh,
                                        self.num_micro_batches, self.as_option,
                                        self.ms_option, *avals)


class DataParallel(ShardParallel):
    """
    Use vanilla data parallelism.
    This method syncs gradients by using all-reduce.
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
                pxla.ShardingSpec]]] = None,
            manual_sharding_option: ManualShardingOption = None):
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
                cached_profile_result=None,
            )
        elif stage_option == "uniform":
            stage_option = UniformStageOption()
        self.stage_option = stage_option or UniformStageOption()
        self.stage_input_shardings = stage_input_shardings
        assert not (stage_input_shardings is not None and
                    manual_sharding_option is not None)
        self.manual_sharding_option = manual_sharding_option

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
            self.as_option, self.layer_option, self.stage_option, None,
            self.stage_input_shardings, self.manual_sharding_option, *avals)


def get_3d_parallel_method(num_micro_batches: int,
                           data_parallel: int,
                           operator_parallel: int,
                           pipeline_parallel: int,
                           allow_degenerate_into_shard_parallel: bool = True,
                           manual_layer_num: int = None,
                           manual_sharding_option: ManualShardingOption = None):
    """
    Get a parallel method for 3D parallelism, which reguarlly combines
    data parallelism, operator parallelism and pipeline parallelism.
    """
    # Validity check
    virtual_mesh = get_global_virtual_physical_mesh()
    num_devices = virtual_mesh.num_devices
    num_devices_per_host = virtual_mesh.num_devices_per_host
    if data_parallel == -1:
        data_parallel = (num_devices // operator_parallel // pipeline_parallel)
    assert num_devices % data_parallel == 0
    assert num_devices % operator_parallel == 0
    assert num_devices % pipeline_parallel == 0
    assert (num_devices == data_parallel * operator_parallel *
            pipeline_parallel)
    pp = pipeline_parallel

    # Decide logical and physical mesh shapes
    logical_mesh_shape = (data_parallel, operator_parallel)
    num_mesh_devices = np.prod(logical_mesh_shape)
    if num_mesh_devices <= num_devices_per_host:
        physical_mesh_shape = (1, num_mesh_devices)
    else:
        assert num_mesh_devices % num_devices_per_host == 0
        physical_mesh_shape = (num_mesh_devices // num_devices_per_host,
                               num_devices_per_host)

    # If no pipeline parallel, degenerate into shard parallel
    if pp == 1 and allow_degenerate_into_shard_parallel:
        return ShardParallel(num_micro_batches=num_micro_batches,
                             auto_sharding_option=AutoShardingOption(
                                 prefer_reduce_scatter=True,
                                 force_batch_dim_to_mesh_dim=0),
                             devices=get_global_physical_mesh(
                                 create_if_not_exist=True).get_logical_mesh(
                                     [data_parallel, operator_parallel]))

    # Return pipeshard parallel
    if manual_layer_num is not None:
        assert manual_layer_num % pp == 0
        layer_option = ManualLayerOption()
        stage_option = UniformStageOption(pp, physical_mesh_shape,
                                          logical_mesh_shape, {})
    else:
        layer_option = AutoLayerOption(layer_num=pp, eps=0.1)
        stage_option = ManualStageOption(
            forward_stage_layer_ids=[[i] for i in range(pp)],
            submesh_physical_shapes=[physical_mesh_shape] * pp,
            submesh_logical_shapes=[logical_mesh_shape] * pp,
            submesh_autosharding_option_dicts=[{}] * pp)
    return PipeshardParallel(
        devices=virtual_mesh,
        num_micro_batches=num_micro_batches,
        default_auto_sharding_option=AutoShardingOption(
            enable_auto_sharding=manual_sharding_option is None,
            prefer_reduce_scatter=True,
            force_batch_dim_to_mesh_dim=0,
        ),
        layer_option=layer_option,
        stage_option=stage_option,
        manual_sharding_option=manual_sharding_option)


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
        See `tests/test_create_state.py` for example usages.
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


class FollowParallel(ParallelMethod):
    """
    Parallelize a function given its input placement specs.

    Args:
        num_micro_batches: The number of micro batches.
        get_input_placement_specs: A callaback function that returns
          the input placement specs.
        pipeline_schedule: The pipeline schedule.
          Possible choices: {"1f1b", "gpipe", "inference"}
        layer_option: Options of grouping basic operators to layers.
          Possible choices: {"auto", "manual"}.
    """

    def __init__(self,
                 src_func: "ParallelizedFunc",
                 num_micro_batches: Optional[int] = None,
                 get_input_placement_specs: Callable = None,
                 pipeline_schedule: str = "inference",
                 layer_option: str = "follow"):
        self.src_func = src_func
        self.num_micro_batches = num_micro_batches

        if get_input_placement_specs is None:

            def default_get():
                executable = src_func.get_last_executable()
                input_placement_specs = executable.get_input_placement_specs()
                train_state, batch = input_placement_specs
                return train_state.params, batch

            get_input_placement_specs = default_get

        self.get_input_placement_specs = get_input_placement_specs
        self.pipeline_schedule = pipeline_schedule
        self.layer_option = layer_option

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
        input_placement_specs = self.get_input_placement_specs()
        return compile_follow_parallel_executable(
            fun, in_tree, out_tree_thunk, static_argnums, donated_invars,
            batch_invars, self.src_func, self.num_micro_batches,
            input_placement_specs, self.pipeline_schedule, self.layer_option,
            *avals)
