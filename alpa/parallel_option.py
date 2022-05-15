from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Dict, Union

from jax import linear_util as lu
from jax.core import AbstractValue
from jax.tree_util import PyTreeDef

from alpa.device_mesh import (PhysicalDeviceMesh, LocalPhysicalDeviceMesh, DeviceCluster,
                              get_global_cluster, get_global_physical_mesh,
                              set_global_physical_mesh)
from alpa.shard_parallel.auto_sharding import AutoShardingOption, LogicalDeviceMesh
from alpa.shard_parallel.compile_executable import compile_shard_executable


class ParallelOption(ABC):
    """Options for parallelzing a function."""

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
        raise NotImplemented


class ShardParallel(ParallelOption):
    """Use shard parallel with options.
 
    Args:
        devices: Specify the devices to use.
        num_micro_batches: The number of micro batches for gradient accumulation.
        overwrite_auto_sharding_option: Overrite default auto sharding options.
          see also AutoShardingOption for valid options.
    """

    def __init__(self,
                 devices: Optional[Union[LogicalDeviceMesh, PhysicalDeviceMesh]] = None,
                 num_micro_batches: Optional[int] = None,
                 overwrite_auto_sharding_option: Optional[Dict] = None):
        self.devices = devices
        self.num_micro_batches = num_micro_batches
        self.as_option = AutoShardingOption()
        if overwrite_auto_sharding_option:
            self.as_option = self.as_option.copy_and_update(overwrite_auto_sharding_option)

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
            global_physical_mesh = get_global_physical_mesh()
            if global_physical_mesh is None:
                global_cluster = get_global_cluster()
                if global_cluster is None:
                    mesh = LocalPhysicalDeviceMesh()
                else:
                    mesh = global_cluster.get_physical_mesh()
                set_global_physical_mesh(mesh)
            else:
                mesh = global_physical_mesh
        elif isinstance(self.devices, (list, tuple)):
            mesh = LocalPhysicalDeviceMesh(self.devices)
        else:
            mesh = self.devices

        assert isinstance(mesh, (PhysicalDeviceMesh, LogicalDeviceMesh))

        return compile_shard_executable(fun, in_tree, out_tree_thunk,
                                        static_argnums, donated_invars, batch_invars,
                                        mesh, self.num_micro_batches,
                                        self.as_option, *avals)


class PipeShardParallel(ParallelOption):
    pass


class ManualPipeShardParallel(ParallelOption):
    pass

