"""A class that wraps HloModule and records whether the module runs AutoSharding
and SPMD Partitioner or not.
"""
from enum import Enum, auto
from typing import Union

from jax._src.lib import xla_extension as xe
from jax.interpreters import mlir

class HloStatus(Enum):
    """
    The status of an HloModule.
    See also the docstring at the beginning of shard_parallel/auto_sharding.py.
    """
    UNOPTIMIZED = auto()
    SHARDING_ANNOTATED = auto()
    SPMD_PARTITIONED = auto()
    FULLY_OPTIMIZED = auto()


class WrappedHlo:
    """Wrapped HloModule with HloStatus."""

    def __init__(self,
                 module: Union[xe.HloModule, xe.XlaComputation, bytes],
                 status: HloStatus = HloStatus.UNOPTIMIZED):
        if isinstance(module, xe.HloModule):
            self.module = module
        elif isinstance(module, xe.XlaComputation):
            self.module = module.get_hlo_module()
        else:
            assert isinstance(module, bytes)
            self.module = xe.XlaComputation(module).get_hlo_module()
        self.name = self.module.name
        self.status = status
        self.is_manually_annotated = False

    def get_computation(self) -> xe.XlaComputation:
        return xe.XlaComputation(self.module.as_serialized_hlo_module_proto())

    def get_mhlo(self):
        xla_computation = self.get_computation()
        module_str = xe.mlir.xla_computation_to_mlir_module(xla_computation)
        with mlir.make_ir_context():
            mhlo = mlir.ir.Module.parse(module_str)
        return mhlo

    def get_module(self) -> xe.HloModule:
        return self.module

    def get_hlo_proto(self):
        return self.module.as_serialized_hlo_module_proto()

    def program_shape(self):
        return self.module.program_shape()

    def set_input_shardings(self, sharding_protos):
        assert self.is_sharding_annotated() or self.is_unoptimized()
        xe.set_hlo_module_input_shardings(self.module, sharding_protos)

    def set_output_shardings(self, sharding_protos):
        assert self.is_sharding_annotated() or self.is_unoptimized()
        xe.set_hlo_module_output_shardings(self.module, sharding_protos)

    def is_unoptimized(self):
        return self.status == HloStatus.UNOPTIMIZED

    def is_sharding_annotated(self):
        return self.status == HloStatus.SHARDING_ANNOTATED

    def is_spmd_partitioned(self):
        return self.status == HloStatus.SPMD_PARTITIONED

    def to_string(self):
        return self.module.to_string()

    def __getstate__(self):
        return (self.get_hlo_proto(), self.status)

    def __setstate__(self, bytes_and_status):
        b, s = bytes_and_status
        self.__init__(b, s)
