"""Types conversion between different backends."""
from enum import Enum
from dataclasses import dataclass
from datetime import timedelta

_NUMPY_AVAILABLE = True
_TORCH_AVAILABLE = False
_CUPY_AVAILABLE = True

try:
    import cupy as cp  # pylint: disable=unused-import
except ImportError:
    _CUPY_AVAILABLE = False


def cupy_available():
    return _CUPY_AVAILABLE


def torch_available():
    return _TORCH_AVAILABLE


class Backend:
    """A class to represent different backends."""
    NCCL = "nccl"
    MPI = "mpi"
    GLOO = "gloo"
    UNRECOGNIZED = "unrecognized"

    def __new__(cls, name: str):
        backend = getattr(Backend, name.upper(), Backend.UNRECOGNIZED)
        if backend == Backend.UNRECOGNIZED:
            raise ValueError(f"Unrecognized backend: '{name}'. "
                             "Only NCCL is supported")
        if backend == Backend.MPI:
            raise RuntimeError("Ray does not support MPI backend.")
        return backend


class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3


unset_timeout_ms = timedelta(milliseconds=-1)


@dataclass
class AllReduceOptions:
    reduce_op = ReduceOp.SUM
    timeout_ms = unset_timeout_ms


@dataclass
class BarrierOptions:
    timeout_ms = unset_timeout_ms


@dataclass
class ReduceOptions:
    reduce_op = ReduceOp.SUM
    root_rank = 0
    root_tensor = 0  # index for multi-gpu reduce operations
    timeout_ms = unset_timeout_ms


@dataclass
class AllGatherOptions:
    timeout_ms = unset_timeout_ms


#
# @dataclass
# class GatherOptions:
#     root_rank = 0
#     timeout = unset_timeout


@dataclass
class BroadcastOptions:
    comm_key = ""
    world_size = 0
    devices_ids = []
    devices_global_rank = []
    n_elements = 0
    timeout_ms = unset_timeout_ms
    local_start_pos_list = []


@dataclass
class ReduceScatterOptions:
    reduce_op = ReduceOp.SUM
    timeout_ms = unset_timeout_ms


@dataclass
class SendOptions:
    dst_rank = 0
    dst_gpu_index = 0
    n_elements = 0
    timeout_ms = unset_timeout_ms
    start_pos = 0


@dataclass
class RecvOptions:
    src_rank = 0
    src_gpu_index = 0
    n_elements = 0
    unset_timeout_ms = unset_timeout_ms
    start_pos = 0
