from typing import Sequence

import alpa.collective.worker_nccl_util_cupy as cupy_impl
import alpa.collective.worker_nccl_util_xla as xla_impl
from alpa.global_env import global_config

def _switch_impl(cupy_fn, xla_fn, *args):
    if global_config.nccl_mode == "cupy":
        return cupy_fn(*args)
    elif global_config.nccl_mode == "xla_extension":
        return xla_fn(*args)
    else:
        raise ValueError(f"nccl mode {global_config.nccl_mode} is illegal")


def send_tile(worker, uuid: int, device_id: int, offset: Sequence[slice],
              dst_rank: int, dst_gpu_idx: int, group_name: str):
    return _switch_impl(cupy_impl.send_tile, xla_impl.send_tile, worker, uuid,
                        device_id, offset, dst_rank, dst_gpu_idx, group_name)

def recv_tile(worker, uuid: int, device_id: int,
              indices_in_dst_tile: Sequence[slice], src_rank: int,
              src_gpu_idx: int, group_name: str):
    return _switch_impl(cupy_impl.recv_tile, xla_impl.recv_tile, worker, uuid,
                        device_id, indices_in_dst_tile, src_rank, src_gpu_idx,
                        group_name)


def broadcast(worker, uuid: int, comm_key: str, world_size: int,
              devices_ids: Sequence[int], devices_global_rank: Sequence[int],
              tensor_slices: Sequence[Sequence[slice]], group_name: str):
    return _switch_impl(cupy_impl.broadcast, xla_impl.broadcast, worker, uuid,
                        comm_key, world_size, devices_ids, devices_global_rank,
                        tensor_slices, group_name)

def allgather(worker, uuid: int, device_ids: Sequence[int],
              tensor_slices: Sequence[Sequence[slice]], output_slice):
    return _switch_impl(cupy_impl.allgather, xla_impl.allgather, worker, uuid,
                        device_ids, tensor_slices, output_slice)

def to_signal_buffer(jax_tensor):
    return _switch_impl(cupy_impl.to_signal_buffer, xla_impl.to_signal_buffer,
                        jax_tensor)
