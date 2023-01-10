"""Utility functions for device mesh workers to call nccl APIs."""
import logging
from typing import Sequence

import jax.numpy as jnp
from jax import device_put
from jax._src.lib import xla_extension as xe
import numpy as np

import alpa.collective as col
from alpa.util import (jax_tensor_set, jax_tensor_index,
                       xla_buffer_to_jax_tensor, jax_tensor_to_xla_buffer,
                       is_continuous_subset, infer_offset_and_n_elements,
                       infer_start_pos_and_n_elements)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def send_tile(worker, uuid: int, device_id: int, offset: Sequence[slice],
              dst_rank: int, dst_gpu_idx: int, group_name: str):
    buffer = worker.buffers[uuid][device_id]
    tensor_shape = buffer.shape
    if is_continuous_subset(offset, tensor_shape):
        start_pos, n_elements = (infer_start_pos_and_n_elements(
            tensor_shape, offset))
        col.send_multigpu(buffer,
                          dst_rank,
                          dst_gpu_idx,
                          group_name,
                          start_pos=start_pos,
                          n_elements=n_elements)
    else:
        # slower path, because of indexing.
        logger.debug("Send goes along the slowest path. "
                     "If this is for transformers, please check the resharding "
                     "specs.")
        start_indices = tuple(o.start for o in offset)
        slice_sizes = tuple(o.stop - o.start for o in offset)
        src_buffer = jax_tensor_index(xla_buffer_to_jax_tensor(buffer),
                                      start_indices, slice_sizes)
        to_send = jax_tensor_to_xla_buffer(src_buffer)
        n_elements = np.prod(slice_sizes)
        # dummy_compute_on_default_stream(device_id)

        # let send stream wait for compute stream
        col.comm_wait_compute(group_name, True, True, device_id)

        col.send_multigpu(to_send,
                          dst_rank,
                          dst_gpu_idx,
                          group_name,
                          start_pos=0,
                          n_elements=n_elements)


def recv_tile(worker, uuid: int, device_id: int,
              indices_in_dst_tile: Sequence[slice], src_rank: int,
              src_gpu_idx: int, group_name: str):
    buffer = worker.buffers[uuid][device_id]
    tensor_shape = buffer.shape
    slice_shape = tuple(ind.stop - ind.start for ind in indices_in_dst_tile)
    if is_continuous_subset(indices_in_dst_tile, tensor_shape):
        start_pos, n_elements = infer_start_pos_and_n_elements(
            tensor_shape, indices_in_dst_tile)
        col.recv_multigpu(buffer,
                          src_rank,
                          src_gpu_idx,
                          group_name,
                          start_pos=start_pos,
                          n_elements=n_elements)
    else:
        tmp_buffer = device_put(jnp.ones(slice_shape, dtype=buffer.dtype),
                                worker.local_devices[device_id])
        to_recv = jax_tensor_to_xla_buffer(tmp_buffer)
        n_elements = np.prod(slice_shape)
        # let recv stream wait for d2d stream
        col.comm_wait_compute(group_name, False, False, device_id)
        # let recv stream wait for compute stream
        col.comm_wait_compute(group_name, False, True, device_id)

        col.recv_multigpu(to_recv,
                          src_rank,
                          src_gpu_idx,
                          group_name,
                          start_pos=0,
                          n_elements=n_elements)
        # let compute stream wait for recv stream
        col.compute_wait_comm(group_name, False, True, device_id)

        start_indices = tuple(
            ind_in_dst.start for ind_in_dst in indices_in_dst_tile)
        new_buffer = jax_tensor_set(xla_buffer_to_jax_tensor(buffer),
                                    xla_buffer_to_jax_tensor(to_recv),
                                    start_indices)
        worker.buffers[uuid][device_id] = jax_tensor_to_xla_buffer(new_buffer)


def allgather(worker, uuid: int, device_ids: Sequence[int],
              tensor_slices: Sequence[Sequence[slice]], output_slice):
    # FIXME: handle the case that local device ids are the same but global ids
    # are different
    communicators = worker.allgather_communicators[repr(sorted(device_ids))]
    tensor_shape = worker.buffers[uuid][device_ids[0]].shape
    global_start_pos, _ = infer_start_pos_and_n_elements(
        tensor_shape, output_slice)

    buffers = []
    local_start_pos_list = []
    for device_id, tensor_slice in zip(device_ids, tensor_slices):
        xla_buffer = worker.buffers[uuid][device_id]
        start_pos, _ = infer_start_pos_and_n_elements(tensor_shape,
                                                      tensor_slice)
        buffers.append(xla_buffer)
        local_start_pos_list.append(start_pos)

    _, local_n_elements = infer_offset_and_n_elements(tensor_slices[0])
    xe.nccl_local_all_gather(communicators, buffers, local_start_pos_list,
                             global_start_pos, local_n_elements)

    for device_id, buf in zip(device_ids, buffers):
        worker.buffers[uuid][device_id] = buf


def broadcast(worker, uuid, comm_key, world_size, devices_ids,
              devices_global_rank, tensor_slices, group_name):
    buffers = []
    local_start_pos_list = []
    _, n_elements = infer_offset_and_n_elements(tensor_slices[0])
    for device_id, global_rank, tensor_slice in zip(devices_ids,
                                                    devices_global_rank,
                                                    tensor_slices):
        buffer = worker.buffers[uuid][device_id]
        tensor_shape = buffer.shape
        slice_shape = tuple(ind.stop - ind.start for ind in tensor_slice)
        if is_continuous_subset(tensor_slice, tensor_shape):
            # fast path, two cases: (1) same shape, (2) continuous subset.
            start_pos, _ = infer_start_pos_and_n_elements(
                tensor_shape, tensor_slice)
            local_start_pos_list.append(start_pos)
            buffers.append(buffer)
        else:
            tmp = None
            if global_rank == 0:
                start_indices = tuple(o.start for o in tensor_slice)
                tmp = jax_tensor_index(xla_buffer_to_jax_tensor(buffer),
                                       start_indices, slice_shape)
            else:
                tmp = device_put(jnp.ones(slice_shape, dtype=buffer.dtype),
                                 worker.local_devices[device_id])
            # let communicate stream wait for compute stream
            is_send = global_rank == 0
            col.comm_wait_compute(group_name, is_send, True, device_id)
            # let communicate stream wait for d2d stream
            col.comm_wait_compute(group_name, is_send, False, device_id)

            local_start_pos_list.append(0)
            buffers.append(jax_tensor_to_xla_buffer(tmp))

    col.broadcast_partialgpu(buffers, n_elements, comm_key, world_size,
                             devices_ids, devices_global_rank, group_name,
                             local_start_pos_list)

    for xla_buffer, device_id, global_rank, tensor_slice in zip(
            buffers, devices_ids, devices_global_rank, tensor_slices):
        if global_rank == 0:
            continue
        buffer = worker.buffers[uuid][device_id]
        tensor_shape = buffer.shape
        slice_shape = tuple(ind.stop - ind.start for ind in tensor_slice)
        if is_continuous_subset(tensor_slice, tensor_shape):
            new_buffer = xla_buffer
        else:
            start_indices = tuple(
                ind_in_dst.start for ind_in_dst in tensor_slice)
            # let compute stream wait for communicator stream
            is_send = global_rank == 0
            col.compute_wait_comm(group_name, is_send, True, device_id)
            new_buffer = jax_tensor_set(xla_buffer_to_jax_tensor(buffer),
                                        xla_buffer_to_jax_tensor(xla_buffer),
                                        start_indices)
            new_buffer = jax_tensor_to_xla_buffer(new_buffer)
        worker.buffers[uuid][device_id] = new_buffer


to_signal_buffer = jax_tensor_to_xla_buffer
