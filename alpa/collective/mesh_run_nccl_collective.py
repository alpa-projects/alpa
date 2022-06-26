"""Implementation of devive_mesh's send/recv/allgather/broadcast."""

from typing import Sequence
import logging

from jax import device_put
from jax._src.lib import xla_extension as xe

import jax.numpy as jnp
import numpy as np
import cupy

import alpa.collective as col
from alpa.collective.collective_group import nccl_util
from alpa.util import (jax_tensor_to_cupy, cupy_to_jax_tensor, jax_tensor_set,
                       xla_buffer_to_jax_tensor, jax_tensor_to_xla_buffer,
                       xla_buffer_to_cupy, cupy_to_xla_buffer,
                       is_continuous_subset, infer_offset_and_n_elements,
                       jax_tensor_index, infer_start_pos_and_n_elements)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def xla_nccl_send_tile(worker, uuid: int, offset: Sequence[slice],
                       dst_rank: int, dst_gpu_idx: int, group_name: str):

    tensor_shape = worker.buffers[uuid].shape
    if is_continuous_subset(offset, tensor_shape):
        start_pos, n_elements = (infer_start_pos_and_n_elements(
            tensor_shape, offset))
        col.send_multigpu(worker.buffers[uuid],
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
        src_buffer = jax_tensor_index(
            xla_buffer_to_jax_tensor(worker.buffers[uuid]), start_indices,
            slice_sizes)
        to_send = jax_tensor_to_xla_buffer(src_buffer)
        n_elements = np.prod(slice_sizes)
        col.send_multigpu(to_send,
                          dst_rank,
                          dst_gpu_idx,
                          group_name,
                          start_pos=0,
                          n_elements=n_elements)


def xla_nccl_recv_tile(worker, uuid: int, device_id: int,
                       indices_in_dst_tile: Sequence[slice], src_rank: int,
                       src_gpu_idx: int, group_name: str):
    tensor_shape = worker.buffers[uuid].shape
    slice_shape = tuple(ind.stop - ind.start for ind in indices_in_dst_tile)
    is_bool = worker.buffers[uuid].dtype == np.bool_
    if is_continuous_subset(indices_in_dst_tile, tensor_shape):
        start_pos, n_elements = infer_start_pos_and_n_elements(
            tensor_shape, indices_in_dst_tile)
        col.recv_multigpu(worker.buffers[uuid],
                          src_rank,
                          src_gpu_idx,
                          group_name,
                          start_pos=start_pos,
                          n_elements=n_elements)
    else:
        tmp_buffer = device_put(
            jnp.ones(slice_shape, dtype=worker.buffers[uuid].dtype),
            worker.local_devices[device_id])
        to_recv = jax_tensor_to_xla_buffer(tmp_buffer)
        n_elements = np.prod(slice_shape)
        col.recv_multigpu(to_recv,
                          src_rank,
                          src_gpu_idx,
                          group_name,
                          start_pos=0,
                          n_elements=n_elements)
        start_indices = tuple(
            ind_in_dst.start for ind_in_dst in indices_in_dst_tile)
        new_buffer = jax_tensor_set(
            xla_buffer_to_jax_tensor(worker.buffers[uuid]),
            xla_buffer_to_jax_tensor(to_recv), start_indices)
        worker.buffers[uuid] = jax_tensor_to_xla_buffer(new_buffer)
    if is_bool:
        worker.buffers[uuid] = _uint8_to_bool(worker.buffers[uuid])


def xla_nccl_allgather(worker, uuids: Sequence[int], device_ids: Sequence[int],
                       tensor_slices: Sequence[slice], output_slice):

    if repr(sorted(device_ids)) not in worker.allgather_communicators:
        worker.allgather_communicators[repr(
            sorted(device_ids))] = (worker.nccl_local_allgather_init_comms(
                list(device_ids)))

    communicators = worker.allgather_communicators[repr(sorted(device_ids))]
    is_bool = worker.buffers[uuids[device_ids[0]]].dtype == np.bool_
    tensor_shape = worker.buffers[uuids[device_ids[0]]].shape
    global_start_pos, _ = infer_start_pos_and_n_elements(
        tensor_shape, output_slice)

    buffers = []
    local_start_pos_list = []
    for device_id, tensor_slice in zip(device_ids, tensor_slices):
        uuid = uuids[device_id]
        xla_buffer = worker.buffers[uuid]
        start_pos, _ = infer_start_pos_and_n_elements(tensor_shape,
                                                      tensor_slice)
        buffers.append(xla_buffer)
        local_start_pos_list.append(start_pos)

    _, local_n_elements = infer_offset_and_n_elements(tensor_slices[0])
    xe.nccl_local_all_gather(communicators, buffers, local_start_pos_list,
                             global_start_pos, local_n_elements)

    for device_id, buf in zip(device_ids, buffers):
        uuid = uuids[device_id]
        if is_bool:
            buf = _uint8_to_bool(buf)
        worker.buffers[uuid] = buf


def xla_nccl_broadcast(worker, uuids, comm_key, world_size, devices_ids,
                       devices_global_rank, tensor_slices, group_name):
    buffers = []
    local_start_pos_list = []
    is_bool = worker.buffers[uuids[devices_ids[0]]].dtype == np.bool_
    _, n_elements = infer_offset_and_n_elements(tensor_slices[0])
    for device_id, global_rank, tensor_slice in zip(devices_ids,
                                                    devices_global_rank,
                                                    tensor_slices):
        uuid = uuids[device_id]
        tensor_shape = worker.buffers[uuid].shape
        slice_shape = tuple(ind.stop - ind.start for ind in tensor_slice)
        if is_continuous_subset(tensor_slice, tensor_shape):
            # fast path, two cases: (1) same shape, (2) continuous subset.
            start_pos, _ = infer_start_pos_and_n_elements(
                tensor_shape, tensor_slice)
            local_start_pos_list.append(start_pos)
            buffers.append(worker.buffers[uuid])
        else:
            tmp = None
            if global_rank == 0:
                start_indices = tuple(o.start for o in tensor_slice)
                tmp = jax_tensor_index(
                    xla_buffer_to_jax_tensor(worker.buffers[uuid]),
                    start_indices, slice_shape)
            else:
                tmp = device_put(
                    jnp.ones(slice_shape, dtype=worker.buffers[uuid].dtype),
                    worker.local_devices[device_id])
            local_start_pos_list.append(0)
            buffers.append(jax_tensor_to_xla_buffer(tmp))

    col.broadcast_partialgpu(buffers, n_elements, comm_key, world_size,
                             devices_ids, devices_global_rank, group_name,
                             local_start_pos_list)

    for xla_buffer, device_id, global_rank, tensor_slice in zip(
            buffers, devices_ids, devices_global_rank, tensor_slices):
        if global_rank == 0:
            continue
        uuid = uuids[device_id]
        tensor_shape = worker.buffers[uuid].shape
        slice_shape = tuple(ind.stop - ind.start for ind in tensor_slice)
        if is_continuous_subset(tensor_slice, tensor_shape):
            worker.buffers[uuid] = xla_buffer
        else:
            start_indices = tuple(
                ind_in_dst.start for ind_in_dst in tensor_slice)
            new_buffer = jax_tensor_set(
                xla_buffer_to_jax_tensor(worker.buffers[uuid]),
                xla_buffer_to_jax_tensor(xla_buffer), start_indices)
            worker.buffers[uuid] = jax_tensor_to_xla_buffer(new_buffer)
        if is_bool:
            worker.buffers[uuid] = _uint8_to_bool(worker.buffers[uuid])


# Note: in this device mesh code, we will use 3 types of tensors:
# (1) JAX high-level _DeviceArray, which is index-able, has __cuda_array__
#     interface
# (2) XLA low-level PyLocalBuffer, which is not index-able
# (3) cupy array, which is an intermediate format for ray collective
def cupy_nccl_send_tile(worker, uuid: int, offset: Sequence[slice],
                        dst_rank: int, dst_gpu_idx: int, group_name: str):
    """
    Send a slice of a source buffer to a target GPU.

    Args:
        uuid: the uuid of the xla buffers.
        offset: the slice to be sent in the buffer.
        dst_rank: destination rank to send.
        dst_gpu_idx: the gpu index on the destination rank.
        group_name: collective group name
    """
    tensor_shape = worker.buffers[uuid].shape
    if is_continuous_subset(offset, tensor_shape):
        # fast path, two cases: (1) same shape, (2) continuous subset.
        slice_shape = tuple(ind.stop - ind.start for ind in offset)
        to_send = xla_buffer_to_cupy(worker.buffers[uuid])
        if slice_shape == tensor_shape:
            col.send_multigpu(to_send, dst_rank, dst_gpu_idx, group_name)
        else:
            ind, n_elements = infer_offset_and_n_elements(offset)
            col.send_multigpu(to_send[ind],
                              dst_rank,
                              dst_gpu_idx,
                              group_name,
                              n_elements=n_elements)
    else:
        # slower path, because of indexing.
        logger.debug("Send goes along the slowest path. "
                     "If this is for transformers, please check the resharding "
                     "specs.")
        start_indices = tuple(o.start for o in offset)
        slice_sizes = tuple(o.stop - o.start for o in offset)
        src_buffer = jax_tensor_index(
            xla_buffer_to_jax_tensor(worker.buffers[uuid]), start_indices,
            slice_sizes)
        to_send = jax_tensor_to_cupy(src_buffer)
        col.send_multigpu(to_send, dst_rank, dst_gpu_idx, group_name)


def cupy_nccl_recv_tile(worker, uuid: int, device_id: int,
                        indices_in_dst_tile: Sequence[slice], src_rank: int,
                        src_gpu_idx: int, group_name: str):
    """
    Receive a slice from a source GPU and in-place write it on the target
    buffer.

    Args:
        uuid: the uuid of the xla buffers.
        device_id: the device where the buffer is received, used to allocate
            tmp buffer.
        indices_in_dst_tile: the slice index to be written on destination
            buffer.
        src_rank: source rank to receive from.
        src_gpu_idx: the sender gpu index on the source rank.
        group_name: collective group name.
    """

    tensor_shape = worker.buffers[uuid].shape
    slice_shape = tuple(ind.stop - ind.start for ind in indices_in_dst_tile)
    is_bool = worker.buffers[uuid].dtype == np.bool_
    if is_continuous_subset(indices_in_dst_tile, tensor_shape):
        to_recv = xla_buffer_to_cupy(worker.buffers[uuid], take_ownership=True)
        if slice_shape == tensor_shape:
            col.recv_multigpu(to_recv, src_rank, src_gpu_idx, group_name)
        else:
            ind, n_elements = infer_offset_and_n_elements(indices_in_dst_tile)
            col.recv_multigpu(to_recv[ind],
                              src_rank,
                              src_gpu_idx,
                              group_name,
                              n_elements=n_elements)
        worker.buffers[uuid] = cupy_to_xla_buffer(to_recv)
    else:
        # The following call will allocate memory and cause a few H2D and
        # D2D kernels.
        # See: https://github.com/alpa-projects/alpa/issues/145
        logger.debug("Recv goes along the slowest path. "
                     "If this is for transformers, please check the resharding "
                     "specs.")
        tmp_buffer = device_put(
            jnp.ones(slice_shape, dtype=worker.buffers[uuid].dtype),
            worker.local_devices[device_id])
        to_recv = jax_tensor_to_cupy(tmp_buffer, take_ownership=True)
        col.recv_multigpu(to_recv, src_rank, src_gpu_idx, group_name)
        recv_tensor = cupy_to_jax_tensor(to_recv)
        start_indices = tuple(
            ind_in_dst.start for ind_in_dst in indices_in_dst_tile)

        # The following in-place write will cause a D2D copy kernel
        # See: https://github.com/alpa-projects/alpa/issues/144
        # It is unavoidable, but it is better than:
        # new_buffer = dynamic_update_slice(src_buf, update, start_indices)
        # which is not in-place and will cause extra allocation-related
        # kernels.
        new_buffer = jax_tensor_set(
            xla_buffer_to_jax_tensor(worker.buffers[uuid]), recv_tensor,
            start_indices)
        worker.buffers[uuid] = jax_tensor_to_xla_buffer(new_buffer)
    if is_bool:
        worker.buffers[uuid] = _uint8_to_bool(worker.buffers[uuid])


def cupy_nccl_allgather(worker, uuids: Sequence[int], device_ids: Sequence[int],
                        tensor_slices: Sequence[slice], output_slice):

    cupy_buffers = []
    communicators = worker.allgather_communicators[repr(sorted(device_ids))]
    relative_idx = dict(zip(sorted(device_ids), range(len(device_ids))))
    output_idx, _ = infer_offset_and_n_elements(output_slice)
    is_bool = worker.buffers[uuids[0]].dtype == np.bool_
    nccl_util.groupStart()
    for device_id, tensor_slice in zip(device_ids, tensor_slices):
        uuid = uuids[device_id]
        xla_buffer = worker.buffers[uuid]
        cupy_buffer = xla_buffer_to_cupy(xla_buffer, take_ownership=True)
        ind, n_elements = infer_offset_and_n_elements(tensor_slice)
        cupy_slice = cupy_buffer[ind]
        cupy_output_slice = cupy_buffer[output_idx]
        communicators[relative_idx[device_id]].allGather(
            nccl_util.get_tensor_ptr(cupy_slice),
            nccl_util.get_tensor_ptr(cupy_output_slice), n_elements,
            nccl_util.get_nccl_tensor_dtype(cupy_buffer),
            cupy.cuda.Stream.null.ptr)
        cupy_buffers.append(cupy_buffer)
    nccl_util.groupEnd()
    for device_id, cupy_buffer in zip(device_ids, cupy_buffers):
        uuid = uuids[device_id]
        buf = cupy_to_xla_buffer(cupy_buffer)
        if is_bool:
            buf = _uint8_to_bool(buf)
        worker.buffers[uuid] = buf


def cupy_nccl_broadcast(worker, uuids, comm_key, world_size, devices_ids,
                        devices_global_rank, tensor_slices, group_name):
    to_use = []
    for_buffer = []
    is_bool = worker.buffers[uuids[devices_ids[0]]].dtype == np.bool_
    for device_id, global_rank, tensor_slice in zip(devices_ids,
                                                    devices_global_rank,
                                                    tensor_slices):
        uuid = uuids[device_id]
        tensor_shape = worker.buffers[uuid].shape
        slice_shape = tuple(ind.stop - ind.start for ind in tensor_slice)
        if is_continuous_subset(tensor_slice, tensor_shape):
            # fast path, two cases: (1) same shape, (2) continuous subset.
            tmp = xla_buffer_to_cupy(worker.buffers[uuid])
            if slice_shape != tensor_shape:
                ind, _ = infer_offset_and_n_elements(tensor_slice)
                to_use.append(tmp[ind])
            else:
                to_use.append(tmp)
            for_buffer.append(tmp)
        else:
            tmp = None
            if global_rank == 0:
                start_indices = tuple(o.start for o in tensor_slice)
                tmp = jax_tensor_index(
                    xla_buffer_to_jax_tensor(worker.buffers[uuid]),
                    start_indices, slice_shape)
                tmp = jax_tensor_to_cupy(tmp)
            else:
                tmp = device_put(
                    jnp.ones(slice_shape, dtype=worker.buffers[uuid].dtype),
                    worker.local_devices[device_id])
                tmp = jax_tensor_to_cupy(tmp, take_ownership=True)
            to_use.append(tmp)
            for_buffer.append(tmp)

    _, n_elements = infer_offset_and_n_elements(tensor_slices[0])
    col.broadcast_partialgpu(to_use, n_elements, comm_key, world_size,
                             devices_ids, devices_global_rank, group_name)

    for for_buffer_tensor, device_id, global_rank, tensor_slice in zip(
            for_buffer, devices_ids, devices_global_rank, tensor_slices):
        if global_rank == 0:
            continue
        uuid = uuids[device_id]
        tensor_shape = worker.buffers[uuid].shape
        slice_shape = tuple(ind.stop - ind.start for ind in tensor_slice)
        if is_continuous_subset(tensor_slice, tensor_shape):
            worker.buffers[uuid] = cupy_to_xla_buffer(for_buffer_tensor)
        else:
            recv_tensor = cupy_to_jax_tensor(for_buffer_tensor)
            start_indices = tuple(
                ind_in_dst.start for ind_in_dst in tensor_slice)
            new_buffer = jax_tensor_set(
                xla_buffer_to_jax_tensor(worker.buffers[uuid]), recv_tensor,
                start_indices)
            worker.buffers[uuid] = jax_tensor_to_xla_buffer(new_buffer)
        if is_bool:
            worker.buffers[uuid] = _uint8_to_bool(worker.buffers[uuid])


# in XLA pred(bool) and uint8 are different, but xla->dlpack->xla
# turns a bool into uint8. This implementation is slow.
def _uint8_to_bool(xla_buffer):
    buf = xla_buffer_to_jax_tensor(xla_buffer).astype(np.bool_)
    return jax_tensor_to_xla_buffer(buf)
