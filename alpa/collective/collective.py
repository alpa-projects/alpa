"""APIs exposed under the namespace ray.util.collective."""
import logging
import os
from typing import List

import numpy as np
import ray
from jax._src.lib import xla_extension as xe

from alpa.collective import types
from alpa.global_env import global_config
from alpa.util import try_import_ray_worker

ray_worker = try_import_ray_worker()

_CUPY_NCCL_AVAILABLE = True
_XLA_NCCL_AVAILABLE = True
_GLOO_AVAILABLE = True

logger = logging.getLogger(__name__)

try:
    from alpa.collective.collective_group.nccl_collective_group import (
        NCCLGroup as CupyNcclGroup)
except ImportError:
    _CUPY_NCCL_AVAILABLE = False

try:
    from alpa.collective.collective_group.xla_nccl_collective_group import (
        XLANCCLGroup as XlaNcclGroup)
except AttributeError:
    _XLA_NCCL_AVAILABLE = False

try:
    from alpa.collective.collective_group.gloo_collective_group import (
        GLOOGroup)
except ImportError:
    _GLOO_AVAILABLE = False


def nccl_available():
    if global_config.nccl_mode == "cupy":
        if not _CUPY_NCCL_AVAILABLE:
            logger.warning("NCCL seems unavailable. Please install Cupy "
                           "following the guide at: "
                           "https://docs.cupy.dev/en/stable/install.html.")
        return _CUPY_NCCL_AVAILABLE
    elif global_config.nccl_mode == "xla_extension":
        if not _XLA_NCCL_AVAILABLE:
            logger.warning("NCCL from xla_extention seems unavailable! "
                           "Please check whether your local tensorflow-alpa "
                           "has already been up-to-date. You could also set "
                           "global_config.nccl_mode == \"cupy\" to "
                           "use another set of nccl apis from cupy. ")
        return _XLA_NCCL_AVAILABLE
    else:
        raise ValueError(f"nccl mode {global_config.nccl_mode} is illegal")


def get_nccl_group(world_size, rank, group_name):
    assert nccl_available()
    if global_config.nccl_mode == "cupy":
        return CupyNcclGroup(world_size, rank, group_name)
    elif global_config.nccl_mode == "xla_extension":
        return XlaNcclGroup(world_size, rank, group_name)
    else:
        raise ValueError(f"nccl mode {global_config.nccl_mode} is illegal")


def gloo_available():
    return _GLOO_AVAILABLE


class GroupManager:
    """Use this class to manage the collective groups we created so far.

    Each process will have an instance of `GroupManager`. Each process
    could belong to multiple collective groups. The membership information
    and other metadata are stored in the global `_group_mgr` object.
    """

    def __init__(self):
        self._name_group_map = {}
        self._group_name_map = {}

    def create_collective_group(self, backend, world_size, rank, group_name):
        """The entry to create new collective groups in the manager.

        Put the registration and the group information into the manager
        metadata as well.
        """
        backend = types.Backend(backend)
        if backend == types.Backend.MPI:
            raise RuntimeError("Ray does not support MPI.")
        if backend == types.Backend.GLOO:
            logger.debug(f"Creating GLOO group: '{group_name}'...")
            g = GLOOGroup(world_size,
                          rank,
                          group_name,
                          store_type="redis",
                          device_type="tcp")
            self._name_group_map[group_name] = g
            self._group_name_map[g] = group_name
        if backend == types.Backend.NCCL:
            logger.debug(f"Creating NCCL group: '{group_name}'...")
            g = get_nccl_group(world_size, rank, group_name)
            self._name_group_map[group_name] = g
            self._group_name_map[g] = group_name
        return self._name_group_map[group_name]

    def is_group_exist(self, group_name):
        return group_name in self._name_group_map

    def get_group_by_name(self, group_name):
        """Get the collective group handle by its name."""
        if not self.is_group_exist(group_name):
            logger.warning(f"The group '{group_name}' is not initialized.")
            return None
        return self._name_group_map[group_name]

    def destroy_collective_group(self, group_name):
        """Group destructor."""
        if not self.is_group_exist(group_name):
            logger.warning(f"The group '{group_name}' does not exist.")
            return

        # release the collective group resource
        g = self._name_group_map[group_name]
        # clean up the dicts
        del self._group_name_map[g]
        del self._name_group_map[group_name]
        # Release the communicator resources
        g.destroy_group()

        # Release the detached actors spawned by `create_collective_group()`
        name = "info_" + group_name
        try:
            store = ray.get_actor(name)
            ray.kill(store)
        except ValueError:
            pass


_group_mgr = GroupManager()


def is_group_initialized(group_name):
    """Check if the group is initialized in this process by the group name."""
    return _group_mgr.is_group_exist(group_name)


def init_collective_group(world_size: int,
                          rank: int,
                          backend=types.Backend.NCCL,
                          group_name: str = "default"):
    """Initialize a collective group inside an actor process.

    Args:
        world_size (int): the total number of processes in the group.
        rank (int): the rank of the current process.
        backend: the CCL backend to use, NCCL or GLOO.
        group_name (str): the name of the collective group.

    Returns:
        None
    """
    _check_inside_actor()
    backend = types.Backend(backend)
    _check_backend_availability(backend)
    # TODO(Hao): implement a group auto-counter.
    if not group_name:
        raise ValueError(f"group_name '{group_name}' needs to be a string.")

    if _group_mgr.is_group_exist(group_name):
        raise RuntimeError("Trying to initialize a group twice.")

    assert world_size > 0
    assert rank >= 0
    assert rank < world_size
    _group_mgr.create_collective_group(backend, world_size, rank, group_name)


def create_collective_group(actors,
                            world_size: int,
                            ranks: List[int],
                            backend=types.Backend.NCCL,
                            group_name: str = "default"):
    """Declare a list of actors as a collective group.

    Note: This function should be called in a driver process.

    Args:
        actors (list): a list of actors to be set in a collective group.
        world_size (int): the total number of processes in the group.
        ranks (List[int]): the rank of each actor.
        backend: the CCL backend to use, NCCL or GLOO.
        group_name (str): the name of the collective group.

    Returns:
        None
    """
    backend = types.Backend(backend)
    _check_backend_availability(backend)

    name = "info_" + group_name
    try:
        ray.get_actor(name)
        raise RuntimeError("Trying to initialize a group twice.")
    except ValueError:
        pass

    if len(ranks) != len(actors):
        raise RuntimeError(
            f"Each actor should correspond to one rank. Got '{len(ranks)}' "
            f"ranks but '{len(actors)}' actors")

    if set(ranks) != set(range(len(ranks))):
        got_ranks = "".join([str(r) for r in ranks])
        raise RuntimeError(
            f"Ranks must be a permutation from 0 to '{len(ranks)}'. "
            f"Got '{got_ranks}'.")

    if world_size <= 0:
        raise RuntimeError("World size must be greater than zero. "
                           f"Got '{world_size}'.")
    if not all(ranks) >= 0:
        raise RuntimeError("Ranks must be non-negative.")
    if not all(ranks) < world_size:
        raise RuntimeError("Ranks cannot be greater than world_size.")

    # avoid a circular dependency
    from alpa.collective.util import Info  # pylint: disable=import-outside-toplevel
    # store the information into a NamedActor that can be accessed later.
    name = "info_" + group_name
    actors_id = [a._ray_actor_id for a in actors]  # pylint: disable=protected-access
    # TODO (Dacheng): how do we recycle this name actor?
    info = Info.options(name=name, lifetime="detached").remote()
    ray.get([info.set_info.remote(actors_id, world_size, ranks, backend)])


# TODO (we need a declarative destroy() API here.)
def destroy_collective_group(group_name: str = "default") -> None:
    """Destroy a collective group given its group name."""
    _check_inside_actor()
    _group_mgr.destroy_collective_group(group_name)


def get_rank(group_name: str = "default") -> int:
    """Return the rank of this process in the given group.

    Args:
        group_name (str): the name of the group to query

    Returns:
        the rank of this process in the named group,
        -1 if the group does not exist or the process does
        not belong to the group.
    """
    _check_inside_actor()
    if not is_group_initialized(group_name):
        return -1
    g = _group_mgr.get_group_by_name(group_name)
    return g.rank


def get_collective_group_size(group_name: str = "default") -> int:
    """Return the size of the collective group with the given name.

    Args:
        group_name: the name of the group to query

    Returns:
        The world size of the collective group, -1 if the group does
            not exist or the process does not belong to the group.
    """
    _check_inside_actor()
    if not is_group_initialized(group_name):
        return -1
    g = _group_mgr.get_group_by_name(group_name)
    return g.world_size


def allreduce(tensor, group_name: str = "default", op=types.ReduceOp.SUM):
    """Collective allreduce the tensor across the group.

    Args:
        tensor: the tensor to be all-reduced on this process.
        group_name (str): the collective group name to perform allreduce.
        op: The reduce operation.

    Returns:
        None
    """
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    opts = types.AllReduceOptions
    opts.reduce_op = op
    g.allreduce([tensor], opts)


def allreduce_multigpu(tensor_list: list,
                       group_name: str = "default",
                       op=types.ReduceOp.SUM):
    """Collective allreduce a list of tensors across the group.

    Args:
        tensor_list (List[tensor]): list of tensors to be allreduced,
            each on a GPU.
        group_name (str): the collective group name to perform allreduce.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError("Multigpu calls requires NCCL and Cupy.")
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)
    opts = types.AllReduceOptions
    opts.reduce_op = op
    g.allreduce(tensor_list, opts)


def barrier(group_name: str = "default"):
    """Barrier all processes in the collective group.

    Args:
        group_name (str): the name of the group to barrier.

    Returns:
        None
    """
    g = _check_and_get_group(group_name)
    g.barrier()


def reduce(tensor,
           dst_rank: int = 0,
           group_name: str = "default",
           op=types.ReduceOp.SUM):
    """Reduce the tensor across the group to the destination rank.

    Args:
        tensor: the tensor to be reduced on this process.
        dst_rank (int): the rank of the destination process.
        group_name (str): the collective group name to perform reduce.
        op: The reduce operation.

    Returns:
        None
    """
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)

    # check dst rank
    _check_rank_valid(g, dst_rank)
    opts = types.ReduceOptions()
    opts.reduce_op = op
    opts.root_rank = dst_rank
    opts.root_tensor = 0
    g.reduce([tensor], opts)


def reduce_multigpu(tensor_list: list,
                    dst_rank: int = 0,
                    dst_tensor: int = 0,
                    group_name: str = "default",
                    op=types.ReduceOp.SUM):
    """Reduce the tensor across the group to the destination rank
    and destination tensor.

    Args:
        tensor_list: the list of tensors to be reduced on this process;
            each tensor located on a GPU.
        dst_rank (int): the rank of the destination process.
        dst_tensor: the index of GPU at the destination.
        group_name (str): the collective group name to perform reduce.
        op: The reduce operation.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError("Multigpu calls requires NCCL and Cupy.")
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)

    # check dst rank
    _check_rank_valid(g, dst_rank)
    _check_root_tensor_valid(len(tensor_list), dst_tensor)
    opts = types.ReduceOptions()
    opts.reduce_op = op
    opts.root_rank = dst_rank
    opts.root_tensor = dst_tensor
    g.reduce(tensor_list, opts)


def broadcast(tensor, src_rank: int = 0, group_name: str = "default"):
    """Broadcast the tensor from a source process to all others.

    Args:
        tensor: the tensor to be broadcasted (src) or received (destination).
        src_rank (int): the rank of the source process.
        group_name (str): the collective group name to perform broadcast.

    Returns:
        None
    """
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)

    # check src rank
    _check_rank_valid(g, src_rank)
    opts = types.BroadcastOptions()
    opts.root_rank = src_rank
    opts.root_tensor = 0
    g.broadcast([tensor], opts)


def broadcast_partialgpu(tensor_list,
                         n_elements,
                         comm_key,
                         world_size,
                         devices_ids,
                         devices_global_rank,
                         group_name: str = "default",
                         local_start_pos_list=None):
    """Broadcast the tensor from a source GPU to some other GPUs.
    This function is different from broadcast_multigpu that it only
    uses a subset of gpus in one host.

    Args:
        tensor_list: the tensors to broadcast (src) or receive (dst).
        n_elements: total number of elements involved in this broadcast.
        comm_key: an unique identifier for this cross-host collective group.
        world_size: total number of devices in this cross-host collective group.
        devices_ids: local devices in this cross-host collective group.
        devices_global_rank: the corresponding global rank for local devices.
        group_name (str): the collective group name to perform broadcast.
        local_start_pos_list (list[int]): the list contains starting positions
        of the contiguous data to be sent in every tensor.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError("Multigpu calls requires NCCL and Cupy.")
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)

    opts = types.BroadcastOptions()
    opts.n_elements = n_elements
    opts.comm_key = comm_key
    opts.world_size = world_size
    opts.devices_ids = devices_ids
    opts.devices_global_rank = devices_global_rank
    opts.local_start_pos_list = (local_start_pos_list
                                 if local_start_pos_list is not None else [])
    g.broadcast_partialgpu(tensor_list, opts)


def broadcast_multigpu(tensor_list,
                       src_rank: int = 0,
                       src_tensor: int = 0,
                       group_name: str = "default"):
    """Broadcast the tensor from a source GPU to all other GPUs.

    Args:
        tensor_list: the tensors to broadcast (src) or receive (dst).
        src_rank (int): the rank of the source process.
        src_tensor (int): the index of the source GPU on the source process.
        group_name (str): the collective group name to perform broadcast.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError("Multigpu calls requires NCCL and Cupy.")
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)

    # check src rank
    _check_rank_valid(g, src_rank)
    _check_root_tensor_valid(len(tensor_list), src_tensor)
    opts = types.BroadcastOptions()
    opts.root_rank = src_rank
    opts.root_tensor = src_tensor
    g.broadcast(tensor_list, opts)


def allgather(tensor_list: list, tensor, group_name: str = "default"):
    """Allgather tensors from each process of the group into a list.

    Args:
        tensor_list (list): the results, stored as a list of tensors.
        tensor: the tensor (to be gathered) in the current process
        group_name (str): the name of the collective group.

    Returns:
        None
    """
    _check_single_tensor_input(tensor)
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)
    if len(tensor_list) != g.world_size:
        # Typically CLL lib requires len(tensor_list) >= world_size;
        # Here we make it more strict: len(tensor_list) == world_size.
        raise RuntimeError(
            "The length of the tensor list operands to allgather "
            "must be equal to world_size.")
    opts = types.AllGatherOptions()
    g.allgather([tensor_list], [tensor], opts)


def allgather_multigpu(output_tensor_lists: list,
                       input_tensor_list: list,
                       group_name: str = "default"):
    """Allgather tensors from each gpus of the group into lists.

    Args:
        output_tensor_lists (List[List[tensor]]): gathered results, with shape
            must be num_gpus * world_size * shape(tensor).
        input_tensor_list: (List[tensor]): a list of tensors, with shape
            num_gpus * shape(tensor).
        group_name (str): the name of the collective group.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError("Multigpu calls requires NCCL and Cupy.")
    _check_tensor_lists_input(output_tensor_lists)
    _check_tensor_list_input(input_tensor_list)
    g = _check_and_get_group(group_name)
    opts = types.AllGatherOptions()
    g.allgather(output_tensor_lists, input_tensor_list, opts)


def reducescatter(tensor,
                  tensor_list: list,
                  group_name: str = "default",
                  op=types.ReduceOp.SUM):
    """Reducescatter a list of tensors across the group.

    Reduce the list of the tensors across each process in the group, then
    scatter the reduced list of tensors -- one tensor for each process.

    Args:
        tensor: the resulted tensor on this process.
        tensor_list (list): The list of tensors to be reduced and scattered.
        group_name (str): the name of the collective group.
        op: The reduce operation.

    Returns:
        None
    """
    _check_single_tensor_input(tensor)
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)
    if len(tensor_list) != g.world_size:
        raise RuntimeError(
            "The length of the tensor list operands to reducescatter "
            "must not be equal to world_size.")
    opts = types.ReduceScatterOptions()
    opts.reduce_op = op
    g.reducescatter([tensor], [tensor_list], opts)


def reducescatter_multigpu(output_tensor_list,
                           input_tensor_lists,
                           group_name: str = "default",
                           op=types.ReduceOp.SUM):
    """Reducescatter a list of tensors across all GPUs.

    Args:
        output_tensor_list: the resulted list of tensors, with
            shape: num_gpus * shape(tensor).
        input_tensor_lists: the original tensors, with shape:
            num_gpus * world_size * shape(tensor).
        group_name (str): the name of the collective group.
        op: The reduce operation.

    Returns:
        None.
    """
    if not types.cupy_available():
        raise RuntimeError("Multigpu calls requires NCCL and Cupy.")
    _check_tensor_lists_input(input_tensor_lists)
    _check_tensor_list_input(output_tensor_list)
    g = _check_and_get_group(group_name)
    opts = types.ReduceScatterOptions()
    opts.reduce_op = op
    g.reducescatter(output_tensor_list, input_tensor_lists, opts)


def send(tensor, dst_rank: int, group_name: str = "default"):
    """Send a tensor to a remote process synchronously.

    Args:
        tensor: the tensor to send.
        dst_rank (int): the rank of the destination process.
        group_name (str): the name of the collective group.

    Returns:
        None
    """
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, dst_rank)
    if dst_rank == g.rank:
        raise RuntimeError(f"The destination rank '{dst_rank}' is self.")
    opts = types.SendOptions()
    opts.dst_rank = dst_rank
    g.send([tensor], opts)


def send_multigpu(tensor,
                  dst_rank: int,
                  dst_gpu_index: int,
                  group_name: str = "default",
                  start_pos=0,
                  n_elements=0):
    """Send a tensor to a remote GPU synchronously.

    The function asssume each process owns >1 GPUs, and the sender
    process and receiver process has equal nubmer of GPUs.

    Args:
        tensor: the tensor to send, located on a GPU.
        dst_rank (int): the rank of the destination process.
        dst_gpu_index (int): the destination gpu index.
        group_name (str): the name of the collective group.
        start_pos (int): the starting position of the contiguous
        data to be sent in this tensor.
        n_elements (int): if specified, send the next n elements
            from the starting address of tensor.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError("send_multigpu call requires NCCL.")
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, dst_rank)
    if dst_rank == g.rank:
        raise RuntimeError(f"The dst_rank '{dst_rank}' is self. Considering "
                           "doing GPU to GPU memcpy instead?")
    if n_elements < 0:
        raise RuntimeError(f"The n_elements '{n_elements}' should >= 0.")
    opts = types.SendOptions()
    opts.dst_rank = dst_rank
    opts.dst_gpu_index = dst_gpu_index
    opts.start_pos = start_pos
    opts.n_elements = n_elements
    g.send([tensor], opts)


def recv(tensor, src_rank: int, group_name: str = "default"):
    """Receive a tensor from a remote process synchronously.

    Args:
        tensor: the received tensor.
        src_rank (int): the rank of the source process.
        group_name (str): the name of the collective group.

    Returns:
        None
    """
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, src_rank)
    if src_rank == g.rank:
        raise RuntimeError(f"The destination rank '{src_rank}' is self.")
    opts = types.RecvOptions()
    opts.src_rank = src_rank
    g.recv([tensor], opts)


def recv_multigpu(tensor,
                  src_rank: int,
                  src_gpu_index: int,
                  group_name: str = "default",
                  start_pos=0,
                  n_elements=0):
    """Receive a tensor from a remote GPU synchronously.

    The function asssume each process owns >1 GPUs, and the sender
    process and receiver process has equal nubmer of GPUs.

    Args:
        tensor: the received tensor, located on a GPU.
        src_rank (int): the rank of the source process.
        src_gpu_index (int): the index of the source gpu on the src process.
        start_pos (int): the starting position of the contiguous
        data to be sent in this tensor.
        group_name (str): the name of the collective group.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError("recv_multigpu call requires NCCL.")
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, src_rank)
    if src_rank == g.rank:
        raise RuntimeError(f"The dst_rank '{src_rank}' is self. Considering "
                           "doing GPU to GPU memcpy instead?")
    if n_elements < 0:
        raise RuntimeError(f"The n_elements '{n_elements}' should be >= 0.")
    opts = types.RecvOptions()
    opts.src_rank = src_rank
    opts.src_gpu_index = src_gpu_index
    opts.start_pos = start_pos
    opts.n_elements = n_elements
    g.recv([tensor], opts)


def synchronize(gpu_id: int):
    """Synchronize the current process to a give device.

    Args:
        gpu_id (int): the GPU device id to synchronize.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError("synchronize call requires CUDA and NCCL.")
    import cupy as cp  # pylint: disable=import-outside-toplevel
    cp.cuda.Device(gpu_id).synchronize()


def _check_and_get_group(group_name):
    """Check the existence and return the group handle."""
    _check_inside_actor()
    if not is_group_initialized(group_name):
        # try loading from remote info store
        try:
            # if the information is stored in an Info object,
            # get and create the group.
            name = "info_" + group_name
            info_actor = ray.get_actor(name=name)
            ids, world_size, rank, backend = ray.get(
                info_actor.get_info.remote())

            # Recycle the info named actor *pro-activately* to avoid named actor
            # leak.
            if ray.get(info_actor.get_access_counter.remote()) == world_size:
                ray.kill(info_actor)
                logger.debug(
                    "Information about the collective group has been "
                    "broadcasted. The Info actor will go out of context and be "
                    "destroyed.")

            worker = ray_worker.global_worker
            id_ = worker.core_worker.get_actor_id()
            r = rank[ids.index(id_)]
            _group_mgr.create_collective_group(backend, world_size, r,
                                               group_name)
        except ValueError as exc:
            # check if this group is initialized using options()
            if ("collective_group_name" in os.environ and
                    os.environ["collective_group_name"] == group_name):
                rank = int(os.environ["collective_rank"])
                world_size = int(os.environ["collective_world_size"])
                backend = os.environ["collective_backend"]
                _group_mgr.create_collective_group(backend, world_size, rank,
                                                   group_name)
            else:
                raise RuntimeError(
                    f"The collective group '{group_name}' is not "
                    "initialized in the process.") from exc
    g = _group_mgr.get_group_by_name(group_name)
    return g


check_and_get_group = _check_and_get_group


def record_events(group_name, uuids, num_devices, is_send):
    g = _check_and_get_group(group_name)
    g.record_events(uuids, num_devices, is_send)


def wait_events(group_name, uuids, num_devices, is_send):
    g = _check_and_get_group(group_name)
    g.wait_events(uuids, num_devices, is_send)


def comm_wait_compute(group_name, is_send, is_compute, device_id):
    g = _check_and_get_group(group_name)
    g.comm_wait_compute(is_send, is_compute, device_id)


def compute_wait_comm(group_name, is_send, is_compute, device_id):
    g = _check_and_get_group(group_name)
    g.compute_wait_comm(is_send, is_compute, device_id)


def _check_single_tensor_input(tensor):
    """Check if the tensor is with a supported type."""
    if isinstance(tensor, (np.ndarray, xe.DeviceArray)):
        return
    if types.cupy_available():
        if isinstance(tensor, types.cp.ndarray):
            return
    if types.torch_available():
        if isinstance(tensor, types.th.Tensor):
            return
    raise RuntimeError(f"Unrecognized tensor type '{type(tensor)}'. "
                       "Supported types are: np.ndarray, torch.Tensor, "
                       "cupy.ndarray.")


def _check_backend_availability(backend: types.Backend):
    """Check whether the backend is available."""
    if backend == types.Backend.GLOO:
        if not gloo_available():
            raise RuntimeError("GLOO is not available.")
    elif backend == types.Backend.NCCL:
        if not nccl_available():
            raise RuntimeError("NCCL is not available.")


def _check_inside_actor():
    """Check if currently it is inside a Ray actor/task."""
    worker = ray_worker.global_worker
    if worker.mode == ray.WORKER_MODE:
        return
    else:
        raise RuntimeError("The collective APIs shall be only used inside "
                           "a Ray actor or task.")


def _check_rank_valid(g, rank: int):
    """Check the rank: 0 <= rank < world_size."""
    if rank < 0:
        raise ValueError(f"rank '{rank}' is negative.")
    if rank >= g.world_size:
        raise ValueError(f"rank '{rank}' must be less than world size "
                         f"'{g.world_size}'")


def _check_tensor_list_input(tensor_list):
    """Check if the input is a list of supported tensor types."""
    if not isinstance(tensor_list, list):
        raise RuntimeError("The input must be a list of tensors. "
                           f"Got '{type(tensor_list)}'.")
    if not tensor_list:
        raise RuntimeError("Got an empty list of tensors.")
    for t in tensor_list:
        _check_single_tensor_input(t)


def _check_tensor_lists_input(tensor_lists):
    """Check if the input is a list of lists of supported tensor types."""
    if not isinstance(tensor_lists, list):
        raise RuntimeError("The input must be a list of lists of tensors. "
                           f"Got '{type(tensor_lists)}'.")
    if not tensor_lists:
        raise RuntimeError(f"Did not receive tensors. Got: {tensor_lists}")
    for t in tensor_lists:
        _check_tensor_list_input(t)


def _check_root_tensor_valid(length, root_tensor):
    """Check the root_tensor device is 0 <= root_tensor < length"""
    if root_tensor < 0:
        raise ValueError(f"root_tensor '{root_tensor}' is negative.")
    if root_tensor >= length:
        raise ValueError(f"root_tensor '{root_tensor}' is greater "
                         f"than the number of GPUs: '{length}'")
