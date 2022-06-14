"""Code to wrap NCCL API calls from XLA."""
from jax._src.lib import xla_extension as xe


def get_nccl_runtime_version():
    return xe.nccl_GetVersion()


def get_nccl_unique_id():
    return xe.nccl_GetUniqueId()


def create_nccl_communicator(world_size, nccl_unique_id, rank):
    """Create an NCCL communicator using NCCL APIs.

    Args:
        world_size (int): the number of processes of this communicator group.
        nccl_unique_id (str): the NCCLUniqueID for this group.
        rank (int): the rank of this process.
    Returns:
        comm (nccl.ncclComm_t): an NCCL communicator.
    """
    comm = xe.nccl_NcclCommunicator(world_size, nccl_unique_id, rank)
    return comm

