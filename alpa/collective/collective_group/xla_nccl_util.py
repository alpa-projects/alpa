"""Code to wrap NCCL API calls from XLA extension."""
from jax._src.lib import xla_extension as xe


def get_nccl_runtime_version():
    return xe.nccl_get_version()


def get_nccl_unique_id():
    return xe.nccl_get_unique_id()
