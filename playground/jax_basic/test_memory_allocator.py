
import os
import jax
from jax import numpy as jnp

def run_cmd(x):
    os.system(x)

def test_platform_allocator():
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    #os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    a = jnp.ones(1 << 30)

    run_cmd("nvidia-smi")

    a = None

    run_cmd("nvidia-smi")


if __name__ == "__main__":
    test_platform_allocator()

