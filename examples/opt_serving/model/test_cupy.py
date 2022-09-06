import numpy as np
import cupy
import jax.numpy as jnp

from jax._src.lib import xla_client as xc

from alpa.collective.worker_nccl_util_cupy import cupy_to_jax_tensor

def jax_to_cupy(jax_array):
    return cupy.from_dlpack(
        xc._xla.buffer_to_dlpack_managed_tensor(jax_array, take_ownership=False))

cache = np.array([1, 2, 3 ,4])
print(cache)
jax_cache = jnp.asarray(cache)
jax_cache_cupy = jax_to_cupy(jax_cache)

# print(f"jax_cache: {jax_cache.__cuda}")

print(f"jax_cache: {jax_cache}, cupy cache: {jax_cache_cupy}")

jax_cache_cupy[0] = 5
print(f"jax_cache: {jax_cache}, cupy cache: {jax_cache_cupy}")

jax_cache = jax_cache + 1
print(f"jax_cache: {jax_cache}, cupy cache: {jax_cache_cupy}")
jax_cache_new = cupy_to_jax_tensor(jax_cache_cupy)
print(f"jax_cache: {jax_cache_new}, cupy cache: {jax_cache_cupy}")


