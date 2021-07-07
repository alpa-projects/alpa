import jax
import numpy
from jax import core, xla
from jax._src.util import (partial, unzip3)
from jax.abstract_arrays import array_types
from jax.interpreters import pxla
from jax.interpreters.pxla import (ShardingSpec, Chunked, NoSharding, Replicated,
                                   ShardedAxis, _as_slice_indices, _hashable_index, ShardedDeviceArray)
import numpy as np
from jax.lib import xla_client, xla_bridge
import jax.numpy as jnp


backend = xla_client.get_local_backend()

A  = backend.buffer_from_pyval(np.ones([5, 4], dtype=numpy.float32), backend.local_devices()[0]) # do not work
# A  = backend.buffer_from_pyval(np.ones([5, 4]), backend.local_devices()[0]) # do not work
# A = jnp.arange(20).reshape(5, 4) # work
offset = [slice(0, 2), slice(2, 3)]
# B = A[tuple(offset)]
# print(B)
C = A[tuple(offset)]
print(A)
print(C)