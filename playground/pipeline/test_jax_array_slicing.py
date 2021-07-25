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
from parax.util import jax_buffer_set


offset = [slice(0, 2), slice(4, 6)]
m = jnp.ones([10, 10], dtype=np.float32)
print(m.__cuda_array_interface__)
n = jnp.ones([2, 2], dtype=np.float32)
print(n.__cuda_array_interface__)
k = jax_buffer_set_v2(m, n, offset)
print(k.__cuda_array_interface__)
