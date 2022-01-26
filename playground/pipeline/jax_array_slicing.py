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
from alpa.util import jax_buffer_set, jax_buffer_set_v2


offset = [0, 4]
m = jnp.zeros([10, 10], dtype=np.float32)
print(m.__cuda_array_interface__)
n = jnp.ones([2, 2], dtype=np.float32)
print(n.__cuda_array_interface__)
k = jax_buffer_set_v2(m, n, tuple(offset))
print(k.__cuda_array_interface__)
print(k)
