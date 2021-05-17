from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.lib import xla_client, xla_bridge

ops = xla_client.ops


def test_simple_graph():
    c = xla_client.XlaBuilder("simple_graph")
    x = ops.Parameter(c, 0, xla_client.shape_from_pyval(np.ones((10, 8), dtype=np.float32)))
    y = ops.Parameter(c, 1, xla_client.shape_from_pyval(np.ones((10, 8), dtype=np.float32)))

    backend = xla_client.get_local_backend("gpu")

    z = ops.Add(x, y)
    z = ops.Add(z, y)

    c = c.build(z)
    print(c.as_hlo_text())

    compiled_c = backend.compile(c)

    print(compiled_c.hlo_modules()[0].to_string())

    x = backend.buffer_from_pyval(np.ones((10, 8), dtype=np.float32))
    y = backend.buffer_from_pyval(np.ones((10, 8), dtype=np.float32))
    ans, = compiled_c.execute([x, y])

    print("ans", ans)


if __name__ == "__main__":
    test_simple_graph()
