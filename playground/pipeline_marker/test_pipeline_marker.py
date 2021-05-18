from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.lib import xla_client, xla_bridge

ops = xla_client.ops

import gpu_custom_call_test
xla_client.register_custom_call_target(b'pipeline_marker',
    gpu_custom_call_test.pipeline_marker(), platform='gpu')


def test_simple_graph():
    c = xla_client.XlaBuilder("simple_graph")
    x = ops.Parameter(c, 0, xla_client.shape_from_pyval(np.ones((10, 8), dtype=np.float32)))
    y = ops.Parameter(c, 1, xla_client.shape_from_pyval(np.ones((10, 8), dtype=np.float32)))

    backend = xla_client.get_local_backend("gpu")

    z = ops.Add(x, y)
    z = ops.Add(z, y)
    out_shape = xla_client.Shape.array_shape(np.dtype(np.float32), (1,), (0,))
    z = xla_client.ops.CustomCall(z,
        b'pipeline_marker',
        operands=(z,),
        shape=out_shape,
        opaque=b"abc"
        )

    c = c.build(z)
    print("=" * 60)
    print(c.as_hlo_text())

    compiled_c = backend.compile(c)

    print("=" * 60)
    print(compiled_c.hlo_modules()[0].to_string())

    x = backend.buffer_from_pyval(np.ones((10, 8), dtype=np.float32))
    y = backend.buffer_from_pyval(np.ones((10, 8), dtype=np.float32))
    ans, = compiled_c.execute([x, y])

    print("=" * 60)
    print("ans", ans)


if __name__ == "__main__":
    test_simple_graph()
