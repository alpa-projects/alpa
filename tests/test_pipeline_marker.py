import unittest

import numpy as np
import jax
from jax.lib import xla_client as xc, xla_bridge as xb
import jax.numpy as jnp

from alpa.pipeline_parallel.primitive_def import mark_pipeline, xla_pipeline_marker
from alpa.testing import assert_allclose

ops = xc.ops


class PipelineMarkerTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(1337)

    def test_xla_graph(self):
        c = xc.XlaBuilder("xla_graph_with_marker")

        parameter_shape = xc.Shape.array_shape(np.dtype(np.float32), (10, 8),
                                               (0, 1))
        x = ops.Parameter(c, 0, parameter_shape)
        y = ops.Parameter(c, 1, parameter_shape)

        backend = xb.get_backend("gpu")

        a = ops.Add(x, y)
        b = ops.Mul(x, y)

        output_tuple = xla_pipeline_marker(c, "start", "1", a, b)
        a = ops.GetTupleElement(output_tuple, 0)
        b = ops.GetTupleElement(output_tuple, 1)

        z = ops.Add(a, b)
        output_tuple = xla_pipeline_marker(c, "end", "1", z)
        z = ops.GetTupleElement(output_tuple, 0)

        c = c.build(z)
        compiled_c = backend.compile(c)

        x_np = np.random.rand(10, 8).astype(np.float32)
        y_np = np.random.rand(10, 8).astype(np.float32)

        x = backend.buffer_from_pyval(x_np)
        y = backend.buffer_from_pyval(y_np)
        z, = compiled_c.execute([x, y])

        a_np = x_np + y_np
        b_np = x_np * y_np
        z_np = a_np + b_np

        assert_allclose(z, z_np)

    def test_jax_graph(self):
        x_np = np.random.rand(10, 8).astype(np.float32)
        y_np = np.random.rand(10, 8).astype(np.float32)
        a_np = x_np + y_np
        b_np = x_np * y_np
        z_np = a_np + b_np

        def f(x, y):
            a = x + y
            b = x * y
            a, b = mark_pipeline(a, b, mark_type="start", name="1")
            z = a + b
            z, = mark_pipeline(z, mark_type="end", name="1")
            return z

        z_without_jit = f(x_np, y_np)
        f = jax.jit(f)
        z_with_jit = f(x_np, y_np)
        assert_allclose(z_with_jit, z_np)
        assert_allclose(z_without_jit, z_np)

    def test_transpose(self):

        def f(x):
            x, = mark_pipeline(x, mark_type="start", name="1")
            x = jnp.transpose(x, axes=(1, 0))
            return x

        x = np.random.rand(2, 4)
        no_jit_result = f(x)
        jit_result = jax.jit(f)(x)
        assert_allclose(no_jit_result, jit_result)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineMarkerTest("test_xla_graph"))
    suite.addTest(PipelineMarkerTest("test_jax_graph"))
    suite.addTest(PipelineMarkerTest("test_transpose"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
