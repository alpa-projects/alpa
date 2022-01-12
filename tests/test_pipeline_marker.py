import unittest

import numpy as np
from jax.lib import xla_client as xc, xla_bridge as xb

from parax.pipeline_parallel.primitive_def import mark_pipeline_xla
from parax.testing import assert_allclose

ops = xc.ops


class PipelineMarkerTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1337)

    def test_xla_graph(self):
        c = xc.XlaBuilder("simple_graph")

        parameter_shape = xc.Shape.array_shape(np.float32, (10, 8), (0, 1))
        x = ops.Parameter(c, 0, parameter_shape)
        y = ops.Parameter(c, 1, parameter_shape)

        backend = xb.get_backend("gpu")

        a = ops.Add(x, y)
        b = ops.Mul(x, y)

        output_tuple = mark_pipeline_xla(c, a, b, mark_type="start", name="1")
        a = ops.GetTupleElement(output_tuple, 0)
        b = ops.GetTupleElement(output_tuple, 1)

        z = ops.Add(a, b)
        output_tuple = mark_pipeline_xla(c, z, mark_type="end", name="1")
        z = ops.GetTupleElement(output_tuple, 0)

        c = c.build(z)
        print("=" * 60)
        print(c.as_hlo_text())

        compiled_c = backend.compile(c)

        print("=" * 60)
        print(compiled_c.hlo_modules()[0].to_string())

        x_np = np.random.rand(10, 8)
        y_np = np.random.rand(10, 8)

        x = backend.buffer_from_pyval(x_np)
        y = backend.buffer_from_pyval(y_np)
        z, = compiled_c.execute([x, y])

        a_np = x_np + y_np
        b_np = x_np * y_np
        z_np = a_np + b_np

        assert_allclose(z, z_np)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineMarkerTest("test_xla_graph"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
