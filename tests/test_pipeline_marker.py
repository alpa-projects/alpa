import unittest

import numpy as np

from parax.pipeline_parallel.primitive_def import mark_pipeline_xla
from jax.lib import xla_client as xc, xla_bridge as xb

ops = xc.ops


class PipelineMarkerTest(unittest.TestCase):

    def test_xla_graph(self):
        c = xc.XlaBuilder("simple_graph")
        x = ops.Parameter(c, 0, xc.shape_from_pyval(
            np.ones((10, 8), dtype=np.float32)))
        y = ops.Parameter(c, 1, xc.shape_from_pyval(
            np.ones((10, 8), dtype=np.float32)))

        backend = xb.get_backend("gpu")

        a = ops.Add(x, y)
        b = ops.Mul(x, y)

        output_tuple = mark_pipeline_xla(c, a, b, mark_type="start")
        a = ops.GetTupleElement(output_tuple, 0)
        b = ops.GetTupleElement(output_tuple, 1)

        z = ops.Add(a, b)

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


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineMarkerTest("test_xla_graph"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
