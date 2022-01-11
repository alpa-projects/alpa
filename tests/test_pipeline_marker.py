import unittest

import numpy as np

from parax.pipeline_parallel.primitive_def import mark_pipeline_xla
from jax.lib import xla_client, xla_bridge

ops = xla_client.ops


class PipelineMarkerTest(unittest.TestCase):

    def test_xla_graph(self):
        c = xla_client.XlaBuilder("simple_graph")
        x = ops.Parameter(c, 0, xla_client.shape_from_pyval(
            np.ones((10, 8), dtype=np.float32)))
        y = ops.Parameter(c, 1, xla_client.shape_from_pyval(
            np.ones((10, 8), dtype=np.float32)))

        backend = xla_client.get_local_backend("gpu")

        a = ops.Add(x, y)
        b = ops.Mul(x, y)

        a, b = mark_pipeline_xla(c, a, b)

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
    suite.addTest(PipelineMarkerTest("test_2_layer_mlp_local_pipeline_parallel"))
    suite.addTest(PipelineMarkerTest("test_2_layer_mlp_3d_parallel"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
