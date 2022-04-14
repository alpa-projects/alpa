import unittest
import os

import jax
import alpa
import ray
import numpy as np
from jax import numpy as jnp
from jax import lax
from alpa.pipeline_parallel.primitive_def import mark_pipeline, mark_gradient
from alpa import automatic_layer_construction
from alpa.util import get_ray_namespace_str


class AutoShardingFwdOnlyTest(unittest.TestCase):
    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        assert len(jax.local_devices()) >= 4

        ray.init(address="auto",
                 namespace=get_ray_namespace_str(prefix="alpa-unittest"))

    def tearDown(self):
        ray.shutdown()

    def test_pipeshard_fwd_only(self):
        device_mesh = alpa.DeviceCluster().get_virtual_physical_mesh()
        alpa.set_parallelize_options(
            devices=device_mesh,
            strategy="pipeshard_parallel",
            pipeline_stage_mode="auto_gpipe",
            num_micro_batches=2,
        )

        def func(x):
            bufs_updated = {}
            bufs_updated["_dummy"] = jnp.ones(1)
            x = jnp.matmul(x, x)
            x = jnp.matmul(x, x)
            y = x + 1
            return bufs_updated, (x, y)

        def wrapper_func(x):
            bufs_updated, (x, y) = mark_gradient(automatic_layer_construction(layer_num=1)(func)(x))
            # Add fake operations so that XLA has something to do in apply_grad stage, which makes it happy.
            bufs_updated["_dummy"] = bufs_updated["_dummy"] + 0
            x = x + 0
            y = y + 0
            return bufs_updated, (x, y)

        x = np.random.rand(16, 16)

        bufs_updated, alpa_output = tuple(alpa.api.parallelize(
            wrapper_func,
            batch_argnums=(1,),  # the 1st argument to JAX func is input batch
            fwd_only=True,
        )(x))

        bufs_updated, jax_output = tuple(func(x))

        assert len(jax_output) == len(alpa_output)
        for a, b in zip(jax_output, alpa_output):
            anp = np.array(a)
            bnp = np.array(b)
            assert np.allclose(anp, bnp, rtol=1e-3)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        AutoShardingFwdOnlyTest("test_pipeshard_fwd_only"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
