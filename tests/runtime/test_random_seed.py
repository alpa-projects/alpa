"""Test random seed."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from alpa import init, grad, parallelize, ShardParallel, set_seed, shutdown
from alpa.parallel_method import PipeshardParallel
from alpa.pipeline_parallel.layer_construction import manual_layer_construction, manual_remat
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary
from alpa.testing import assert_allclose


class RandomSeedTest(unittest.TestCase):

    def test_random_generation(self):

        @parallelize(method=ShardParallel())
        def func():
            rngkey = jax.random.PRNGKey(0)
            x = jax.random.normal(rngkey, (16, 4))
            y = jax.random.normal(rngkey, (16, 4))
            z = jnp.hstack((x, y))
            z = (10000 * z).astype(jnp.int32)
            return z.flatten()

        a = func()
        s = set(np.array(a))

        assert len(a) == len(s)

    def test_set_seed(self):

        @parallelize(method=ShardParallel())
        def func():
            rngkey = jax.random.PRNGKey(0)
            return jax.random.normal(rngkey, (16, 4))

        set_seed(10)
        a = func()
        b = func()
        set_seed(10)
        c = func()

        assert_allclose(a, c)

        allclose = True
        try:
            assert_allclose(a, b)
        except AssertionError:
            allclose = False
        assert not allclose

    def test_remat_rng(self):
        init(cluster="ray")
        shape = (4, 4)
        def get_parallelized_step(method, add_marker=False):
            def forward(params, x, key):
                rns = jax.random.normal(key, shape)
                y = x @ params["x1"]
                y = y @ rns
                mark_pipeline_boundary()
                y = y @ params["x2"]
                return jnp.mean(y), rns
            if add_marker:
                forward = manual_layer_construction(forward, remat_layer=True)
            else:
                forward = manual_remat(forward)
            @parallelize(method=method, donate_argnums=())
            def fn(*args):
                return grad(forward, has_aux=True)(*args)
            return fn
        @jax.grad
        def normal_step(params, x, rns):
            y = x @ params["x1"]
            y = y @ rns
            y = y @ params["x2"]
            return jnp.mean(y)

        remat_pipeline_fn = get_parallelized_step(PipeshardParallel(num_micro_batches=2), True)
        remat_shard_fn = get_parallelized_step(ShardParallel(num_micro_batches=2), False)

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, shape)
        params = {
            "x1": jax.random.normal(key, shape),
            "x2": jax.random.normal(key, shape)
        }
        pipeline_out, rns = remat_pipeline_fn(params, x, key)
        expected_out = normal_step(params, x, rns._value)
        assert_allclose(pipeline_out, expected_out)

        shard_out, rns = remat_shard_fn(params, x, key)
        expected_out = normal_step(params, x, rns._value)
        assert_allclose(shard_out, expected_out)
        shutdown()

def suite():
    suite = unittest.TestSuite()
    suite.addTest(RandomSeedTest("test_random_generation"))
    suite.addTest(RandomSeedTest("test_set_seed"))
    suite.addTest(RandomSeedTest("test_remat_rng"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
