"""Test random seed."""
import unittest
import os

import jax
from jax._src.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp
import numpy as np

from alpa import (init, grad, parallelize, ShardParallel, set_seed, shutdown,
                  AutoShardingOption)
from alpa.parallel_method import PipeshardParallel
from alpa.pipeline_parallel.layer_construction import ManualLayerOption
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary
from alpa.testing import assert_allclose


class RandomSeedTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

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

        # Check all random numbers are unique
        assert len(a) == len(s)

    def test_set_seed(self):

        @parallelize(method=ShardParallel())
        def func():
            rngkey = jax.random.PRNGKey(0)
            return jax.random.normal(rngkey, (16, 4))

        @parallelize(method=ShardParallel())
        def func2():
            rngkey = jax.random.PRNGKey(0)
            return jax.random.normal(rngkey, (16, 4))

        set_seed(10)
        a = func()
        b = func()
        set_seed(10)
        c = func()
        set_seed(10)
        d = func2()

        assert_allclose(a, c)
        assert_allclose(c, d)

        allclose = True
        try:
            assert_allclose(a, b)
        except AssertionError:
            allclose = False
        assert not allclose

    def test_remat_rng(self):
        init(cluster="ray")

        batch_size = 64
        hidden_size = 8
        num_micro_batches = 1
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, hidden_size))
        params = {
            "x1": jax.random.normal(rngkey, (hidden_size, hidden_size)),
            "x2": jax.random.normal(rngkey, (hidden_size, hidden_size)),
        }

        # Run an inference-only forward pass to get rngs
        def gen_rns(params, x, key):
            # NOTE: We minic the real forward pass to make sure
            # the sharding specs are the same. Otherwise, the results of rng
            # do not match.
            y = x @ params["x1"]
            rns = jax.random.normal(key, y.shape)
            y = jax.lax.select(rns > 0, y, jnp.zeros_like(y))
            mark_pipeline_boundary()
            y = y @ params["x2"]
            return rns

        set_seed(10)
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            pipeline_schedule="inference",
            layer_option="manual",
            default_auto_sharding_option=AutoShardingOption(
                force_data_parallel=True))
        p_gen_rns = parallelize(gen_rns, method=method)
        external_rns = np.array(p_gen_rns(params, x, rngkey))

        # Run train step with remat and rng
        def train_step(params, x, key, use_external_rns, external_rns):

            def loss_func(params):
                y = x @ params["x1"]
                if use_external_rns:
                    rns = external_rns
                else:
                    rns = jax.random.normal(key, y.shape)
                y = jax.lax.select(rns > 0, y, jnp.zeros_like(y))
                mark_pipeline_boundary()
                y = y @ params["x2"]
                return jnp.mean(y), rns

            grads, rns = grad(loss_func, has_aux=True)(params)
            # A workaroud to make apply_grad non-empty, otherwise it hits a bug
            # (https://github.com/alpa-projects/alpa/issues/560).
            grads = jax.tree_map(lambda x: x + 1, grads)
            return grads, rns

        set_seed(10)
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            layer_option=ManualLayerOption(remat_layer=True),
            default_auto_sharding_option=AutoShardingOption(
                force_data_parallel=True))
        p_train_step = parallelize(train_step,
                                   method=method,
                                   static_argnums=(3,))

        grads_actual, rns_actual = p_train_step(params, x, rngkey, False,
                                                external_rns)
        grads_expected, rns_expected = train_step(params, x, rngkey, True,
                                                  external_rns)

        assert_allclose(external_rns, rns_actual)
        assert_allclose(external_rns, rns_expected)
        assert_allclose(grads_actual, grads_expected)
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
