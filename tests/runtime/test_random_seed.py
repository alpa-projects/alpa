"""Test random seed."""
import unittest

import jax
from jax._src.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp
import numpy as np

from alpa import init, grad, parallelize, ShardParallel, set_seed, shutdown
from alpa.parallel_method import PipeshardParallel
from alpa.pipeline_parallel.layer_construction import ManualLayerOption
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
        shape = (40, 40)

        # TODO: Stateful rng is correct only if it is not replicated, so we
        # should forcely set the parallel strategy to data parallel.
        def get_train_step(parallel_method):

            def train_step(*args):

                def loss_func(params, x, key):
                    rns = jax.random.normal(key, x.shape)
                    y = x @ params["x1"]
                    y = jax.lax.select(rns > 0, y, jnp.zeros_like(y))
                    mark_pipeline_boundary()
                    y = y @ params["x2"]
                    return jnp.mean(y), rns

                grads, rns = grad(loss_func, has_aux=True)(*args)
                grad_val, tree = tree_flatten(grads)
                return tree_unflatten(tree, [val + 1 for val in grad_val]), rns

            return parallelize(train_step,
                               method=parallel_method,
                               donate_argnums=())

        def normal_step(params, x, rns):

            def forward(params, x, rns):
                y = x @ params["x1"]
                y = jax.lax.select(rns > 0, y, jnp.zeros_like(y))
                y = y @ params["x2"]
                return jnp.mean(y)

            grads = jax.grad(forward)(params, x, rns)
            grad_val, tree = tree_flatten(grads)
            return tree_unflatten(tree, [val + 1 for val in grad_val])

        # the num micrbatch can only be 1 because current runtime does not
        # support
        remat_pipeline_with_rng = get_train_step(
            PipeshardParallel(num_micro_batches=1,
                              layer_option=ManualLayerOption(True)))

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, shape)
        params = {
            "x1": jax.random.normal(key, shape),
            "x2": jax.random.normal(key, shape)
        }
        pipeline_out, rns = remat_pipeline_with_rng(params, x, key)
        expected_out = normal_step(params, x, rns._value)
        assert_allclose(pipeline_out, expected_out)
        executable = remat_pipeline_with_rng.get_executable(params, x, key)
        rng_str = "rng-get-and-update-state"
        for hlo in executable.get_hlo_text()[1:]:
            assert rng_str not in hlo
        assert rng_str in executable.get_hlo_text()[0]

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
