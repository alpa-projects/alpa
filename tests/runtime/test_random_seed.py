"""Test random seed."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from alpa import parallelize, ShardParallel, set_seed
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


def suite():
    suite = unittest.TestSuite()
    suite.addTest(RandomSeedTest("test_random_generation"))
    suite.addTest(RandomSeedTest("test_set_seed"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
