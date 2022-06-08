"""Test auto sharding with simple computational graphs."""
import unittest
import time

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, ShardedAxis, NoSharding, Replicated
from flax import linen as nn
from flax import optim

from alpa import parallelize, ShardParallel
from alpa.util import count_communication_primitives
from alpa.testing import assert_allclose

from test_auto_sharding_mlp import assert_close

MB = 1024**2


class RandomSeedTest(unittest.TestCase):

    def test_donate_buffer(self):
        @parallelize(method=ShardParallel())
        def func():
            rngkey = jax.random.PRNGKey(0)
            x = jax.random.normal(rngkey, (8, 2))
            y = jax.random.normal(rngkey, (8, 2))
            return jnp.vstack((x, y))

        devices = jax.local_devices()

        [d.synchronize_all_activity() for d in devices]
        [d.set_seed(0) for d in devices]
        s = func()
        print(s)

        [d.synchronize_all_activity() for d in devices]
        [d.set_seed(0) for d in devices]
        s = func()
        print(s)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(RandomSeedTest("test_donate_buffer"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

