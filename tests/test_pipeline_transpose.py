import unittest

import jax
import numpy as np
import jax.numpy as jnp

import parax
from parax.testing import assert_allclose


class PipelineTransposeTest(unittest.TestCase):

    def test_transpose(self):

        def f(x):
            x, = parax.mark_pipeline(x, mark_type="start", name="1")
            x = jnp.transpose(x, axes=(1, 0))
            # x, = parax.mark_pipeline(x, mark_type="end", name="1")
            return x

        x = np.random.rand(2, 4)
        print(x)
        no_jit_result = f(x)
        jit_result = jax.jit(f)(x)
        print(no_jit_result)
        print(jit_result)
        assert_allclose(no_jit_result, jit_result)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineTransposeTest("test_transpose"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
