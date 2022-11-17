"""Test CreateStateParallel on TPU."""
import unittest

from alpa import global_config

import tests.runtime.test_create_state as test_create_state
from tests.tpu.test_shard_parallel import has_tpu


class TpuCreateStateTest(test_create_state.CreateStateTest):

    def setUp(self):
        global_config.backend = "tpu"

    def tearDown(self):
        return

    @unittest.skip("unsupported yet.")
    def test_shard_parallel_grad_acc(self):
        super().test_shard_parallel_grad_acc()

    @unittest.skip("unsupported yet.")
    def test_pipeshard_parallel(self):
        super().test_pipeshard_parallel()


def suite():
    suite = unittest.TestSuite()
    if not has_tpu():
        return suite

    suite.addTest(TpuCreateStateTest("test_shard_parallel"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())