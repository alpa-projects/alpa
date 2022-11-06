"""Test CreateStateParallel on TPU."""
import unittest

from alpa import global_config

from tests.runtime.test_create_state import CreateStateTest
from tests.tpu.test_shard_parallel import has_tpu


class TpuCreateStateTest(CreateStateTest):

    def setUp(self):
        global_config.backend = "tpu"

    def tearDown(self):
        return


def suite():
    suite = unittest.TestSuite()
    if not has_tpu():
        return suite

    suite.addTest(TpuCreateStateTest("test_shard_parallel"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())