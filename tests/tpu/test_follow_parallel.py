"""Test FollowParallel on TPU."""
import unittest

from alpa import global_config

from tests.runtime.test_follow_parallel import FollowParallelTest
from tests.tpu.test_shard_parallel import has_tpu

class TpuFollowParallelTest(FollowParallelTest):

    def setUp(self):
        global_config.backend = "tpu"

    def tearDown(self):
        return

def suite():
    suite = unittest.TestSuite()
    if not has_tpu():
        return suite

    suite.addTest(TpuFollowParallelTest("test_shard_parallel"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())