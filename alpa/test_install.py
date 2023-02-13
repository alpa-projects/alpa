"""Some basic tests to test installation."""
import os
import unittest

from alpa import (init, parallelize, ShardParallel, PipeshardParallel,
                  AutoLayerOption, prefetch)
from alpa.device_mesh import get_global_cluster
from alpa.testing import assert_allclose, get_mlp_train_state_and_step


class InstallationTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    def test_1_shard_parallel(self):
        state, batch, train_step = get_mlp_train_state_and_step(batch_size=128,
                                                                hidden_size=128,
                                                                num_layers=4)

        # Serial execution
        expected_output = train_step(state, batch)

        # Parallel execution
        p_train_step = parallelize(train_step,
                                   method=ShardParallel(num_micro_batches=2))
        actual_output = p_train_step(state, batch)

        # Check results
        assert_allclose(expected_output, actual_output)

    def test_2_pipeline_parallel(self):
        init(cluster="ray")

        state, batch, train_step = get_mlp_train_state_and_step(batch_size=128,
                                                                hidden_size=128,
                                                                num_layers=6)

        # Serial execution
        expected_output = train_step(state, batch)

        # Parallel execution
        layer_num = min(get_global_cluster().num_devices, 2)
        p_train_step = parallelize(
            train_step,
            method=PipeshardParallel(
                num_micro_batches=2,
                layer_option=AutoLayerOption(layer_num=layer_num)))
        actual_output = p_train_step(state, batch)

        # Check results
        prefetch(actual_output)
        assert_allclose(expected_output, actual_output)


def suite():
    s = unittest.TestSuite()
    s.addTest(InstallationTest("test_1_shard_parallel"))
    s.addTest(InstallationTest("test_2_pipeline_parallel"))
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

