"""Test the debug information dummping."""
import os
import unittest

from alpa import (init, parallelize, ShardParallel, PipeshardParallel,
                  AutoLayerOption, global_config)
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.device_mesh import get_global_cluster
from alpa.testing import assert_allclose, get_mlp_train_state_and_step


class DebugInfoTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    def test_1_shard_parallel(self):
        state, batch, train_step = get_mlp_train_state_and_step(batch_size=128,
                                                                hidden_size=128,
                                                                num_layers=4)

        # Print auto-sharding intermidiate results
        os.environ["ALPA_DEBUG_PRINT_AS_STRATEGY"] = "1"

        p_train_step = parallelize(train_step,
                                   method=ShardParallel(num_micro_batches=2))
        actual_output = p_train_step(state, batch)
        executable = p_train_step.get_last_executable()
        executable.sync()

        # Dump final HLO and other debug info
        executable.dump_debug_info("alpa_debug_info")


    def test_2_pipeline_parallel(self):
        init(cluster="ray")
        state, batch, train_step = get_mlp_train_state_and_step(batch_size=128,
                                                                hidden_size=128,
                                                                num_layers=6)

        # Print auto-sharding intermidiate results
        global_config.pipeline_distributed_compile = False
        os.environ["ALPA_DEBUG_PRINT_AS_STRATEGY"] = "1"

        layer_num = min(get_global_cluster().num_devices, 2)
        p_train_step = parallelize(
            train_step,
            method=PipeshardParallel(
                num_micro_batches=2,
                layer_option=AutoLayerOption(layer_num=layer_num)))
        actual_output = p_train_step(state, batch)
        executable = p_train_step.get_last_executable()
        executable.sync()

        # Dump final HLO and other debug info
        executable.dump_debug_info("alpa_debug_info")

        # Print auto-stage dynamic programming results if use auto stage partition
        print(get_last_dp_result())


def suite():
    s = unittest.TestSuite()
    s.addTest(DebugInfoTest("test_1_shard_parallel"))
    s.addTest(DebugInfoTest("test_2_pipeline_parallel"))
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
