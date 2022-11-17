"""Test activity tracing."""
import unittest

from alpa import (init, shutdown, parallelize, global_config, PipeshardParallel)
from alpa.global_env import global_config
from alpa.device_mesh import get_global_cluster
from alpa.test_install import get_mlp_train_state_and_step


class TracingTest(unittest.TestCase):

    def setUp(self):
        global_config.collect_trace = True
        init()

    def tearDown(self):
        shutdown()

    def test_trace_pipeshard_execuable(self):
        state, batch, train_step = get_mlp_train_state_and_step(
            batch_size=128, hidden_size=128, add_manual_pipeline_marker=True)

        layer_num = min(get_global_cluster().num_devices, 2)
        train_step = parallelize(
            train_step,
            method=PipeshardParallel(num_micro_batches=2,
                                     layer_option="manual"))

        for i in range(2):
            state, _ = train_step(state, batch)

        executable = train_step.get_last_executable()
        stage_exec_info = executable.get_stage_execution_info()

        assert len(stage_exec_info) == 6  # 6 stages
        assert len(stage_exec_info[0]) == 4  # 4 invocations


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TracingTest("test_trace_pipeshard_execuable"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
