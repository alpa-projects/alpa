"""Some basic tests to test installation."""
import os
import pickle
import unittest

from alpa import (init, shutdown, parallelize, ShardParallel, PipeshardParallel,
                  AutoLayerOption, plan_to_method, AutoShardingOption,
                  AutoStageOption)
from alpa.device_mesh import get_global_cluster
from alpa.testing import assert_allclose, get_mlp_train_state_and_step


class ParallelPlanTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def tearDown(self):
        shutdown()

    def test_shard_parallel(self):
        state, batch, train_step = get_mlp_train_state_and_step(batch_size=128,
                                                                hidden_size=128,
                                                                num_layers=4)

        method = ShardParallel(
            num_micro_batches=2,
            auto_sharding_option=AutoShardingOption(force_data_parallel=True))
        p_train_step = parallelize(train_step, method=method)

        executable1 = p_train_step.get_executable(state, batch)
        plan = executable1.get_parallel_plan()

        with open("tmp_plan.pkl", "wb") as fout:
            pickle.dump(plan, fout)
        with open("tmp_plan.pkl", "rb") as fin:
            plan = pickle.load(fin)

        p_train_step = parallelize(train_step, method=plan_to_method(plan))
        executable2 = p_train_step.get_executable(state, batch)

        assert (executable1.auto_sharding_objective ==
                executable2.auto_sharding_objective)

    def test_pipeshard_parallel(self):
        state, batch, train_step = get_mlp_train_state_and_step(batch_size=128,
                                                                hidden_size=128,
                                                                num_layers=4)

        method = PipeshardParallel(num_micro_batches=2,
                                   layer_option=AutoLayerOption(layer_num=2),
                                   stage_option="uniform")
        p_train_step = parallelize(train_step, method=method)

        executable1 = p_train_step.get_executable(state, batch)
        plan = executable1.get_parallel_plan()

        with open("tmp_plan.pkl", "wb") as fout:
            pickle.dump(plan, fout)
        with open("tmp_plan.pkl", "rb") as fin:
            plan = pickle.load(fin)

        p_train_step = parallelize(train_step, method=plan_to_method(plan))
        executable2 = p_train_step.get_executable(state, batch)

        assert (executable1.get_input_placement_specs() ==
                executable2.get_input_placement_specs())


def suite():
    s = unittest.TestSuite()
    s.addTest(ParallelPlanTest("test_shard_parallel"))
    s.addTest(ParallelPlanTest("test_pipeshard_parallel"))
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
