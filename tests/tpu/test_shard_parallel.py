"""Test auto sharding with MLP and MoE on TPU."""
import unittest

import jax

from alpa import global_config

from tests.shard_parallel.test_mlp import AutoShardingMLPTest
from tests.shard_parallel.test_moe import AutoShardingMoETest

with_device = {}


def has_device(name):
    global with_device
    if name in with_device:
        return with_device[name]
    try:
        jax.devices(name)
        with_device[name] = True
    except RuntimeError:
        with_device[name] = False
    return with_device[name]


def has_tpu():
    return has_device("tpu")


def has_gpu():
    return has_device("gpu")


class AutoShardingTpuMlpTest(AutoShardingMLPTest):

    def setUp(self):
        global_config.backend = "tpu"
        super().setUp()


class AutoShardingTpuMoeTest(AutoShardingMoETest):

    def setUp(self):
        global_config.backend = "tpu"
        super().setUp()


def suite():
    suite = unittest.TestSuite()
    if not has_tpu():
        return suite

    def add_mlp(name):
        suite.addTest(AutoShardingTpuMlpTest(name))

    def add_moe(name):
        suite.addTest(AutoShardingTpuMoeTest(name))

    add_mlp("test_n_layer_mlp_data_parallel")
    add_mlp("test_n_layer_mlp_model_parallel")
    add_mlp("test_n_layer_mlp_2d_mesh")
    add_mlp("test_n_layer_mlp_force_data_parallel")
    add_mlp("test_n_layer_mlp_force_batch_dim_mapping")

    add_moe("test_moe_layer")
    add_moe("test_moe_layer_2d")
    add_moe("test_moe_lm")
    add_moe("test_moe_lm_2d")
    add_moe("test_moe_lm_data_parallel")

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())