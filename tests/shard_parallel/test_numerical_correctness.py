"""Test the numerical correctness of shard parallel."""
import unittest

from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
import ray

import alpa
from alpa import parallelize, LocalPhysicalDeviceMesh
from alpa.model.bert_model import BertConfig, FlaxBertLayer, TrainState
from alpa.testing import (assert_allclose, create_train_state,
                          get_bert_layer_train_state_and_step)


class AutoShardingCorrectnessTest(unittest.TestCase):

    def test_2_layer_bert_shard_parallel(self):
        physical_mesh = LocalPhysicalDeviceMesh(jax.local_devices()[:4])
        logical_mesh = physical_mesh.get_logical_mesh([2, 2])

        # Init model
        state, batch, train_step = get_bert_layer_train_state_and_step(
            batch_size=16,
            seq_len=8,
            num_layers=2,
            hidden_size=256,
            num_heads=8,
            clip_by_global_norm=False,
            use_dynamic_scale=False,
            add_manual_pipeline_marker=False)

        # Train one step
        p_train_step = parallelize(train_step)
        expected_state, expected_grads = train_step(state, batch)
        actual_state, actual_grads = p_train_step(state, batch)

        #print(expected_state)
        #print(actual_state)

        # print("group 1:")
        # print("expected param example: ", jax.tree_util.tree_flatten(expected_params.params)[0][0][0:10])
        # print("actual param example: ", jax.tree_util.tree_flatten(actual_params.params)[0][0]._value[0:10])
        # print("expected grad example: ", jax.tree_util.tree_flatten(expected_grads)[0][0][0:10])
        # print("actual grad example: ", jax.tree_util.tree_flatten(actual_grads)[0][0]._value[0:10])

        # print("group 2:")
        # print("expected param example: ", jax.tree_util.tree_flatten(expected_params.params)[0][-1][0:100])
        # print("actual param example: ", jax.tree_util.tree_flatten(actual_params.params)[0][-1]._value[0:100])
        # print("expected grad example: ", jax.tree_util.tree_flatten(expected_grads)[0][-1][0:100])
        # print("actual grad example: ", jax.tree_util.tree_flatten(actual_grads)[0][-1]._value[0:100])

        assert_allclose(expected_state, actual_state, rtol=5e-4, atol=5e-4)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        AutoShardingCorrectnessTest("test_2_layer_bert_shard_parallel"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
