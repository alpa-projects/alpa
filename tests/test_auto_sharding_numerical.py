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
from alpa.testing import assert_allclose


def create_train_state(rngkey, model, batch_args):
    params = model.init(rngkey, *batch_args)
    tx = optax.adam(learning_rate=1e-2)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              dynamic_scale=None)
    return state


class BertLayerModel(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer0 = FlaxBertLayer(config=self.config, dtype=self.dtype)
        self.layer1 = FlaxBertLayer(config=self.config, dtype=self.dtype)

    def __call__(self, x, attention_mask):
        out = x
        out = self.layer0(out, attention_mask)[0]
        out = self.layer1(out, attention_mask)[0]
        return out


class AutoShardingCorrectnessTest(unittest.TestCase):

    def test_2_layer_bert_shard_parallel(self):
        physical_mesh = LocalPhysicalDeviceMesh(jax.local_devices()[:4])
        logical_mesh = physical_mesh.get_logical_mesh([2, 2], [2, 2], [1, 0.1])

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"],
                                     batch["attention_mask"])
                loss = jnp.mean((out - batch["y"])**2)
                return loss

            grads = alpa.grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state, grads

        batch_size = 16
        seq_len = 8
        hidden_size = 256
        num_heads = 8
        dtype = jnp.float32

        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=dtype)
        y = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=dtype)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=dtype)

        # Init model and optimizer
        model = BertLayerModel(config=BertConfig(hidden_size=hidden_size,
                                                 intermediate_size=hidden_size *
                                                 4,
                                                 num_attention_heads=num_heads),
                               dtype=dtype)
        batch = {"x": x, "y": y, "attention_mask": attention_mask}
        state = create_train_state(rngkey, model, [x, attention_mask])

        # Train one step
        parallel_train_step = parallelize(train_step)
        expected_params, expected_grads = train_step(state, batch)
        actual_params, actual_grads = parallel_train_step(state, batch)

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

        assert_allclose(expected_params.params,
                        actual_params.params,
                        rtol=5e-4,
                        atol=5e-4)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        AutoShardingCorrectnessTest("test_2_layer_bert_shard_parallel"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
