import jax
import jax.numpy as jnp
import optax
import ray
import unittest
from flax import linen as nn

import parax
from parax import (parallelize, global_config, set_parallelize_options,
                   DeviceCluster)
from parax.model.bert_model import BertConfig, FlaxBertLayer, TrainState
from parax.testing import assert_allclose


def create_train_state(rngkey, model, params):
    params = model.init(rngkey, *params)
    tx = optax.adam(learning_rate=1e-2)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dynamic_scale=None)
    return state


class BertLayer_Model(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer0 = FlaxBertLayer(config=self.config, dtype=self.dtype)
        self.layer1 = FlaxBertLayer(config=self.config, dtype=self.dtype)

    def __call__(self, x, attention_mask):
        layer_outputs = self.layer0(x, attention_mask)
        x = layer_outputs[0]
        layer_outputs = self.layer1(x, attention_mask)
        x = layer_outputs[0]
        return x


class AccumulateGradTest(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto")
        jax.config.update('jax_platform_name', 'cpu')
        self.physical_mesh = DeviceCluster().get_physical_mesh()
        logical_mesh = self.physical_mesh.get_logical_mesh([2, 2], [2, 2], [1, 0.1])
        self.dtype = jnp.float32
        set_parallelize_options(logical_mesh)

    def tearDown(self):
        ray.shutdown()

    def test_2_layer_bert_shard_parallel(self):

        def train_step(state, batch, apply_fn):

            def loss_func(params, x, y, attention_mask):
                out = apply_fn(params, x, attention_mask)
                loss = jnp.mean((out - y)**2)
                return loss

            grad_param = parax.grad(loss_func)(state.params, batch['x'],
                                               batch['y'],
                                               batch['attention_mask'])
            new_state = state.apply_gradients(grads=grad_param)
            return new_state, grad_param

        batch_size = 16
        seq_len = 8
        hidden_size = 512
        num_heads = 8

        x = jnp.ones((batch_size, seq_len, hidden_size), dtype=self.dtype)
        y = jnp.ones(
            (batch_size, seq_len, hidden_size),
            dtype=self.dtype)  # * np.arange(hidden_size)[None, None, :]
        attention_mask = jnp.ones((batch_size, seq_len), dtype=self.dtype)

        # Init model and optimizer
        model = BertLayer_Model(
            config=BertConfig(hidden_size=hidden_size,
                              intermediate_size=hidden_size * 4,
                              num_attention_heads=num_heads),
            dtype=self.dtype)
        rngkey = jax.random.PRNGKey(0)
        batch = {"x": x, "y": y, "attention_mask": attention_mask}
        state = create_train_state(rngkey, model, [x, attention_mask])

        global_config.num_micro_batches = 1

        parallel_train_step = parallelize(train_step)
        expected_params, expected_grads = train_step(state, batch, model.apply)
        actual_params, actual_grads = parallel_train_step(state, batch, model.apply)

        print("group 1:")
        print("expected param example: ", jax.tree_util.tree_flatten(expected_params.params)[0][0][0:10])
        print("actual param example: ", jax.tree_util.tree_flatten(actual_params.params)[0][0]._value[0:10])
        print("expected grad example: ", jax.tree_util.tree_flatten(expected_grads)[0][0][0:10])
        print("actual grad example: ", jax.tree_util.tree_flatten(actual_grads)[0][0]._value[0:10])

        print("group 2:")
        print("expected param example: ", jax.tree_util.tree_flatten(expected_params.params)[0][-1][0:100])
        print("actual param example: ", jax.tree_util.tree_flatten(actual_params.params)[0][-1]._value[0:100])
        print("expected grad example: ", jax.tree_util.tree_flatten(expected_grads)[0][-1][0:100])
        print("actual grad example: ", jax.tree_util.tree_flatten(actual_grads)[0][-1]._value[0:100])

        assert_allclose(expected_params.params, actual_params.params)
        self.physical_mesh.shutdown()

def suite():
    suite = unittest.TestSuite()
    suite.addTest(AccumulateGradTest("test_2_layer_bert_shard_parallel"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
