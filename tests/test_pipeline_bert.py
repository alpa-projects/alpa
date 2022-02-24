import unittest
import os

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import optax
import ray

from alpa import (parallelize, set_parallelize_options, mark_pipeline,
                  DeviceCluster, manual_layer_construction)
from alpa.testing import BertLayerModel, assert_allclose
from alpa.model.model_util import TrainState
from alpa.model.bert_model import BertConfig
from alpa.util import get_ray_namespace_str


class PipelineBERTTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        assert len(jax.local_devices()) >= 4

        ray.init(address="auto",
                 namespace=get_ray_namespace_str(prefix="alpa-unittest"))
        device_cluster = DeviceCluster()
        self.devices = device_cluster.get_virtual_physical_mesh()

    def tearDown(self):
        ray.shutdown()

    def train_2_layer_bert(self, devices, strategy):
        set_parallelize_options(devices=devices, strategy=strategy)

        def train_step(state, batch):

            def loss_func(params, x, y, attention_mask):
                out = state.apply_fn(params, x, attention_mask)
                loss = jnp.mean((out - y)**2)
                mark_pipeline(name="2", mark_type="end")
                return loss

            loss_func = manual_layer_construction(loss_func)
            grads = jax.grad(loss_func)(state.params, batch["x"], batch["y"],
                                        batch["attention_mask"])
            return grads

        batch_size = 16
        seq_len = 8
        hidden_size = 512
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
                                                 num_attention_heads=num_heads,
                                                 num_hidden_layers=2))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x, attention_mask)
        tx = optax.sgd(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=tx,
                                  dynamic_scale=None)

        # Train step
        batch = {"x": x, "y": y, "attention_mask": attention_mask}
        gradients = train_step(state, batch)
        pipelined_train_step = parallelize(donate_argnums=())(train_step)
        gradients_with_pipeline = pipelined_train_step(state, batch)

        # Check results
        assert_allclose(gradients, gradients_with_pipeline)
        pipelined_train_step.get_executable(state, batch).shutdown()

    def test_2_layer_bert_local_pipeline_parallel(self):
        self.train_2_layer_bert(self.devices, "local_pipeline_parallel")

    @unittest.skip("This test is failing because it's not using apply grad")
    def test_2_layer_bert_pipeshard_parallel(self):
        self.train_2_layer_bert(self.devices, "pipeshard_parallel")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineBERTTest("test_2_layer_bert_local_pipeline_parallel"))
    suite.addTest(PipelineBERTTest("test_2_layer_bert_pipeshard_parallel"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
