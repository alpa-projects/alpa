import unittest
import os

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
from jax.experimental.maps import FrozenDict
import ray

from parax import parallelize, set_parallelize_options, mark_pipeline, DeviceCluster
from parax.testing import assert_allclose
from parax.model.bert_model import BertConfig, FlaxBertLayer

MB = 1024 ** 2

class PipelineBERTTest(unittest.TestCase):
    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
        assert len(jax.local_devices()) >= 4
        ray.init(address='auto')
        device_cluster = DeviceCluster()
        mesh = device_cluster.get_virtual_mesh()
        self.devices = mesh

    def tearDown(self):
        ray.shutdown()

    def train_2_layer_bert(self, devices, strategy):
        set_parallelize_options(devices=devices, strategy=strategy)

        class Model(nn.Module):
            config: BertConfig
            dtype: jnp.dtype = jnp.float32

            def setup(self):
                self.layer0 = FlaxBertLayer(config=self.config, dtype=self.dtype)
                self.layer1 = FlaxBertLayer(config=self.config, dtype=self.dtype)

            def __call__(self, x, attention_mask):
                # FIXME (zhuohan): if don't require the gradient of x here, the
                #                  backward pass of the pipeline start will not
                #                  be generated.
                x, = mark_pipeline(x, name='1', mark_type='start')
                layer_outputs = self.layer0(x, attention_mask)
                x = layer_outputs[0]
                x, = mark_pipeline(x, name='1', mark_type='end')
                x, = mark_pipeline(x, name='2', mark_type='start')
                layer_outputs = self.layer1(x, attention_mask)
                x = layer_outputs[0]
                return x

        def train_step(optimizer, batch, apply_fn):
            def loss_func(params, x, y, attention_mask):
                out = apply_fn(params, x, attention_mask)
                loss = jnp.mean((out - y) ** 2)
                loss, = mark_pipeline(loss, name='2', mark_type='end')
                return loss

            grad_param, grad_x = jax.grad(loss_func, argnums = (0, 1))(optimizer.target, batch['x'], batch['y'], batch['attention_mask'])
            # FIXME (zhuohan): make the pipeline work with apply_gradient
            # new_optimizer = optimizer.apply_gradient(grad_param)
            return grad_param

        batch_size = 4
        seq_len = 128
        hidden_size = 128
        num_heads = 4

        x = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
        y = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32) * 23 # * np.arange(hidden_size)[None, None, :]
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        # Init model and optimizer
        model = Model(config=BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x, attention_mask)
        optimizer = optim.GradientDescent(1e-2).create(params)
        gradients = train_step(optimizer, {"x": x, "y": y, "attention_mask": attention_mask}, model.apply)
        pipelined_train_step = parallelize(donate_argnums=())(train_step)
        gradients_with_pipeline = pipelined_train_step(optimizer, {"x": x, "y": y, "attention_mask": attention_mask}, model.apply)
        assert_allclose(gradients, gradients_with_pipeline)

    def test_2_layer_bert_pipeline_parallel(self):
        self.train_2_layer_bert(self.devices, "pipeline_parallel")

    def test_2_layer_bert_3d_parallel(self):
        self.train_2_layer_bert(self.devices, "3d_parallel")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineBERTTest("test_2_layer_bert_pipeline_parallel"))
    suite.addTest(PipelineBERTTest("test_2_layer_bert_3d_parallel"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
