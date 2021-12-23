import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import os
import ray
from flax import linen as nn
from flax import optim

import parax
from benchmark.parax.benchmark_transformer_layer_3d import create_train_state
from parax import (parallelize, set_parallelize_options, mark_pipeline,
                   DeviceCluster, manual_layer_slicing)
from parax.global_env import global_config
from parax.model.bert_model import FlaxBertLayerCollection, BertConfig
from parax.testing import assert_allclose

MB = 1024**2

# Note(Hao):
# In this test case, we turn on/off aggressively sync and compared the pipelined training results.
# We expect the results should be same.


class PipelineMLP(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        mark_pipeline(name='1', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=False)(x)
        x = nn.relu(x)
        mark_pipeline(name='1', mark_type='end')
        mark_pipeline(name='2', mark_type='start')
        x = nn.Dense(features=self.output_dim, use_bias=False)(x)
        return x


def train_mlp_step(optimizer, batch, apply_fn, grad_fn):

    def loss_func(params, x, y):
        out = apply_fn(params, x)
        loss = jnp.mean((out - y)**2)
        mark_pipeline(name='2', mark_type='end')
        return loss

    loss_func = manual_layer_slicing(loss_func)
    grad_param = grad_fn(loss_func)(optimizer.target, batch['x'], batch['y'])
    return grad_param


Transformer = FlaxBertLayerCollection


def get_train_transformer_step(grad_func, pipeline_mp_size=2):

    @partial(parallelize, donate_argnums=())
    def train_step(state, batch, rng_key):

        def loss_func(params):
            rngs = {"dropout": rng_key}
            if pipeline_mp_size > 1:
                mark_pipeline(name="0", mark_type="start")
            out = state.apply_fn(params,
                                 batch["hidden_states"],
                                 batch["attention_mask"],
                                 deterministic=True,
                                 rngs=rngs)[0]
            loss = jnp.mean((out - batch["label"])**2)
            if pipeline_mp_size > 1:
                mark_pipeline(name=str(pipeline_mp_size - 1), mark_type="end")
            return loss

        if pipeline_mp_size > 1:
            loss_func = manual_layer_slicing(loss_func)
        # grad, grad_x = jax.grad(loss_func, argnums=(0, 1))(optimizer.target, batch["hidden_states"])
        grad = grad_func(loss_func)(state.params)
        # Do not apply grads now
        # new_state = state.apply_gradients(grads=grads)
        return grad

    return train_step


class PipelineCorrectnessTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
        # Note(Hao): I tested on my own cluster.
        assert len(jax.local_devices()) >= 4
        ray.init(address='auto', namespace="parax-unittest")
        device_cluster = DeviceCluster()
        mesh = device_cluster.get_virtual_physical_mesh()
        self.devices = mesh

    def tearDown(self):
        ray.shutdown()

    def train_2_layer_mlp(self, num_microbatches=1):
        set_parallelize_options(devices=self.devices,
                                strategy="3d_parallel",
                                num_micro_batches=num_microbatches)

        batch_size = 128
        hidden_dim = 2048
        input_dim = output_dim = hidden_dim
        if num_microbatches == 1:
            grad_func = jax.grad
        else:
            grad_func = parax.grad

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

        batch = {"x": x, "y": y}
        # Init model and optimizer
        model = PipelineMLP(hidden_dim=hidden_dim, output_dim=output_dim)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)

        pipelined_train_step = parallelize(donate_argnums=(
        ))(lambda optimizer, batch, apply_fn: train_mlp_step(
            optimizer, batch, apply_fn, grad_func))

        executable = pipelined_train_step.get_executable(
            optimizer, batch, model.apply)

        # Run sync + no compile
        global_config.pipeline_aggressively_sync = True
        global_config.precompile_resharding_tasks = False
        gradients_sync_no_compile = pipelined_train_step(
            optimizer, batch, model.apply)

        # Run async + no compile
        global_config.pipeline_aggressively_sync = False
        global_config.precompile_resharding_tasks = False
        gradients_async_no_compile = pipelined_train_step(
            optimizer, batch, model.apply)
        assert_allclose(gradients_sync_no_compile, gradients_async_no_compile)

        # Run sync + compile
        global_config.pipeline_aggressively_sync = True
        global_config.precompile_resharding_tasks = True
        gradients_sync_compile = pipelined_train_step(optimizer, batch,
                                                      model.apply)
        assert_allclose(gradients_sync_no_compile, gradients_sync_compile)

        # Run sync + compile
        global_config.pipeline_aggressively_sync = False
        global_config.precompile_resharding_tasks = True
        gradients_async_compile = pipelined_train_step(optimizer, batch,
                                                       model.apply)
        assert_allclose(gradients_sync_no_compile, gradients_async_compile)

        executable.shutdown()

    def train_2_layer_transformer(self, num_microbatches=1):
        set_parallelize_options(devices=self.devices,
                                strategy="3d_parallel",
                                num_micro_batches=num_microbatches)

        batch_size = 8
        seq_len = 128
        hidden_size = 384
        num_layers = 2
        num_heads = 1536 // 96
        pipeline_mp_size = 2
        global_config.force_data_parallel = False
        global_config.prefer_reduce_scatter = False

        if num_microbatches == 1:
            grad_func = jax.grad
        else:
            grad_func = parax.grad

        model = Transformer(
            BertConfig(num_hidden_layers=num_layers,
                       hidden_size=hidden_size,
                       intermediate_size=hidden_size * 4,
                       num_attention_heads=num_heads,
                       pipeline_mp_size=pipeline_mp_size))

        batch = {
            "hidden_states": jnp.ones((batch_size, seq_len, hidden_size),
                                      dtype=np.float32),
            "attention_mask": jnp.ones((batch_size, seq_len), dtype=np.int32),
            "label": jnp.ones((batch_size, seq_len, hidden_size),
                              dtype=np.float32),
        }
        rngkey = jax.random.PRNGKey(0)
        state = create_train_state(rngkey, model, batch)

        pipelined_train_step = get_train_transformer_step(
            grad_func, pipeline_mp_size=pipeline_mp_size)
        executable = pipelined_train_step.get_executable(state, batch, rngkey)

        # sync + no compile
        global_config.pipeline_aggressively_sync = True
        global_config.precompile_resharding_tasks = False
        gradients_sync_no_compile = pipelined_train_step(state, batch, rngkey)

        # async + no compile
        global_config.pipeline_aggressively_sync = False
        global_config.precompile_resharding_tasks = False
        gradients_async_no_compile = pipelined_train_step(state, batch, rngkey)
        assert_allclose(gradients_sync_no_compile, gradients_async_no_compile)

        # sync + compile
        global_config.pipeline_aggressively_sync = True
        global_config.precompile_resharding_tasks = True
        gradients_sync_compile = pipelined_train_step(state, batch, rngkey)
        assert_allclose(gradients_sync_no_compile, gradients_sync_compile)

        # sync + compile
        global_config.pipeline_aggressively_sync = False
        global_config.precompile_resharding_tasks = True
        gradients_async_compile = pipelined_train_step(state, batch, rngkey)
        assert_allclose(gradients_sync_no_compile, gradients_async_compile)

        executable.shutdown()

    def test_pipeline_correctness_mlp_nmb_1(self):
        self.train_2_layer_mlp(num_microbatches=1)

    def test_pipeline_correctness_mlp_nmb_2(self):
        self.train_2_layer_mlp(num_microbatches=2)

    def test_pipeline_correctness_mlp_nmb_4(self):
        self.train_2_layer_mlp(num_microbatches=4)

    def test_pipeline_correctness_mlp_nmb_8(self):
        self.train_2_layer_mlp(num_microbatches=8)

    def test_pipeline_correctness_transformer_layers_nmb_1(self):
        self.train_2_layer_transformer(num_microbatches=1)

    def test_pipeline_correctness_transformer_layers_nmb_2(self):
        self.train_2_layer_transformer(num_microbatches=2)

    def test_pipeline_correctness_transformer_layers_nmb_4(self):
        self.train_2_layer_transformer(num_microbatches=4)

    def test_pipeline_correctness_transformer_layers_nmb_8(self):
        self.train_2_layer_transformer(num_microbatches=8)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        PipelineCorrectnessTest("test_pipeline_correctness_mlp_nmb_1"))
    suite.addTest(
        PipelineCorrectnessTest("test_pipeline_correctness_mlp_nmb_2"))
    suite.addTest(
        PipelineCorrectnessTest("test_pipeline_correctness_mlp_nmb_4"))
    suite.addTest(
        PipelineCorrectnessTest("test_pipeline_correctness_mlp_nmb_8"))
    suite.addTest(
        PipelineCorrectnessTest(
            "test_pipeline_correctness_transformer_layers_nmb_1"))
    suite.addTest(
        PipelineCorrectnessTest(
            "test_pipeline_correctness_transformer_layers_nmb_2"))
    suite.addTest(
        PipelineCorrectnessTest(
            "test_pipeline_correctness_transformer_layers_nmb_4"))
    suite.addTest(
        PipelineCorrectnessTest(
            "test_pipeline_correctness_transformer_layers_nmb_8"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
