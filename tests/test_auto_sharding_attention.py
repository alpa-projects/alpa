"""
Test auto sharding with attention and transformer layers.

Usage:
python3 -m unittest -bv test_auto_sharding_attention.py
"""
from functools import partial
import os
import unittest

import numpy as np

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, NoSharding, Replicated, ShardedAxis
from flax import linen as nn
from flax import optim

from parax import parallelize, SingleHostDeviceMesh, global_config, testing
from parax.model.bert_model import BertConfig, FlaxBertAttention, FlaxBertLayerCollection

from test_auto_sharding_mlp import (assert_close, all_reduce_cost, map_to_shape,
    assert_all_replicated, assert_column_partitioned, assert_row_partitioned,
    assert_replicated_column_partitioned, assert_replicated_row_partitioned)

MB = 1024 ** 2

class AutoShardingAttentionTest(unittest.TestCase):
    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = tuple(jax.local_devices()[:4])
        global_config.shard_parallel_strategy = 'auto_sharding'

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        device_mesh = SingleHostDeviceMesh(self.devices)
        return device_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_attention(self, batch_size, seq_len, hidden_size, num_heads,
                      dropout_rate, device_mesh):
        @parallelize(devices=device_mesh)
        def train_step(optimizer, batch, apply_fn):
            def loss_func(params):
                rngs = {"dropout": batch['rng']}
                out = apply_fn(params,
                               batch['hidden_states'], batch['attention_mask'],
                               rngs=rngs)[0]
                return jnp.mean((out - batch['label']) ** 2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        hidden_states = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        label = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)

        # Init model and optimizer
        model = FlaxBertAttention(BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            attention_probs_dropout_prob=0))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, hidden_states, attention_mask)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # JIT compile
        optimizer = train_step(optimizer,
                               {"hidden_states": hidden_states,
                                "attention_mask": attention_mask,
                                "label": label,
                                "rng": rngkey},
                               model.apply)

        # Get optimized HLO IR
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()

        return optimizer, hlo_ir, testing.last_compiled_auto_sharding_objective

    def run_bert_layers(self, num_layers, batch_size, seq_len, hidden_size,
                        num_heads, dropout_rate, device_mesh):
        @parallelize(devices=device_mesh)
        def train_step(optimizer, batch, apply_fn):
            def loss_func(params):
                rngs = {"dropout": batch['rng']}
                out = apply_fn(params,
                               batch['hidden_states'], batch['attention_mask'],
                               rngs=rngs)[0]
                return jnp.mean((out - batch['label']) ** 2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        hidden_states = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        label = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)

        # Init model and optimizer
        model = FlaxBertLayerCollection(BertConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            attention_probs_dropout_prob=0))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, hidden_states, attention_mask)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # JIT compile
        optimizer = train_step(optimizer,
                               {"hidden_states": hidden_states,
                                "attention_mask": attention_mask,
                                "label": label,
                                "rng": rngkey},
                               model.apply)

        # Get optimized HLO IR
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()

        return optimizer, hlo_ir, testing.last_compiled_auto_sharding_objective

    def test_attention_data_parallel(self):
        batch_size = 32
        seq_len = 32
        hidden_size = 64
        num_heads = 8
        dropout_rate = 0.0

        # Test on different device meshes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_attention(
                batch_size, seq_len, hidden_size, num_heads, dropout_rate, device_mesh)

            # Check communication cost
            params = jax.tree_util.tree_leaves(optimizer.target)
            expected = sum(device_mesh.all_reduce_cost(np.prod(x.shape) * 4, i)
                           for x in params)
            assert_close(objective, expected)

            # Check sharding specification
            weight0 = optimizer.target["params"]["self"]["qvk_combined"]["kernel"]
            weight1 = optimizer.target["params"]["output"]["dense"]["kernel"]
            assert_all_replicated(weight0, np.prod(mesh_shape))
            assert_all_replicated(weight1, np.prod(mesh_shape))

    def test_attention_model_parallel(self):
        batch_size = 8
        seq_len = 8
        hidden_size = 256
        num_heads = 8
        dropout_rate = 0.0

        # Test on different device meshes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_attention(
                batch_size, seq_len, hidden_size, num_heads, dropout_rate, device_mesh)

            # Check communication cost
            expected = device_mesh.all_reduce_cost(
                batch_size * seq_len * hidden_size * 4, i)
            assert_close(objective, expected)

            assert hlo_ir.count("channel_id") == 1
            assert hlo_ir.count("all-reduce(") == 1

            # Check sharding specification
            weight0 = optimizer.target["params"]["self"]["qvk_combined"]["kernel"]
            weight1 = optimizer.target["params"]["output"]["dense"]["kernel"]
            assert_column_partitioned(weight0, mesh_shape[i], i)
            assert_row_partitioned(weight1, mesh_shape[i], i)

    def test_attention_2d_mesh(self):
        batch_size = 8
        seq_len = 8
        hidden_size = 128
        num_heads = 8
        dropout_rate = 0.0

        mesh_shape = [2, 2]
        device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 0.01])
        optimizer, hlo_ir, objective = self.run_attention(
            batch_size, seq_len, hidden_size, num_heads, dropout_rate, device_mesh)

        # Check communication cost
        params = jax.tree_util.tree_leaves(optimizer.target)
        expected = sum(device_mesh.all_reduce_cost(
            np.prod(x.shape) * 4 / mesh_shape[1], 0) for x in params) +\
            device_mesh.all_reduce_cost(
            batch_size * seq_len * hidden_size * 4 / mesh_shape[0], 1)
        assert_close(objective, expected)

        # Check sharding specification
        weight0 = optimizer.target["params"]["self"]["qvk_combined"]["kernel"]
        weight1 = optimizer.target["params"]["output"]["dense"]["kernel"]
        assert_replicated_column_partitioned(weight0, mesh_shape)
        assert_replicated_row_partitioned(weight1, mesh_shape)

    def test_bert_layer_data_parallel(self):
        num_layers = 2
        batch_size = 64
        seq_len = 64
        hidden_size = 32
        num_heads = 8
        dropout_rate = 0.0

        # Test on different device meshes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_bert_layers(
                num_layers, batch_size, seq_len, hidden_size,
                num_heads, dropout_rate, device_mesh)

            # Check communication cost
            params = jax.tree_util.tree_leaves(optimizer.target)
            expected = sum(device_mesh.all_reduce_cost(np.prod(x.shape) * 4, i)
                           for x in params)
            assert_close(objective, expected)

            for weight in params:
                assert_all_replicated(weight, np.prod(mesh_shape))

    def test_bert_layer_model_parallel(self):
        num_layers = 2
        batch_size = 8
        seq_len = 8
        hidden_size = 128
        num_heads = 8
        dropout_rate = 0.0

        # Test on different device meshes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_bert_layers(
                num_layers, batch_size, seq_len, hidden_size,
                num_heads, dropout_rate, device_mesh)

            # Check communication cost
            expected = (num_layers * 4 - 1) * device_mesh.all_reduce_cost(
                batch_size * seq_len * hidden_size * 4, i)
            assert_close(objective, expected)

            assert hlo_ir.count("channel_id") == num_layers * 4 - 1
            assert hlo_ir.count("all-reduce(") == num_layers * 4 - 1

            # Check sharding specification
            for k in range(num_layers):
                params = optimizer.target["params"][str(k)]
                weights = [
                    params["attention"]["self"]["qvk_combined"]["kernel"],
                    params["attention"]["output"]["dense"]["kernel"],
                    params["intermediate"]["dense"]["kernel"],
                    params["output"]["dense"]["kernel"],
                ]

                for j in range(len(weights)):
                    if j % 2 == 0:
                        assert_column_partitioned(weights[j], mesh_shape[i], i)
                    else:
                        assert_row_partitioned(weights[j], mesh_shape[i], i)

    def test_bert_layer_2d_mesh(self):
        num_layers = 2
        batch_size = 8
        seq_len = 8
        hidden_size = 128
        num_heads = 8
        dropout_rate = 0.0

        # Test on different device meshes
        mesh_shape = [2, 2]
        device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 0.01])
        optimizer, hlo_ir, objective = self.run_bert_layers(
            num_layers, batch_size, seq_len, hidden_size,
            num_heads, dropout_rate, device_mesh)

        # Check communication cost
        params = jax.tree_util.tree_leaves(optimizer.target)
        expected = sum(device_mesh.all_reduce_cost(
            np.prod(x.shape) * 4 / mesh_shape[1], 0) for x in params) +\
            device_mesh.all_reduce_cost(
            batch_size * seq_len * hidden_size * 4 / mesh_shape[0], 1)
        assert_close(objective, expected)

        # Check sharding specification
        for k in range(num_layers):
            params = optimizer.target["params"][str(k)]
            weights = [
                params["attention"]["self"]["qvk_combined"]["kernel"],
                params["attention"]["output"]["dense"]["kernel"],
                params["intermediate"]["dense"]["kernel"],
                params["output"]["dense"]["kernel"],
            ]

            for j in range(len(weights)):
                if j % 2 == 0:
                    assert_replicated_column_partitioned(weights[j], mesh_shape)
                else:
                    assert_replicated_row_partitioned(weights[j], mesh_shape)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingAttentionTest('test_attention_data_parallel'))
    suite.addTest(AutoShardingAttentionTest('test_attention_model_parallel'))
    suite.addTest(AutoShardingAttentionTest('test_attention_2d_mesh'))

    suite.addTest(AutoShardingAttentionTest('test_bert_layer_data_parallel'))
    suite.addTest(AutoShardingAttentionTest('test_bert_layer_model_parallel'))
    suite.addTest(AutoShardingAttentionTest('test_bert_layer_2d_mesh'))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

