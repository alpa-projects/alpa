"""
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

from parax import parallelize, global_config, testing, DeviceMesh

from bert_model import BertConfig, FlaxBertAttention, FlaxBertLayerCollection
from test_auto_sharding_basic import assert_close, all_reduce_cost, map_to_shape

MB = 1024 ** 2

class AutoShardingAttentionTest(unittest.TestCase):
    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = tuple(jax.local_devices()[:4])
        global_config.shard_parallel_strategy = 'auto_sharding'

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        devices = np.array(self.devices).reshape(shape)
        return DeviceMesh(devices, mesh_alpha, mesh_beta)

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
            assert weight0.sharding_spec.mesh_mapping == (Replicated(np.prod(mesh_shape)),)
            assert weight1.sharding_spec.mesh_mapping == (Replicated(np.prod(mesh_shape)),)

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
            # Column partitioned
            assert weight0.sharding_spec == pxla.ShardingSpec(
                sharding=(Chunked([1]), Chunked([np.prod(mesh_shape)])),
                mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
            )
            # Row partitioned
            assert weight1.sharding_spec == pxla.ShardingSpec(
                sharding=(Chunked([np.prod(mesh_shape)]), Chunked([1])),
                mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
            )

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
        #weight0 = optimizer.target["params"]["self"]["qvk_combined"]["kernel"]
        #weight1 = optimizer.target["params"]["output"]["dense"]["kernel"]
        #print(weight0.sharding_spec)
        #print(weight1.sharding_spec)
        ## Column partitioned
        #assert weight0.sharding_spec == pxla.ShardingSpec(
        #    sharding=(Chunked([1]), Chunked([np.prod(mesh_shape)])),
        #    mesh_mapping=(ShardedAxis(0), ShardedAxis(1), ),
        #)
        ## Row partitioned
        #assert weight1.sharding_spec == pxla.ShardingSpec(
        #    sharding=(Chunked([np.prod(mesh_shape)]), Chunked([1])),
        #    mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
        #)

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

            for x in params:
                assert x.sharding_spec.mesh_mapping == (Replicated(np.prod(mesh_shape)),)

    def test_bert_layer_model_parallel(self):
        num_layers = 3
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
            for i in range(num_layers):
                params = optimizer.target["params"][str(i)]
                weights = [
                    params["attention"]["self"]["qvk_combined"]["kernel"],
                    params["attention"]["output"]["dense"]["kernel"],
                    params["intermediate"]["dense"]["kernel"],
                    params["output"]["dense"]["kernel"],
                ]

                for j in range(len(weights)):
                    if j % 2 == 0:
                        # Column partitioned
                        assert weights[j].sharding_spec == pxla.ShardingSpec(
                            sharding=(Chunked([1]), Chunked([np.prod(mesh_shape)])),
                            mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
                        )
                    else:
                        # Row partitioned
                        assert weights[j].sharding_spec == pxla.ShardingSpec(
                            sharding=(Chunked([np.prod(mesh_shape)]), Chunked([1])),
                            mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
                        )

    def test_bert_layer_2d_mesh(self):
        num_layers = 1
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

        ## Check sharding specification
        #for i in range(num_layers):
        #    params = optimizer.target["params"][str(i)]
        #    weights = [
        #        params["attention"]["self"]["qvk_combined"]["kernel"],
        #        params["attention"]["output"]["dense"]["kernel"],
        #        params["intermediate"]["dense"]["kernel"],
        #        params["output"]["dense"]["kernel"],
        #    ]

        #    for j in range(len(weights)):
        #        if j % 2 == 0:
        #            # Column partitioned
        #            assert weights[j].sharding_spec == pxla.ShardingSpec(
        #                sharding=(Chunked([1]), Chunked([np.prod(mesh_shape)])),
        #                mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
        #            )
        #        else:
        #            # Row partitioned
        #            assert weights[j].sharding_spec == pxla.ShardingSpec(
        #                sharding=(Chunked([np.prod(mesh_shape)]), Chunked([1])),
        #                mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
        #            )


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

