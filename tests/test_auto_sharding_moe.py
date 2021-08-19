"""Test auto sharding with MLP."""

import copy
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import optim
from jax.interpreters.pxla import Chunked, NoSharding, Replicated, ShardedAxis

from parax import parallelize, set_parallelize_options, testing, PhysicalDeviceMesh
from parax.global_env import global_config
from parax.util import map_to_shape, count_communication_primitives
from parax.model.moe import FlaxMoELayer, FlaxMoEForLMModule, MoEConfig

from test_auto_sharding_mlp import (assert_all_replicated,
                                    assert_close,
                                    assert_expert_partitioned)

class AutoShardingMoETest(unittest.TestCase):
    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = jax.local_devices()[:4]

        # Backup global config
        self.old_global_config = copy.deepcopy(global_config.__dict__)

    def tearDown(self):
        # Restore global config
        global_config.__dict__ = self.old_global_config

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        device_mesh = PhysicalDeviceMesh(self.devices)
        return device_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_moe_layer(self, batch_size, seq_len, hidden_size,
            num_heads, S, E, deterministic, device_mesh):
        set_parallelize_options(devices=device_mesh)

        @parallelize
        def train_step(optimizer, batch, deterministic, apply_fn):
            def loss_func(params):
                rngs = {"dropout": batch["rng"]}
                out = apply_fn(params,
                               batch["hidden_states"], batch["attention_mask"],
                               deterministic, rngs=rngs)[0]
                return jnp.mean((out - batch["labels"]) ** 2) * 0.12345

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        hidden_states = jnp.ones((batch_size, seq_len, hidden_size))
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        labels = jnp.ones((batch_size, seq_len, hidden_size))

        # Init model and optimizer
        model = FlaxMoELayer(MoEConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size*4,
            num_attention_heads=num_heads,
            expert_group_size=S,
            expert_number=E,
        ))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, hidden_states, attention_mask)
        optimizer = optim.Adam(1e-2).create(params)

        # JIT compile
        optimizer = train_step(optimizer,
                               {"hidden_states": hidden_states,
                                "attention_mask": attention_mask,
                                "labels": labels,
                                "rng": rngkey},
                               deterministic,
                               model.apply)

        # Get optimized HLO IR
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()
        return optimizer, hlo_ir, testing.last_compiled_auto_sharding_objective

    def run_moe_lm(self, batch_size, seq_len, num_layers, hidden_size, num_heads,
                   vocab_size, S, E, deterministic, device_mesh):
        set_parallelize_options(devices=device_mesh)

        @parallelize
        def train_step(optimizer, batch):
            def loss_func(params):
                rngs = {"dropout": batch["rng"]}
                logits = model.apply(params,
                                     batch["input_ids"],
                                     batch["attention_mask"],
                                     batch["token_type_ids"],
                                     batch["position_ids"],
                                     deterministic=deterministic,
                                     rngs=rngs)[0]
                label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
                labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
                loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
                return (label_mask * loss).sum() / label_mask.sum() * 0.1234

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        # Init model and optimizer
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        token_type_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        position_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        labels = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        model = FlaxMoEForLMModule(MoEConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size*4,
            num_attention_heads=num_heads,
            max_position_embeddings=seq_len,
            vocab_size=vocab_size,
            expert_group_size=S,
            expert_number=E,
        ))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, input_ids, attention_mask, token_type_ids, position_ids)
        optimizer = optim.Adam(1e-2).create(params)

        # JIT compile
        optimizer = train_step(optimizer,
                               {"input_ids": input_ids,
                                "attention_mask": attention_mask,
                                "token_type_ids": token_type_ids,
                                "position_ids": position_ids,
                                "labels": labels,
                                "rng": rngkey})

        # Get optimized HLO IR
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()
        return optimizer, hlo_ir, testing.last_compiled_auto_sharding_objective

    def test_moe_layer(self):
        batch_size = 64
        seq_len = 16
        hidden_size = 64
        num_heads = 16
        S = 32
        E = 16
        deterministic = True

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_moe_layer(
                batch_size, seq_len, hidden_size,
                num_heads, S, E, deterministic, device_mesh)

            # Check communication cost
            # all-to-all + data-parallel on attention_w_i, attention_w_o, layer_norm, moe_w_g
            expected = device_mesh.all_to_all_cost(batch_size * seq_len * hidden_size * 2 * 4, i) * 4 +\
                       device_mesh.all_reduce_cost(hidden_size * hidden_size * 3 * 4, i) +\
                       device_mesh.all_reduce_cost(hidden_size * 3 * 4, i) +\
                       device_mesh.all_reduce_cost(hidden_size * hidden_size * 4, i) +\
                       device_mesh.all_reduce_cost(hidden_size * 4, i) +\
                       device_mesh.all_reduce_cost(hidden_size * 4, i) * 4 +\
                       device_mesh.all_reduce_cost(hidden_size * E * 4, i)
            assert_close(expected, objective)

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all =\
                count_communication_primitives(hlo_ir)
            assert n_all_reduce == 1
            assert n_all_to_all == 4
            assert n_total == n_all_reduce + n_all_to_all

            # Check sharding specification
            num_devices = np.prod(device_mesh.id_mesh.shape)
            assert_all_replicated(optimizer.target["params"]["attention"]["output"]\
                    ["dense"]["kernel"], num_devices)
            assert_all_replicated(optimizer.target["params"]["attention"]["self"]\
                    ["qvk_combined"]["kernel"], num_devices)
            assert_all_replicated(optimizer.target["params"]["moe"]["wg"], num_devices)
            assert_expert_partitioned(optimizer.target["params"]["moe"]["wi"], num_devices, i)
            assert_expert_partitioned(optimizer.target["params"]["moe"]["wo"], num_devices, i)


    def test_moe_lm(self):
        num_layers = 2
        batch_size = 64
        seq_len = 16
        hidden_size = 64
        num_heads = 16
        vocab_size = 32
        S = 32
        E = 16
        deterministic = True

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_moe_lm(
                batch_size, seq_len, num_layers, hidden_size, num_heads,
                vocab_size, S, E, deterministic, device_mesh)

            # Check communication cost
            # all-to-all + data-parallel on attention_w_i, attention_w_o, layer_norm, moe_w_g
            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all =\
                count_communication_primitives(hlo_ir)
            assert n_all_reduce == 1
            assert n_all_to_all == 4
            assert n_total == n_all_reduce + n_all_to_all


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingMoETest("test_moe_layer"))
    suite.addTest(AutoShardingMoETest("test_moe_lm"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

