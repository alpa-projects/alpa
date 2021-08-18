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
from parax.model.moe import FlaxMoeLayer, MoEConfig

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

        G = batch_size * seq_len // S

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
        model = FlaxMoeLayer(MoEConfig(
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


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingMoETest("test_moe_layer"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

