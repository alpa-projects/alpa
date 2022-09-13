"""Test auto sharding with MoE."""

import unittest

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.interpreters.pxla import Chunked, NoSharding, Replicated, ShardedAxis
import numpy as np
import optax

from alpa import parallelize, ShardParallel, LocalPhysicalDeviceMesh, AutoShardingOption
from alpa.util import map_to_shape, count_communication_primitives
from alpa.model.moe import FlaxMoELayer, FlaxMoEForLMModule, MoEConfig, TrainState

from tests.shard_parallel.test_mlp import (assert_all_replicated, assert_close,
                                           assert_expert_partitioned,
                                           assert_sharding_zero_stage_3)


class AutoShardingMoETest(unittest.TestCase):

    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.physical_mesh = LocalPhysicalDeviceMesh(jax.local_devices()[:4])
        self.as_option = AutoShardingOption()

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        return self.physical_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_moe_layer(self, batch_size, seq_len, hidden_size, num_heads, S, E,
                      deterministic, device_mesh):

        @parallelize(method=ShardParallel(devices=device_mesh,
                                          auto_sharding_option=self.as_option))
        def train_step(state, batch, deterministic):

            def loss_func(params):
                rngs = {"dropout": batch["rng"]}
                out = state.apply_fn(params,
                                     batch["hidden_states"],
                                     batch["attention_mask"],
                                     deterministic,
                                     rngs=rngs)[0]
                return jnp.mean((out - batch["labels"])**2)

            grads = jax.grad(loss_func)(state.params)
            return state.apply_gradients(grads=grads)

        dtype = jnp.float32
        hidden_states = jnp.ones((batch_size, seq_len, hidden_size),
                                 dtype=dtype)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        labels = jnp.ones((batch_size, seq_len, hidden_size), dtype=dtype)

        # Init model and optimizer
        model = FlaxMoELayer(MoEConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            expert_group_size=S,
            expert_number=E,
        ),
                             dtype=dtype)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, hidden_states, attention_mask)
        tx = optax.adam(1e-2)
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=tx,
                                  dynamic_scale=None)

        # JIT compile
        state = train_step(
            state, {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "labels": labels,
                "rng": rngkey
            }, deterministic)

        # Get optimized HLO IR
        executable = train_step.get_last_executable()
        return (state, executable.get_hlo_text(),
                executable.auto_sharding_objective)

    def run_moe_lm(self, batch_size, seq_len, num_layers, hidden_size,
                   num_heads, vocab_size, S, E, deterministic, device_mesh):

        @parallelize(method=ShardParallel(devices=device_mesh,
                                          auto_sharding_option=self.as_option))
        def train_step(state, batch, deterministic, rng_key):

            def loss_func(params):
                rngs = {"dropout": rng_key}
                logits = state.apply_fn(params,
                                        batch["input_ids"],
                                        batch["attention_mask"],
                                        batch["token_type_ids"],
                                        batch["position_ids"],
                                        deterministic=deterministic,
                                        rngs=rngs)[0]
                label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
                labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
                loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1),
                                axis=-1)
                loss = (label_mask * loss).sum() / label_mask.sum()
                return loss

            grads = jax.grad(loss_func)(state.params)
            return state.apply_gradients(grads=grads)

        # Init model and optimizer
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        token_type_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        position_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        labels = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        dtype = jnp.float32

        model = FlaxMoEForLMModule(MoEConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            max_position_embeddings=seq_len,
            vocab_size=vocab_size,
            expert_group_size=S,
            expert_number=E,
        ),
                                   dtype=dtype)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, input_ids, attention_mask, token_type_ids,
                            position_ids)

        def weight_decay_mask(pytree):
            # do not use weight decay on layer norm and bias.
            return jax.tree_map(lambda x: x.ndim > 1, pytree)

        tx = optax.adafactor(
            learning_rate=1e-2,
            weight_decay_mask=weight_decay_mask,
            min_dim_size_to_factor=4,
        )
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=tx,
                                  dynamic_scale=None,
                                  use_master_copy=(dtype == jnp.float16))

        # JIT compile
        state = train_step(
            state, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "labels": labels,
            }, deterministic, rngkey)

        # Get optimized HLO IR
        executable = train_step.get_last_executable()
        return (state, executable.get_hlo_text(),
                executable.auto_sharding_objective)

    def test_moe_layer(self):
        batch_size = 64
        seq_len = 16
        hidden_size = 64
        num_heads = 16
        S = 32
        E = 16
        deterministic = True

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_moe_layer(
                batch_size, seq_len, hidden_size, num_heads, S, E,
                deterministic, device_mesh)

            # Check communication cost
            # all-to-all + data-parallel on attention_w_i, attention_w_o, layer_norm, moe_w_g
            expected = (
                device_mesh.all_to_all_cost(
                    batch_size * seq_len * hidden_size * 2 * 4, i) * 4 +
                device_mesh.all_reduce_cost(hidden_size * hidden_size * 3 * 4,
                                            i) +
                device_mesh.all_reduce_cost(hidden_size * 3 * 4, i) +
                device_mesh.all_reduce_cost(hidden_size * hidden_size * 4, i) +
                device_mesh.all_reduce_cost(hidden_size * 4, i) +
                device_mesh.all_reduce_cost(hidden_size * 4, i) * 4 +
                device_mesh.all_reduce_cost(hidden_size * E * 4, i))
            assert_close(expected, objective)

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all = (
                count_communication_primitives(hlo_ir))
            assert n_all_reduce == 1
            assert n_all_to_all == 4
            assert n_total == n_all_reduce + n_all_to_all

            # Check sharding specification
            num_devices = np.prod(device_mesh.shape)
            assert_all_replicated(
                state.params["params"]["attention"]["output"]["dense"]
                ["kernel"], num_devices)
            assert_all_replicated(
                state.params["params"]["attention"]["self"]["qvk_combined"]
                ["kernel"], num_devices)
            assert_all_replicated(state.params["params"]["moe"]["wg"],
                                  num_devices)
            assert_expert_partitioned(state.params["params"]["moe"]["wi"],
                                      num_devices, i)
            assert_expert_partitioned(state.params["params"]["moe"]["wo"],
                                      num_devices, i)

    def test_moe_layer_2d(self):
        batch_size = 64
        seq_len = 16
        hidden_size = 64
        num_heads = 16
        S = 32
        E = 16
        deterministic = True
        self.as_option.allow_mixed_mesh_shape = True
        self.as_option.allow_all_gather = False

        # Test on different logical mesh shapes
        device_mesh = self.get_device_mesh([2, 2], [1, 1], [1, 1])
        state, hlo_ir, objective = self.run_moe_layer(batch_size, seq_len,
                                                      hidden_size, num_heads, S,
                                                      E, deterministic,
                                                      device_mesh)

        # Check communication cost
        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all = (
            count_communication_primitives(hlo_ir))
        assert n_all_reduce == 2  # one data-parallel for experts weights,
        # one data-parallel for normal weights
        assert n_all_to_all > 0
        assert n_total == n_all_reduce + n_all_to_all

    def test_moe_layer_2d_reduce_scatter(self):
        batch_size = 64
        seq_len = 16
        hidden_size = 64
        num_heads = 16
        S = 32
        E = 16
        deterministic = True
        self.as_option.allow_mixed_mesh_shape = True
        self.as_option.allow_all_gather = False
        self.as_option.prefer_reduce_scatter = True

        # Test on different logical mesh shapes
        device_mesh = self.get_device_mesh([2, 2], [1, 1], [1, 1])
        state, hlo_ir, objective = self.run_moe_layer(batch_size, seq_len,
                                                      hidden_size, num_heads, S,
                                                      E, deterministic,
                                                      device_mesh)

        # Check communication cost
        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all = (
            count_communication_primitives(hlo_ir))
        assert n_all_to_all > 0
        assert n_reduce_scatter > 0
        assert n_all_reduce == 0
        assert n_total == n_all_reduce + n_reduce_scatter + n_all_to_all + n_all_gather

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
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_moe_lm(batch_size, seq_len,
                                                       num_layers, hidden_size,
                                                       num_heads, vocab_size, S,
                                                       E, deterministic,
                                                       device_mesh)

            # Check communication cost
            # all-to-all + data-parallel on attention_w_i, attention_w_o, layer_norm, moe_w_g
            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all = (
                count_communication_primitives(hlo_ir,
                                               ignore_scalar_all_reduce=True))

            # Special case: zero stage 3
            if self.as_option.force_zero_stage_3:
                assert n_total == n_all_reduce + n_all_gather + n_reduce_scatter + n_all_to_all
                assert_sharding_zero_stage_3(state, 4)
                continue

            # Normal cases
            if self.as_option.prefer_reduce_scatter:
                if self.as_option.force_data_parallel:
                    assert 0 < n_reduce_scatter <= 2
                    assert n_total == n_all_reduce + n_all_gather + n_reduce_scatter
                else:
                    assert n_reduce_scatter == 1
                    assert n_all_to_all == 4
                    assert n_total == n_all_reduce + n_all_gather + n_reduce_scatter + n_all_to_all
            else:
                if self.as_option.force_data_parallel:
                    assert n_all_reduce == 1
                    assert n_total == n_all_reduce
                else:
                    assert n_all_reduce <= 4
                    assert n_all_to_all == 4
                    assert n_total == n_all_reduce + n_all_to_all

    def test_moe_lm_2d(self):
        num_layers = 2
        batch_size = 64
        seq_len = 16
        hidden_size = 64
        num_heads = 16
        vocab_size = 32
        S = 32
        E = 16
        deterministic = True
        self.as_option.allow_mixed_mesh_shape = True

        mesh_shape = (2, 2)
        device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
        state, hlo_ir, objective = self.run_moe_lm(batch_size, seq_len,
                                                   num_layers, hidden_size,
                                                   num_heads, vocab_size, S, E,
                                                   deterministic, device_mesh)

        # Check communication cost
        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all = (
            count_communication_primitives(hlo_ir))
        if self.as_option.prefer_reduce_scatter:
            assert n_reduce_scatter > 0
            assert n_total == n_all_reduce + n_all_gather + n_reduce_scatter + n_all_to_all
        else:
            assert n_all_to_all == 4
            assert n_total == n_all_reduce + n_all_to_all

    def test_moe_lm_data_parallel(self):
        self.as_option.force_data_parallel = True
        self.test_moe_lm()

    def test_moe_lm_reduce_scatter(self):
        self.as_option.prefer_reduce_scatter = True
        self.test_moe_lm()

    def test_moe_lm_2d_reduce_scatter(self):
        self.as_option.prefer_reduce_scatter = True
        self.test_moe_lm_2d()

    def test_moe_lm_data_parallel_reduce_scatter(self):
        self.as_option.prefer_reduce_scatter = True
        self.as_option.force_data_parallel = True
        self.test_moe_lm()

    def test_moe_lm_data_parallel_reduce_scatter_zero_3(self):
        self.as_option.force_zero_stage_3 = True
        self.as_option.force_zero_stage_3_all_gather_threshold = 1
        self.test_moe_lm()


def suite():
    suite = unittest.TestSuite()

    def add(name):
        suite.addTest(AutoShardingMoETest(name))

    add("test_moe_layer")
    add("test_moe_layer_2d")
    add("test_moe_layer_2d_reduce_scatter")

    add("test_moe_lm")
    add("test_moe_lm_2d")
    add("test_moe_lm_data_parallel")

    add("test_moe_lm_reduce_scatter")
    add("test_moe_lm_2d_reduce_scatter")
    add("test_moe_lm_data_parallel_reduce_scatter")
    add("test_moe_lm_data_parallel_reduce_scatter_zero_3")

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
