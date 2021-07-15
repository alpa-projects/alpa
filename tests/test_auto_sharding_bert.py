"""
Test auto sharding with attention and transformer layers.

Usage:
python3 -m unittest -bv test_auto_sharding_attention.py
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import optim, linen as nn

from parax import parallelize, set_parallelize_options, testing, PhysicalDeviceMesh, global_config
from parax.model.bert_model import (BertConfig, FlaxBertAttention, FlaxBertLayerCollection,
                                    FlaxBertForMaskedLMModule)
from test_auto_sharding_mlp import (assert_close, assert_all_replicated,
                                    assert_column_partitioned,
                                    assert_only_has_allreduce,
                                    assert_replicated_column_partitioned,
                                    assert_replicated_row_partitioned,
                                    assert_row_partitioned)


def inspect_params(optimizer):
    """For debug usage."""
    print(jax.tree_util.tree_map(lambda x: x.shape, optimizer.target))


class AutoShardingAttentionTest(unittest.TestCase):
    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = jax.local_devices()[:4]

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        device_mesh = PhysicalDeviceMesh(self.devices)
        return device_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_attention(self, batch_size, seq_len, hidden_size, num_heads,
                      deterministic, device_mesh):
        set_parallelize_options(devices=device_mesh)

        @parallelize
        def train_step(optimizer, batch, deterministic, apply_fn):
            def loss_func(params):
                rngs = {"dropout": batch["rng"]}
                out = apply_fn(params,
                               batch["hidden_states"], batch["attention_mask"],
                               deterministic, rngs=rngs)[0]
                return jnp.mean((out - batch["label"]) ** 2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        # Init model and optimizer
        hidden_states = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        label = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)

        model = FlaxBertAttention(BertConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, hidden_states, attention_mask)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # JIT compile
        optimizer = train_step(optimizer,
                               {"hidden_states": hidden_states,
                                "attention_mask": attention_mask,
                                "label": label,
                                "rng": rngkey},
                               deterministic,
                               model.apply)

        # Get optimized HLO IR
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()

        return optimizer, hlo_ir, testing.last_compiled_auto_sharding_objective

    def run_bert_layers(self, num_layers, batch_size, seq_len, hidden_size,
                        num_heads, deterministic, device_mesh):
        set_parallelize_options(devices=device_mesh)

        @parallelize
        def train_step(optimizer, batch, deterministic, apply_fn):
            def loss_func(params):
                rngs = {"dropout": batch["rng"]}
                out = apply_fn(params,
                               batch["hidden_states"], batch["attention_mask"],
                               deterministic, rngs=rngs)[0]
                return jnp.mean((out - batch["label"]) ** 2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        # Init model and optimizer
        hidden_states = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        label = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)

        model = FlaxBertLayerCollection(BertConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, hidden_states, attention_mask)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # JIT compile
        optimizer = train_step(optimizer,
                               {"hidden_states": hidden_states,
                                "attention_mask": attention_mask,
                                "label": label,
                                "rng": rngkey},
                               deterministic,
                               model.apply)

        # Get optimized HLO IR
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()

        return optimizer, hlo_ir, testing.last_compiled_auto_sharding_objective

    def run_bert_mlm(self, num_layers, batch_size, seq_len, hidden_size,
                     num_heads, vocab_size, deterministic, device_mesh):
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

        model = FlaxBertForMaskedLMModule(BertConfig(
            num_hidden_layers=num_layers,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
        ))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, input_ids, attention_mask, token_type_ids, position_ids)
        optimizer = optim.GradientDescent(1e-2).create(params)

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

    def test_attention_data_parallel(self):
        batch_size = 32
        seq_len = 32
        hidden_size = 64
        num_heads = 8
        deterministic = False

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_attention(
                batch_size, seq_len, hidden_size, num_heads, deterministic, device_mesh)

            # Check communication cost
            params = jax.tree_util.tree_leaves(optimizer.target)
            expected = sum(device_mesh.all_reduce_cost(np.prod(x.shape) * 4, i)
                           for x in params)
            assert_close(objective, expected)
            assert_only_has_allreduce(hlo_ir)

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
        deterministic = False

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_attention(
                batch_size, seq_len, hidden_size, num_heads, deterministic, device_mesh)

            # Check communication cost
            expected = device_mesh.all_reduce_cost(
                batch_size * seq_len * hidden_size * 4, i)
            assert_close(objective, expected)
            assert_only_has_allreduce(hlo_ir)

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
        deterministic = False

        mesh_shape = [2, 2]
        device_mesh = self.get_device_mesh(mesh_shape, [2, 2], [1, 0.1])
        optimizer, hlo_ir, objective = self.run_attention(
            batch_size, seq_len, hidden_size, num_heads, deterministic, device_mesh)

        # Check communication cost
        params = jax.tree_util.tree_leaves(optimizer.target)
        expected = sum(device_mesh.all_reduce_cost(
            np.prod(x.shape) * 4 / mesh_shape[1], 0) for x in params) +\
            device_mesh.all_reduce_cost(
            batch_size * seq_len * hidden_size * 4 / mesh_shape[0], 1)
        assert_close(objective, expected)
        assert_only_has_allreduce(hlo_ir)

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
        deterministic = False

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_bert_layers(
                num_layers, batch_size, seq_len, hidden_size,
                num_heads, deterministic, device_mesh)

            # Check communication cost
            params = jax.tree_util.tree_leaves(optimizer.target)
            expected = sum(device_mesh.all_reduce_cost(np.prod(x.shape) * 4, i)
                           for x in params)
            assert_close(objective, expected)
            assert_only_has_allreduce(hlo_ir)

            # Check sharding specification
            for weight in params:
                assert_all_replicated(weight, np.prod(mesh_shape))

    def test_bert_layer_model_parallel(self):
        num_layers = 2
        batch_size = 8
        seq_len = 8
        hidden_size = 128
        num_heads = 8
        deterministic = False

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_bert_layers(
                num_layers, batch_size, seq_len, hidden_size,
                num_heads, deterministic, device_mesh)

            # Check communication cost
            expected = (num_layers * 4 - 1) * device_mesh.all_reduce_cost(
                batch_size * seq_len * hidden_size * 4, i)
            assert_close(objective, expected)
            assert_only_has_allreduce(hlo_ir)

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
        deterministic = False

        # Test on different logical mesh shapes
        mesh_shape = [2, 2]
        device_mesh = self.get_device_mesh(mesh_shape, [2, 2], [1, 0.1])
        optimizer, hlo_ir, objective = self.run_bert_layers(
            num_layers, batch_size, seq_len, hidden_size,
            num_heads, deterministic, device_mesh)

        # Check communication cost
        params = jax.tree_util.tree_leaves(optimizer.target)
        expected = sum(device_mesh.all_reduce_cost(
            np.prod(x.shape) * 4 / mesh_shape[1], 0) for x in params) +\
            device_mesh.all_reduce_cost(
            batch_size * seq_len * hidden_size * 4 / mesh_shape[0], 1) * (num_layers * 4 - 1)
        assert_close(objective, expected)
        assert_only_has_allreduce(hlo_ir)

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

    def test_embedding_2d_mesh(self):
        vocab_size = 1024
        hidden_size = 8
        batch_size = 8
        seq_len = 8
        mesh_shape = [2, 2]

        logical_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
        set_parallelize_options(devices=logical_mesh)

        # Model and training step definition
        class Model(nn.Module):
            """Tied input and output embedding."""

            def setup(self):
                self.embed = nn.Embed(vocab_size, hidden_size)
                self.dense = nn.Dense(hidden_size, use_bias=False)

            def __call__(self, x):
                x = self.embed(x)
                embed = self.embed.variables["params"]["embedding"]
                x = x @ embed.T
                return x

        @parallelize
        def func(optimizer, x, y):
            def loss_func(params):
                out = model.apply(params, x)
                y_ = jax.nn.one_hot(y, out.shape[-1])
                loss = -jnp.sum(y_ * jax.nn.log_softmax(out, axis=-1), axis=-1)
                return loss.sum()
            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        # Init model and optimizer
        x = jnp.ones((batch_size, seq_len), np.int32)
        y = jnp.ones((batch_size, seq_len), np.int32)

        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # JIT Compile
        optimize = func(optimizer, x, y)

        # Check communication cost
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()

        params = jax.tree_util.tree_leaves(optimizer.target)
        objective = testing.last_compiled_auto_sharding_objective
        expected = \
            logical_mesh.all_reduce_cost(
                vocab_size * hidden_size * 4 / mesh_shape[1], 0) * 2 +\
            logical_mesh.all_reduce_cost(
                batch_size * seq_len * hidden_size * 4 / mesh_shape[0], 1) * 2 +\
            logical_mesh.all_reduce_cost(
                batch_size * seq_len * 4 / mesh_shape[0], 1) * 2

        assert_close(objective, expected)
        assert_only_has_allreduce(hlo_ir)

    def test_bert_mlm_data_parallel(self):
        batch_size = 32
        seq_len = 32
        hidden_size = 16
        num_heads = 4
        num_layers = 2
        vocab_size = 128
        deterministic = True

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_bert_mlm(
                num_layers, batch_size, seq_len, hidden_size,
                num_heads, vocab_size, deterministic, device_mesh)

            # Check communication cost
            # Cost = all-reduce on all parameters + one extra all-redcue for the shared embedding.
            params = jax.tree_util.tree_leaves(optimizer.target)
            expected = sum(device_mesh.all_reduce_cost(np.prod(x.shape) * 4, i)
                           for x in params) + \
                       device_mesh.all_reduce_cost(vocab_size * hidden_size * 4, i)

            assert_close(objective, expected)
            assert_only_has_allreduce(hlo_ir)

            # Check sharding specification
            for weight in params:
                assert_all_replicated(weight, np.prod(mesh_shape))

    def test_bert_mlm_model_parallel(self):
        batch_size = 16
        seq_len = 16
        hidden_size = 128
        num_heads = 4
        num_layers = 2
        vocab_size = 512
        deterministic = False
        global_config.allow_all_gather = False  # Temporary hack

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_bert_mlm(
                num_layers, batch_size, seq_len, hidden_size,
                num_heads, vocab_size, deterministic, device_mesh)

            # Check communication cost
            # expected_cost = embed.forward (2) + embed.backward(2) +
            #                 LM_head.forward (1) + LM_head.backward (1) +
            #                 LM_head.weight.backward (1) +  log_softmax.forward (2) + 
            #                 transformer.backward (2 * num_layers) + transformer.backward (2 * num_layers)
            #
            # Note that the final cost is different from this estimated cost in ILP solver.
            # The SPMD partitioner will eliminate some unnecessary communication in favor of
            # redundant computation (e.g., it will elimiate the all-reduce in embed.backward).
            expected = \
              device_mesh.all_reduce_cost(batch_size * seq_len * hidden_size * 4, i) * 6 + \
              device_mesh.all_reduce_cost(hidden_size * hidden_size * 4, i) + \
              device_mesh.all_reduce_cost(batch_size * seq_len * 4, i) * 2 + \
              device_mesh.all_reduce_cost(batch_size * seq_len * hidden_size * 4, i) * num_layers * 4

            assert_close(objective, expected)
            assert_only_has_allreduce(hlo_ir)

            # Check sharding specification
            embed_weight = optimizer.target["params"]["bert"]["embeddings"]["word_embeddings"]["embedding"]
            lm_head = optimizer.target["params"]["cls"]["predictions"]["transform"]["dense"]["kernel"]
            assert_row_partitioned(embed_weight, mesh_shape[i], i)
            assert_all_replicated(lm_head, np.prod(mesh_shape))

            for k in range(num_layers):
                params = optimizer.target["params"]["bert"]["encoder"]["layer"][str(k)]
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

    def test_bert_mlm_2d_mesh(self):
        batch_size = 4
        seq_len = 4
        hidden_size = 512
        num_heads = 4
        num_layers = 2
        vocab_size = 4096
        deterministic = False
        global_config.allow_all_gather = False  # Temporary hack
        global_config.allow_recompute_heavy_op = True

        mesh_shape = [2, 2]
        device_mesh = self.get_device_mesh(mesh_shape, [2, 2], [1, 0.1])

        optimizer, hlo_ir, objective = self.run_bert_mlm(
            num_layers, batch_size, seq_len, hidden_size,
            num_heads, vocab_size, deterministic, device_mesh)

        # Check communication cost.
        # All communiation primitives are all-reduce, except one all-gather
        # used for token_type_embeddings (generated by the optimization in SPMD partitioner).
        assert hlo_ir.count("channel_id") == hlo_ir.count("all-reduce(") + 1

        # Check sharding specification
        embed_weight = optimizer.target["params"]["bert"]["embeddings"]["word_embeddings"]["embedding"]
        lm_head = optimizer.target["params"]["cls"]["predictions"]["transform"]["dense"]["kernel"]

        assert_replicated_row_partitioned(embed_weight, mesh_shape)
        assert_all_replicated(lm_head, np.prod(mesh_shape))

        for k in range(num_layers):
            params = optimizer.target["params"]["bert"]["encoder"]["layer"][str(k)]
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
    suite.addTest(AutoShardingAttentionTest("test_attention_data_parallel"))
    suite.addTest(AutoShardingAttentionTest("test_attention_model_parallel"))
    suite.addTest(AutoShardingAttentionTest("test_attention_2d_mesh"))

    suite.addTest(AutoShardingAttentionTest("test_bert_layer_data_parallel"))
    suite.addTest(AutoShardingAttentionTest("test_bert_layer_model_parallel"))
    suite.addTest(AutoShardingAttentionTest("test_bert_layer_2d_mesh"))

    suite.addTest(AutoShardingAttentionTest("test_embedding_2d_mesh"))

    suite.addTest(AutoShardingAttentionTest("test_bert_mlm_data_parallel"))
    suite.addTest(AutoShardingAttentionTest("test_bert_mlm_model_parallel"))
    suite.addTest(AutoShardingAttentionTest("test_bert_mlm_2d_mesh"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

