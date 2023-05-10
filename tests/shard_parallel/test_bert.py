"""Test auto sharding on transformer layers and bert models."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

from alpa import parallelize, ShardParallel, LocalPhysicalDeviceMesh, AutoShardingOption
from alpa.model.bert_model import (BertConfig, FlaxBertLayerCollection,
                                   FlaxBertForMaskedLMModule)
from alpa.util import count_communication_primitives
from tests.shard_parallel.test_mlp import (
    assert_all_replicated, assert_close, assert_column_partitioned,
    assert_data_parallel_cost, assert_fully_sharded, assert_less_equal,
    assert_sharded, assert_replicated_column_partitioned,
    assert_replicated_row_partitioned, assert_row_partitioned, is_fully_sharded,
    assert_sharding_zero_stage_3)


class AutoShardingAttentionTest(unittest.TestCase):

    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.physical_mesh = LocalPhysicalDeviceMesh(jax.local_devices()[:4])
        self.as_option = AutoShardingOption()

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        return self.physical_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_bert_layers(self, batch_size, seq_len, num_layers, hidden_size,
                        num_heads, deterministic, use_remat, device_mesh):

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
                return jnp.mean((out - batch["label"])**2)

            grads = jax.grad(loss_func)(state.params)
            return state.apply_gradients(grads=grads)

        # Init model and optimizer
        hidden_states = jnp.ones((batch_size, seq_len, hidden_size),
                                 dtype=jnp.float32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        label = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)

        model = FlaxBertLayerCollection(
            BertConfig(num_hidden_layers=num_layers,
                       hidden_size=hidden_size,
                       intermediate_size=hidden_size * 4,
                       num_attention_heads=num_heads,
                       gradient_checkpointing=use_remat))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, hidden_states, attention_mask)
        tx = optax.adam(1e-2)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        # JIT compile
        state = train_step(
            state, {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "label": label,
                "rng": rngkey
            }, deterministic)

        # Get optimized HLO IR
        executable = train_step.get_last_executable()
        return (state, executable.get_hlo_text(),
                executable.auto_sharding_objective)

    def run_bert_mlm(self, batch_size, seq_len, num_layers, hidden_size,
                     num_heads, vocab_size, deterministic, device_mesh):

        @parallelize(method=ShardParallel(devices=device_mesh,
                                          auto_sharding_option=self.as_option))
        def train_step(state, batch):

            def loss_func(params):
                rngs = {"dropout": batch["rng"]}
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
                return (label_mask * loss).sum() / label_mask.sum() * 0.1234

            grads = jax.grad(loss_func)(state.params)
            return state.apply_gradients(grads=grads)

        # Init model and optimizer
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        token_type_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        position_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        labels = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        model = FlaxBertForMaskedLMModule(
            BertConfig(
                num_hidden_layers=num_layers,
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 4,
                num_attention_heads=num_heads,
                vocab_size=vocab_size,
                max_position_embeddings=seq_len,
            ))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, input_ids, attention_mask, token_type_ids,
                            position_ids)
        tx = optax.adam(1e-2)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        # JIT compile
        state = train_step(
            state, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "labels": labels,
                "rng": rngkey
            })

        # Get optimized HLO IR
        executable = train_step.get_last_executable()
        return (state, executable.get_hlo_text(),
                executable.auto_sharding_objective)

    def test_bert_layer_data_parallel(self):
        batch_size = 64
        seq_len = 64
        num_layers = 2
        hidden_size = 32
        num_heads = 8
        deterministic = False
        use_remat = False

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_bert_layers(
                batch_size, seq_len, num_layers, hidden_size, num_heads,
                deterministic, use_remat, device_mesh)

            assert_data_parallel_cost(state, hlo_ir, objective, device_mesh,
                                      self.as_option, i)

    def test_bert_layer_model_parallel(self):
        batch_size = 8
        seq_len = 8
        num_layers = 2
        hidden_size = 128
        num_heads = 8
        deterministic = False
        use_remat = False

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_bert_layers(
                batch_size, seq_len, num_layers, hidden_size, num_heads,
                deterministic, use_remat, device_mesh)

            # Check communication cost
            expected = (num_layers * 4 - 1) * device_mesh.all_reduce_cost(
                batch_size * seq_len * hidden_size * 4, i)
            assert_close(objective, expected)

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
                count_communication_primitives(hlo_ir))
            if self.as_option.prefer_reduce_scatter:
                assert n_total == num_layers * 4 - 1
                assert n_all_reduce == num_layers * 4 - 1
                assert n_total == n_all_reduce
            else:
                assert n_total == num_layers * 4 - 1
                assert n_all_reduce == num_layers * 4 - 1
                assert n_total == n_all_reduce

            # Check sharding specification
            for k in range(num_layers):
                params = state.params["params"][str(k)]
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
        batch_size = 8
        seq_len = 8
        num_layers = 2
        hidden_size = 128
        num_heads = 8
        deterministic = False
        use_remat = False

        # Test on different logical mesh shapes
        mesh_shape = [2, 2]
        device_mesh = self.get_device_mesh(mesh_shape, [2, 2], [1, 0.1])
        state, hlo_ir, objective = self.run_bert_layers(batch_size, seq_len,
                                                        num_layers, hidden_size,
                                                        num_heads,
                                                        deterministic,
                                                        use_remat, device_mesh)

        # Check communication cost
        params = jax.tree_util.tree_leaves(state.params)
        expected = (sum(
            device_mesh.all_reduce_cost(
                np.prod(x.shape) * 4 / mesh_shape[1], 0)
            for x in params) + device_mesh.all_reduce_cost(
                batch_size * seq_len * hidden_size * 4 / mesh_shape[0], 1) *
                    (num_layers * 4 - 1))
        assert_close(objective, expected)

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_ir,
                                           ignore_scalar_all_reduce=True))
        if self.as_option.prefer_reduce_scatter:
            assert n_all_reduce == num_layers * 4 - 1
            assert n_reduce_scatter == 2
            assert n_all_gather <= 2
            assert n_total == n_all_reduce + n_reduce_scatter + n_all_gather
        else:
            assert n_all_reduce == num_layers * 4
            assert n_total == n_all_reduce

        # Check sharding specification
        if self.as_option.prefer_reduce_scatter:
            for weight in jax.tree_util.tree_leaves(state.opt_state):
                if len(weight.shape) > 1:
                    assert_fully_sharded(weight)
        else:
            for k in range(num_layers):
                params = state.params["params"][str(k)]
                weights = [
                    params["attention"]["self"]["qvk_combined"]["kernel"],
                    params["attention"]["output"]["dense"]["kernel"],
                    params["intermediate"]["dense"]["kernel"],
                    params["output"]["dense"]["kernel"],
                ]

                for j in range(len(weights)):
                    if j % 2 == 0:
                        assert_replicated_column_partitioned(
                            weights[j], mesh_shape)
                    else:
                        assert_replicated_row_partitioned(
                            weights[j], mesh_shape)

    def test_bert_layer_force_batch_dim_mapping(self):
        batch_size = 64
        seq_len = 64
        num_layers = 2
        hidden_size = 32
        num_heads = 8
        deterministic = False
        use_remat = False
        self.as_option.force_batch_dim_to_mesh_dim = 0

        # data parallel
        device_mesh = self.get_device_mesh([4, 1], [1, 1], [1, 1])
        state, hlo_ir, objective = self.run_bert_layers(batch_size, seq_len,
                                                        num_layers, hidden_size,
                                                        num_heads,
                                                        deterministic,
                                                        use_remat, device_mesh)
        assert_data_parallel_cost(state, hlo_ir, objective, device_mesh,
                                  self.as_option, 0)

        # model parallel (case 1)
        device_mesh = self.get_device_mesh([1, 4], [1, 1], [1, 1])
        state, hlo_ir, objective = self.run_bert_layers(batch_size, seq_len,
                                                        num_layers, hidden_size,
                                                        num_heads,
                                                        deterministic,
                                                        use_remat, device_mesh)
        expected = (num_layers * 4 - 1) * device_mesh.all_reduce_cost(
            batch_size * seq_len * hidden_size * 4, 1)
        assert_close(objective, expected)

        # model parallel (case 2)
        batch_size = 1
        device_mesh = self.get_device_mesh([1, 4], [1, 1], [1, 1])
        state, hlo_ir, objective = self.run_bert_layers(batch_size, seq_len,
                                                        num_layers, hidden_size,
                                                        num_heads,
                                                        deterministic,
                                                        use_remat, device_mesh)
        expected = (num_layers * 4 - 1) * device_mesh.all_reduce_cost(
            batch_size * seq_len * hidden_size * 4, 1)
        assert_close(objective, expected)

    def test_embedding_2d_mesh(self):
        vocab_size = 1024
        hidden_size = 8
        batch_size = 8
        seq_len = 8
        mesh_shape = [2, 2]

        # Model and training step definition
        class Model(nn.Module):
            """Tied input and output embedding."""

            def setup(self):
                self.embed = nn.Embed(vocab_size, hidden_size)

            def __call__(self, x):
                x = self.embed(x)
                embed = self.embed.variables["params"]["embedding"]
                x = x @ embed.T
                return x

        logical_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])

        @parallelize(method=ShardParallel(devices=logical_mesh))
        def func(state, x, y):

            def loss_func(params):
                out = state.apply_fn(params, x)
                y_ = jax.nn.one_hot(y, out.shape[-1])
                loss = -jnp.sum(y_ * jax.nn.log_softmax(out, axis=-1), axis=-1)
                return loss.sum()

            grads = jax.grad(loss_func)(state.params)
            return state.apply_gradients(grads=grads)

        # Init model and optimizer
        x = jnp.ones((batch_size, seq_len), np.int32)
        y = jnp.ones((batch_size, seq_len), np.int32)

        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        tx = optax.adam(1e-2)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        # JIT Compile
        state = func(state, x, y)

        # Check communication cost
        executable = func.get_last_executable()
        hlo_ir = executable.get_hlo_text()
        objective = executable.auto_sharding_objective

        expected = (
            logical_mesh.all_reduce_cost(
                vocab_size * hidden_size * 4 / mesh_shape[1], 0) +
            logical_mesh.all_reduce_cost(
                batch_size * seq_len * hidden_size * 4 / mesh_shape[0], 1) * 2 +
            logical_mesh.all_reduce_cost(
                batch_size * seq_len * 4 / mesh_shape[0], 1) * 2)

        assert_close(objective, expected)
        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_ir))
        assert n_total == n_all_reduce

    def test_bert_mlm_data_parallel(self):
        batch_size = 32
        seq_len = 32
        num_layers = 2
        hidden_size = 16
        num_heads = 4
        vocab_size = 128
        deterministic = False

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_bert_mlm(
                batch_size, seq_len, num_layers, hidden_size, num_heads,
                vocab_size, deterministic, device_mesh)

            if self.as_option.force_zero_stage_3:
                # only the weight and opt_state of token_embed is not sharded
                assert_sharding_zero_stage_3(state, 3)
                continue

            assert_data_parallel_cost(state, hlo_ir, objective, device_mesh,
                                      self.as_option, i, 1)

    @unittest.skip("This test is broken after we disallow some replicated iota")
    def test_bert_mlm_model_parallel(self):
        batch_size = 16
        seq_len = 16
        num_layers = 2
        hidden_size = 128
        num_heads = 4
        vocab_size = 512
        deterministic = False
        self.as_option.allow_all_gather = False  # Temporary hack
        self.as_option.allow_all_to_all = False  # Temporary hack

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_bert_mlm(
                batch_size, seq_len, num_layers, hidden_size, num_heads,
                vocab_size, deterministic, device_mesh)

            # Check communication cost
            # expected_cost = embed.forward (1) + embed.backward(2) +
            #                 LM_head.forward (1) + LM_head.backward (1) +
            #                 LM_head.weight.backward (1) +  log_softmax.forward (2) +
            #                 transformer.forward (2 * num_layers) + transformer.backward (2 * num_layers)
            #
            # Note that the final cost is different from this estimated cost in ILP solver.
            # The SPMD partitioner will eliminate some unnecessary communication in favor of
            # redundant computation (e.g., it will elimiate the all-reduce in embed.backward).
            expected = (
                device_mesh.all_reduce_cost(
                    batch_size * seq_len * hidden_size * 4, i) * 5 +
                device_mesh.all_reduce_cost(hidden_size * hidden_size * 4, i) +
                device_mesh.all_reduce_cost(batch_size * seq_len * 4, i) * 2 +
                device_mesh.all_reduce_cost(
                    batch_size * seq_len * hidden_size * 4, i) * num_layers * 4)
            assert_close(objective, expected)

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
                count_communication_primitives(hlo_ir))

            # real number of all-reduce = transformers (4 * num_layers) + log_softmax (2) +
            #                             embed.forward (1) + embad.backward (1)
            assert n_all_reduce == num_layers * 4 + 4
            assert n_total == n_all_reduce

            # Check sharding specification
            embed_weight = state.params["params"]["bert"]["embeddings"][
                "word_embeddings"]["embedding"]
            lm_head = state.params["params"]["cls"]["predictions"]["transform"][
                "dense"]["kernel"]
            assert_row_partitioned(embed_weight, mesh_shape[i], i)
            assert_all_replicated(lm_head, np.prod(mesh_shape))

            for k in range(num_layers):
                params = state.params["params"]["bert"]["encoder"]["layer"][str(
                    k)]
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
        num_layers = 2
        hidden_size = 512
        num_heads = 4
        vocab_size = 4096
        deterministic = False
        # To generate the desired strategy, we have to turn off mixed mesh shape and all-gather
        # and enable recomputing heavy ops.
        self.as_option.allow_recompute_heavy_op = True
        self.as_option.allow_all_gather = False
        self.as_option.allow_mixed_mesh_shape = False

        mesh_shape = [2, 2]
        device_mesh = self.get_device_mesh(mesh_shape, [2, 2], [1, 0.1])

        state, hlo_ir, objective = self.run_bert_mlm(batch_size, seq_len,
                                                     num_layers, hidden_size,
                                                     num_heads, vocab_size,
                                                     deterministic, device_mesh)

        # Check communication cost.
        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_ir,
                                           ignore_scalar_all_reduce=True))
        if self.as_option.prefer_reduce_scatter:
            assert n_all_reduce == 4 * num_layers + 2 + 2
            assert n_reduce_scatter <= 3  # The correct number should be 2,
            # but GpuMultiOutputFusion can make
            # some reduce-scatter unable to be combined
            assert n_all_gather <= 2
            assert n_total == n_all_reduce + n_all_gather + n_reduce_scatter
        else:
            # real number of all-reduce = transformers (4 * num_layers) + log_softmax (2) +
            #                             embed.forward (1) + embad.backward (1) + weights (1)
            assert n_all_reduce == 4 * num_layers + 2 + 2 + 1
            assert n_total == n_all_reduce

        # Check sharding specification
        assert "s32[4,4,4096]{2,1,0} iota()" not in hlo_ir
        assert "s32[2,4,2048]{2,1,0} iota()" in hlo_ir

        if self.as_option.prefer_reduce_scatter:
            num_not_sharded = 0  # allow the token_type_embeddings not partitioned.
            for weight in jax.tree_util.tree_leaves(state.opt_state):
                if len(weight.shape) > 1:
                    if not is_fully_sharded(weight):
                        num_not_sharded += 1
            assert num_not_sharded <= 2
        else:
            embed_weight = (state.params["params"]["bert"]["embeddings"]
                            ["word_embeddings"]["embedding"])
            lm_head = (state.params["params"]["cls"]["predictions"]["transform"]
                       ["dense"]["kernel"])

            assert_replicated_row_partitioned(embed_weight, mesh_shape)
            assert_all_replicated(lm_head, np.prod(mesh_shape))

            for k in range(num_layers):
                params = state.params["params"]["bert"]["encoder"]["layer"][str(
                    k)]
                weights = [
                    params["attention"]["self"]["qvk_combined"]["kernel"],
                    params["attention"]["output"]["dense"]["kernel"],
                    params["intermediate"]["dense"]["kernel"],
                    params["output"]["dense"]["kernel"],
                ]

                for j in range(len(weights)):
                    if j % 2 == 0:
                        assert_replicated_column_partitioned(
                            weights[j], mesh_shape)
                    else:
                        assert_replicated_row_partitioned(
                            weights[j], mesh_shape)

    def test_bert_layer_data_parallel_reduce_scatter(self):
        self.as_option.prefer_reduce_scatter = True
        self.test_bert_layer_data_parallel()

    def test_bert_layer_model_parallel_reduce_scatter(self):
        self.as_option.prefer_reduce_scatter = True
        self.test_bert_layer_model_parallel()

    def test_bert_layer_2d_mesh_reduce_scatter(self):
        self.as_option.prefer_reduce_scatter = True
        self.test_bert_layer_2d_mesh()

    def test_bert_mlm_data_parallel_reduce_scatter(self):
        self.as_option.prefer_reduce_scatter = True
        self.test_bert_mlm_data_parallel()

    def test_bert_mlm_data_parallel_reduce_scatter_zero_3(self):
        self.as_option.force_zero_stage_3 = True
        self.as_option.force_zero_stage_3_all_gather_threshold = 1
        self.test_bert_mlm_data_parallel()

    @unittest.skip("This test is broken after we disallow some replicated iota."
                  )
    def test_bert_mlm_model_parallel_reduce_scatter(self):
        self.as_option.prefer_reduce_scatter = True
        self.test_bert_mlm_model_parallel()

    def test_bert_mlm_2d_mesh_reduce_scatter(self):
        self.as_option.prefer_reduce_scatter = True
        self.test_bert_mlm_2d_mesh()

    def test_bert_layer_model_parallel_remat(self):
        batch_size = 8
        seq_len = 8
        num_layers = 2
        hidden_size = 128
        num_heads = 8
        deterministic = False
        use_remat = True

        # Test on different logical mesh shapes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_bert_layers(
                batch_size, seq_len, num_layers, hidden_size, num_heads,
                deterministic, use_remat, device_mesh)

            expected = (num_layers * 6 - 1) * device_mesh.all_reduce_cost(
                batch_size * seq_len * hidden_size * 4, i)
            assert_close(objective, expected)

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
                count_communication_primitives(hlo_ir))
            assert n_total == num_layers * 6 - 1
            assert n_all_reduce == num_layers * 6 - 1
            assert n_total == n_all_reduce


def suite():
    suite = unittest.TestSuite()

    def add(name):
        suite.addTest(AutoShardingAttentionTest(name))

    add("test_bert_layer_data_parallel")
    add("test_bert_layer_model_parallel")
    add("test_bert_layer_2d_mesh")
    add("test_bert_layer_force_batch_dim_mapping")

    add("test_embedding_2d_mesh")

    add("test_bert_mlm_data_parallel")
    add("test_bert_mlm_model_parallel")
    add("test_bert_mlm_2d_mesh")

    add("test_bert_layer_data_parallel_reduce_scatter")
    add("test_bert_layer_model_parallel_reduce_scatter")
    add("test_bert_layer_2d_mesh_reduce_scatter")

    add("test_bert_mlm_data_parallel_reduce_scatter")
    add("test_bert_mlm_model_parallel_reduce_scatter")
    add("test_bert_mlm_2d_mesh_reduce_scatter")
    add("test_bert_mlm_data_parallel_reduce_scatter_zero_3")

    add("test_bert_layer_model_parallel_remat")

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
