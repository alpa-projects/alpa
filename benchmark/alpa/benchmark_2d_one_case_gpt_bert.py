"""Benchmark one case of intra-op only parallelism."""
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time

import alpa
from alpa import parallelize, global_config, ShardParallel, AutoShardingOption
from alpa.model.bert_model import BertConfig, FlaxBertForMaskedLMModule, TrainState
from alpa.model.gpt_model import FlaxGPTForLMModule
from alpa.util import map_to_shape, count_communication_primitives, print_used_time, GB

from benchmark.util import compute_gpt_parameter_count, compute_gpt_tflops
from benchmark_3d_one_case_gpt_bert import create_train_state_aval


def create_train_state(rngkey, model, dtype, batch):
    params = model.init_dummy(rngkey, batch["input_ids"],
                              batch["attention_mask"], batch["token_type_ids"],
                              batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.chain(
        #optax.clip_by_global_norm(1.0),  # TODO(lmzheng): fix reduce-scatter for this
        optax.adamw(learning_rate=1e-8, mask=weight_decay_mask))

    mixed_precision = (dtype == jnp.float16)

    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              mixed_precision=mixed_precision,
                              dynamic_scale=None)
    return state


def get_train_step(grad_func, method):

    @parallelize(method=method)
    def train_step(state, batch, rng_key):

        def loss_func(params):
            rngs = {"dropout": rng_key}
            logits = state.apply_fn(params,
                                    batch["input_ids"],
                                    batch["attention_mask"],
                                    batch["token_type_ids"],
                                    batch["position_ids"],
                                    deterministic=True,
                                    rngs=rngs)[0]
            label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
            labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
            loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1),
                            axis=-1)
            loss = (label_mask * loss).sum() / label_mask.sum()
            return loss

        grads = grad_func(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        # TODO(lmzheng): add dynamic scaling for mixed-precision training
        return new_state

    return train_step


def benchmark_gpt_bert_internal(physical_mesh, model_type, benchmark_case,
                                niter):
    print_used_time(None)

    # Model configs
    (batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,
     num_micro_batches, parallel_mode, parallel_args) = benchmark_case
    (prefer_reduce_scatter, use_remat, (dp, op, pp),
     force_batch_dim_mapping) = parallel_args

    dtype = jnp.float16

    # Parallel configs
    assert pp == 1, "Do not support pipeline parallelism"
    if num_micro_batches > 1:
        grad_func = alpa.grad
    else:
        num_micro_batches = None
        grad_func = jax.grad

    as_option = AutoShardingOption()
    if force_batch_dim_mapping:  # Always map batch dim to mesh dim 0
        as_option.force_batch_dim_to_mesh_dim = 0
    as_option.prefer_reduce_scatter = prefer_reduce_scatter
    if parallel_mode == "zero-3":
        as_option.force_zero_stage_3 = True
    elif parallel_mode in ["shard-largest"]:
        as_option.force_simple_heuristic = other
        global_config.remat_using_while = True

    logical_mesh = physical_mesh.get_logical_mesh([dp, op])
    method = ShardParallel(devices=logical_mesh,
                           num_micro_batches=num_micro_batches,
                           auto_sharding_option=as_option)
    print_used_time("Setup device mesh")

    # Prepare input batch
    batch = {
        "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "position_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
    }
    print_used_time("Prepare input")

    # Init train state
    if model_type == "gpt":
        model = FlaxGPTForLMModule(BertConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,
            type_vocab_size=0,
            gradient_checkpointing=use_remat,
        ),
                                   dtype=dtype)
    elif model_type == "bert":
        model = FlaxBertForMaskedLMModule(BertConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,
            type_vocab_size=0,
            gradient_checkpointing=use_remat,
        ),
                                          dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    if global_config.use_dummy_value_for_benchmarking:
        state = create_train_state_aval(rngkey, model, batch, dtype)
    else:
        state = create_train_state(rngkey, model, dtype, batch)
    print_used_time("Create train state")

    # Compile executable
    train_step = get_train_step(grad_func, method=method)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    physical_mesh.sync_workers()
    print_used_time("Compile (workers)")

    # Check sharding strategy
    alloc_mem = executable.get_total_allocation_size()
    ilp_objective = executable.auto_sharding_objective or 0.0
    executable.dump_debug_info("tmp")
    hlo_text = executable.get_hlo_text()
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all =\
        count_communication_primitives(hlo_text)

    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}, "
          f"#all-to-all: {n_all_to_all}")
    print(f"alloc_mem: {alloc_mem / GB:.2f} GB")

    # Benchmark step time
    warmup = 2 if niter >= 5 else 1

    if alloc_mem > physical_mesh.get_available_memory():
        latencies = [-1]
    else:
        # Benchmark latency without driver overhead
        for i in range(niter):
            print(f"Iteration {i} ...")
            state = train_step(state, batch, rngkey)
            physical_mesh.sync_workers()

        latencies = executable.get_execution_time_costs(warmup=warmup)

        # Benchmark latency with driver overhead
        if False:
            global_config.use_dummy_value_for_benchmarking = False
            global_config.shard_parallel_sync_for_timer = False
            number = niter
            physical_mesh.sync_workers()
            tic = time.time()
            for i in range(number):
                state = train_step(state, batch, rngkey)
            physical_mesh.sync_workers()
            e2e_latency = (time.time() - tic) / number
            print(f"latency with dirver overhead: {e2e_latency:.3f}")

    print_used_time("Benchmark")

    # Compute statistics
    tflops = compute_gpt_tflops(batch_size, seq_len, num_layers, hidden_size,
                                vocab_size, physical_mesh.num_devices,
                                np.mean(latencies), use_remat)
    param_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                              vocab_size)
    peak_mem = max(physical_mesh.get_max_memory_allocated(), alloc_mem)

    return param_count, ilp_objective, peak_mem, latencies, tflops
