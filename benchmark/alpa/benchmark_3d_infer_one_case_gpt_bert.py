"""Benchmark one case of inter-op + intra-op parallelism."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time

from alpa import (parallelize, global_config, get_global_cluster,
                  set_global_virtual_physical_mesh, PipeshardParallel,
                  ManualPipeshardParallel, AutoShardingOption,
                  manual_layer_construction)
from alpa.model.bert_model import BertConfig, FlaxBertForMaskedLMModule
from alpa.model.model_util import TrainState
from alpa.model.gpt_model import FlaxGPTForLMModule
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.timer import timers
from alpa.util import print_used_time, to_str_round, GB

from benchmark.util import compute_gpt_parameter_count, compute_inference_gpt_tflops


def create_infer_state(rngkey, model, batch, dtype):
    params = model.init_dummy(rngkey, batch["input_ids"],
                              batch["attention_mask"], batch["token_type_ids"],
                              batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.chain(
        #optax.clip_by_global_norm(1.0),  # TODO(lmzheng): fix reduce-scatter for this
        optax.adamw(learning_rate=1e-2, mask=weight_decay_mask))
    mixed_precision = (dtype == jnp.float16)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              mixed_precision=mixed_precision,
                              dynamic_scale=None)
    return state


def create_infer_state_aval(rngkey, model, batch, dtype):
    params = jax.eval_shape(model.init, rngkey, batch["input_ids"],
                            batch["attention_mask"], batch["token_type_ids"],
                            batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.chain(
        #optax.clip_by_global_norm(1.0),  # TODO(lmzheng): fix reduce-scatter for this
        optax.adamw(learning_rate=1e-2, mask=weight_decay_mask))
    mixed_precision = (dtype == jnp.float16)
    state = TrainState.create_aval(apply_fn=model.apply,
                                   params=params,
                                   tx=tx,
                                   mixed_precision=mixed_precision,
                                   dynamic_scale=None)
    return state


def get_infer_step(parallel_method):

    def infer_step(state, batch, rng_key):
        rngs = {"dropout": rng_key}
        logits = state.apply_fn(state.params,
                                batch["input_ids"],
                                batch["attention_mask"],
                                batch["token_type_ids"],
                                batch["position_ids"],
                                deterministic=True,
                                rngs=rngs)[0]
        label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
        labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
        loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
        loss = (label_mask * loss).sum() / label_mask.sum()
        return loss

    infer_step = manual_layer_construction(infer_step)
    return parallelize(infer_step, method=parallel_method, donate_argnums=())


def benchmark_gpt_bert_internal(model_type,
                                benchmark_case,
                                niter,
                                num_hosts,
                                num_devices_per_host,
                                aval_train_state=True):
    print_used_time(None)

    # Model configs
    (_, batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,
     num_micro_batches, parallel_mode, parallel_args) = benchmark_case
    dtype = jnp.float16
    tie_word_embeddings = False

    # Connect to the cluster
    virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    set_global_virtual_physical_mesh(virtual_mesh)

    # Parallel configs
    if parallel_mode == "load_solution":
        prefer_reduce_scatter, _, num_auto_layers, manual_stage_option = parallel_args
        add_manual_remat = False
        num_manual_pipeline_stages = num_auto_layers
        add_manual_layer_marker = True
        method = ManualPipeshardParallel(
            *manual_stage_option,
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter),
            pipeline_schedule="inference")
    else:
        raise ValueError(f"Invalid mode: {parallel_mode}")

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
    if model_type == "bert":
        model = FlaxBertForMaskedLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
            tie_word_embeddings=tie_word_embeddings,
            gradient_checkpointing=add_manual_remat,
            add_manual_pipeline_markers=add_manual_layer_marker,
            pipeline_mp_size=num_manual_pipeline_stages,
        ),
                                          dtype=dtype)
    elif model_type == "gpt":
        model = FlaxGPTForLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
            tie_word_embeddings=tie_word_embeddings,
            gradient_checkpointing=add_manual_remat,
            add_manual_pipeline_markers=add_manual_layer_marker,
            pipeline_mp_size=num_manual_pipeline_stages,
        ),
                                   dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    if aval_train_state:
        state = create_infer_state_aval(rngkey, model, batch, dtype)
    else:
        state = create_infer_state(rngkey, model, batch, dtype)
    print_used_time("Create train state")

    # Compile executable
    infer_step = get_infer_step(method)
    executable = infer_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    if parallel_mode == "search":
        compilation_times = {
            k: timers(k).elapsed() for k in [
                "stage-construction", "stage-construction-dp",
                "stage-construction-compilation", "stage-construction-profiling"
            ]
        }
        print(
            f"compilation time breakdown: {to_str_round(compilation_times, 2)}")
    else:
        compilation_times = None

    # Dump hlo ir for debugging
    stage_hlo_texts = executable.get_hlo_text()
    for i in range(len(stage_hlo_texts)):
        with open(f"tmp/stage_{i}.hlo", "w") as fout:
            fout.write(stage_hlo_texts[i])
    with open(f"tmp/resharding_tasks.txt", "w") as fout:
        fout.write(executable.print_resharding_tasks())

    executable.sync()
    print_used_time("Compile (worker)")

    # Warmup for e2e latency
    _ = infer_step(state, batch, rngkey)
    executable.sync()

    # Benchmark latency
    tic = time.time()
    for i in range(niter):
        print(f"Iteration {i} ...")
        _ = infer_step(state, batch, rngkey)
        executable.sync()
    e2e_latency = (time.time() - tic) / niter

    timer_types = [
        "overall",
        "compute",
        "resharding_send",
        "resharding_recv",
        "resharding_broadcast",
        "free",
    ]

    latencies = []
    for timer_type in timer_types:
        latencies.append(
            np.mean(
                executable.get_execution_time_costs(warmup=1,
                                                    timer_name=timer_type)))

    max_mem_allocated = executable.mesh_group.get_max_memory_allocated()

    print_used_time("Benchmark")

    # Compute statistics
    tflops = compute_inference_gpt_tflops(batch_size, seq_len, num_layers,
                                          hidden_size, vocab_size,
                                          virtual_mesh.num_devices,
                                          latencies[0])
    parameter_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                                  vocab_size)

    return (parameter_count, max_mem_allocated, latencies, e2e_latency, tflops,
            compilation_times) + get_last_dp_result()
