"""Benchmark one case of inter-op + intra-op parallelism."""
import jax
import jax.numpy as jnp
from jax._src.tree_util import tree_flatten, tree_leaves, tree_unflatten
import numpy as np
import optax
import time

from alpa import (parallelize, global_config, get_global_cluster,
                  set_global_virtual_physical_mesh, PipeshardParallel,
                  ManualPipeshardParallel, AutoShardingOption,
                  manual_layer_construction)
from alpa.model.bert_model import BertConfig, FlaxBertLayerCollection
from alpa.model.model_util import TrainState
from alpa.model.gpt_model import FlaxGPTForLMModule
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.timer import timers
from alpa.util import print_used_time, to_str_round, GB

from benchmark.util import compute_gpt_parameter_count, compute_inference_gpt_tflops


def create_infer_params_aval(rngkey, model, batch, no_embedding):
    if no_embedding:
        params = jax.eval_shape(model.init, rngkey, batch["x"],
                                batch["attention_mask"])
    else:
        params = jax.eval_shape(model.init, rngkey, batch["input_ids"],
                                batch["attention_mask"],
                                batch["token_type_ids"], batch["position_ids"])
    return params


def get_infer_step(parallel_method, model, no_embedding):

    def infer_step_with_embedding(params, batch, rng_key):
        rngs = {"dropout": rng_key}
        logits = model.apply(params,
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

    def infer_step_without_embedding(params, batch, rng_key):
        out = model.apply(params,
                          batch["x"],
                          batch["attention_mask"],
                          output_attentions=True,
                          output_hidden_states=True)
        loss = jnp.mean((out.last_hidden_state - batch["y"])**2)
        return loss

    if no_embedding:
        infer_step = manual_layer_construction(infer_step_without_embedding)
    else:
        infer_step = manual_layer_construction(infer_step_with_embedding)
    return parallelize(infer_step, method=parallel_method, donate_argnums=())


def benchmark_gpt_bert_internal(model_type,
                                benchmark_case,
                                niter,
                                num_hosts,
                                num_devices_per_host,
                                aval_infer_state=True):
    print_used_time(None)

    # Model configs
    (_, no_embedding, batch_size, seq_len, hidden_size, num_layers, num_heads,
     vocab_size, num_micro_batches, parallel_mode,
     parallel_args) = benchmark_case
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
    rngkey = jax.random.PRNGKey(0)
    if no_embedding:
        batch = {
            "x":
                jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                                  dtype=dtype),
            "y":
                jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                                  dtype=dtype),
            "attention_mask":
                jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        }
    else:
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
        if no_embedding:
            model = FlaxBertLayerCollection(BertConfig(
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 4,
                num_attention_heads=num_heads,
                num_hidden_layers=num_layers,
                gradient_checkpointing=add_manual_remat,
                add_manual_pipeline_markers=add_manual_layer_marker,
                pipeline_mp_size=num_manual_pipeline_stages),
                                            dtype=dtype)

        else:
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

    if aval_infer_state:
        params = create_infer_params_aval(rngkey, model, batch, no_embedding)
    else:
        raise RuntimeError(f"only support aval infer_state")
    print_used_time("Create infer state")

    # Compile executable
    infer_step = get_infer_step(method, model, no_embedding)
    executable = infer_step.get_executable(params, batch, rngkey)
    print_used_time("Compile (driver)")

    # Preshard params
    params_ps, _, _ = executable.get_input_placement_specs()
    flat_params, in_tree = tree_flatten(params)
    flat_ps = tree_leaves(params_ps)
    params = tree_unflatten(
        in_tree,
        executable.mesh_group.shard_args_to_arrays(flat_ps, flat_params))
    print_used_time("Preshard (driver)")

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

    executable.sync()
    print_used_time("Compile (worker)")

    # Warmup for e2e latency
    _ = infer_step(params, batch, rngkey)
    executable.sync()

    # Benchmark latency
    tic = time.time()
    for i in range(niter):
        print(f"Iteration {i} ...")
        _ = infer_step(params, batch, rngkey)
        executable.sync()
    e2e_latency = (time.time() - tic) / niter

    overall_latency = np.mean(executable.get_execution_time_costs(warmup=1))

    max_mem_allocated = executable.mesh_group.get_max_memory_allocated()

    print_used_time("Benchmark")

    # Compute statistics
    tflops = compute_inference_gpt_tflops(batch_size, seq_len, num_layers,
                                          hidden_size, vocab_size,
                                          virtual_mesh.num_devices,
                                          overall_latency)
    parameter_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                                  vocab_size)

    return (parameter_count, max_mem_allocated, overall_latency, e2e_latency,
            tflops, compilation_times) + get_last_dp_result()
