"""Benchmark one case of inter-op + intra-op parallelism."""
import os

import jax
import jax.numpy as jnp
import numpy as np

from alpa import (parallelize, get_global_cluster,
                  set_global_virtual_physical_mesh)
from alpa.model.bert_model import BertConfig, FlaxBertLayerCollection
from alpa.model.gpt_model import FlaxGPTForLMModule
from alpa.util import print_used_time, GB, write_tsv

from util import compute_gpt_parameter_count, compute_gpt_tflops
from benchmark_parallel_utils import (
    get_pipeshard_parallel_method,
    compile_and_benchmark_pipeshard_inference_executable,
    compute_avg_stage_latencies)


def create_infer_params_aval(rngkey, model, batch, model_type):
    if model_type == "gpt_no_embedding_inference":
        params = jax.eval_shape(model.init, rngkey, batch["x"],
                                batch["attention_mask"])
    elif model_type == "gpt_inference":
        params = jax.eval_shape(model.init, rngkey, batch["input_ids"],
                                batch["attention_mask"],
                                batch["token_type_ids"], batch["position_ids"])
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    params = jax.eval_shape(
        lambda p: jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=jnp.float16), p), params)
    return params


def get_infer_step(parallel_method, model, model_type):

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

    if model_type == "gpt_no_embedding_inference":
        infer_step = infer_step_without_embedding
    elif model_type == "gpt_inference":
        infer_step = infer_step_with_embedding
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return parallelize(infer_step, method=parallel_method, donate_argnums=())


def prepare_gpt_inference_input_and_model(model_type,
                                          benchmark_case,
                                          add_manual_layer_marker=None,
                                          num_manual_pipeline_stages=None,
                                          tie_word_embeddings=False):
    print_used_time(None)
    batch_size = benchmark_case.batch_size
    (seq_len, hidden_size, num_layers, num_heads,
     vocab_size) = benchmark_case.model_config
    dtype = jnp.float16

    bert_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_layers,
        type_vocab_size=0,
        tie_word_embeddings=tie_word_embeddings,
        add_manual_pipeline_markers=add_manual_layer_marker,
        pipeline_mp_size=num_manual_pipeline_stages,
    )

    # Init train state
    if model_type == "gpt_no_embedding_inference":
        batch = {
            "x": jnp.ones((batch_size, seq_len, hidden_size), dtype=dtype),
            "y": jnp.ones((batch_size, seq_len, hidden_size), dtype=dtype),
            "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        }
        model = FlaxBertLayerCollection(bert_config, dtype=dtype)
    elif model_type == "gpt_inference":
        batch = {
            "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "position_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        }

        model = FlaxGPTForLMModule(bert_config, dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    params = create_infer_params_aval(rngkey, model, batch, model_type)
    print_used_time("Create infer state")
    return model, params, batch, rngkey


def compute_gpt_inference_statistics(benchmark_case, latencies, num_devices):
    batch_size = benchmark_case.batch_size
    (seq_len, hidden_size, num_layers, num_heads,
     vocab_size) = benchmark_case.model_config
    use_remat = benchmark_case.parallel_args.use_remat

    tflops = compute_gpt_tflops(batch_size,
                                seq_len,
                                num_layers,
                                hidden_size,
                                vocab_size,
                                num_devices,
                                np.mean(latencies),
                                backward=False)
    parameter_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                                  vocab_size)
    return tflops, parameter_count


def benchmark_gpt_inference_internal(model_type,
                                     benchmark_case,
                                     niter,
                                     num_hosts,
                                     num_devices_per_host,
                                     profile_driver_time=False,
                                     profile_stage_execution_time=False):
    # Connect to the cluster
    virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    set_global_virtual_physical_mesh(virtual_mesh)

    (method, _, add_manual_layer_marker,
     num_manual_pipeline_stages) = get_pipeshard_parallel_method(
         benchmark_case,
         virtual_mesh.num_devices_per_host,
         pipeline_schedule="inference")

    model, params, batch, rngkey = prepare_gpt_inference_input_and_model(
        model_type, benchmark_case, add_manual_layer_marker,
        num_manual_pipeline_stages)

    infer_step = get_infer_step(method, model, model_type)

    (latencies, max_mem_allocated, compilation_times, executable,
     per_stage_weight_mem,
     per_stage_peak_mem) = compile_and_benchmark_pipeshard_inference_executable(
         benchmark_case.parallel_mode,
         niter,
         infer_step,
         params, (batch, rngkey),
         profile_driver_time=profile_driver_time)

    # Compute statistics
    tflops, parameter_count = compute_gpt_inference_statistics(
        benchmark_case, latencies, virtual_mesh.num_devices_per_host)

    # Log per-stage execution information if needed
    if profile_stage_execution_time:
        model_name = f"bert-{parameter_count/1e9:.1f}b"
        # dump chrome trace
        executable.dump_stage_execution_trace(
            f"./chrome_trace/{model_name},bs={benchmark_case.batch_size},op={benchmark_case.parallel_args.op},pp={benchmark_case.parallel_args.pp}.json"
        )
        # compute and log per-stage latency/memory statistics
        exec_info = executable.get_stage_execution_info()
        timelines = list(zip(*exec_info))
        # drop warmup case
        timelines = timelines[3:]
        avg_stage_latencies = compute_avg_stage_latencies(timelines)
        assert len(avg_stage_latencies) == num_manual_pipeline_stages
        parallel_args = benchmark_case.parallel_args
        dp, op, pp = parallel_args.dp, parallel_args.op, parallel_args.pp
        heads = [
            "ModelName", "BS", "#Microbatch", "DP", "OP", "PP", "#GPU",
            "MeanTime(s)", "StdTime(s)", "TFLOPs", "StageWeights(B)",
            "StagePeakMem(B)", "StageLatencies(s)"
        ]
        values = [
            model_name, benchmark_case.batch_size,
            benchmark_case.num_micro_batches, dp, op, pp, dp * op * pp,
            f"{np.mean(latencies):.3f}", f"{np.std(latencies):.3f}",
            f"{tflops:.2f}", f"{per_stage_weight_mem}", f"{per_stage_peak_mem}",
            list(avg_stage_latencies)
        ]
        write_tsv(heads, values, f"inference_prof_res.tsv")

    metadata = {
        "compilation_times": compilation_times,
    }
    return parameter_count, max_mem_allocated, latencies, tflops, metadata
