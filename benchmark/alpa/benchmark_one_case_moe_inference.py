"""Benchmark one case of inter-op + intra-op parallelism."""
import jax
import jax.numpy as jnp
import numpy as np

from alpa import parallelize, get_global_cluster, set_global_virtual_physical_mesh
from alpa.model.moe import FlaxMoEForLMModule, MoEConfig, TrainState
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.util import print_used_time, GB, write_tsv

from benchmark_one_case_gpt_bert import get_train_step
from util import compute_moe_parameter_count, compute_moe_tflops
from benchmark_parallel_utils import (
    get_pipeshard_parallel_method, get_shard_parallel_method,
    compile_and_benchmark_pipeshard_inference_executable,
    compute_avg_stage_latencies)


def create_infer_params_aval(rngkey, model, batch):
    params = jax.eval_shape(model.init, rngkey, batch["input_ids"],
                            batch["attention_mask"], batch["token_type_ids"],
                            batch["position_ids"])
    params = jax.eval_shape(
        lambda p: jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=jnp.float16), p), params)
    return params


def get_infer_step(parallel_method, model):

    def infer_step(params, batch, rng_key):
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

    return parallelize(infer_step, method=parallel_method, donate_argnums=())


def prepare_moe_inference_input_and_model(benchmark_case,
                                          add_manual_remat=None,
                                          add_manual_layer_marker=None,
                                          num_manual_pipeline_stages=None,
                                          correct_expert_group_size=True):
    print_used_time(None)
    batch_size = benchmark_case.batch_size
    (seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts,
     expert_group_size) = benchmark_case.model_config
    dtype = jnp.float16
    tie_word_embeddings = False

    if correct_expert_group_size:
        rang_factor = 1
        expected_expert_group_size = min(
            expert_group_size, batch_size * seq_len //
            benchmark_case.num_micro_batches // 1 // rang_factor)
        if expected_expert_group_size != expert_group_size:
            print("- Expected expert group size should be {}, "
                  "but got {}. Will reset it".format(expected_expert_group_size,
                                                     expert_group_size))
            expert_group_size = expected_expert_group_size

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
    model = FlaxMoEForLMModule(
        MoEConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 8,  # this is specific to gspmd.
            num_attention_heads=num_heads,
            max_position_embeddings=seq_len,
            vocab_size=vocab_size,
            expert_group_size=expert_group_size,
            expert_number=num_experts,
            tie_word_embeddings=tie_word_embeddings,
            gradient_checkpointing=add_manual_remat,
            add_manual_pipeline_markers=add_manual_layer_marker,
            pipeline_mp_size=num_manual_pipeline_stages,
        ),
        dtype=dtype)

    rngkey = jax.random.PRNGKey(0)
    params = create_infer_params_aval(rngkey, model, batch)
    print_used_time("Create train state")
    return model, params, batch, rngkey


def compute_moe_statistics(benchmark_case, latencies, num_devices):
    batch_size = benchmark_case.batch_size
    (seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts,
     expert_group_size) = benchmark_case.model_config
    use_remat = benchmark_case.parallel_args.use_remat

    tflops = compute_moe_tflops(batch_size,
                                seq_len,
                                num_layers,
                                hidden_size,
                                expert_group_size,
                                vocab_size,
                                num_experts,
                                num_devices,
                                np.mean(latencies),
                                checkpoint_activations=use_remat)
    parameter_count = compute_moe_parameter_count(num_layers,
                                                  hidden_size,
                                                  vocab_size,
                                                  num_experts,
                                                  mlp_factor=8)
    return tflops, parameter_count


def benchmark_moe_inference_internal(benchmark_case,
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

    # Parallel configs
    (method, _, add_manual_layer_marker,
     num_manual_pipeline_stages) = get_pipeshard_parallel_method(
         benchmark_case,
         virtual_mesh.num_devices_per_host,
         pipeline_schedule="inference")

    model, params, batch, rngkey = prepare_moe_inference_input_and_model(
        benchmark_case,
        add_manual_layer_marker=add_manual_layer_marker,
        num_manual_pipeline_stages=num_manual_pipeline_stages)

    infer_step = get_infer_step(method, model)

    (latencies, max_mem_allocated, compilation_times, executable,
     per_stage_weight_mem,
     per_stage_peak_mem) = compile_and_benchmark_pipeshard_inference_executable(
         benchmark_case.parallel_mode,
         niter,
         infer_step,
         params, (batch, rngkey),
         profile_driver_time=profile_driver_time)

    # compute statistics
    tflops, parameter_count = compute_moe_statistics(benchmark_case, latencies,
                                                     virtual_mesh.num_devices)

    # Log per-stage execution information if needed
    if profile_stage_execution_time:
        model_name = f"moe-{parameter_count/1e9:.1f}b"
        # dump chrome trace
        executable.dump_stage_execution_trace(
            f"./chrome_trace/{model_name},bs={benchmark_case.batch_size},op={benchmark_case.parallel_args.op},pp={benchmark_case.parallel_args.pp}.json"
        )
        # compute and log per-stage latency/memory statistics
        exec_info = executable.get_stage_execution_info()
        timelines = list(zip(*exec_info))
        # drop warmup case
        timelines = timelines[1:]
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
            avg_stage_latencies
        ]
        write_tsv(heads, values, f"benchmark_results.tsv")

    metadata = {
        "compilation_times": compilation_times,
    }

    return parameter_count, max_mem_allocated, latencies, tflops, metadata
