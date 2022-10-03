"""
A simpler benchmark script that benchmarks the latency of alpa execution
without the huggingface generator interface.
"""

import argparse
import os
import time

import alpa
from alpa.util import write_tsv
import jax
import jax.numpy as jnp
import numpy as np

from llm_serving.model import opt_model, bloom_model
from llm_serving.model.wrapper import set_skip_shard_args_check


def run_benchmark(args):
    name = args.model.split("/")[1].lower()
    path = os.path.join(args.path, f"{name}-np")

    alpa.global_config.shard_parallel_sync_for_timer = True
    alpa.global_config.pipeline_check_alive = False
    alpa.global_config.pipeline_sync_for_timer = True
    alpa.global_config.delete_remote_arrays_threshold = 100

    batch_size = args.batch_size
    seq_len = 10
    dummy = args.dummy
    if "opt" in name:
        m = opt_model
        def inference_step_with_cache(params, batch):
            output = model.apply(params,
                                 batch["input_ids"],
                                 batch["position_ids"],
                                 attention_mask=batch["mask"],
                                 attention_cache=batch["cache"])
            return output.logits, output.attention_cache
    else:
        m = bloom_model
        def inference_step_with_cache(params, batch):
            output = model.apply(params,
                                 batch["input_ids"],
                                 attention_mask=batch["mask"],
                                 attention_cache=batch["cache"])
            return output.logits, output.attention_cache

    if args.parallel_method == "jit":
        config = m.get_config(name)
        model, params_aval = m.init_model_aval(config)
        params = m.load_params_np(params_aval, path, config, dummy)
        cache = m.init_cache_np(config, batch_size)
        params, cache = jax.tree_map(jnp.array, (params, cache))

        infer_step = jax.jit(inference_step_with_cache)
        sync_func = lambda: jax.local_devices()[0].synchronize_all_activity()
        executable = None
        num_gpus = 1
    else:
        if args.parallel_method in ["shard_local", "shard_ray"]:
            assert dummy == True, 'Only support dummy weights. Plasese add "--dummy".'

            config = m.get_config(name)
            model, params_aval = m.init_model_aval(config)
            if args.parallel_method == "shard_local":
                alpa.init(cluster="local")
            else:
                alpa.init(cluster="ray")
            num_gpus = alpa.get_global_num_devices()

            method = alpa.ShardParallel(
                auto_sharding_option=alpa.AutoShardingOption())
            infer_step = alpa.parallelize(inference_step_with_cache,
                                          method=method)
        else:
            assert args.parallel_method == "pipeshard"
            alpa.init(cluster="ray")
            num_gpus = alpa.get_global_num_devices()
            num_pp_stages = max(2, alpa.get_global_cluster().num_hosts)
            config = m.get_config(name, num_pp_stages=num_pp_stages)
            model, params_aval = m.init_model_aval(config)

            method = alpa.PipeshardParallel(
                num_micro_batches=1,
                pipeline_schedule="inference",
                layer_option="manual",
                default_auto_sharding_option=alpa.AutoShardingOption(
                    # Force operator model parallel
                    force_batch_dim_to_mesh_dim=None if batch_size == 1 else 0,
                    # Disabling all-to-all and all-gather generates better intra-op strategies.
                    allow_all_to_all=False,
                    allow_all_gather=False,
                ))
            infer_step = alpa.parallelize(inference_step_with_cache, method=method)
            alpa.global_config.always_donate_micro_batch_vars = False

        executable = infer_step.get_executable(
            params_aval, {
                "input_ids":
                    jax.core.ShapedArray((batch_size, 1), jnp.int32),
                "position_ids":
                    jax.core.ShapedArray((batch_size, 1), jnp.int32),
                "cache":
                    m.init_cache_aval(config, batch_size),
                "mask":
                    m.init_mask_aval(config, batch_size),
            })
        executable.dump_debug_info("tmp")

        params = m.load_params_dis_array(path, executable, params_aval, config,
                                         dummy)
        cache = m.init_cache_dis_array(executable, config, batch_size, dummy)
        set_skip_shard_args_check(cache)
        infer_step = executable
        if args.parallel_method == "local_shard":
            # Already synced by the local timer
            sync_func = lambda: None
        else:
            sync_func = lambda: executable.sync()

    input_ids = np.random.randint(0,
                                  10000,
                                  size=(batch_size, seq_len),
                                  dtype=np.int32)
    position_ids = opt_model.build_position_ids(input_ids, config.pad)
    mask = np.ones((batch_size, 1, 1, config.max_seq_len), dtype=np.int8)

    step_latencies = []
    compute_latencies = []
    shard_args_latencies = []
    for i in range(input_ids.shape[1]):
        input_ids_step = input_ids[:, i:i + 1]
        position_ids_step = np.full_like(input_ids_step, i + config.pad + 1)

        sync_func()
        start_time = time.time()
        infer_step(
            params, {
                "input_ids": input_ids_step,
                "position_ids": position_ids_step,
                "mask": mask,
                "cache": cache,
            })
        sync_func()
        end_time = time.time()

        step_latencies.append(end_time - start_time)
        if executable:
            compute_latencies.append(executable.get_execution_time_costs()[-1])
            shard_args_latencies.append(
                executable.get_shard_args_time_costs()[-1])
        else:
            compute_latencies.append(step_latencies[-1])
            shard_args_latencies.append(0)

        print(f"{i}, step_latency: {step_latencies[-1] * 1000:.2f} ms")

    warmup = 3
    heads = [
        "Model", "Parallel Method", "Dummy", "#gpu", "Step Latency (ms)",
        "Compute Latency (ms)", "ShardArgs Latency (ms)"
    ]
    values = [
        args.model, args.parallel_method, args.dummy, num_gpus,
        f"{np.mean(step_latencies[warmup:]) * 1e3:.2f}",
        f"{np.mean(compute_latencies[warmup:]) * 1e3:.2f}",
        f"{np.mean(shard_args_latencies[warmup:]) * 1e3:.2f}"
    ]
    write_tsv(heads, values, "result_step_func.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpa/opt-2.7b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--path", type=str, default="/home/ubuntu/opt_weights/")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument(
        "--parallel-method",
        type=str,
        required=True,
        choices=["jit", "shard_local", "shard_ray", "pipeshard"])
    args = parser.parse_args()

    run_benchmark(args)
