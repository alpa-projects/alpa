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

from examples.opt_serving.model.opt_model import (
    get_opt_config, get_pipeshard_executable, load_params_dis_array,
    init_cache_dis_array, load_params_np, init_cache_np, get_jax_executable,
    build_position_ids, init_model_aval, init_cache_aval)
from examples.opt_serving.model.wrapper import set_skip_shard_args_check


def run_benchmark(args):
    name = args.model.split("-")[1].upper()
    path = os.path.join(args.path, f"{name}_np")

    alpa.global_config.shard_parallel_sync_for_timer = True

    config = get_opt_config(name)

    batch_size = 1
    seq_len = 8
    dummy = args.dummy

    input_ids = np.random.randint(0, 10000, size=(batch_size, seq_len), dtype=np.int32)
    position_ids = build_position_ids(input_ids, config.pad)

    def inference_step_with_cache(params, batch):
        output = model.apply(params,
                             batch["input_ids"],
                             batch["position_ids"],
                             attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    if args.parallel_method == "jit":
        model, params_aval = init_model_aval(config)
        params = load_params_np(params_aval, path, config, dummy)
        cache = init_cache_np(config, batch_size)
        params, cache = jax.tree_map(jnp.array, (params, cache))

        infer_step = jax.jit(inference_step_with_cache)
        sync_func = lambda : jax.local_devices()[0].synchronize_all_activity()
        executable = None
        num_gpus = 1
    elif args.parallel_method == "local_shard":
        model, params_aval = init_model_aval(config)

        method = alpa.ShardParallel(devices=jax.local_devices(),
                                    auto_sharding_option=alpa.AutoShardingOption())
        infer_step = alpa.parallelize(inference_step_with_cache, method=method)
        executable = infer_step.get_executable(
            params_aval, {
                "input_ids": jax.core.ShapedArray((batch_size, 1), jnp.int32),
                "position_ids": jax.core.ShapedArray((batch_size, 1), jnp.int32),
                "cache": init_cache_aval(config, batch_size),
            })
        executable.dump_debug_info("tmp")

        assert dummy == True, 'Only support dummy weights. Plasese add "--dummy".'
        params = load_params_dis_array(path, executable, params_aval, config, dummy)
        cache = init_cache_dis_array(executable, config, batch_size, dummy)
        set_skip_shard_args_check(cache)
        sync_func = lambda : executable.sync()
        infer_step = executable
        num_gpus = len(method.devices)
    else:
        raise ValueError("Invalid parallel method: {args.parallel_method}")

    step_latencies = []
    compute_latencies = []
    for i in range(input_ids.shape[1]):
        input_ids_step = input_ids[:, i:i + 1]
        position_ids_step = np.full_like(input_ids_step, i + config.pad + 1)

        sync_func()
        start_time = time.time()
        logits_step, cache = infer_step(
            params, {
                "input_ids": input_ids_step,
                "position_ids": position_ids_step,
                "cache": cache,
            })
        sync_func()
        end_time = time.time()
        step_latencies.append(end_time - start_time)
        if executable:
            compute_latencies.append(executable.get_execution_time_costs(0)[-1])
        else:
            compute_latencies.append(step_latencies[-1])

        print(f"{i}, step_latency: {step_latencies[-1] * 1000:.2f} ms")

    heads = ["Model", "Parallel Method", "Dummy", "#gpu",
             "Step Latency (ms)", "Compute Latency (ms)"]
    values = [args.model, args.parallel_method, args.dummy, num_gpus,
              f"{np.mean(step_latencies[3:]) * 1e3:.2f}",
              f"{np.mean(compute_latencies[3:]) * 1e3:.2f}"]
    write_tsv(heads, values, "result_step_func.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpa/opt-2.7b")
    parser.add_argument("--path", type=str, default="/home/ubuntu/opt_weights/")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--parallel-method", type=str, required=True,
        choices=["jit", "local_shard", "ray_shard", "pipeshard"])
    args = parser.parse_args()

    run_benchmark(args)
