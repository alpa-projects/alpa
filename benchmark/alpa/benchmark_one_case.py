"""Benchmark one case of inter-op + intra-op parallelism."""
import os
import argparse
import multiprocessing as mp

import jax

from alpa import (init, global_config, get_global_cluster,
                  LocalPhysicalDeviceMesh)
from alpa.util import disable_tqdm_globally

from benchmark_one_case_gpt_bert import (benchmark_gpt_bert_3d_internal,
                                         benchmark_gpt_bert_2d_internal)
from benchmark_one_case_moe import (benchmark_moe_3d_internal,
                                    benchmark_moe_2d_internal)
from benchmark_one_case_unet import benchmark_unet_3d_internal
from benchmark_one_case_wresnet import (benchmark_wresnet_3d_internal,
                                        benchmark_wresnet_2d_internal)
from benchmark_one_case_gpt_bert_inference import (
    benchmark_gpt_inference_internal)
from benchmark_one_case_moe_inference import (benchmark_moe_inference_internal)


def benchmark_one_case_internal(model,
                                case,
                                niter,
                                num_hosts,
                                num_devices_per_host,
                                profile_driver_time=False,
                                profile_stage_execution_time=False,
                                shard_only=False,
                                local=False,
                                disable_tqdm=False):
    if disable_tqdm:
        disable_tqdm_globally()

    # local mode does not support dummy value
    global_config.use_dummy_value_for_benchmarking = not local

    if shard_only:
        global_config.shard_parallel_sync_for_timer = True
        if local:
            assert num_hosts == 1
            physical_mesh = LocalPhysicalDeviceMesh(
                jax.local_devices()[:num_devices_per_host])
        else:
            init(cluster="ray")
            physical_mesh = get_global_cluster().get_physical_mesh(
                list(range(num_hosts)), num_devices_per_host)

        # Run benchmark
        if model in ["gpt", "bert"]:
            result = benchmark_gpt_bert_2d_internal(
                physical_mesh,
                model,
                case,
                niter,
                profile_driver_time=profile_driver_time)
        elif model == "moe":
            result = benchmark_moe_2d_internal(
                physical_mesh,
                case,
                niter,
                profile_driver_time=profile_driver_time)
        elif model == "wresnet":
            global_config.xla_client_mem_fraction = 0.88
            # Due to legacy issues, we turn off auto-tuning. Although the
            # performance will be much better if we turn it on
            global_config.xla_gpu_autotune_level = 0
            result = benchmark_wresnet_2d_internal(
                physical_mesh,
                case,
                niter,
                profile_driver_time=profile_driver_time)
        else:
            raise ValueError(f"Invalid model: {model}")

    else:
        global_config.pipeline_sync_for_timer = True
        if profile_stage_execution_time:
            global_config.collect_trace = True
        init(cluster="ray")

        # Run benchmark
        if model in ["gpt", "bert"]:
            result = benchmark_gpt_bert_3d_internal(
                model,
                case,
                niter,
                num_hosts,
                num_devices_per_host,
                profile_driver_time=profile_driver_time)
        elif model == "moe":
            result = benchmark_moe_3d_internal(
                case,
                niter,
                num_hosts,
                num_devices_per_host,
                profile_driver_time=profile_driver_time)
        elif model == "wresnet":
            global_config.xla_client_mem_fraction = 0.88
            # Due to legacy issues, we turn off auto-tuning. Although the
            # performance will be much better if we turn it on
            global_config.xla_gpu_autotune_level = 0
            result = benchmark_wresnet_3d_internal(
                case,
                niter,
                num_hosts,
                num_devices_per_host,
                profile_driver_time=profile_driver_time)
        elif model == "unet":
            global_config.xla_client_mem_fraction = 0.88
            global_config.xla_gpu_autotune_level = 0
            result = benchmark_unet_3d_internal(
                case,
                niter,
                num_hosts,
                num_devices_per_host,
                profile_driver_time=profile_driver_time)
        elif model in ["gpt_inference", "gpt_no_embedding_inference"]:
            result = benchmark_gpt_inference_internal(
                model,
                case,
                niter,
                num_hosts,
                num_devices_per_host,
                profile_driver_time=profile_driver_time,
                profile_stage_execution_time=profile_stage_execution_time)
        elif model in ["moe_inference"]:
            result = benchmark_moe_inference_internal(
                case,
                niter,
                num_hosts,
                num_devices_per_host,
                profile_driver_time=profile_driver_time,
                profile_stage_execution_time=profile_stage_execution_time)
        else:
            raise ValueError(f"Invalid model: {model}")

    return result


def benchmark_and_write_to_namespace(result_namespace, *args, **kwargs):
    result = benchmark_one_case_internal(*args, **kwargs)
    result_namespace.result = result


def benchmark_one_case(*args, use_separate_process=False, **kwargs):
    if not use_separate_process:
        return benchmark_one_case_internal(*args, **kwargs)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    result_namespace = manager.Namespace()
    p = ctx.Process(target=benchmark_and_write_to_namespace,
                    args=(result_namespace, *args),
                    kwargs=kwargs)
    p.start()
    p.join()
    if p.exitcode != 0:
        return -1, -1, [-1], -1, None
    return result_namespace.result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--niter", type=int)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--num-hosts", type=int)
    parser.add_argument("--num-devices-per-host", type=int)
    parser.add_argument("--shard-only",
                        action="store_true",
                        help="Only profile the 2D case. No pipeline "
                        "parallelism.")
    parser.add_argument("--local",
                        action="store_true",
                        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--profile-driver-time",
                        action="store_true",
                        help="Profile the execution time on the driver instead "
                        "of the workers.")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    os.makedirs("tmp", exist_ok=True)

    # Make eval work smoothly
    from benchmark_parallel_utils import *
    from suite_manual_gpt import GPTModelConfig
    from suite_manual_moe import MoEModelConfig
    from suite_wresnet import WResNetModelConfig
    from suite_unet import UNetModelConfig
    case = eval(args.case)

    result = benchmark_one_case(args.model,
                                case,
                                args.niter,
                                args.num_hosts,
                                args.num_devices_per_host,
                                shard_only=args.shard_only,
                                local=args.local,
                                profile_driver_time=args.profile_driver_time,
                                disable_tqdm=args.disable_tqdm)

    print(result)
