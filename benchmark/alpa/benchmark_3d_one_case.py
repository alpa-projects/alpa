"""Benchmark one case of inter-op + intra-op parallelism."""
import os
import argparse
import multiprocessing as mp

from alpa import init, global_config
from alpa.util import disable_tqdm_globally

from benchmark_3d_one_case_gpt_bert import benchmark_gpt_bert_internal
from benchmark_3d_one_case_moe import benchmark_moe_internal
from benchmark_3d_one_case_wresnet import benchmark_wresnet_internal


def benchmark_one_case_internal(model,
                                case,
                                niter,
                                num_hosts,
                                num_devices_per_host,
                                disable_tqdm=False):
    if disable_tqdm:
        disable_tqdm_globally()

    init(cluster="ray")

    global_config.use_dummy_value_for_benchmarking = True
    global_config.pipeline_sync_for_timer = True

    # Run benchmark
    if model in ["gpt", "bert"]:
        result = benchmark_gpt_bert_internal(model, case, niter, num_hosts,
                                             num_devices_per_host)
    elif model == "moe":
        result = benchmark_moe_internal(case, niter, num_hosts,
                                        num_devices_per_host)
    elif model == "wresnet":
        global_config.xla_client_mem_fraction = 0.88
        # Due to legacy issues, we turn off auto-tuning. Although the
        # performance will be much better if we turn it on
        global_config.xla_gpu_autotune_level = 0
        result = benchmark_wresnet_internal(case, niter, num_hosts,
                                            num_devices_per_host)
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
        return -1, -1, [-1], -1, -1, None, None, None, None, None, None
    return result_namespace.result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--niter", type=int)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--num-hosts", type=int)
    parser.add_argument("--num-devices-per-host", type=int)
    parser.add_argument("--dump-result",
                        action="store_true",
                        help="Dump results into a temporary pickle file")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    os.makedirs("tmp", exist_ok=True)

    # Make eval work smoothly
    from parallel_option import *
    from suite_manual_gpt import GPTModelConfig
    from suite_manual_moe import MoEModelConfig
    from suite_wresnet import WResNetModelConfig
    case = eval(args.case)

    benchmark_one_case(args.model,
                       case,
                       args.niter,
                       args.num_hosts,
                       args.num_devices_per_host,
                       disable_tqdm=args.disable_tqdm)
