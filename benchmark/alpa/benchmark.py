"""The entry point of intra-op + inter-op parallelism benchmark."""
import os
import argparse
from datetime import datetime
import time

import numpy as np

from alpa.util import (write_tsv, get_num_hosts_and_num_devices, to_str_round,
                       GB)
from collections import namedtuple
from benchmark_one_case import benchmark_one_case
import suite_auto_gpt
import suite_auto_moe
import suite_manual_gpt
import suite_manual_moe
import suite_wresnet
import suite_inference_gpt
from benchmark_parallel_utils import (BenchmarkCase, ShardParallelArgs, UniformParallelArgs)
#from suite_manual_gpt import GPTModelConfig

#B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
GPTModelConfig = namedtuple(
    "GPTModelConfig",
    ["seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size"])

gpt_specs = {
    #                      S，   H,   L,  head,   V,
    "125M": GPTModelConfig(1024, 768, 12, 12, 51200),
    "350M": GPTModelConfig(1024, 1024, 24, 16, 51200),
    "760M": GPTModelConfig(1024, 1536, 24, 16, 51200),
    "1.3B": GPTModelConfig(1024, 2048, 24, 32, 51200),
    "2.6B": GPTModelConfig(1024, 2560, 32, 32, 51200),
    "6.7B": GPTModelConfig(1024, 4096, 32, 32, 51200),
    "15B": GPTModelConfig(1024, 5120, 48, 40, 51200),
    "39B": GPTModelConfig(1024, 8192, 48, 64, 51200),
    "76B": GPTModelConfig(1024, 10240, 60, 80, 51200),
}


benchmark_suites = {
    "gpt.tmp": suite_manual_gpt.tmp_suite,
    "gpt.tmp_auto": suite_auto_gpt.tmp_suite,
    "gpt.perf_test_fast_2d": suite_manual_gpt.perf_test_fast_2d_suite,
    "gpt.perf_test_manual": suite_manual_gpt.perf_test_suite,
    "gpt.perf_test_auto": suite_auto_gpt.perf_test_suite,
    "gpt.grid_search_auto": suite_auto_gpt.grid_search_suite,
    "gpt.correctness_test_auto": suite_auto_gpt.correctness_test_suite,
    "gpt_inference.profile": suite_inference_gpt.profile_suite,
    "gpt_no_embedding_inference.profile": suite_inference_gpt.profile_suite,
    "moe.tmp": suite_manual_moe.tmp_suite,
    "moe.tmp_auto": suite_auto_moe.tmp_suite,
    "moe.perf_test_fast_2d": suite_manual_moe.perf_test_fast_2d_suite,
    "moe.perf_test_auto": suite_auto_moe.perf_test_suite,
    "moe.grid_search_auto": suite_auto_moe.grid_search_suite,
    "wresnet.perf_test_2d": suite_wresnet.perf_test_2d_suite,
    "wresnet.perf_test_auto": suite_wresnet.perf_test_auto_suite,
    "wresnet.grid_search_auto": suite_wresnet.grid_search_auto_suite,
}


def benchmark_suite(suite_name,
                    num_hosts,
                    num_devices_per_host,
                    input_gpt_layer,
                    input_batch_size,
                    input_micro_batches,
                    reduce_scatter,
                    dp,op,recomputation, change_gpt_layer,
                    exp_name="default",
                    niter=3,
                    shard_only=False,
                    local=False,
                    profile_driver_time=False,
                    profile_stage_execution_time=False,
                    disable_tqdm=False,
                    use_separate_process=True):
    num_gpus = num_hosts * num_devices_per_host

    if local:
        assert shard_only, ("Only shard-only mode is supported for execution "
                            "on local GPUs.")

    if num_gpus not in benchmark_suites[suite_name]:
        return
    suite = benchmark_suites[suite_name][num_gpus]
    #print("suit is {},suit[0]is {}".format(suite,benchmark_case))
    os.makedirs("tmp", exist_ok=True)

    model_type = suite_name.split(".")[0]
    output_name = f"{exp_name}.tsv"

    # Run all cases
    for benchmark_case in suite:
        
        if shard_only:
            assert dp*op == num_gpus, ("dp*op != num_gpus.")

            if change_gpt_layer == True:
                gpt_config = GPTModelConfig(1024, 4096, input_gpt_layer, 32, 51200)

            else:
                if num_gpus == 1:
                    gpt_config = gpt_specs["350M"]
                if num_gpus == 2:
                    gpt_config = gpt_specs["760M"]
                if num_gpus == 4:
                    gpt_config = gpt_specs["1.3B"]
                if num_gpus == 8:
                    gpt_config = gpt_specs["2.6B"]
                if num_gpus == 16:
                    gpt_config = gpt_specs["6.7B"]     
                               
            # B, model, NB, PM, (RS, Remat, 3D Config, FM)
            benchmark_case_new= BenchmarkCase(input_batch_size,
                                      gpt_config,
                                      input_micro_batches,
                                      "uniform",
                                      UniformParallelArgs(reduce_scatter, recomputation, dp, op, 1, True))
        
        else:
            benchmark_case_new=benchmark_case
        

        model_config = benchmark_case_new.model_config
        num_micro_batches = benchmark_case_new.num_micro_batches
        parallel_args = benchmark_case_new.parallel_args


        # Run one case
        print("Working on case: {}".format(str(benchmark_case_new)))
        
        
        result = benchmark_one_case(model_type,
                                    benchmark_case_new,
                                    niter,
                                    num_hosts,
                                    num_devices_per_host,
                                    shard_only=shard_only,
                                    local=local,
                                    profile_driver_time=profile_driver_time,
                                    disable_tqdm=disable_tqdm,
                                    use_separate_process=use_separate_process)

        (parameter_count, peak_mem, latencies, tflops, metadata) = result

        heads = [
            "Type", "Model Config", "#Microbatch", "#GPU", "Parallel Config",
            "Mean Time (s)", "Std Time (s)", "#Params (Billion)", "TFLOPs",
            "Peak Mem (GB)", "Metadata"
        ]
        values = [
            model_type, model_config, num_micro_batches, num_gpus,
            parallel_args, f"{np.mean(latencies):.3f}",
            f"{np.std(latencies):.3f}", f"{parameter_count/1e9:.3f}B",
            f"{tflops:.2f}", f"{peak_mem/GB:.3f}",
            to_str_round(metadata, 2)
        ]
        write_tsv(heads, values, output_name)

        time.sleep(0.1)  # for ctrl+c to work


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",
                        choices=list(benchmark_suites.keys()),
                        type=str,
                        required=True)
    parser.add_argument("--niter",
                        type=int,
                        default=3,
                        help="The number of benchmark iterations")
    parser.add_argument("--num-hosts", type=int, default=None)
    parser.add_argument("--num-devices-per-host", type=int, default=None)
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
    parser.add_argument(
        "--profile-stage-execution-time",
        action="store_true",
        help="Profile the execution timestamps of each pipeline "
        "stage")
    parser.add_argument("--no-separate-process",
                        action="store_false",
                        help="Do not launch separate processes for benchmark. "
                        "Errors in a single case will terminate this "
                        "script.",
                        dest="use_separate_process")
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--num_gpt_layer", type=int, default=1)
    parser.add_argument("--num_batch_size", type=int, default=4)
    parser.add_argument("--num_micro_batches", type=int, default=1)
    parser.add_argument("--reduce_scatter",
                        action="store_true",
                        help="Prefer_reduce_scatter = True.")
    parser.add_argument("--dp", type=int, default=4)
    parser.add_argument("--op", type=int, default=1)
    parser.add_argument("--recomputation",
                        action="store_true",
                        help="remat = True.")
    parser.add_argument("--change_gpt_layer",
                        action="store_true",
                        help="if change_gpt_layer = False, it will use the default gpt config")
    args = parser.parse_args()

    num_hosts, num_devices_per_host = get_num_hosts_and_num_devices(args)

    benchmark_suite(args.suite, num_hosts, num_devices_per_host,
                    args.num_gpt_layer, args.num_batch_size,
                    args.num_micro_batches,
                    args.reduce_scatter,args.dp,args.op,
                    args.recomputation, args.change_gpt_layer,
                    args.exp_name,
                    args.niter, args.shard_only, args.local,
                    args.profile_driver_time, args.disable_tqdm,
                    args.use_separate_process,)
