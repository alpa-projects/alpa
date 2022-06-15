"""Benchmark one case of inter-op + intra-op parallelism."""
import argparse
import pickle

import jax
import ray

from alpa import init, global_config
from alpa.util import run_cmd, disable_tqdm_globally

from benchmark_3d_one_case_gpt_bert import benchmark_gpt_bert_internal
from benchmark_3d_one_case_moe import benchmark_moe_internal
from benchmark_3d_one_case_wresnet import benchmark_wresnet_internal

TMP_PICKLE_FILE_NAME = "/tmp/tmp_transfer.pkl"


def benchmark_one_case(model,
                       case,
                       niter,
                       num_hosts,
                       num_devices_per_host,
                       use_separate_process=False,
                       dump_result=False,
                       disable_tqdm=False):
    if disable_tqdm:
        disable_tqdm_globally()

    if not use_separate_process:
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
            # Due to legacy issues, we turn off auto-tuning. Although the performance
            # will be much better if we turn it on
            global_config.xla_gpu_autotune_level = 0
            result = benchmark_wresnet_internal(case, niter, num_hosts,
                                                num_devices_per_host)
        else:
            raise ValueError(f"Invalid model: {model}")
    else:
        # Launch a new process for benchmark to isolate errors.
        # Get the return data via pickle.
        run_cmd(f"rm -rf {TMP_PICKLE_FILE_NAME}")
        cmd = (f"python3 -u benchmark_3d_one_case.py "
               f"--model {model} "
               f"--niter {niter} "
               f'--case "{case}" '
               f"--num-hosts {num_hosts} "
               f"--num-devices-per-host {num_devices_per_host} "
               f"--dump-result ")
        if disable_tqdm:
            cmd += "--disable-tqdm "
        ret = run_cmd(cmd)
        if ret == 0:
            result = pickle.load(open(TMP_PICKLE_FILE_NAME, "rb"))
        else:
            result = -1, -1, [-1], -1, -1, None, None, None, None, None, None

    if dump_result:
        pickle.dump(result, open(TMP_PICKLE_FILE_NAME, "wb"))

    return result


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

    run_cmd("mkdir -p tmp")
    case = eval(args.case)
    benchmark_one_case(args.model,
                       case,
                       args.niter,
                       args.num_hosts,
                       args.num_devices_per_host,
                       use_separate_process=False,
                       dump_result=args.dump_result,
                       disable_tqdm=args.disable_tqdm)
