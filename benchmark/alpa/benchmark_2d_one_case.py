"""Benchmark one case of intra-op only parallelism."""
import argparse
import pickle

import jax
import ray

from alpa import global_config, LocalPhysicalDeviceMesh, DeviceCluster
from alpa.util import run_cmd

from benchmark_2d_one_case_gpt_bert import benchmark_gpt_bert_internal
from benchmark_2d_one_case_moe import benchmark_moe_internal
from benchmark_2d_one_case_wresnet import benchmark_wresnet_internal


TMP_PICKLE_FILE_NAME = "/tmp/tmp_transfer.pkl"


def benchmark_one_case(model, case, niter,
                       num_hosts, num_devices_per_host,
                       local, use_separate_process,
                       dump_result=False):
    if not use_separate_process:
        if model == "wresnet":
            global_config.xla_client_mem_fraction = 0.88
            global_config.xla_gpu_autotune_level = 0

        # Launch physical mesh
        if local:
            assert num_hosts == 1
            physical_mesh = LocalPhysicalDeviceMesh(jax.local_devices()[:num_devices_per_host])
        else:
            ray.init(address="auto", ignore_reinit_error=True)
            device_cluster = DeviceCluster()
            physical_mesh = device_cluster.get_physical_mesh(
                list(range(num_hosts)), num_devices_per_host)
            jax.config.update('jax_platform_name', 'cpu')

        global_config.use_dummy_value_for_benchmarking = True
        global_config.shard_parallel_sync_for_timer = True

        # Run benchmark
        if model in ["gpt", "bert"]:
            result = benchmark_gpt_bert_internal(physical_mesh, model, case, niter)
        elif model == "moe":
            result = benchmark_moe_internal(physical_mesh, case, niter)
        elif model == "wresnet":
            result = benchmark_wresnet_internal(physical_mesh, case, niter)
        else:
            raise ValueError(f"Invalid model: {model}")

        physical_mesh.shutdown()
    else:
        # Launch a new process for benchmark to isolate errors.
        # Get the return data via pickle.
        run_cmd(f"rm -rf {TMP_PICKLE_FILE_NAME}")
        ret = run_cmd("python3 benchmark_2d_one_case.py "
                     f"--model {model} "
                     f"--niter {niter} "
                     f'--case "{case}" '
                     f"--num-hosts {num_hosts} "
                     f"--num-devices-per-host {num_devices_per_host} "
                     f"{'--local' if local else ''} "
                     f"--dump-result ")
        if ret == 0:
            result = pickle.load(open(TMP_PICKLE_FILE_NAME, "rb"))
        else:
            result = -1, -1, -1, [-1], -1

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
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--dump-result", action="store_true",
        help="Dump results into a temporary pickle file")
    args = parser.parse_args()

    run_cmd("mkdir -p tmp")
    case = eval(args.case)
    benchmark_one_case(args.model, case, args.niter, args.num_hosts, args.num_devices_per_host,
                       args.local, False, args.dump_result)
