"""Benchmark one case of inter-op + intra-op parallelism."""
import argparse
import pickle

import jax
import ray

from alpa import global_config
from alpa.util import run_cmd, get_ray_namespace_str, disable_tqdm_globally

from unet_benchmark_3d_one_case_unet import benchmark_unet_internal

import tensorflow as tf

def benchmark_one_case(model, case, niter,
                       num_hosts, num_devices_per_host,
                       use_separate_process=False,
                       dump_result=False, disable_tqdm=False):

    disable_tqdm_globally()
    TMP_PICKLE_FILE_NAME = f"./tmp_transfer_{num_hosts}_{num_devices_per_host}.pkl"
    if not use_separate_process:
        ray.init(address="auto", ignore_reinit_error=True,
                    namespace=get_ray_namespace_str())
        tf.config.experimental.set_visible_devices([], 'GPU')
        jax.config.update('jax_platform_name', 'cpu')
        global_config.use_dummy_value_for_benchmarking = True

        result = benchmark_unet_internal(case, niter, num_hosts, num_devices_per_host)

        ray.shutdown()
    else:
        # Launch a new process for benchmark to isolate errors.
        # Get the return data via pickle.
        print("run in seperate process")
        run_cmd(f"rm -rf {TMP_PICKLE_FILE_NAME}")
        cmd = (f"python3 -u unet_benchmark_3d_one_case.py "
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
            result = -1, -1, -1, [-1], -1, -1, None, None, None, None, None, None

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
    parser.add_argument("--dump-result", action="store_true",
        help="Dump results into a temporary pickle file")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    run_cmd("mkdir -p tmp")
    case = eval(args.case)
    benchmark_one_case(args.model, case, args.niter,
                       args.num_hosts, args.num_devices_per_host,
                       use_separate_process=False, dump_result=args.dump_result,
                       disable_tqdm=args.disable_tqdm)
