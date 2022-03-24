"""Generate the profiling result database.

Usage:
AWS p3.16:
python3 gen_prof_database.py --comm-size-max 30

AWS p4.24:
python3 gen_prof_database.py --efa --comm-size-max 32
"""

import ray
import argparse

import jax
from alpa import DeviceCluster, ProfilingResultDatabase, global_config
from alpa.util import run_cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-key", type=str, default="default")
    parser.add_argument("--filename", type=str, default="prof_database.pkl")
    parser.add_argument("--efa", action="store_true")
    parser.add_argument("--comm-size-max", type=int, required=True)
    parser.add_argument("--max-fail-retry", type=int, default=5)
    args = parser.parse_args()

    run_cmd("mkdir -p tmp")
    if args.efa:
        global_config.use_aws_efa = True

    # Initialize a useless jax GPU backend in the driver script.
    # This GPU backend takes 300MB GPU memory to store the CUDA context.
    # This simulates the environment of our benchmark scripts and
    # can make the profiling of available memory more accurate.
    # TODO(lmzheng): Modify jax so it does not allocate this useless CUDA context.
    jax.config.update('jax_platform_name', 'cpu')
    _ = jax.numpy.ones(1)

    # Connect to a ray cluster
    ray.init(address="auto")
    cluster = DeviceCluster()

    # Must use an absolute efs filename because ray actors are on distributed workers.
    prof_database = cluster.profile_all(
        args.cluster_key,
        comm_size_range=(0, args.comm_size_max + 1),
        max_fail_retry=args.max_fail_retry,
        cache_filename="/home/ubuntu/efs/alpa/benchmark/alpa/tmp/hlo_op_cost_dict.pkl")
    prof_database.save(args.filename)
    print(f"Save profiling database to {args.filename}")
