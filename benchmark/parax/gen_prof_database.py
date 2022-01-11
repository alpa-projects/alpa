"""Generate the profiling result database."""
import ray
import argparse

import jax
from parax import DeviceCluster, ProfilingResultDatabase
from parax.util import run_cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_key", type=str, default="default")
    parser.add_argument("--filename", type=str, default="prof_database.pkl")
    args = parser.parse_args()
    ray.init(address="auto")

    # Initialize a useless jax GPU backend in the driver script.
    # This GPU backend takes 300MB GPU memory to store the CUDA context.
    # This simulates the environment of our benchmark scripts and
    # can make the profiling of available memory more accurate.
    # TODO(lmzheng): Modify jax so it does not allocate this useless CUDA context.
    jax.config.update('jax_platform_name', 'cpu')
    _ = jax.numpy.ones(1)

    run_cmd("mkdir -p tmp")

    comm_size_range = (0, 30)
    cluster = DeviceCluster()
    # Must use an absolute efs filename because ray actors are on distributed workers.
    prof_database = cluster.profile_all(args.cluster_key, comm_size_range=comm_size_range,
        cache_filename="/home/ubuntu/efs/parax/benchmark/parax/tmp/hlo_op_cost_dict.pkl")
    prof_database.save(args.filename)
    print(f"Save profiling database to {args.filename}")
