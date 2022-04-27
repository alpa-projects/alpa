"""Generate the profiling result database.

Usage:
AWS p3.16:
python3 gen_prof_database.py --max-comm-size-intra-node 32 --max-comm-size-inter-node 29

AWS p4.24:
python3 gen_prof_database.py --efa --max-comm-size-intra-node 33 --max-comm-size-inter-node 30 --max-fail-retry 8
"""

import ray
import argparse

import jax
from alpa import DeviceCluster, ProfilingResultDatabase, global_config
from alpa.util import run_cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-key", type=str, default="default")
    parser.add_argument("--efa", action="store_true")
    parser.add_argument("--filename", type=str, default="prof_database.pkl",
        help="The filename of the output database")
    parser.add_argument("--max-comm-size-intra-node", type=int, required=True,
        help="Run profiling for communication up to 2^x bytes within a node, where x "
             "is this argument")
    parser.add_argument("--max-comm-size-inter-node", type=int, required=True,
        help="Run profiling for communication up to 2^x bytes cross nodes, where x "
             "is this argument")
    parser.add_argument("--cache-filename", type=str,
        default="/home/ubuntu/efs/alpa/benchmark/alpa/tmp/hlo_op_cost_dict.pkl",
        help="The filename of the temporary cache. This should be an "
             "absolute path on a network file system that can be accessed by "
             "ray workers on all nodes.")
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

    prof_database = cluster.profile_all(
        args.cluster_key,
        args.max_comm_size_intra_node,
        args.max_comm_size_inter_node,
        max_fail_retry=args.max_fail_retry,
        cache_filename=args.cache_filename)
    prof_database.save(args.filename)
    print(f"Save profiling database to {args.filename}")
