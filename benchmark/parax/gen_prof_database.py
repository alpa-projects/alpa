import ray
import argparse

from parax import DeviceCluster, ProfilingResultDatabase


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_key", type=str, default="p3.16")
    parser.add_argument("--filename", type=str, default="prof_database.pkl")
    args = parser.parse_args()
    ray.init(address="auto")

    comm_size_range = (0, 29)
    cluster = DeviceCluster()
    prof_database = cluster.profile_all(args.cluster_key, comm_size_range=comm_size_range)
    prof_database.save(args.filename)
    print(f"Save profiling database to {args.filename}")
