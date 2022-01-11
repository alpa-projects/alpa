"""Inspect and edit a profiling database."""
import argparse

from parax import DeviceCluster, ProfilingResultDatabase
from parax.util import run_cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="prof_database.pkl")
    args = parser.parse_args()

    prof_database = ProfilingResultDatabase()
    prof_database.load(args.filename)

    # Do some editing
    #prof_database.insert_dummy_mesh_result("default", (8, 8))
    #prof_database.save(args.filename)

    # Print results
    print("Meshes:")
    print(list(prof_database.data.keys()))
