"""Inspect and edit a profiling database."""
import argparse

from alpa import DeviceCluster, ProfilingResultDatabase
from alpa.util import run_cmd

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
    print()

    mesh_result = prof_database.query("default", (2, 8))
    print(mesh_result)
