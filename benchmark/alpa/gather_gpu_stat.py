"""Gather gpu utilization from all nodes."""

import os
import tempfile

import gpustat
import ray


def call_nvidia_smi():
    gpus = gpustat.new_query().gpus
    return [g.utilization for g in gpus]


if __name__ == "__main__":
    ray.init(address="auto")

    host_info = []
    for node in ray.nodes():
        for key in node["Resources"]:
            if key.startswith("node:"):
                host_info.append(node)

    results = []
    for i in range(len(host_info)):
        # Launch a ray actor
        node_resource = "node:" + host_info[i]["NodeManagerAddress"]
        func = ray.remote(resources={node_resource: 1e-3})(call_nvidia_smi)
        results.append(func.remote())
    results = ray.get(results)

    for i in range(len(host_info)):
        print(host_info[i]["NodeManagerAddress"])
        print(results[i])
