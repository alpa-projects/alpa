import numpy as np
import os
from metaseq.checkpoint_utils import torch_load_cpu

# Replace this with the path to the eight parts
path = "/home/ubuntu/dataset/175B-resharded/renamed/"
weight_paths = [os.path.join(path, f"reshard-model_part-{part_idx}.pt") for part_idx in range(8)]

result_file = "/tmp/results"
np.random.seed(9)

with open(result_file, "w") as rf:
    rf.write(f"All MP parts: {weight_paths}.\n")
    for i, weight_path in enumerate(weight_paths):
        rf.write(f"> Processing MP part {i}...\n")
        part = torch_load_cpu(weight_path)
        assert "model" in part.keys()
        flat_params = part["model"]
        for param in flat_params:
            param_size = flat_params[param].size()
            rf.write(f">> Verifying {param} with size: {param_size}:\n")
            positions = np.random.randint(low=0, high=param_size, size=50)
            rf.write(f">>> positions:: {positions}\n")
            rf.write(f">>> weights: {flat_params[param][positions]}\n")
