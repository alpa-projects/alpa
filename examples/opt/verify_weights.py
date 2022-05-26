import os

import torch
import numpy as np
from metaseq.checkpoint_utils import load_checkpoint_to_cpu, torch_load_cpu


# Replace this with the path to the eight parts
path = "/home/ubuntu/dataset/175B-resharded/renamed/"
weigth_paths = [os.path.join(path, f"reshard-model_part-{part_idx}.pt") for part_idx in range(8)]

result_file = "/tmp/results"
np.random.seed(9)


with open(result_file, "w") as rf:
    for i, weight_path in enumerate(weigth_paths):
        print(f"Processing MP part {i}...")
        part = torch_load_cpu(weight_path)
        assert "model" in part.keys()
        flat_params = part["model"]
        for param in flat_params:
            rf.write(f"Verifying {param} with size: {param.size}:")
            positions = np.random.randint(low=0, high=param.size, size=500)
            rf.write(f">>> positions:: {positions}")
            rf.write(f">>> weights: {param[positions]}")
