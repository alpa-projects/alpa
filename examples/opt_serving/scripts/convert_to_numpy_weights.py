"""Convert Metaseq's OPT model weights into Alpa numpy weights."""
import numpy as np
import os

from metaseq.file_io import torch_load_cpu


def save_numpy(weight_dict, to_folder):
    os.makedirs(os.path.dirname(to_folder), exist_ok=True)
    weights_folder = os.path.join(to_folder, "numpy_weights")
    os.makedirs(weights_folder, exist_ok=True)
    for tensor_name, tensor in weight_dict.items():
        print(f"writing tensor {tensor_name} with shape {tensor.shape}")
        t = tensor.cpu().detach().numpy()
        with open(weights_folder + "/" + tensor_name, "wb") as g:
            np.save(g, t)


def worker_main():
    PATH = "/home/ubuntu/parax-efs/pycharm/opt/opt_metaseq_30000m/model/"
    # Path to the single
    consolidated_weight = os.path.join(PATH, "restored.pt")
    state = torch_load_cpu(consolidated_weight)
    to_folder = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/30B_resharded/"
    save_numpy(state["model"], to_folder)


if __name__ == "__main__":
    worker_main()
