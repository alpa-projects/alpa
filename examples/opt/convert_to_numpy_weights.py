import numpy as np
import os

from metaseq.dataclass.configs import MetaseqConfig
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


def worker_main(cfg: MetaseqConfig):
    raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B_resharded/consolidated.pt"
    state = torch_load_cpu(raw_weights_path)
    to_folder = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B_resharded/"
    save_numpy(state["model"], to_folder)


if __name__ == "__main__":
    worker_main(None)
