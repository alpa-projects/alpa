"""Convert Metaseq's OPT model weights into Alpa numpy weights."""
import time

import argparse
import os
import logging

import numpy as np
from .utils import torch_load_cpu


logger = logging.getLogger(__name__)


def save_numpy(weight_dict, to_folder):
    os.makedirs(os.path.dirname(to_folder), exist_ok=True)
    for tensor_name, tensor in weight_dict.items():
        logger.info(f"- writing tensor {tensor_name} with shape {tensor.shape}")
        t = tensor.cpu().detach().numpy()
        with open(to_folder + "/" + tensor_name, "wb") as g:
            np.save(g, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--singleton-path", type=str, default="/home/ubuntu/consolidated")
    parser.add_argument("--output-folder", type=str, default="/home/ubuntu/175B_np")
    args = parser.parse_args()
    start_time = time.time()
    logger.info("- Reading the weight into memory")
    state = torch_load_cpu(args.sigleton_path)
    logger.info(f"- Done in {time.time() - start_time} seconds")
    save_numpy(state["model"], args.output_folder)
