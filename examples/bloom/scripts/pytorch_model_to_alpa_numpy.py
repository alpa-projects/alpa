"""Convert Bloom model weights into Alpa numpy weights."""
import time

import argparse
import os

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig

def torch_load_cpu(path):
    state = torch.load(path, map_location=torch.device("cpu"))
    # If model was trained with fp16, model from loaded state_dict can be moved to fp16
    if not isinstance(state, dict):
        return state
    if "cfg" in state:
        state["cfg"] = recursively_cast_dictconfigs(state["cfg"])
        if (
            state["cfg"]["common"]["fp16"]
            or state["cfg"]["common"]["memory_efficient_fp16"]
        ):
            state["model"] = {k: v.half() for k, v in state["model"].items()}
    return state

def save_numpy(weight_dict, to_folder):
    os.makedirs(to_folder, exist_ok=True)
    for tensor_name, tensor in weight_dict.items():
        print(f"- Writing tensor {tensor_name} with shape {tensor.shape}")
        t = tensor.cpu().detach().numpy()
        if "model." == tensor_name[:6]:
            tensor_name = tensor_name[6:]
        with open(to_folder + "/" + tensor_name, "wb") as g:
            np.save(g, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="/home/ubuntu/consolidated")
    parser.add_argument("--output-folder", type=str, default="/home/ubuntu/175B_np")
    args = parser.parse_args()
    start_time = time.time()
    print("- Reading the weight into memory")
    state = torch_load_cpu(args.ckpt_path)
    #print(state.keys())
    print(f"Done with reading: {time.time() - start_time} seconds")
    save_numpy(state, args.output_folder)
