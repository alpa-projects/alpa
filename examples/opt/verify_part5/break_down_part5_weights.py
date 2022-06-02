import os

import numpy as np
from metaseq.checkpoint_utils import torch_load_cpu

part5_path = "/home/ubuntu/dataset/175B-resharded/reshard-model_part-5-shard0.pt"
part5_metadata = "/home/ubuntu/dataset/175B-resharded/metadata/metadata_part-5"

state = torch_load_cpu(part5_path)
metadata = torch_load_cpu(part5_metadata)

assert len(metadata["param_metadata"]) + 1 == len(state["model"]), "size is not matching"

save_path = "/tmp/part5_tensors"
os.makedirs(save_path, exist_ok=True)
for index, m in enumerate(metadata["param_metadata"]):
    fsdp_path = m["fsdp_path"]
    names = m["params"]["flat_param_0"]["names"]
    shapes = m["params"]["flat_param_0"]["shapes"]
    numels = m["params"]["flat_param_0"]["numels"]
    if len(fsdp_path) > 0:
        flat_param_name = fsdp_path + "." +  "flat_param_0"
    else:
        flat_param_name = "flat_param_0"
    print(f"Processing flat param: {flat_param_name}")
    assert flat_param_name in state["model"], f"Missing a key {flat_param_name}"
    flat_param_state = state["model"][flat_param_name]
    assert flat_param_state.numel() == sum(numels), "the flat param size does not match metadata."

    # split the flat params into tensors
    start = 0
    for i, name in enumerate(names):
        tensor_name = fsdp_path + "." + name if len(fsdp_path) > 0 else name
        tensor_shape = shapes[i]
        tensor_numel = numels[i]
        tensor = flat_param_state[start:start+tensor_numel].reshape(tensor_shape).numpy()
        start = start + tensor_numel
        tensor_save_path = os.path.join(save_path, tensor_name)
        print(f"> Save tensor {tensor_name} at {tensor_save_path}")
        with open(tensor_save_path, "wb") as f:
            np.save(f, tensor)

    assert start == flat_param_state.numel()
