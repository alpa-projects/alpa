import numpy as np
import os
from metaseq.checkpoint_utils import torch_load_cpu

weight_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B_resharded/numpy_weights/decoder.layers.15.fc1.weight"
weight_name = weight_path.split("/")[-1]
save_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B_resharded"

full_weight = np.load(weight_path)
shape0 = full_weight.shape[0]
assert shape0 % 8 == 0, "wrong!"

s = shape0 // 8

problem_positions = [
(1537, 5172),
(1537, 5173),
(1537, 5174),
(1537, 5175),
(1537, 5176),
(1537, 5177),
(1537, 5178),
(1537, 5179),
(1537, 5180),
(1537, 5181),
(1537, 5182),
(1537, 5183),
(1537, 5184),
(1537, 5185),
(1537, 5186),
(1537, 5187),
(1537, 5188),
(1537, 5189),
]

for p in problem_positions:
    dim0, dim1 = p
    print(full_weight[5*s+dim0, dim1])
    full_weight[5*s+dim0, dim1] = np.float16(0.0)

with open(os.path.join(save_path, weight_name), "wb") as f:
    np.save(f, full_weight)



# part5_metadata_path = "/home/ubuntu/parax-efs/pycharm/alpa/examples/opt/verify_part5/metadata_part-5"
# metadata = torch_load_cpu(part5_metadata_path)
#
# names = metadata["param_metadata"][16]["params"]["flat_param_0"]["names"]
# numels = metadata["param_metadata"][16]["params"]["flat_param_0"]["numels"]
#
# total_params = sum(numels)
# total_params_before = 0
# for name, numel in zip(names, numels):
#     if name == "fc1.weight":
#         break
#     total_params_before += numel
#
# num_fsdp_shard = 124
# pad = num_fsdp_shard - total_params % num_fsdp_shard
# total_params = total_params + pad
# num_params_per_shard = total_params // num_fsdp_shard
#
# total_params_before += 1537 * full_weight.shape[1]+ 5172
#
# shard_idx = total_params_before // num_params_per_shard
# element_idx = total_params_before % num_params_per_shard
#
# shard_path = f"/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-5-shard{shard_idx}.pt"
# shard = torch_load_cpu(shard_path)
# shard_flat_param = shard["model"]["decoder.layers.15.flat_param_0"]
#
# for ele in range(element_idx, element_idx + len(problem_positions)):
#     print(shard_flat_param[ele].numpy())
