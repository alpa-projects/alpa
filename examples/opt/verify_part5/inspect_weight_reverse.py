import numpy as np
from metaseq.checkpoint_utils import torch_load_cpu

weight_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B_resharded_v2/numpy_weights/decoder.layers.15.fc1.weight"

full_weight = np.load(weight_path)
shape0 = full_weight.shape[0]
assert shape0 % 8 == 0, "wrong!"

s = shape0 // 8
mp5_weight = full_weight[5*s:6*s, :]

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
    print(mp5_weight[p])


part5_metadata_path = "/home/ubuntu/parax-efs/pycharm/alpa/examples/opt/verify_part5/metadata_part-5"
metadata = torch_load_cpu(part5_metadata_path)

names = metadata["param_metadata"][16]["params"]["flat_param_0"]["names"]
numels = metadata["param_metadata"][16]["params"]["flat_param_0"]["numels"]

total_params = sum(numels)
total_params_before = 0
for name, numel in zip(names, numels):
    if name == "fc1.weight":
        break
    total_params_before += numel

# total_params_before += 1537 * 6144 + 5172



num_fsdp_shard = 124

pad = num_fsdp_shard - total_params % num_fsdp_shard
total_params = total_params + pad
num_params_per_shard = total_params // num_fsdp_shard

total_params_before += 1537 * full_weight.shape[1]+ 5172

shard_idx = total_params_before // num_params_per_shard
element_idx = total_params_before % num_params_per_shard

shard_path = f"/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-5-shard{shard_idx}.pt"
shard = torch_load_cpu(shard_path)
shard_flat_param = shard["model"]["decoder.layers.15.flat_param_0"]

for i, p in enumerate(problem_positions):
    print(f"numpy at {p} is {mp5_weight[p]}, at shard position {element_idx + i} is {shard_flat_param[element_idx+i].numpy()}")
