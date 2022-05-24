import os

import torch
import numpy as np
from metaseq.checkpoint_utils import load_checkpoint_to_cpu, torch_load_cpu



meta_data_path = "/home/ubuntu/dataset/175B-resharded/metadata/metadata_part-7"
m = torch.load(meta_data_path)

key = "decoder.layers.95.self_attn.k_proj.weight"

weight = np.load(f"/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B_resharded/numpy_weights/{key}")
# print(weight)
print(weight.shape)
shape = (1536, 12288)
print(weight[1536*7:, :])




# weight_paths = [
#     "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-0.pt",
#     "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-1.pt",
#     "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-2.pt",
#     "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-3.pt",
#     "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-4.pt",
#     "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-5.pt",
#     "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-6.pt",
#     "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-7.pt",
# ]
# # raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/"
# # raw_weights_path = "/home/ubuntu/opt/reshard.pt"
# # raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/test2/checkpoint_last-model_part-0-shard0.pt"
# # raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B_resharded/consolidated.pt"
# # raw_weights_path = "/home/ubuntu/dataset/175B-resharded/reshard-model_part-0-shard0.pt"
# # # raw_weights_path1 = "/home/ubuntu/parax-efs/pycharm/opt/opt_metaseq_125m/model/restored.pt"
# # # raw_weights_path2 = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/30B/reshard-model_part-0.pt"
# # # raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/opt_metaseq_2700m/model/restored.pt"
# # # raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/1.3B/checkpoint_last-model_part-0-shard0.pt"
# # # raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/2.7B/reshard-model_part-0.pt"
# # vocab_file = os.path.join("/home/ubuntu/parax-efs/pycharm/opt/raw_weights/1.3B/gpt2-vocab.json")
# # merges_file = os.path.join("/home/ubuntu/parax-efs/pycharm/opt/raw_weights/1.3B/gpt2-merges.txt")
# # arg_overrides={"vocab_filename": vocab_file, "merge_filename": merges_file}
#
#
# printkeys = ["flat_param_0",
#              "decoder.layers.0.flat_param_0",
#              "decoder.layers.6.flat_param_0",
#              "decoder.layers.17.flat_param_0",
#              "decoder.layers.25.flat_param_0",
#              "decoder.layers.38.flat_param_0",
#              "decoder.layers.44.flat_param_0",
#              "decoder.layers.51.flat_param_0",
#              "decoder.layers.69.flat_param_0",
#              "decoder.layers.73.flat_param_0",
#              "decoder.layers.84.flat_param_0",
#              "decoder.layers.95.flat_param_0"]
# for i, part_path in enumerate(weight_paths):
#     part = torch_load_cpu(part_path)
#     print(f"Inspect model part {i} at {part_path}...")
#     for key in printkeys:
#         print(part["model"][key][1:15])
