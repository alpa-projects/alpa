from metaseq.checkpoint_utils import load_checkpoint_to_cpu, torch_load_cpu
import torch
import json


weight_path = [
    "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-0.pt",
    "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-1.pt",
    "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-2.pt",
    "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-3.pt",
    "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-4.pt",
    "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-5.pt",
    "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-6.pt",
    "/home/ubuntu/dataset/175B-resharded/renamed/reshard-model_part-7.pt",
]
weight_shard_path = [
    "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-0-shard0.pt",
    "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-1-shard0.pt",
    "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-2-shard0.pt",
    "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-3-shard0.pt",
    "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-4-shard0.pt",
    "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-5-shard0.pt",
    "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-6-shard0.pt",
    "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-7-shard0.pt",
]

for i, shard_path in enumerate(weight_shard_path):
    part_path = weight_path[i]
    shard = torch_load_cpu(shard_path)
    metadata = shard["shard_metadata"]
    save_path = f"/home/ubuntu/dataset/175B-resharded/metadata_part-{i}"
    torch.save(metadata, save_path)
