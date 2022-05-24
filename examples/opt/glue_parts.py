import os


import torch
from metaseq.checkpoint_utils import load_checkpoint_to_cpu, torch_load_cpu

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

meta_data_path = [
    "/home/ubuntu/dataset/175B-resharded/metadata/metadata_part-0",
    "/home/ubuntu/dataset/175B-resharded/metadata/metadata_part-1",
    "/home/ubuntu/dataset/175B-resharded/metadata/metadata_part-2",
    "/home/ubuntu/dataset/175B-resharded/metadata/metadata_part-3",
    "/home/ubuntu/dataset/175B-resharded/metadata/metadata_part-4",
    "/home/ubuntu/dataset/175B-resharded/metadata/metadata_part-5",
    "/home/ubuntu/dataset/175B-resharded/metadata/metadata_part-6",
    "/home/ubuntu/dataset/175B-resharded/metadata/metadata_part-7",
]


metadatas =[]
for meta_data_p in meta_data_path:
    metadata = torch.load(meta_data_p)
    metadatas.append(metadata)

print(metadatas)