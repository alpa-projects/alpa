from metaseq.checkpoint_utils import torch_load_cpu
import torch


for shard_idx in range(124):
    shard_path = f"/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-5-shard{shard_idx}.pt"
    state = torch_load_cpu(shard_path)
    flat_param = state["model"]["decoder.layers.15.flat_param_0"]
    print(f"Shard {shard_idx}, max: {torch.max(flat_param)}")