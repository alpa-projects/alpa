import os

from metaseq.checkpoint_utils import load_checkpoint_to_cpu, torch_load_cpu

raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/test/checkpoint_last-model_part-0-shard0.pt"
# raw_weights_path = "/home/ubuntu/opt/reshard.pt"
# raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/1.3B/checkpoint_last-model_part-0-shard0.pt"
vocab_file = os.path.join("/home/ubuntu/parax-efs/pycharm/opt/raw_weights/1.3B/gpt2-vocab.json")
merges_file = os.path.join("/home/ubuntu/parax-efs/pycharm/opt/raw_weights/1.3B/gpt2-merges.txt")
arg_overrides={"vocab_filename": vocab_file, "merge_filename": merges_file}

state = load_checkpoint_to_cpu(raw_weights_path, arg_overrides=arg_overrides)
print(state)
with open("info.txt", "w") as info:
    for k, v in state["model"].items():
        info.write(f"{k}\t{v.shape}\n")



print(state["model"])





