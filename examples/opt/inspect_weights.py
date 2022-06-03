import os

from metaseq.checkpoint_utils import load_checkpoint_to_cpu, torch_load_cpu

# raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/"
# raw_weights_path = "/home/ubuntu/opt/reshard.pt"
# raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/test2/checkpoint_last-model_part-5-shard0.pt"
# raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B_resharded/consolidated.pt"
# raw_weights_path = "/home/ubuntu/dataset/175B-resharded/reshard-model_part-0-shard0.pt"
# raw_weights_path1 = "/home/ubuntu/parax-efs/pycharm/opt/opt_metaseq_125m/model/restored.pt"
# raw_weight_path1 = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-0-shard0.pt"
raw_weight_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/175B/checkpoint_last-model_part-5-shard1.pt"
# raw_weights_path2 = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/30B/reshard-model_part-0.pt"
# raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/opt_metaseq_2700m/model/restored.pt"
# raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/1.3B/checkpoint_last-model_part-0-shard0.pt"
# raw_weights_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/2.7B/reshard-model_part-0.pt"
vocab_file = os.path.join("/home/ubuntu/parax-efs/pycharm/opt/raw_weights/1.3B/gpt2-vocab.json")
merges_file = os.path.join("/home/ubuntu/parax-efs/pycharm/opt/raw_weights/1.3B/gpt2-merges.txt")
arg_overrides={"vocab_filename": vocab_file, "merge_filename": merges_file}

# state = load_checkpoint_to_cpu(raw_weights_path, arg_overrides=arg_overrides)
state = torch_load_cpu(raw_weight_path)

# state2 = torch_load_cpu(raw_weight_path2)
# state1 = load_checkpoint_to_cpu(raw_weights_path1, arg_overrides=arg_overrides)
# state2 = load_checkpoint_to_cpu(raw_weights_path2, arg_overrides=arg_overrides)
# state = state2
print(state)
with open("info.txt", "w") as info:
    for k, v in state["model"].items():
        info.write(f"{k}\t{v.shape}\n")



print(state["model"])





