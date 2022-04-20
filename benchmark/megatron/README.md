# Benchmark Megatron-LM

## Requirements
```
# torch 1.8.0 and CUDA 11.1
pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install ninja

# Install Megatron
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
echo 'export PYTHONPATH=$PYTHONPATH:~/efs/Megatron-LM' >> ~/.bashrc   # use your own path
source ~/.bashrc

# Install Apex
git clone https://github.com/NVIDIA/apex
cd apex
# Comment out the raised RuntimeError in setup.py if you get errors running the following command.
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Instructions
### Single Node
```
# MLP
python3 benchmark_mlp.py --nproc_per_node 4
# Transfomer layer
python3 benchmark_transformer_layer.py --nproc_per_node 4
# GPT
python3 benchmark_gpt_bert.py --nproc_per_node 1 --suite gpt.tmp
python3 benchmark_gpt_bert.py --nproc_per_node 8 --suite gpt.tmp
```

### Multiple Nodes
```
# on node 0
python3 benchmark_gpt_bert.py --suite gpt.tmp --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_port 11000 --master_addr 172.31.16.139
# on node 1
python3 benchmark_gpt_bert.py --suite gpt.tmp --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_port 11000 --master_addr 172.31.16.139
```

For other models, replace `benchmark_gpt_bert.py` with the corresponding filenames.

### With nvprof
```
nvprof --profile-child-processes python3 benchmark_mlp.py --nproc_per_node 4 &> megatron.prof
```
