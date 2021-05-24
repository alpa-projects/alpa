# Benchmark Megatron-LM

## Requirements
```
pip3 install ninja

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip3 install -e .
echo 'export PYTHONPATH=$PYTHONPATH:~/Megatron-LM' >> ~/.bashrc
```

## Instructions
### Single Node
```
# Transfomer layer
python3 benchmark_transformer_layer.py --nproc_per_node 4
# MLP
python3 benchmark_mlp.py --nproc_per_node 4
```

### Multiple Nodes
```
# on node 0
python3 benchmark_transformer_layer.py --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr 172.31.16.139 --master_port 11000
# on node 1
python3 benchmark_transformer_layer.py --nproc_per_node 4 --nnodes 2 --node_rank 1 --master_addr 172.31.16.139 --master_port 11000
```

### With nvprof
```
nvprof --profile-child-processes python3 benchmark_mlp.py &> megatron.prof
```
