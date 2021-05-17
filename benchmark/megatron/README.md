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
- Megatron-LM
  - MLP
  ```
  python3 -m torch.distributed.launch --nproc_per_node 4 benchmark_mlp.py
  ```

## With nvprof
- Megatron-LM
```
nvprof --profile-child-processes python3 benchmark_mlp.py &> megatron.nvprof
```
- Parax
```
nvprof python3 benchmark_parax.py &> parax.nvprof
```
