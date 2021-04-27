# Benchmark A single transformer layer
- Megatron-LM
```
python3 -m torch.distributed.launch --nproc_per_node 4 benchmark_megatron.py
```

- Parax
```
python3 benchmark_parax.py
```


## With nvprof
- Megatron-LM
```
nvprof --profile-child-processes python3 -m torch.distributed.launch --nproc_per_node 4 benchmark_megatron.py &> megatron.nvprof
```
- Parax
```
nvprof python3 benchmark_parax.py &> parax.nvprof
```
