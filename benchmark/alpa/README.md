## Instructions

1. Start ray cluster.
```
# On head node
ray start --head
# (Optional) : launch worker nodes
```

2. Run benchmark.
```
python3 benchmark_2d.py --suite gpt.fast_test
```

## Generate Profiling Database
1. Start ray cluster.

2. Run profiling to generate the database for the HLO instruction cost model.
```
python3 gen_prof_database.py
```


## Santity Checks

```
python3 benchmark_3d.py --suite gpt.test
```

Expected output on AWS p3.16 (Feb.17, 2022, commit: d7dbb2b3ef)
```
mkdir -p tmp
Working on case: (32, 1024, 2560, 32, 32, 51200, 2, 2, 1, 4, 2, 4, False, True, True, 'uniform_layer_gpipe', None)
rm -rf /tmp/tmp_transfer.pkl
python3 -u benchmark_3d_one_case.py --model gpt --niter 3 --case "(32, 1024, 2560, 32, 32, 51200, 2, 2, 1, 4, 2, 4, False, True, True, 'uniform_layer_gpipe', None)" --num-hosts 1 --num-devices-per-host 8 --dump-result
mkdir -p tmp
2022-02-17 18:17:32,255 INFO worker.py:840 -- Connecting to existing Ray cluster at address: 172.31.31.37:6379
 - Prepare input: 2.83 s
 - Create train state: 3.56 s
-------------------- Layer slicing stats --------------------
layer_num: 2
 - Number of Jaxpr eqns in each stage:
Layer 0: #eqns=41, flop=24.245 TFlop, #heavy_ops=98
Layer 1: #eqns=49, flop=24.227 TFlop, #heavy_ops=97
 - Invars of each stage:
Layer 0 has inputs:
Layer 1 has inputs:
qt (8, 1024, 2560) from layer 0
-------------------------------------------------------------
 - Compile (driver): 72.40 s
 - Compile (worker): 44.54 s
Iteration 0
Iteration 1
Iteration 2
 - Benchmark: 16.90 s
Type: gpt  Model Config: (32, 1024, 2560, 32, 32)  Parallel Config: (2, 2, 2)  P-mesh shape: (1, 4)  #Microbatch: 4  Force Mapping: False  Remat: True  Reduce-scatter: True  Mean Time: 2.437s  Std Time: 0.001  #Params: 2.649B  TFLOPs: 28.40  TFLOPs (ckpt): 37.42  Peak Mem: 8.745G  overwrite_global_config_dict: None
```
