## Instructions

1. Start ray cluster.
```
# On head node
ray start --head
# (Optional) : launch worker nodes
```

2. Run benchmark.
```
python3 benchmark_2d.py --suite gpt.fast_perf_test
```

## Santity Checks

```
python3 benchmark_3d.py --suite gpt.perf_test
```

Expected output on AWS p3.16 (Mar.25, 2022, commit: b789b35c667)
```
mkdir -p tmp
Working on case: (32, 1024, 2560, 32, 32, 51200, 2, 2, 1, 4, 2, 4, False, True, True, 'uniform_layer_gpipe', None)
rm -rf /tmp/tmp_transfer.pkl
python3 -u benchmark_3d_one_case.py --model gpt --niter 3 --case "(32, 1024, 2560, 32, 32, 51200, 2, 2, 1, 4, 2, 4, False, True, True, 'uniform_layer_gpipe', None)" --num-hosts 1 --num-devices-per-host 8 --dump-result
mkdir -p tmp
2022-03-25 10:21:19,889 INFO worker.py:840 -- Connecting to existing Ray cluster at address: 172.31.24.40:6379
 - Prepare input: 2.83 s
 - Create train state: 3.57 s
 - Compile (driver): 62.07 s
 - Compile (worker): 58.23 s
Iteration 0
Iteration 1
Iteration 2
 - Benchmark: 17.36 s
Type: gpt  Model Config: (32, 1024, 2560, 32, 32)  Parallel Config: (2, 2, 2)  P-mesh shape: (1, 4)  #Microbatch: 4  Force Mapping: False  Remat: True  Reduce-scatter: True  Mean Time: 2.465s  Std Time: 0.001  #Params: 2.649B  TFLOPs: 28.07  TFLOPs (ckpt): 37.00  Peak Mem: 8.745G  overwrite_global_config_dict: None
```

## Generate Profiling Database
1. Start ray cluster.

2. Run profiling to generate the database for the HLO instruction cost model.
```
python3 gen_prof_database.py
```
