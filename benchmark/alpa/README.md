## Instructions

1. Start ray cluster.
```
# On head node
ray start --head

# (Optional) : launch more worker nodes and connect them to the head node
# ray start --address='172.31.31.37:6379' --redis-password='5241590000000000'
```

2. Run benchmark.
```
python3 benchmark_3d.py --suite gpt.perf_test_auto
```

## Sanity Check
```
python3 benchmark_3d.py --suite gpt.perf_test_manual
```

Expected output on AWS p3.16 (April 24, 2022)
```
mkdir -p tmp
Working on case: (32, 1024, 2560, 32, 32, 51200, 2, 2, 1, 4, 2, 4, False, True, True, 'uniform_layer_gpipe', None)
rm -rf /tmp/tmp_transfer.pkl
python3 -u benchmark_3d_one_case.py --model gpt --niter 3 --case "(32, 1024, 2560, 32, 32, 51200, 2, 2, 1, 4, 2, 4, False, True, True, 'uniform_layer_gpipe', None)" --num-hosts 1 --num-devices-per-host 8 --dump-result
mkdir -p tmp
2022-04-24 10:41:32,482 INFO worker.py:840 -- Connecting to existing Ray cluster at address: 172.31.24.40:6379
 - Prepare input: 2.85 s
 - Create train state: 3.03 s
 - Compile (driver): 49.68 s
 - Compile (worker): 58.72 s
Iteration 0
Iteration 1
Iteration 2
 - Benchmark: 18.85 s
Type: gpt  Model Config: (32, 1024, 2560, 32, 32)  Parallel Config: (2, 2, 2)  P-mesh shape: (1, 4)  #Microbatch: 4  Force Mapping: False  Remat: True  Reduce-scatter: True  Mean Time: 2.425s  Std Time: 0.000  #Params: 2.649B  TFLOPs: 28.53  TFLOPs (ckpt): 37.60  Peak Mem: 8.745G  overwrite_global_config_dict: None
```

## Generate Profiling Database
1. Start ray cluster.

2. Run profiling to generate the database for the HLO instruction cost model.
```
# for AWS p3.16:
python3 gen_prof_database.py --max-comm-size-intra-node 32 --max-comm-size-inter-node 29

# for AWS p4.24:
python3 gen_prof_database.py --efa --max-comm-size-intra-node 33 --max-comm-size-inter-node 30 --max-fail-retry 8
```

