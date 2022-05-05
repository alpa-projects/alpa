# Benchmark
To achieve the best performance with Alpa, one needs to run a full auto-parallelization search for the target model on a target cluster.
The search procedure can take a significant amount of time.
To make the benchmark feasible in a short amount of time, this documentation provides:
- Instructions for benchmarking the solutions found on an AWS p3.16xlarge cluster.  
  You can use these to quickly run Alpa, see how Alpa works, and get an estimation of the performance.
  The performance may not be the best if your cluster is not an AWS p3.16xlarge cluster.
- Instructions for running the full search.  
  You can use these to fully benchmark the auto-parallelization ability of Alpa.

## Benchmark Pre-found Solutions

### Start a Ray Cluster
Alpa uses Ray as the distributed compute framework.
Here, we provide instructions for manually launching a ray cluster.
You can also refer to the Ray [documentation](https://docs.ray.io/en/latest/cluster/quickstart.html#) for more methods on launching and managing ray clusters. 

1. Pick one node as the head node and run the command below on it
    ```
    ray start --head
    ```
2. For all other nodes, connect them to the head node following the instructions printed by the previous command. Skip this step if you only have one node.
    ```
    # The command should look like this, but with the ip address and password printed by the previous command. 
    ray start --address='172.31.31.37:6379' --redis-password='5241590000000000'
    ```

You can check the cluster status by 
```
ray status
```
You should be able to see the number of CPUs and GPUs available on your cluster.
All nodes should have alpa installed.

### GPT-3
Run the benchmark with all GPUs in your cluster.
```
python3 benchmark_3d.py --suite gpt.perf_test_auto
```

You can also specify the number of hosts and the number of devices per host.
```
python3 benchmark_3d.py --suite gpt.perf_test_auto --num-hosts 2 --num-devices-per-host 8
```

### Mixture-of-Expert Transformer
Similar to the previous subsection.
```
python3 benchmark_3d.py --suite moe.perf_test_auto
```

### Wide-ResNet
Similar to the previous subsection.
```
python3 benchmark_3d.py --suite wresnet.perf_test_auto
```

## Run Full Search

### Generate Profiling Database
Alpa requires a cost model to estimate the performance of different parallelization strategies.
This cost model is based on profiling results on the target cluster.
We can generate a profiling database with the following commands, which profiles the time costs of various computation and communication patterns.
Note that this procedure is very slow and can take hours, but you only need to do it once for your cluster.

1. Start a Ray cluster
2. Generate the profiling database
  ```
  # for AWS p3.16:
  python3 gen_prof_database.py --max-comm-size-intra-node 32 --max-comm-size-inter-node 29
  
  # for AWS p4.24 with EFA:
  python3 gen_prof_database.py --efa --max-comm-size-intra-node 33 --max-comm-size-inter-node 30 --max-fail-retry 8
  ```

### Run Search
```
python3 benchmark_3d.py --suite gpt.grid_search_auto
```

## A Quick Performance Test
This is a quick test for checking performance regressions.
Developers should at least run this test to make sure their modifications do not introduce performance regressions.

```
python3 benchmark_3d.py --suite gpt.perf_test_manual
```

Expected output on AWS p3.16 (May 2, 2022)
```
mkdir -p tmp
Working on case: (32, 1024, 2560, 32, 32, 51200, 4, 'manual', (True, True, (2, 2, 2), True))
rm -rf /tmp/tmp_transfer.pkl
python3 -u benchmark_3d_one_case.py --model gpt --niter 3 --case "(32, 1024, 2560, 32, 32, 51200, 4, 'manual', (True, True, (2, 2, 2), True))" --num-hosts 1 --num-devices-per-host 8 --dump-result
mkdir -p tmp
2022-05-02 22:12:06,445 INFO worker.py:840 -- Connecting to existing Ray cluster at address: 172.31.36.188:6379
 - Prepare input: 2.86 s
 - Create train state: 3.17 s
 - Compile (driver): 50.08 s
 - Compile (worker): 57.25 s
Iteration 0
Iteration 1
Iteration 2
 - Benchmark: 18.89 s
Type: gpt  Model Config: (32, 1024, 2560, 32, 32)  #Microbatch: 4  #GPU: 8  Parallel Config: (True, True, (2, 2, 2), True)  Mean Time: 2.429s  Std Time: 0.004  #Params: 2.649B  TFLOPs: 28.49  TFLOPs (ckpt): 37.54  Peak Mem: 8.745G  Compilation Time: None
```
