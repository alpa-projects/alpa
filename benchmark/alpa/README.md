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
Alpa uses a distributed framework Ray to manage the cluster and distributed workers.
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
python3 benchmark.py --suite gpt.perf_test_auto
```

You can also specify the number of hosts and the number of devices per host.
```
python3 benchmark.py --suite gpt.perf_test_auto --num-hosts 2 --num-devices-per-host 8
```

### Mixture-of-Expert Transformer
Similar to the previous subsection.
```
python3 benchmark.py --suite moe.perf_test_auto
```

### Wide-ResNet
Similar to the previous subsection.
```
python3 benchmark.py --suite wresnet.perf_test_auto
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
python3 benchmark.py --suite gpt.grid_search_auto
```

## A Quick Performance Test
This is a quick test for checking performance regressions.
Developers should at least run this test to make sure their modifications do not introduce performance regressions.

```
python3 benchmark.py --suite gpt.perf_test_manual
```

Expected output on AWS p3.16 (07/27/2022)
```
$ python3 benchmark.py --suite gpt.perf_test_manual
2022-07-27 07:00:18,603 INFO worker.py:852 -- Connecting to existing Ray cluster at address: 172.31.31.193:6379
Working on case: BenchmarkCase(batch_size=32, model_config=GPTModelConfig(seq_len=1024, hidden_size=2560, num_layers=32, num_heads=32, vocab_size=51200), num_micro_batches=4, parallel_mode='uniform', parallel_args=UniformParallelArgs(prefer_reduce_scatter=True, use_remat=True, dp=2, op=2, pp=2, force_batch_dim_mapping=True))
2022-07-27 07:00:23,344 INFO worker.py:852 -- Connecting to existing Ray cluster at address: 172.31.31.193:6379
 - Prepare input: 5.31 s
 - Create train state: 2.56 s
 - Compile (driver): 66.46 s
 - Compile (worker): 79.97 s
Iteration 0 ...
Iteration 1 ...
Iteration 2 ...
 - Benchmark: 18.06 s
Type: gpt  Model Config: GPTModelConfig(seq_len=1024, hidden_size=2560, num_layers=32, num_heads=32, vocab_size=51200)  #Microbatch: 4  #GPU: 8  Parallel Config: UniformParallelArgs(prefer_reduce_scatter=True, use_remat=True, dp=2, op=2, pp=2, force_batch_dim_mapping=True)  Mean Time (s): 2.454  Std Time (s): 0.000  #Params (Billion): 2.649B  TFLOPs: 37.16  Peak Mem (GB): 8.745  Metadata: {'compilation_times': 'None', 'compute_cost_file_name': 'None', 'forward_stage_layer_ids': 'None', 'submesh_shapes': 'None', 'logical_mesh_shapes': 'None', 'autosharding_option_dicts': 'None'}
```

## Advanced Usage
Benchmark pipeshard parallel case:
```
python benchmark.py --suite gpt.perf_test_auto
```

Benchmark shard parallel case (i.e. only intra-opeartor parallelism, no pipeline parallelism). Add `--local` in the end to run the benchmark with the local cluster without ray.
```
python benchmark.py --suite gpt.perf_test_fast_2d --shard-only [--local]
```

Some benchmarks are inference benchmarks:
```
python benchmark.py --suite gpt_inference.profile
```

Add `--profile-driver-time` to derive the latency from the driver. This flag will also turn off the synchronization barrier after each benchmarking step. Specially, for inference case, this turns streaming inference on and the model will pipeline different input batches (in addition to pipelining different micro-batches).
```
python benchmark.py --suite gpt_inference.profile --profile-driver-time
```

We also include a convenient script `run_exp.py` to run multiple benchmarks with different cluster configurations. For example, to run all gpt search cases:
```
python run_exp.py gpt
```
