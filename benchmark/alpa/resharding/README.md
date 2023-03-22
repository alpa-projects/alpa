# Benchmark
This folder contains benchmarking code for cross mesh resharding, corresponding to the experiment section in [On Optimizing the Communication of Model Parallelism](https://arxiv.org/abs/2211.05322). 

To make the benchmark feasible in a short amount of time, this documentation provides: Instructions for benchmarking on an AWS p3.8xlarge cluster. You can use these to quickly run cross mesh resharding using Alpa and get the statistics of the performance. The statistics may be different from that in our papaer if your cluster is not an AWS p3.8xlarge cluster. 
There are two types of experiments for benchmarking:
- Single device to multiple devices microbenchmark: corronspond to section 5.1.1 in [On Optimizing the Communication of Model Parallelism](https://arxiv.org/abs/2211.05322). 
- Multiple devices to multiple devices microbenchmark: corronspond to section 5.1.2 and 5.3.1 in [On Optimizing the Communication of Model Parallelism](https://arxiv.org/abs/2211.05322). 

## Benchmark Steps

### Cluster Preparation

Prepare 5 AWS p3.8xlarge instances and put them in the same Placement Group. 

### Start a Ray Cluster
Alpa uses a distributed framework Ray to manage the cluster and distributed workers.
Here, we provide instructions for manually launching a ray cluster.
You can also refer to the Ray [documentation](https://docs.ray.io/en/latest/cluster/quickstart.html#) for more methods on launching and managing ray clusters. 

1. Pick one node as the head node and run the command below on it
    ```
    ray start --head
    ```
2. For all other 4 nodes, connect them to the head node following the instructions printed by the previous command. 
    ```
    # The command should look like this, but with the ip address and password printed by the previous command. 
    ray start --address='172.31.31.37:6379' --redis-password='5241590000000000'
    ```

You can check the cluster status by 
```
ray status
```
You should be able to see the number of CPUs and GPUs available on your cluster. We should have 20 GPUs to proceed. 
All nodes should have alpa installed.

### Single device to multiple devices microbenchmark
Run all benchmark tests with all GPUs in your cluster. 
```
python3 benchmark.py --suite 1-to-m
```
The result will be saved in `tmp/1_to_m_result.json`. In this set of experiment, the sender mesh has only 1 GPU. We vary the number of GPUs in the receiver mesh. In the first half of benchmark tests, the receiver mesh has 1 node and the number of GPUs in this node varies from 1 to 4. In the second half of benchmark tests, the number of GPUs per node is fixed at 2, but the number of nodes in receiver mesh grows from 1 to 4. For more details, please refer to `perf_1_to_m_suite` in `suite.py`.

If you only want to run one test case,
```
python3 benchmark_cross_mesh_resharding.py --suite 1-to-m --n-nodes 1 --gpu-per-node 4 --resharding-mode send_recv --resharding-loadbalance-mode normal
```
Here, I take dst mesh to be (1, 4) as example and you could also choose other cases.
You could use `--resharding-mode`, `--resharding-loadbalance-mode`, `--use-local-allgather` flags 
to specify the configurations for cross mesh resharding. 

### Multiple devices to multiple devices microbenchmark
Similar to the previous subsection. 
```
python3 benchmark.py --suite n-to-m
```
The result will be saved in `tmp/n_to_m_result.json`. In this set of experiment, we move to more complicated cases where both the sender mesh and receiver mesh have multiple nodes. For more details, please refer to `perf_n_to_m_suite` in `suite.py`.

If you only want to run one test case,
```
python3 benchmark_cross_mesh_resharding.py --suite n-to-m --case case1 --resharding-mode send_recv --resharding-loadbalance-mode normal
```
Here, I take case1 as example and you could choose other cases by referring to `suite.py`. Same as above, you could 
specify the configurations for cross mesh resharding.

## Result

By using the above benchmark scripts, you could compare the time spent among different resharding configurations.
And then we could see conclusions in [On Optimizing the Communication of Model Parallelism](https://arxiv.org/abs/2211.05322) from 
these statistics.
