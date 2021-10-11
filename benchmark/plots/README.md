## Format
We save all experimental results into a single tsv file `results.tsv`, so we can share and query it easily.

The format for each line in this tsv file is
```
exp_name instance num_nodes num_gpu_per_node network_name system algorithm value time_stamp
```

Example
```
weak_scaling_model  p3.24xlarge  1  4  GPT  parax  auto-sharding  {"latency": [0.12, 0.34], "mem": 12.3} 99285.17
weak_scaling_model  p3.24xlarge  1  8  GPT  parax  auto-sharding  {"latency": [0.12, 0.34], "mem": 12.3} 99285.17
weak_scaling_model  p3.24xlarge  2  8  GPT  parax  auto-sharding  {"latency": [0.12, 0.34], "mem": 12.3} 99285.17
```

