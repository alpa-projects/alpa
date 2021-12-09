## Format
We save all experimental results into a single tsv file `results.tsv`, so we can share and query it easily.

The format for each line in this tsv file is
```
exp_name instance num_hosts num_devices_per_host model_name method value time_stamp
```

Example
```
weak_scaling_model  p3.24xlarge  1  4  gpt  parax.auto_sharding  {"latency": [0.12, 0.34], "mem": 12.3} 99285.17
weak_scaling_model  p3.24xlarge  1  8  gpt  parax.auto_sharding  {"latency": [0.12, 0.34], "mem": 12.3} 99285.17
weak_scaling_model  p3.24xlarge  2  8  gpt  parax.auto_sharding  {"latency": [0.12, 0.34], "mem": 12.3} 99285.17
```
