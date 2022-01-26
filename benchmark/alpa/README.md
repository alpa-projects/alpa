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
