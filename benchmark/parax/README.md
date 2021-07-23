## Instructions

1. Start ray cluster
```
# On head node
ray start --head
# (Optional) : launch worker nodes
```

2. Run benchmark
- Run benchmark with Ray
  ```
  python3 benchmark_transformer_layer.py
  python3 benchmark_gpt_bert.py
  ```

- Run benchmark without Ray
  ```
  python3 benchmark_transformer_layer.py --local
  python3 benchmark_gpt_bert.py --local
  ```
