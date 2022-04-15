# Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning
This is the artifact for the paper "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning".
We are going to reproduce the main results in the paper.

## Setting up the Environment
Please use the provided instructions to log in to the AWS cluster set up by authors.
Then go to this folder.

### Check cluster status (5 min)
1. Run
  ```
  python3 -c "import ray; ray.init(address='auto'); assert ray.cluster_resources()['GPU'] == 32"
  ```
  
  You should be able to see it outputs
  ```
  ...
  #GPU: 32.0
  ```

2. Run
  ```
  python3 test_install.py
  ```

  You should be able to see it outputs
  ```
  ...
  Ran 2 tests in 54.444s

  OK
  ```

## End-to-end Performance (Figure. 8)
The profiling and optimization for all data points in our paper takes more than 10 hours to run.
To make the artifact evaluation feasible in a reasonable amount of time, we skip some profiling and
optimization precedures, but directly load the solutions found by our system and measure their throughputs.
If you want to run the profling and optimization from scratch by yourself, follow the instructions in section
"Running the Search".

### GPT
```
python3 gen_data_e2e.py --model gpt
python3 plot_e2e.py --model gpt
```

### MoE
```
python3 gen_data_e2e.py --model moe
python3 plot_e2e.py --model moe
```

### Wide-ResNet
```
python3 gen_data_e2e.py --model wresnet
python3 plot_e2e.py --model wresnet
```

## Intra-op Ablation Study (Figure. 9)

### GPT  (30 mins)
```
python3 gen_data_intra_ab.py --model gpt
python3 plot_intra_ab.py --model gpt
```

### MoE (30 mins)
```
python3 gen_data_intra_ab.py --model moe
python3 plot_intra_ab.py --model moe
```

### Wide-ResNet (30 mins)
```
python3 gen_data_intra_ab.py --model wresnet
python3 plot_intra_ab.py --model wresnet
```

## Inter-op Ablation Study (Figure. 10)

```
python3 gen_data_inter_ab.py
python3 plot_inter_ab.py
```

## Running the Search

