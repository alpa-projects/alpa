# Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning
This is the artifact for the paper "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning".
We are going to reproduce the main results in the paper.

## Setting up the Environment
Please use the provided instructions to log in to the AWS cluster set up by authors.
Then go to this folder.

### Check cluster status (5 mins)
1. Run
  ```
  python3 -c "import ray; ray.init(address='auto'); print('#GPU:', ray.cluster_resources()['GPU'])"
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
The profiling and optimization for all data points in our paper take more than 10 hours to run.
To make the artifact evaluation feasible in a reasonable amount of time, we skip some profiling and
optimization procedures, but directly load the solutions found by our system and measure their throughputs.
Optionally, if you want to run the profiling and optimization from scratch by yourself, follow the
instructions in the section "Running the Search" below.

### GPT (30 mins)
```
python3 gen_data_e2e.py --model gpt
python3 plot_e2e.py --model gpt
```
This outputs Figure 8 (a).

### MoE (30 mins)
```
python3 gen_data_e2e.py --model moe
python3 plot_e2e.py --model moe
```
This outputs Figure 8 (b).

### Wide-ResNet (30 mins)
```
python3 gen_data_e2e.py --model wresnet
python3 plot_e2e.py --model wresnet
```
This outputs Figure 8 (c).


Since Megatron-LM and DeepSpeed requires different setups, we provide instructions to generate their performance numbers 
separately as below.
### Megatron-LM Performance on GPT (30 mins)
Following the instructions in [megatron/setup.md](setup/setup.md) to install Megatron-LM. 
Then, run the benchmarking scripts to obtain the Megatron-LM performance on GPT:
```python
python3 megatron/gen_data_e2e_megatron.py --model gpt
```
This outputs the red bar corresponding to `Megatron-LM` in Figure 8 (a).

### DeepSpeed Performance on MoE (30 mins)
Following the instructions in [deepspeed/setup.md](setup/setup.md) to install DeepSpeed.
Then, run the benchmarking scripts to obtain the DeepSpeed performance on :
```python
python3 deepspeed/gen_data_e2e_megatron.py --model gpt
```
This outputs the brown bar corresponding to `DeepSpeed` in Figure 8 (b).

## Intra-op Ablation Study (Figure. 9)

### GPT  (30 mins)
```
python3 gen_data_intra_ab.py --model gpt
python3 plot_intra_ab.py --model gpt
```
This outputs Figure 9 (a).

### MoE (45 mins)
```
python3 gen_data_intra_ab.py --model moe
python3 plot_intra_ab.py --model moe
```
This outputs Figure 9 (b).

### Wide-ResNet (30 mins)
```
python3 gen_data_intra_ab.py --model wresnet
python3 plot_intra_ab.py --model wresnet
```
This outputs Figure 9 (c).

## Inter-op Ablation Study (Figure. 10)

```
python3 gen_data_inter_ab.py
python3 plot_inter_ab.py
```

## (Optional) Running the Search

