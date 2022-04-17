# Setup Megatron-LM and benchmark it on GPT
To run the Megatron-LM benchmarking, we assume you have set up the AWS cluster following the [setup guide](../README.md).  


## Step 1: Verify the Megatron-LM code
Verify the Megatron-LM repository on EFS we have pre-cloned:
```python
cd ~/efs/Megatron-LM & git log
```
You should be able to see the Megatron-LM as a fork from the official [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) at 
[commit b31e1296354e979722627a6c4dedafe19b51fa97])(https://github.com/NVIDIA/Megatron-LM/tree/b31e1296354e979722627a6c4dedafe19b51fa97) dated at ***Oct 7, 2021***.

The only code difference between this version we used to produce baseline results and [commit b31e12](https://github.com/NVIDIA/Megatron-LM/tree/b31e1296354e979722627a6c4dedafe19b51fa97) 
is that we disable the `tie_embedding` option in Megatron-LM, because at the time of the OSDI submission, we had not figured out an efficient implementation in Alpa to tie embedding variables.

For fair comparisons, we disable the `tie_embedding` in Megatron-LM. Per our experiments, disabling this option in has very little performance 
impact on Megatron-LM on the GPT benchmarking.

## Step 2: Check the environment
We have prepared a Python virtual environment at `~/efs/megatron-env` which installs this Megatron-LM version at
```python
conda deactivate
# Switch to the Python environment for Megatron-LM
source ~/efs/megatron-env/bin/activate
```

Run the following command to check that the environment is working.
```python
python -c "import megatron; import apex"
```

## Step 3: Benchmark
Run the benchmarking using the provided bash script:
```python
cd ~/efs/alpa/osdi22_artifact/megatron

# Replace the [NUM_GPUS] with the number of gpus you want to benchmark with, e.g., 1, 4, 8, 16, 32.
./run_megatron_benchmark.sh [NUM_GPUS]
# For example, benchmark Megatron on GPT on a 32-GPU cluster
./run_megatron_benchmark.sh 32
```

## Trouble Shooting
### First run takes very long?
This is normal. In the first run, Megatron-LM needs to compile several hand-written kernel implementations for SoftMax and GeLU, 
which will take a few minutes. 

### The program hangs somewhere at the startup?
If the benchmark programs hang after launching the program, try to delete the compiled output in this folder `Megatron-LM/megatron/fused_kernels/build`, 
and relaunch. This is a known issue related with Megatron-LM's customized kernel compilation.


## Other notes
Since Megatron-LM does not have auto-parallelization functionalities, given a GPT model specification and a cluster setup, the only way 
to find the best-performing strategy for Megatron-LM is by manual search.

While in [Step 3 of this guide](#step-2-benchmark) we have unveiled the best-performing strategy (a.k.a. DP, TMP, PP combination) straightforwardly, 
[this file](../../benchmark/alpa/) contains all strategy cases we have explored in this manual search process. The exhausted search can be acclerated 
by a few expert-designed (Alpa team) heuristics for GPT-3:
- Try to figure out the largest possible micro-batchsize that do not cause OOM (via trial-and-error).
- Try to figure out the largest possible value of `DP` given this largest possible micro-batchsize.
- Try to figure out a suitable value of `TMP`, while restrict the value of `TMP` to be less than the number of GPUs per node;
- Manually adjust the value of DP and PP further to find the best-performing combination of `(DP, TMP, PP)`





