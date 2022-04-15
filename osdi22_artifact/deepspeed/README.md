# Setup DeepSpeed and benchmark it on MoE
To run the Megatron-LM benchmarking, we assume you have set up the AWS cluster following the [setup guide](../README.md).

## Step 1: Check the DeepSpeed code and versions
Verify the DeepSpeed repository on EFS:
```bash
cd ~/efs/DeepSpeed && git log
```

We use the official DeepSpeed v0.5.4, at commit [bd3ebd](https://github.com/microsoft/DeepSpeed/tree/bd3ebddf3628f3f77d3460e49626c8af7825a92c), 
dated as Nov/10/2021.

At the time of Alpa paper submission, DeepSpeed has provided an official implementation for [expert parallelism](https://www.deepspeed.ai/tutorials/mixture-of-experts/), but 
has not provided implementations for the MoE transformer language model used in [GShard](https://arxiv.org/pdf/2006.16668.pdf) and [GSPMD](https://arxiv.org/pdf/2105.04663.pdf).
Hence, we implemented the following components on top of DeepSpeed's expert parallelism in order to make the comparisons possible:
- an MoE transformer language model implementation consistent with the models used in GShard and GSPMD;
- the AdaFactor optimizer provided by [torch-optimizer](https://github.com/jettify/pytorch-optimizer) project, used for MoE training.

The code modifications can be found under [benchmark/deepspeed/patch](../../benchmark/deepspeed/patch).

## Step 2: Prepare hostfiles
```bash
cd ~/efs/alpa/osdi22_artifact

# Generate the hostfiles
python get_hostfiles.py
```
The above scripts will generate multiple hostfiles in the folder `deepspeed/`, which are required by DeepSpeed for distributed execution.

## Step 3: Check the environment
We have prepared a Python virtual environment at `~/efs/deepspeed-env` which installs this DeepSpeed version.

Activate the environment via:
```bash
conda deactivate
# Switch to the Python environment for DeepSpeed
source ~/efs/deepspeed-env/bin/activate
```
Run the following command to check that the environment is working:
```bash
python -c "import deepspeed"
```

## Step 4: Benchmark
Run the benchmarking:
```bash
cd ~/efs/alpa/osdi22_artifact/deepspeed

# Replace the [NUM_GPUS] with the number of gpus you want to benchmark with, e.g., 1, 4, 8, 16, 32.
./run_deepspeed_benchmark.sh [NUM_GPUS]
# For example, benchmark DeepSpeed on MoE on a 32-GPU cluster
./run_deepspeed_benchmark.sh 32
```

## Notes

### Zero-1, Zero-2, and Zero-3?
DeepSpeed MoE is not compatible with ZeRO-3. It, however, can support ZeRO-2 and ZeRO-1. We have tested both ZeRO-1 and ZeRO-2; In our experiments, ZeRO-2 demonstrates much better performance. Hence, in our artifact evaluation and the paper, we report the ZeRO-2 performance by default. 
