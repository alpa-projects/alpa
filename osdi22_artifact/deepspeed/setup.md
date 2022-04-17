# Setup DeepSpeed and benchmark it on MoE

## Step 1: Check the DeepSpeed code and versions
We use the official DeepSpeed v0.5.6, at commit [bd3ebddf3628f3f77d3460e49626c8af7825a92c](https://github.com/microsoft/DeepSpeed/tree/bd3ebddf3628f3f77d3460e49626c8af7825a92c), 
dated as Nov/10/2021.

At the time of our paper submission, DeepSpeed has provided an official implementation for [``expert parallelism``](https://www.deepspeed.ai/tutorials/mixture-of-experts/), but 
no implementation for the MoE transformer language model used in GShard and GSPMD paper. 

Hence, we implemented the following components on top of DeepSpeed's expert parallelism implementation in order to make the comparisons fair:
- an MoE transformer language model implementation consistent with the models used in GShard and GSPMD
- We borrow the AdaFactor optimizer provided by the [torch-optimizer](https://github.com/jettify/pytorch-optimizer) project, which is a required optimizer for MoE.

The code modifications can be found under [benchmark/deepspeed/patch](../../benchmark/deepspeed/patch).


## Step 2: Check the environment


## Step 3: Benchmark


## Notes

### DeepSpeed prints inconsistent TFlops
DeepSpeed's logger calculates the TFlops incorrectly. Please ignore the values it reported and observe the final value
reported by Alpa benchmarking code.

### Zero-1, Zero-2, and Zero-3?
DeepSpeed MoE is not compatible with ZeRO-3. It, however, can support ZeRO-2 and ZeRO-1. We have tested both ZeRO-1 and ZeRO-2; 
it turns out that ZeRO-2 demonstrates much better performance. Hence, in our artifact evaluation and the paper, we report the ZeRO-2 performance by default. 