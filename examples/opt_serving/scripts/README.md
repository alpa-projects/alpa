This document contains information on how to convert the original weights released by Metaseq into numpy weight that 
can be used by Alpa.

### What is `convert_to_sington.py`
This program runs in SPMD and parallel.

This scripts supposes there is no zero-3 sharding anymore in saved checkpoint files, i.e., Zero-3 shards have been merged, 
It assumes only model parallel parts exist, in the format of `part-0`, `part-1`, ..., `part-7`, etc.

It first use FSDP `load_model_ensemble_and_task` to load model parallel parts into each worker.
It then uses `FSDP.summon_all_params`, and `torch.allgather` to gather the parameters of model parallel parts into each worker.
It then uses the native `glue_megatron_parts()` to recover them into a single entity.
It then moves the glued checkpoint to CPU on rank 0 and saves it on disk as `restored.pt`. 

Notes: `load_model_ensemble_and_task()` is a SPMD distributed program that can support loading in parallel on `MP` workers for different model parallel parts.


### What is `scripts/consolidate_fsdp_shards.py`

It calls `distributed/stitch_fsdp_ckpt.py`.
`distributed/stitch_fsdp_ckpt.py` is a single-node single-thread program. 

It assumes the checkpoint files still preserve both Zero-3 sharding and model-parallel sharding.
It then loads the checkpoint files, and tries to consolidate them into one file. 

My understanding is this file does the same thing as `convert_to_singleton.py`, but 
requires more raw checkpoint formats (2 dimensions of sharding exist).

It has an options `no_stitch_megatron`. If True, it will only consolidate by removing the Zero-3 sharding dimension using the function `FSDP.consolidate_shard_weight`. 
If False, it first consolidate along the Zero-3 dimension using `FSDP.consolidate_shard_weight`, then it uses `glue_megatron_parts()` to consolidate along the model
parallel dimension.

It provides the important function `glue_megatron_parts()` which can be used to merge the megatron MP dimension.


### What is `scripts/reshard_mp.py`
This scripts is single-node and single-threaded.

`reshard_all_parts` loops over each model parallel set of checkpoints, and call `reshard_mp` for each set.
Each set contains *N* shards where N is the number of data-parallel workers (the dimension of Zero-3).

`resharding_mp` has an argument `target_ddp_size`, it reads all the shards corresponding to one model parallel part into
memory, merges them using `_merge_flat_fsdp_shards()`, then resharding them into `target_ddp_size` new chunks, and then save to disk. 
There is an option `drop_optimizer_state` which controls if to drop optimizer state, which might not be relevant with inference/serving.


### Part 5 wrong positions
```python
problem_positions = [
(1537, 5172),
(1537, 5173),
(1537, 5174),
(1537, 5175),
(1537, 5176),
(1537, 5177),
(1537, 5178),
(1537, 5179),
(1537, 5180),
(1537, 5181),
(1537, 5182),
(1537, 5183),
(1537, 5184),
(1537, 5185),
(1537, 5186),
(1537, 5187),
(1537, 5188),
(1537, 5189),
]
```
