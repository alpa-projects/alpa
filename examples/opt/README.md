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


## Differences between metaseq.opt and alpa.gpt

### Computing

#### alpa.gpt 
alpa.gpt computes following the order below:
0. input 
1. word_embedding + position_embedding + type_embedding (optional) -> dropout -> LayerNorm 
2. Attention: self_attention ->  dense -> dropout -> LayerNorm 
3. Intermediate: dense -> activation -> dense -> dropout -> LayerNorm
4. Repeat 2 and 3 many times
5. Decoding: tied_embedding * logits + decoder_bias

#### metaseq.opt
0. input
1. word_embedding + position embedding -> dropout (train: 0.1, inference: return x) 
2. LayerNorm -> 


incremental state?


#### Sompe params
- token embedding: vocab_size = 50272
- position embedding: 2048 (seq length ) + 1 (pad_size) + 1 
- 


### Differences in layers
- positional embedding
- token embedding
- 

### Weight mapping
See the state dict of the 350M OPT model in opt/350M_state_dict.txt.
Some mappings between the `state.params["params"]` (notated as `A`) in alpa.gpt and `state["model"]` (notated as `M`) in metaseq.opt

| metaseq.opt                                                                                                                                                              | alpa.gpt                                                                                    |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `M['decoder.layers.0.self_attn.k_proj.weight']`<br/> `M['decoder.layers.0.self_attn.v_proj.weight']`<br/> `M['decoder.layers.0.self_attn.q_proj.weight']`                | `A['transformers']['encoder']['layer']['0']['attention']['self']['qkv_combined']['kernel']` |
| `M['decoder.layers.0.self_attn.k_proj.bias']`<br/> `M['decoder.layers.0.self_attn.v_proj.bias']`<br/> `M['decoder.layers.0.self_attn.q_proj.bias']`                      | `A['transformers']['encoder']['layer']['0']['attention']['self']['qkv_combined']['bias']`   |                                                                                         |
| `M['decoder.layers.0.self_attn.out_proj.weight']`                                                                                                                        | `A['transformers']['encoder']['layer']['0']['attention']['output']['dense']['kernel']`      | 
| `M['decoder.layers.0.self_attn.out_proj.bias']`                                                                                                                          | `A['transformers']['encoder']['layer']['0']['attention']['output']['dense']['bias']`        | 
| `M['decoder.layers.0.self_attn_layer_norm.weight']`                                                                                                                      | `A['transformers']['encoder']['layer']['0']['attention']['output']['LayerNorm']['scale']`   |
| `M['decoder.layers.0.self_attn_layer_norm.bias']`                                                                                                                        | `A['transformers']['encoder']['layer']['0']['attention']['output']['LayerNorm']['bias']`    |
| `M['decoder.layers.0.fc1.weight']`                                                                                                                                       | `A['transformers']['encoder']['layer']['0']['intermediate']['dense']['kernel']`             |
| `M['decoder.layers.0.fc1.bias']`                                                                                                                                         | `A['transformers']['encoder']['layer']['0']['intermediate']['dense']['bias']`               |
| `M['decoder.layers.0.fc2.weight']`                                                                                                                                       | `A['transformers']['encoder']['layer']['0']['output']['dense']['kernel']`                   |
| `M['decoder.layers.0.fc2.bias']`                                                                                                                                         | `A['transformers']['encoder']['layer']['0']['output']['dense']['bias']`                     |
| `M['decoder.layers.0.final_layer_norm.weight']`                                                                                                                          | `A['transformers']['encoder']['layer']['0']['output']['LayerNorm']['scale']`                |
| `M['decoder.layers.0.final_layer_norm.bias']`                                                                                                                            | `A['transformers']['encoder']['layer']['0']['output']['LayerNorm']['bias']`                     |


### How does the flask serving system work
TODO(Hao)