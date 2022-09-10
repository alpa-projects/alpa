"""Use huggingface/transformers interface and Alpa backend for distributed inference."""
from transformers import AutoTokenizer
# from opt_serving.model.wrapper import get_model
from opt_serving.model import opt_model
import numpy as np
import jax
import jax.numpy as jnp
import time

# Load the tokenizer. We have to use the 30B version because
# other versions have some issues. The 30B version works for all OPT models.
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
tokenizer.add_bos_token = False

generate_params = {"do_sample": False, "num_beams": 1, "num_return_sequences": 1}


def init_2d_inference_step(name, np_weights_folder, batch_size=1):
    # Init 2D model
    config = opt_model.get_opt_config(name, dtype=jnp.float32)
    model_2d, params_2d = opt_model.init_model_aval(config)
    params_2d = opt_model.load_params_np(params_2d, np_weights_folder, config)
    params_2d = jax.tree_map(jnp.array, params_2d)
    cache_2d = opt_model.init_cache_np(config, batch_size)

    @jax.jit
    def inference_step_2d(params, batch):
        output = model_2d.apply(params,
                                batch["input_ids"],
                                batch["position_ids"],
                                attention_cache=batch["cache"])
        return output.logits, output.attention_cache

    return inference_step_2d, params_2d, cache_2d, config


np_weights_folder = f"/home/ubuntu/opt_weights/125M_np"
batch_size = 16
model = init_2d_inference_step("125M", np_weights_folder, batch_size=batch_size)

# Generate
prompts = [
    "Paris is the capital city of",
    "Today is a good day and I'd like to",
    "Computer Science studies the area of",
    "University of California Berkeley is a public university"
]

factor = 100
prompts = [" ".join(prompt.split(" ") * 100) for prompt in prompts]
prompts = prompts * 20
# prompts = [
#     [45942, 2866, 16, 5, 892, 9, 44042, 8],
#     [100, 261, 23888, 2426, 16, 10, 21624, 12, 4310, 3034, 9744, 25526, 11],
#     [133, 589, 9, 886, 6, 10817, 16, 10, 285],
#     [5625, 16, 10, 205, 183, 8, 38, 236, 7],
#     [2264, 16, 5, 7440, 9, 16673, 873, 24214, 116],
#     [32826, 16, 5, 812, 343, 9],
#     [2264, 109, 47, 206, 59, 5, 499, 9, 28850, 1975, 37079, 116],
#     [2264, 109, 47, 206, 59, 5, 3099, 9, 301, 116],
#     [19195, 140, 16, 5, 394, 9],
#     [534, 10311, 12, 246, 16, 10, 739, 2777, 1421, 14, 16, 4453, 9],
# ]

def sync():
    jax.devices()[0].synchronize_all_activity()

input_ids = tokenizer(prompts, return_tensors="np", padding="longest").input_ids

print(f"shape: {input_ids.shape}, #samples: {len(input_ids)}")
position_ids = opt_model.build_position_ids(input_ids, model[-1].pad)

def run(model, input_ids, position_ids):
    inference_step, params, cache, config = model
    num_batch = len(input_ids) // batch_size
    print(f"num batch {num_batch}, batch size {batch_size}")
    for i in range(num_batch):
        inference_step(params, {
                "input_ids": input_ids[i*batch_size:(i+1)*batch_size,:],
                "position_ids": position_ids[i*batch_size:(i+1)*batch_size,:],
                "cache": cache,
            })


n_warmup = 10
for i in range(n_warmup):
    sync()
    tic = time.time()
    run(model, input_ids, position_ids)
    sync()
    toc = time.time()
    print(f"warmup iter {i}, time {toc - tic}")


sync()
tic = time.time()
run(model, input_ids, position_ids)
sync()
toc = time.time()
print(f">> time: {toc - tic}")
