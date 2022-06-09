import argparse
import time

import numpy as np
import torch
import jax.numpy as jnp

import alpa
from alpa.util import write_tsv
from examples.opt.model.opt_model import get_pipeshard_executable, get_config, load_params_dis_array
from examples.opt.model.opt_utils import compute_gpt_tflops_inference_with_padding
from transformers import GPT2Tokenizer


def get_and_load_model(model_name, dummy, cluster="aws",
              support_output_attentions=False,
              support_output_hidden_states=False):
    alpa.init()
    num_pp_stages = max(2, alpa.get_global_cluster().num_hosts)

    name = model_name.split("-")[1].upper()
    config = get_config(name, num_pp_stages=num_pp_stages)

    if cluster == "aws":
        path = f"/home/ubuntu/opt_weights/{name}_np"
    elif cluster == "mbzuai":
        path = f"/dataset/opt_weights/{name}_np"
    else:
        raise RuntimeError("Unrecognized cluster.")

    executable, params_aval = get_pipeshard_executable(
        config,
        support_output_attentions=support_output_attentions,
        support_output_hidden_states=support_output_hidden_states,
        autoregressive=False)
    params = load_params_dis_array(path, executable, params_aval, config, dummy)
    # init_cache = init_cache_dis_array(executable, config, 1, dummy)
    executable.sync()

    return executable, params, config


# def inference_func(input_ids, past_key_values, output_attentions=False,
#                    output_hidden_states=False):
#     nonlocal step_ct
#
#     if past_key_values is None:
#         past_key_values = init_cache
#         step_ct = 0
#
#     input_ids_step = input_ids.cpu().numpy()
#     position_ids_step = np.full_like(input_ids_step, step_ct + config.pad + 1)
#
#     output = executable(params, {
#         "input_ids": input_ids_step,
#         "position_ids": position_ids_step,
#         "cache": past_key_values,
#     })
#     logits_step = torch.from_numpy(np.array(output.logits)).to(device)
#
#     step_ct += 1
#     return InferenceFuncOutput(logits_step,
#                                output.attention_cache,
#                                output.hidden_states,
#                                output.attentions)
#
#
# inference_func_config = InferenceFuncConfig()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpa/opt-125m")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cluster", type=str, default="aws")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")

    tic = time.time()

    model, params, opt_config = get_and_load_model(args.model, args.dummy, args.cluster)
    load_time = time.time() - tic

    prompts = [
        "Computer science is the study of computation and",
        "Ion Stoica is a Romanian-American computer scientist specializing in",
        "The University of California, Berkeley is a public",
        # "Today is a good day and I want to",
        # "What is the valuation of Databricks?",
        # "Paris is the capital city of",
        # "Which country has the most population?",
        # "What do you think about the future of Cryptocurrency?"
    ]

    H = opt_config.decoder_input_dim
    L = opt_config.decoder_layers
    num_head = opt_config.decoder_attention_heads


    batch_size = 512
    seq_len = 1024
    batch = {
        "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "position_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
    }


    for prompt in prompts:
        tic = time.time()
        torch.manual_seed(8)
        # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)


        input_ids = batch["input_ids"]
        position_ids = batch["position_ids"]
        model(params, {
            "input_ids": batch["input_ids"],
            "position_ids": batch["position_ids"]

        })

        # output = model.generate(input_ids=input_ids, max_length=256, do_sample=False,
        #                         return_dict_in_generate=True, output_hidden_states=False)
        # generated_ids = output.sequences
        # generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        num_gpus = alpa.get_global_cluster().num_devices
        print(f"input length: {input_ids.shape[1]}, output_length: {input_ids.shape[1]}, num_gpus: {num_gpus}")
        print(f"hidden size: {H}, num layers: {L}, num attention heads: {num_head}")
        latency = time.time() - tic
        gen_len = input_ids.shape[1]

        exec_flops = model.flop_count / 1e12 / latency / num_gpus * gen_len
        # print(model.executable.flop_count )

        tflops = compute_gpt_tflops_inference_with_padding(1, gen_len, 2048, L, H, 50272, num_gpus, latency)
        speed = np.prod(input_ids.shape) / latency

        # print(f"{generated_string}")
        print(f"speed: {speed:.2f} token/s, tflops: {tflops} tflops/s, exec_flops: {exec_flops}")

    heads = ["Model", "Device", "Dummy", "Load (s)", "Speed (token/s)", "TFlops (TFlops/s)"]
    values = [args.model, args.device, args.dummy, f"{load_time:.2f}", f"{speed}", f"{tflops}"]
    write_tsv(heads, values, "results.tsv")
