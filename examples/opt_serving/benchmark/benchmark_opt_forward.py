import argparse
import time

import numpy as np
import torch
import jax.numpy as jnp
from transformers import GPT2Tokenizer
from jax import tree_flatten

import alpa
from alpa.util import write_tsv
from examples.opt_serving.model.opt_model import get_pipeshard_executable, get_config, load_params_dis_array
from examples.opt_serving.model.opt_utils import compute_gpt_tflops_inference_with_padding


def get_and_load_model(model_name,
                       dummy,
                       cluster="aws",
                       batch_size=1,
                       num_micro_batches=1,
                       gen_len=1024,
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
        batch_size=batch_size,
        num_micro_batches=num_micro_batches,
        gen_len=gen_len,
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
    parser.add_argument("--nb", type=int, default=1)
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")
    num_micro_batches = args.nb

    batch_size = 16
    gen_len = 1

    tic = time.time()
    model, params, opt_config = get_and_load_model(args.model, args.dummy, args.cluster,
                                                   num_micro_batches=num_micro_batches,
                                                   batch_size=batch_size,
                                                   gen_len=gen_len)
    load_time = time.time() - tic

    # Hao: create the batch after alpa.init()
    input_ids = jnp.ones((batch_size, gen_len), dtype=jnp.int32)
    position_ids = jnp.ones((batch_size, gen_len), dtype=jnp.int32)

    H = opt_config.decoder_input_dim
    L = opt_config.decoder_layers
    num_head = opt_config.decoder_attention_heads
    num_gpus = alpa.get_global_cluster().num_devices

    # warm up
    warmup_iters = 10
    for _ in range(warmup_iters):
        forward_results = model(params, {"input_ids": input_ids, "position_ids": position_ids})
        model.sync()

    n_iter = 10
    speeds = []
    tflopss = []
    compute_tflopss = []
    for i in range(n_iter):
        torch.manual_seed(8)

        tic = time.time()
        forward_results = model(params, {"input_ids": input_ids, "position_ids": position_ids})
        model.sync()
        # a = np.array(forward_results)
        # print(a)
        latency = time.time() - tic

        compute_latency = model.get_execution_time_costs(warmup=0)[-1]
        # print(f"input length: {input_ids.shape[1]}, output_length: {input_ids.shape[1]}, num_gpus: {num_gpus}")
        assert gen_len == input_ids.shape[1]

        memory_allocated = model.mesh_group.get_memory_allocated() / 1e9
        max_memory_allocated = model.mesh_group.get_max_memory_allocated() / 1e9

        tflops = compute_gpt_tflops_inference_with_padding(batch_size, gen_len, 2048, L, H, 50272, num_gpus, latency)
        compute_tflops = compute_gpt_tflops_inference_with_padding(batch_size, gen_len, 2048, L, H, 50272, num_gpus, compute_latency)
        speed = np.prod(input_ids.shape) / latency

        print(f"speed: {speed:.2f} token/s, tflops: {tflops:.4f}, compute tflops: {compute_tflops:.4f}, "
              f"memory: {memory_allocated}, max memory: {max_memory_allocated}")
        speeds.append(speed)
        tflopss.append(tflops)
        compute_tflopss.append(compute_tflops)

    avg_speed = sum(speeds) / n_iter
    avg_tflops = sum(tflopss) / n_iter
    avg_compute_tflops = sum(compute_tflopss) / n_iter
    latency_32_token = 32.0 / (avg_speed / batch_size)
    heads = ["Model", "Device", "Dummy", "Load (s)", "Batchsize", "#Microbatches", "#Stages", "Decoder step",
             "Speed (token/s)", "TFlops", "Compute TFlops", "latency (32 token)"]
    values = [args.model, args.device, args.dummy, f"{load_time:.2f}", f"{batch_size}", f"{num_micro_batches}", "2", f"{gen_len}",
              f"{avg_speed:.2f}", f"{avg_tflops:.4f}", f"{avg_compute_tflops:.4f}", f"{latency_32_token:.2f}"]
    print(len(heads))
    print(len(values))
    write_tsv(heads, values, "results.tsv")
