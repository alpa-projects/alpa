"""Benchmark generation performance.

Usages:
1. benchmark huggingface torch-based OPT or GPT-2 generation:
python benchmark_text_gen.py --model facebook/opt-125m --debug

2. benchmark jax.jit based OPT generation without alpa, on a single GPU:
python benchmark_text_gen.py --model jax/opt-125m

3. benchmark alpa parallelized OPT generation:
python benchmark_text_gen.py --model alpa/opt-2.7b --debug

4. benchmark alpa parallelized OPT forward computation, batch_size, decoder length, and #micro_batches can be configured.
python benchmark_text_gen.py --model alpa/opt-2.7b --forward
    --decoder_length 1024 --nb 1 --batch-size 256 --debug

Notes:
1. fp32 does not work now because of embedding
"""
import argparse

import alpa
from alpa.global_env import global_config
from alpa.util import write_tsv
import jax.numpy as jnp
import numpy as np
import time
import torch
from transformers import AutoTokenizer

from examples.opt_serving.model.opt_utils import compute_gpt_tflops_inference_with_padding
from examples.opt_serving.model.wrapper import get_model

test_prompts = [
    "Computer science is the study of computation and",
    "Ion Stoica is a Romanian-American computer scientist specializing in",
    "The University of California, Berkeley is a public",
    "Today is a good day and I want to", "What is the valuation of Databricks?",
    "Paris is the capital city of", "Which country has the most population?",
    "What do you think about the future of Cryptocurrency?",
    "What do you think about the meaning of life?",
    "Donald Trump is the president of",
    "GPT-3 is a large language model that is capable of"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpa/opt-125m")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--path", type=str, default="/home/ubuntu/opt_weights/")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--decoder-length", type=int, default=1)
    parser.add_argument("--nb", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dtype", type=str, default="fp16")
    args = parser.parse_args()

    # Some global params
    warmup_iters = 5
    n_iters = 10
    global_config.pipeline_sync_for_timer = True
    global_config.shard_parallel_sync_for_timer = True

    # Note(Hao): we need to use "opt-30b" and disable "add_bos_token".
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b",
                                              use_fast=False)
    tokenizer.add_bos_token = False

    # Do some param check
    num_micro_batches = args.nb
    decoder_length_per_step = args.decoder_length
    batch_size = args.batch_size
    num_beams = args.num_beams
    autoregressive = not args.forward
    dtype = jnp.float16 if args.dtype == "fp16" else jnp.float32

    if autoregressive:
        assert num_micro_batches == 1, "we only support num_micro_batches=1 for autoregressive!"
        assert decoder_length_per_step == 1, "Decoding one token at a time!"
        assert batch_size == 1, "batch_size > 1 in autoregressive is not tested!"

    decode_speeds = []
    tflopss = []
    compute_tflopss = []

    if not autoregressive:
        # Increase the frequency of deleting buffers to avoid OOM.
        global_config.delete_remote_arrays_threshold = 1

        # forward mode
        tic = time.time()
        model, params, transformer_config = get_model(
            args.model,
            args.device,
            args.path,
            autoregressive,
            dtype=dtype,
            dummy=args.dummy,
            batch_size=batch_size,
            decoding_length_per_step=decoder_length_per_step,
            num_micro_batches=num_micro_batches)
        load_time = time.time() - tic

        # create batch
        input_ids = jnp.ones((batch_size, decoder_length_per_step),
                             dtype=jnp.int32)
        position_ids = jnp.ones((batch_size, decoder_length_per_step),
                                dtype=jnp.int32)

        # get model config
        H = transformer_config.H
        L = transformer_config.L
        seq_len = transformer_config.seq_len
        vocab_size = transformer_config.vocab_size

        num_gpus = alpa.get_global_cluster(
        ).num_devices if "alpa" in args.model else 1

        # warm up
        for _ in range(warmup_iters):
            forward_results = model(params, {
                "input_ids": input_ids,
                "position_ids": position_ids
            })
            model.sync()

        # benchmark
        for i in range(n_iters):
            torch.manual_seed(8)

            tic = time.time()
            forward_results = model(params, {
                "input_ids": input_ids,
                "position_ids": position_ids
            })
            model.sync()
            # a = np.array(forward_results)
            # print(a)
            latency = time.time() - tic

            compute_latency = model.get_execution_time_costs()[-1]
            # print(f"input length: {input_ids.shape[1]}, output_length: {input_ids.shape[1]}, num_gpus: {num_gpus}")
            assert decoder_length_per_step == input_ids.shape[1]

            memory_allocated = model.mesh_group.get_memory_allocated() / 1e9
            max_memory_allocated = model.mesh_group.get_max_memory_allocated(
            ) / 1e9

            tflops = compute_gpt_tflops_inference_with_padding(
                batch_size, decoder_length_per_step, seq_len, L, H, vocab_size,
                num_gpus, latency)
            compute_tflops = compute_gpt_tflops_inference_with_padding(
                batch_size, decoder_length_per_step, seq_len, L, H, vocab_size,
                num_gpus, compute_latency)
            speed = np.prod(input_ids.shape) / latency

            if args.debug:
                print(
                    f"speed: {speed:.2f} token/s, E2E tflops: {tflops:.4f}, compute tflops: {compute_tflops:.4f}, "
                    f"memory: {memory_allocated}, max memory: {max_memory_allocated}"
                )
            decode_speeds.append(speed)
            tflopss.append(tflops)
            compute_tflopss.append(compute_tflops)
    else:
        # generation mode
        tic = time.time()
        model = get_model(args.model,
                          args.device,
                          args.path,
                          autoregressive,
                          dtype=dtype,
                          dummy=args.dummy,
                          num_beams=num_beams)
        load_time = time.time() - tic

        # warm up
        input_ids = tokenizer("Paris is the capital city of",
                              return_tensors="pt").input_ids.to(args.device)
        output = model.generate(input_ids=input_ids,
                                max_length=256,
                                do_sample=False,
                                return_dict_in_generate=True,
                                output_hidden_states=False,
                                num_beams=num_beams)

        H = model.transformer_config.H
        L = model.transformer_config.L
        seq_len = model.transformer_config.seq_len
        vocab_size = model.transformer_config.vocab_size
        if "alpa" in args.model:
            if alpa.get_global_cluster():
                num_gpus = alpa.get_global_cluster().num_devices
            else:
                num_gpus = alpa.get_global_physical_mesh().num_devices
        else:
            num_gpus = 1

        # benchmark
        for i in range(min(n_iters, len(test_prompts))):
            prompt = test_prompts[i]
            torch.manual_seed(8)
            input_ids = tokenizer(prompt,
                                  return_tensors="pt").input_ids.to(args.device)
            tic = time.time()
            output = model.generate(input_ids=input_ids,
                                    max_length=256,
                                    do_sample=False,
                                    return_dict_in_generate=True,
                                    output_hidden_states=False,
                                    num_beams=num_beams)
            latency = time.time() - tic
            generated_ids = output.sequences
            generated_string = tokenizer.batch_decode(generated_ids,
                                                      skip_special_tokens=True)

            gen_len = generated_ids.shape[1]

            if "alpa" in args.model:
                compute_latency = sum(
                    model.executable.get_execution_time_costs()[-gen_len:])
            else:
                compute_latency = latency
            tflops = compute_gpt_tflops_inference_with_padding(
                num_beams * batch_size, gen_len, seq_len, L, H, vocab_size,
                num_gpus, latency)
            compute_tflops = compute_gpt_tflops_inference_with_padding(
                num_beams * batch_size, gen_len, seq_len, L, H, vocab_size,
                num_gpus, compute_latency)
            speed = np.prod(generated_ids.shape) / latency
            if args.debug:
                print(
                    f"input length: {input_ids.shape[1]}, output_length: {generated_ids.shape[1]}, "
                    f"num_gpus: {num_gpus}, speed: {speed:.2f} tokens/s, tflops: {tflops:.4f} tflops/s"
                )
                print(generated_string)
            decode_speeds.append(speed)
            tflopss.append(tflops)
            compute_tflopss.append(compute_tflops)

    avg_speed = np.mean(decode_speeds)
    avg_tflops = np.mean(tflopss)
    avg_compute_tflops = np.mean(compute_tflopss)
    latency_32_tokens = 32.0 / (avg_speed / batch_size)

    heads = [
        "Model", "Device", "Dummy", "Load (s)", "Autoregressive", "Batchsize",
        "#Microbatches", "#Beams", "#Stages", "Decoder step length", "TFlops",
        "Compute TFlops", "Speed (token/s)", "latency (32 token)"
    ]
    values = [
        args.model, args.device, args.dummy, f"{load_time:.2f}",
        f"{autoregressive}", f"{batch_size}", f"{num_micro_batches}",
        f"{num_beams}", "2", f"{decoder_length_per_step}", f"{avg_tflops:.4f}",
        f"{avg_compute_tflops:.4f}", f"{avg_speed:.2f}",
        f"{latency_32_tokens:.2f}"
    ]
    write_tsv(heads, values, "results.tsv")
