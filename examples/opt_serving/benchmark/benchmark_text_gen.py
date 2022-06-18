"""Benchmark generation performance.

Usages:
1. benchmark huggingface torch-based OPT or GPT-2 generation:
python benchmark_text_gen.py --model facebook/opt-125m --cluster aws --debug

2. benchmark alpa parallelized OPT generation:
python benchmark_text_gen.py --model alpa/opt-2.7b --cluster aws --debug

3. benchmark jax.jit based OPT generation without alpa, on a single GPU:
python benchmark_text_gen.py --model jax/opt-125m --cluster aws

4. benchmark alpa parallelized OPT forward computation, batch_size, decoder length, and #micro_batches can be configured.
python benchmark_text_gen.py --model alpa/opt-27.b --cluster aws --forward
    --decoder_length 1024 --nb 1 --batch-size 256 --debug
"""
import argparse
import jax.numpy as jnp
import numpy as np
import time
import torch

import alpa
from alpa.util import write_tsv
from examples.opt_serving.model.opt_utils import compute_gpt_tflops_inference_with_padding, test_prompts
from examples.opt_serving.model.wrapper import get_model
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpa/opt-125m")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cluster", type=str, default="aws")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--decoder-length", type=int, default=1)
    parser.add_argument("--nb", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Some global params
    warmup_iters = 5
    n_iters = 10

    # Note(Hao): we need to use "opt-30b" and disable "add_bos_token".
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
    tokenizer.add_bos_token = False

    # Do some param check
    num_micro_batches = args.nb
    decoder_length_per_step = args.decoder_length
    batch_size = args.batch_size
    autoregressive = not args.forward

    if autoregressive:
        assert num_micro_batches == 1, "we only support num_micro_batches=1 for autoregressive!"
        assert decoder_length_per_step == 1, "Decoding one token at a time!"
        assert batch_size == 1, "batch_size > 1 in autoregressive is not tested!"

    decode_speeds = []
    tflopss = []
    compute_tflopss = []

    if not autoregressive:
        # forward mode
        tic = time.time()
        model, params, transformer_config = get_model(args.model,
                                                      args.device,
                                                      args.cluster,
                                                      autoregressive,
                                                      dummy=args.dummy,
                                                      decoding_length_per_step=decoder_length_per_step,
                                                      num_micro_batches=num_micro_batches)
        load_time = time.time() - tic

        # create batch
        input_ids = jnp.ones((batch_size, decoder_length_per_step), dtype=jnp.int32)
        position_ids = jnp.ones((batch_size, decoder_length_per_step), dtype=jnp.int32)

        # get model config
        H = transformer_config.H
        L = transformer_config.L
        num_head = transformer_config.n_head
        seq_len = transformer_config.seq_len
        vocab_size = transformer_config.vocab_size
        num_gpus = alpa.get_global_cluster().num_devices

        # warm up
        for _ in range(warmup_iters):
            forward_results = model(params, {"input_ids": input_ids, "position_ids": position_ids})
            model.sync()

        # benchmark
        for i in range(n_iters):
            torch.manual_seed(8)

            tic = time.time()
            forward_results = model(params, {"input_ids": input_ids, "position_ids": position_ids})
            model.sync()
            # a = np.array(forward_results)
            # print(a)
            latency = time.time() - tic

            compute_latency = model.get_execution_time_costs(warmup=0)[-1]
            # print(f"input length: {input_ids.shape[1]}, output_length: {input_ids.shape[1]}, num_gpus: {num_gpus}")
            assert decoder_length_per_step == input_ids.shape[1]

            memory_allocated = model.mesh_group.get_memory_allocated() / 1e9
            max_memory_allocated = model.mesh_group.get_max_memory_allocated() / 1e9

            tflops = compute_gpt_tflops_inference_with_padding(batch_size, decoder_length_per_step, seq_len,
                                                               L, H, vocab_size, num_gpus,
                                                               latency)
            compute_tflops = compute_gpt_tflops_inference_with_padding(batch_size, decoder_length_per_step, seq_len,
                                                                       L, H, vocab_size, num_gpus,
                                                                       compute_latency)
            speed = np.prod(input_ids.shape) / latency

            if args.debug:
                print(f"speed: {speed:.2f} token/s, E2E tflops: {tflops:.4f}, compute tflops: {compute_tflops:.4f}, "
                      f"memory: {memory_allocated}, max memory: {max_memory_allocated}")
            decode_speeds.append(speed)
            tflopss.append(tflops)
            compute_tflopss.append(compute_tflops)

    # Warm up
    else:
        tic = time.time()
        model = get_model(args.model,
                          args.device,
                          args.cluster,
                          autoregressive,
                          dummy=args.dummy)
        load_time = time.time() - tic


        input_ids = tokenizer("Paris is the capital city of", return_tensors="pt").input_ids.to(args.device)
        output = model.generate(input_ids=input_ids, max_length=256, do_sample=False,
                                return_dict_in_generate=True, output_hidden_states=False)

        H = model.transformer_config.H
        L = model.transformer_config.L
        n_head = model.transformer_config.n_head
        seq_len = model.transformer_config.seq_len
        vocab_size = model.transformer_config.vocab_size

        if "alpa" in model:
            num_gpus = alpa.get_global_cluster().num_devices
        else:
            num_gpus = 1

        for i in range(n_iters):
            prompt = test_prompts[i]
            torch.manual_seed(8)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
            tic = time.time()
            output = model.generate(input_ids=input_ids, max_length=256, do_sample=False,
                                    return_dict_in_generate=True, output_hidden_states=False)
            latency = time.time() - tic
            generated_ids = output.sequences
            generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            gen_len = generated_ids.shape[1]

            if "alpa" in model:
                compute_latency = sum(model.executable.get_execution_time_costs(warmup=0)[-gen_len:])
            else:
                compute_latency = latency
            tflops = compute_gpt_tflops_inference_with_padding(batch_size, gen_len, seq_len, L, H, vocab_size, num_gpus, latency)
            compute_tflops = compute_gpt_tflops_inference_with_padding(batch_size, gen_len, seq_len,
                                                                       L, H, vocab_size, num_gpus,
                                                                       compute_latency)
            speed = np.prod(generated_ids.shape) / latency
            if args.debug:
                print(f"input length: {input_ids.shape[1]}, output_length: {generated_ids.shape[1]}, "
                      f"num_gpus: {num_gpus}, speed: {speed:.2f} tokens/s, tflops: {tflops:.4f} tflops/s")
            decode_speeds.append(speed)
            tflopss.append(tflops)

    avg_speed = sum(decode_speeds) / n_iters
    avg_tflops = sum(tflopss) / n_iters
    avg_compute_tflops = sum(compute_tflopss) / n_iters
    latency_32_tokens = 32.0 / (avg_speed / batch_size)

    heads = ["Model", "Device", "Dummy", "Load (s)", "Batchsize", "#Microbatches", "#Stages", "Decoder step-len",
             "Speed (token/s)", "TFlops", "Compute TFlops", "latency (32 token)"]
    values = [args.model, args.device, args.dummy, f"{load_time:.2f}", f"{batch_size}", f"{num_micro_batches}", "2", f"{decoder_length_per_step}",
              f"{avg_speed:.2f}", f"{avg_tflops:.4f}", f"{avg_compute_tflops:.4f}", f"{latency_32_tokens:.2f}"]
    write_tsv(heads, values, "results.tsv")
