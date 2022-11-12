import argparse
import math
import time

import random

import copy
import numpy as np

from alpa.timer import timers
from alpa.util import write_tsv
from examples.llm_serving.generator import pad_batch
from transformers import AutoTokenizer
from llm_serving.model.wrapper import get_model as get_model_2d
from llm_serving.model.wrapper_1d import get_model as get_model_1d


input_id_list = [
    [45942, 2866, 16, 5, 892, 9, 44042, 8],
    [100, 261, 23888, 2426, 16, 10, 21624, 12, 4310, 3034, 9744, 25526, 11],
    [133, 589, 9, 886, 6, 10817, 16, 10, 285],
    [5625, 16, 10, 205, 183, 8, 38, 236, 7],
    [2264, 16, 5, 7440, 9, 16673, 873, 24214, 116],
    [32826, 16, 5, 812, 343, 9],
    [2264, 109, 47, 206, 59, 5, 499, 9, 28850, 1975, 37079, 116],
    [2264, 109, 47, 206, 59, 5, 3099, 9, 301, 116],
    [19195, 140, 16, 5, 394, 9],
    [534, 10311, 12, 246, 16, 10, 739, 2777, 1421, 14, 16, 4453, 9],
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="opt-1.3b")
    parser.add_argument("--backend", type=str, default="jax")
    parser.add_argument("--path", type=str, default="~/opt_weights/")
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--n-prompt", type=int, default=10)
    parser.add_argument("--batch-size-2d", type=int, default=8)
    parser.add_argument("--batch-size-1d", type=int, default=128)
    parser.add_argument("--cache-size", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--print-results", action="store_true")
    args = parser.parse_args()

    def extend_input(input_list):
        if args.n_prompt < len(input_list):
            ret = input_list[:args.n_prompt]
        else:
            factor = math.ceil(float(args.n_prompt) / float(len(input_list)))
            ret = input_list * factor
            random.shuffle(ret)
        return ret


    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
    tokenizer.add_bos_token = False

    input = extend_input(input_id_list)
    n_prompts = len(input)
    n_batch_2d = math.ceil(float(n_prompts) / float(args.batch_size_2d))


    def runner_2d(model, input):
        output = []
        latency = []
        total_time = 0.0
        start_idx = 0
        for i in range(n_batch_2d):
            end_idx = start_idx + args.batch_size_2d
            end_idx = min(len(input), end_idx)
            cur_batch = input[start_idx:end_idx]

            effective_num_seq = len(cur_batch)
            if len(cur_batch) < args.batch_size_2d:
                cur_batch = pad_batch(cur_batch, 1, args.batch_size_2d)

            tic = time.time()
            output_ids = model.generate(input_ids=cur_batch,
                                        max_new_tokens=args.max_new_tokens,
                                        do_sample=False)
            toc = time.time()
            batch_latency = toc - tic
            total_time += batch_latency
            latency.extend([batch_latency] * effective_num_seq)
            output.extend(output_ids)

        if args.print_results:
            outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
            print("Outputs:\n" + 100 * '-')
            for i, out in enumerate(outputs):
                print(out[i])
                print(f"{i + 1}: {out}")
                print(100 * '-')

        return latency, total_time, output

    def runner_1d(model, input):
        output = []
        latency = []
        tic = time.time()
        output_ids = model.generate(input,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=False)
        toc = time.time()
        total_time = toc - tic

        if args.print_results:
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            print("Outputs:\n" + 100 * '-')
            for i, output in enumerate(outputs):
                print(output_ids[i])
                print(f"{i + 1}: {output}")
                print(100 * '-')
        return latency, total_time, output

    def benchmark(model, runner, input):
        for _ in range(args.n_warmup):
            runner(model, input)
        latencies = np.zeros((args.n_iter, len(input)), dtype=np.float)
        for i in range(args.n_iter):
            latencies[i, :] = np.array(runner(model, input))
        mean_latency = np.mean(latencies)
        return mean_latency

    def estimate_throughput(input, output, latency, total_time):
        req_per_sec = len(input) / total_time
        decoded_tokens = [out[len(input[i]):] for i, out in enumerate(output)]
        decode_token_per_sec = sum(len(seq) for seq in decoded_tokens) / total_time
        return req_per_sec, decode_token_per_sec

    model_name_2d = args.backend + "/" + args.model
    model_2d = get_model_2d(model_name=model_name_2d,
                         path="~/opt_weights",
                         batch_size=args.batch_size_2d)

    model_name_1d = "alpa/" + args.model.replace("-", "-1d-")
    model_1d = get_model_1d(model_name=model_name_1d,
                            path="~/opt_weights",
                            batch_size=args.batch_size_1d,
                            cache_size=args.cache_size)

    latency_2d, total_time_2d, output_2d = benchmark(model_2d, runner_2d, input)
    latency_1d, total_time_1d, output_1d = benchmark(model_1d, runner_1d, input)

    # Compare throughput
    rps_2d, tps_2d = estimate_throughput(input, output_2d, latency_2d, total_time_2d)
    rps_1d, tps_1d = estimate_throughput(input, output_1d, latency_1d, total_time_1d)

    # Compare latency
    latency_diff = (latency_2d - latency_1d)
    latency_improvement = latency_2d / latency_1d
    mean_latency_1d = np.mean(latency_1d)
    mean_latency_2d = np.mean(latency_2d)
    median_latency_1d = np.median(latency_1d)
    median_latency_2d = np.median(latency_2d)

    heads = [
        "Model", "#Prompts", "batch size (2D)", "batch size (1D)", "Max new tokens",
        "rps (2D)", "rps (1D)", "tps (2D)", "tps (1D)",
        "mean latency (2D)", "mean latency (1D)", "median latency (2D)", "median latency (1D)"
    ]
    values = [
        args.model, args.n_prompt, args.batch_size_2d, args.batch_size_1d, args.max_new_tokens,
        rps_2d, rps_1d, tps_2d, tps_1d,
        mean_latency_2d, mean_latency_1d, median_latency_2d, median_latency_1d
    ]
    write_tsv(heads, values, "1d-vs-2d.tsv")


