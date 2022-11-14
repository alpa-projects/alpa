import argparse
import math
import time
import random

import numpy as np
import torch

from alpa.util import write_tsv
from llm_serving.generator import pad_batch
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


def synthesize_inputs(low=32, high=512, n_prompt=256):
    vocab_size = 50272
    ret = []
    prompt_length = np.random.randint(low, high, (n_prompt,))
    for i in range(n_prompt):
        p = np.random.randint(low=4, high=vocab_size, size=prompt_length[i]).tolist()
        ret.append(p)
    min_length = min(len(p) for p in ret)
    max_length = max(len(p) for p in ret)
    mean_length = sum(len(p) for p in ret) / len(ret)
    print(f"- Synthetic dataset, size {len(ret)}, min {min_length}, max {max_length}, mean {mean_length}")
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="opt-1.3b")
    parser.add_argument("--backend", type=str, default="jax")
    parser.add_argument("--path", type=str, default="~/opt_weights/")
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--n-prompt", type=int, default=8)
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--low", type=int, default=16)
    parser.add_argument("--high", type=int, default=128)
    parser.add_argument("--batch-size-2d", type=int, default=4)
    parser.add_argument("--batch-size-1d", type=int, default=256)
    parser.add_argument("--cache-size", type=int, default=4096 * 8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--tail-percentage", type=float, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    def extend_input(input_list):
        if args.n_prompt <= len(input_list):
            ret = input_list[:args.n_prompt]
        else:
            factor = math.ceil(float(args.n_prompt) / float(len(input_list)))
            ret = input_list * factor
            random.shuffle(ret)
            ret = ret[:args.n_prompt]
        return ret

    if not args.use_synthetic:
        input = extend_input(input_id_list)
    else:
        input = synthesize_inputs(low=args.low, high=args.high, n_prompt=args.n_prompt)
    n_batch_2d = math.ceil(len(input) / float(args.batch_size_2d))

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
            cur_batch = pad_batch(cur_batch, 1, args.batch_size_2d)
            cur_batch = torch.from_numpy(np.array(cur_batch))

            tic = time.time()
            output_ids = model.generate(input_ids=cur_batch,
                                        max_new_tokens=args.max_new_tokens,
                                        do_sample=False)
            toc = time.time()
            batch_latency = toc - tic
            total_time += batch_latency
            latency.extend([batch_latency] * effective_num_seq)
            output.extend(output_ids[:effective_num_seq])
            start_idx += args.batch_size_2d

        return latency, total_time, output

    def runner_1d(model, input):
        tic = time.time()
        output_ids, latency = model.generate(input,
                                             max_new_tokens=args.max_new_tokens,
                                             do_sample=False)
        toc = time.time()
        total_time = toc - tic

        return latency, total_time, output_ids

    def benchmark(model, runner, input):
        for i in range(args.n_warmup):
            print(f"  Warm-up iter {i}")
            runner(model, input)
        latencies = np.zeros((args.n_iter, len(input)), dtype=float)
        total_times = []
        for i in range(args.n_iter):
            latency, total_time, output = runner(model, input)
            print(f"  Benchmark iter {i}")
            if args.verbose:
                print(f"  {latency}")
            latencies[i, :] = latency
            total_times.append(total_time)
        mean_latency = np.mean(latencies, axis=0)
        return mean_latency, sum(total_times) / args.n_iter, output

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

    num_tail = int(args.tail_percentage / 100.0  * len(input))

    print("- Benchmark 2D...")
    latency_2d, total_time_2d, output_2d = benchmark(model_2d, runner_2d, input)
    rps_2d, tps_2d = estimate_throughput(input, output_2d, latency_2d, total_time_2d)
    mean_latency_2d = np.mean(latency_2d)
    tail_latency_2d = np.mean(latency_2d[np.argsort(latency_2d)[-num_tail:]])

    print("- Benchmark 1D...")
    latency_1d, total_time_1d, output_1d = benchmark(model_1d, runner_1d, input)
    rps_1d, tps_1d = estimate_throughput(input, output_1d, latency_1d, total_time_1d)
    mean_latency_1d = np.mean(latency_1d)
    tail_latency_1d = np.mean(latency_1d[np.argsort(latency_1d)[-num_tail:]])

    heads = [
        "Model", "#Prompts", "BS (2D)", "BS (1D)", "Max new tokens",
        "RPS (1D vs. 2D)", "TPS (1D vs. 2D)",
        "Mean Latency (1D vs. 2D)", "Tail latency (1D vs. 2D)"
    ]
    values = [
        args.model, args.n_prompt, args.batch_size_2d, args.batch_size_1d, args.max_new_tokens,
        f"{rps_1d:.2f}/{rps_2d:.2f} ({rps_1d / rps_2d:.2f}x)", f"{tps_1d:.2f}/{tps_2d:.2f} ({tps_1d / tps_2d:.2f}x)",
        f"{mean_latency_1d:.2f}/{mean_latency_2d:.2f} ({mean_latency_2d / mean_latency_1d:.1f}x)",
        f"{tail_latency_1d:.2f}/{tail_latency_2d:.2f} ({tail_latency_2d / tail_latency_1d:.1f}x)"
    ]
    write_tsv(heads, values, "1d-vs-2d.tsv")
