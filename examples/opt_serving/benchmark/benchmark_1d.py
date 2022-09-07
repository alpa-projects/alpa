import argparse
import copy
import time
from examples.opt_serving.model.test_1d import setup, runner_1d, runner_2d
import jax.numpy as jnp
from examples.opt_serving.benchmark.benchmark_text_gen import test_prompts
from alpa.util import print_used_time

input_id_list = [
    [45942, 2866, 16, 5, 892, 9, 44042, 8],
    [100, 261, 23888, 2426, 16, 10, 21624, 12, 4310, 3034, 9744, 25526, 11],
    [133, 589, 9, 886, 6, 10817, 16, 10, 285],
    [5625, 16, 10, 205, 183, 8, 38, 236, 7],
    [2264, 16, 5, 7440, 9, 16673, 873, 24214, 116],
    # [32826, 16, 5, 812, 343, 9],
    # [2264, 109, 47, 206, 59, 5, 499, 9, 28850, 1975, 37079, 116],
    # [2264, 109, 47, 206, 59, 5, 3099, 9, 301, 116],
    # [19195, 140, 16, 5, 394, 9],
    # [534, 10311, 12, 246, 16, 10, 739, 2777, 1421, 14, 16, 4453, 9],
]


def print_execution_cost(execution_cost):
    output = []
    for key, value in execution_cost.items():
        output.append(f"{key}: {value:.4f}")
    print(", ".join(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="jax/opt-125m")
    parser.add_argument("--path", type=str, default="~/opt_weights/")
    parser.add_argument("--n-warmup", type=int, default=1)

    args = parser.parse_args()
    model_1d, input_pool_1d = setup("1d", input_id_list, np_weights_folder=args.path)
    model_2d, input_pool_2d = setup("2d", input_id_list, np_weights_folder=args.path)

    # Warm up
    for i in range(args.n_warmup):
        warmup_pool_2d = copy.deepcopy(input_pool_2d)
        _, cost = runner_2d(model_2d, warmup_pool_2d)
        print_execution_cost(cost)
    for i in range(args.n_warmup):
        warmup_pool_1d = copy.deepcopy(input_pool_1d)
        warmup_pool_1d.kv_caches = [(jnp.asarray(k), jnp.asarray(v)) for k, v in warmup_pool_1d.kv_caches]
        _, cost = runner_1d(model_1d, warmup_pool_1d)
        print_execution_cost(cost)
    print(" === Warm up done! ===")

    # benchmark prompt
    print("Run 1D")
    tic = time.time()
    ret_pool_1d, _ = runner_1d(model_1d, input_pool_1d)
    latency_1d = time.time() - tic
    print("Run 2D")
    tic = time.time()
    ret_pool_2d, _ = runner_2d(model_2d, input_pool_2d)
    latency_2d = time.time() - tic
    print(f"Results 3 prompts: 1d {latency_1d}, 2d {latency_2d}")
    exit(1)
    #
    # # benchmark mixed
    # tic = time.time()
    # ret_pool_1d, _ = runner_1d(input_pool_1d[:2] + ret_pool_1d)
    # latency_1d = time.time() - tic
    # tic = time.time()
    # ret_pool_2d = runner_2d(input_pool_2d[:2] + ret_pool_2d)
    # latency_2d = time.time() - tic
    # print(f"- Benchmark 2 prompts + 3 decode: 1d {latency_1d}, 2d {latency_2d}")
    #
    # # benchmark decoding only
    # tic = time.time()
    # runner_1d(ret_pool_1d)
    # latency_1d = time.time() - tic
    # tic = time.time()
    # runner_2d(ret_pool_2d)
    # latency_2d = time.time() - tic
    # print(f"- Benchmark 5 decode: 1d {latency_1d}, 2d {latency_2d}")
