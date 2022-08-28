import argparse
import copy
import time
from examples.opt_serving.model.test_1d import init_1d_runner, init_2d_runner
from examples.opt_serving.benchmark.benchmark_text_gen import test_prompts
from alpa.util import print_used_time

input_id_list = [
    [45942, 2866, 16, 5, 892, 9, 44042, 8],
    [100, 261, 23888, 2426, 16, 10, 21624, 12, 4310, 3034, 9744, 25526, 11],
    [133, 589, 9, 886, 6, 10817, 16, 10, 285],
    [5625, 16, 10, 205, 183, 8, 38, 236, 7],
    [2264, 16, 5, 7440, 9, 16673, 873, 24214, 116],
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="jax/opt-125m")
    parser.add_argument("--path", type=str, default="~/opt_weights/")
    # parser.add_argument("--dummy", action="store_true")
    # parser.add_argument("--forward", action="store_true")
    # parser.add_argument("--forward-encoder-length", type=int, default=1024)
    # parser.add_argument("--nb", type=int, default=1)
    # parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-warmup", type=int, default=1)
    # parser.add_argument("--n-iter", type=int, default=10)
    # parser.add_argument("--max-length", type=int, default=256)
    # parser.add_argument("--pad-to-max-length", type=int)
    # parser.add_argument("--num-beams", type=int, default=1)
    # parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--dtype", type=str, default="fp16")

    args = parser.parse_args()
    name = args.model.split("-")[1].upper()

    runner_1d, input_pool_1d = init_1d_runner(name, args.path, input_id_list)
    runner_2d, input_pool_2d = init_2d_runner(name, args.path, input_id_list)

    # Warm up
    for i in range(args.n_warmup):
        warmup_pool_2d = copy.deepcopy(input_pool_2d)
        runner_2d(warmup_pool_2d[2:])
    for i in range(args.n_warmup):
        warmup_pool_1d = copy.deepcopy(input_pool_1d)
        runner_1d(warmup_pool_1d[2:])
    print(" === Warm up done! ===")

    # benchmark prompt
    print("Run 1D")
    tic = time.time()
    ret_pool_1d = runner_1d(input_pool_1d[2:])
    latency_1d = time.time() - tic
    print("Run 2D")
    tic = time.time()
    ret_pool_2d = runner_2d(input_pool_2d[2:])
    latency_2d = time.time() - tic
    print(f"Results 3 prompts: 1d {latency_1d}, 2d {latency_2d}")
    exit(1)

    # benchmark mixed
    tic = time.time()
    ret_pool_1d = runner_1d(input_pool_1d[:2] + ret_pool_1d)
    latency_1d = time.time() - tic
    tic = time.time()
    ret_pool_2d = runner_2d(input_pool_2d[:2] + ret_pool_2d)
    latency_2d = time.time() - tic
    print(f"- Benchmark 2 prompts + 3 decode: 1d {latency_1d}, 2d {latency_2d}")

    # benchmark decoding only
    tic = time.time()
    runner_1d(ret_pool_1d)
    latency_1d = time.time() - tic
    tic = time.time()
    runner_2d(ret_pool_2d)
    latency_2d = time.time() - tic
    print(f"- Benchmark 5 decode: 1d {latency_1d}, 2d {latency_2d}")
