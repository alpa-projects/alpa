import argparse

import copy
import numpy as np

from alpa.timer import timers
from examples.llm_serving.model.test_1d import setup, runner_1d, runner_2d, Jax1DInput

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

input_id_list = input_id_list * 4


def print_execution_time_costs(timer_name="overall", return_all_costs=False):
    """Get the execution time costs with internal timers."""
    if timer_name not in timers:
        raise RuntimeError()
    if return_all_costs:
        print(f"{timer_name}: {timers(timer_name).costs}")
    else:
        print(f"{timer_name}: {timers(timer_name).costs[-1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="jax/opt-125m")
    parser.add_argument("--path", type=str, default="~/opt_weights/")
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--batch-size-2d", type=int, default=1)
    parser.add_argument("--batch-size-1d", type=int, default=128)
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument("--cache--size-per-seq", type=int, default=32)
    args = parser.parse_args()

    def extend_input(input_lists, min_factor=1, max_factor=40):
        ret = []
        for seq in input_lists:
            f = np.random.randint(min_factor, high=max_factor+1)
            ret.append(seq * f)
        return ret

    def benchmark_2d(model, input_pool):
        timer_names = ["2d_overall", "2d_compute"]
        for _ in range(args.n_warmup):
            runner_2d(model, input_pool)
            # print_execution_time_costs("2d_overall")
            # print_execution_time_costs("2d_compute")
        for timer_name in timer_names:
            timers(timer_name).reset()
        for _ in range(5):
            runner_2d(model, input_pool)
        print("Results: ", end="")
        timers.log(["2d_overall", "2d_compute"])

    def benchmark_1d(model, input_pool):
        timer_names =  ["1d_overall", "1d_compute", "1d_update_cache"]
        for _ in range(args.n_warmup):
            warmup_pool_1d = copy.deepcopy(input_pool)
            runner_1d(model, warmup_pool_1d)
            # print_execution_time_costs("1d_overall")
            # print_execution_time_costs("1d_compute")
        for timer_name in timer_names:
            timers(timer_name).reset()
        for _ in range(5):
            temp_pool_1d = copy.deepcopy(input_pool)
            runner_1d(model, temp_pool_1d)
        print("Results: ", end="")
        timers.log(timer_names)

    def benchmark_prompt(input_list, mode, batch_size, cache_size=None, max_cache_len_per_seq=None):
        if mode not in ["1d", "2d"]:
            raise RuntimeError()
        num_seq = len(input_list)
        seq_lens = [len(seq) for seq in input_list]
        max_seq_len = max(seq_lens)
        min_seq_len = min(seq_lens)
        print(f"- Benchmark {mode}, {num_seq} prompts with "
              f"bs_{mode} = {batch_size} "
              f"seq_len = [{min_seq_len}->{max_seq_len}] ", end="")
        if mode == "1d":
            if not cache_size:
                cache_size = (max_seq_len + 2) * len(input_list)
            if not max_cache_len_per_seq:
                max_cache_len_per_seq = cache_size // len(input_list)
            print(f"cache = {cache_size} ", end="")
        model, input_pool = setup(mode, input_list,
                                  np_weights_folder=args.path,
                                  batch_size=batch_size,
                                  cache_size=cache_size,
                                  max_cache_len_per_seq=max_cache_len_per_seq)
        if mode == "2d":
            benchmark_2d(model, input_pool)
        else:
            benchmark_1d(model, input_pool)


    def benchmark_mixed(input_list, mode, batch_size=1, num_decoding=None, cache_size=None, max_cache_len_per_seq=None):
        if mode not in ["1d", "2d"]:
            raise RuntimeError()
        num_seq = len(input_list)
        if not num_decoding:
            num_decoding = num_seq // 2
        num_prompt = num_seq - num_decoding
        if mode == "2d":
            batch_size = 1
        seq_lens = [len(seq) for seq in input_list[num_decoding:]]
        min_seq_len = min(seq_lens)
        max_seq_len = max(seq_lens)
        print(f"- Benchmark {mode}, {num_prompt} prompts, {num_decoding} decoding with "
              f"bs_{mode} = {batch_size} "
              f"seq_len = [{min_seq_len}->{max_seq_len}] ", end="")
        if mode == "2d":
            model, input_pool = setup(mode, input_list,
                                      np_weights_folder=args.path,
                                      batch_size=batch_size)
            output_pool = runner_2d(model, input_pool[:num_decoding])
            mixed_pool = output_pool + input_pool[num_decoding:]
            benchmark_2d(model, mixed_pool)
        else:
            if not cache_size:
                cache_size = (max_seq_len + 2) * len(input_list)
            if not max_cache_len_per_seq:
                max_cache_len_per_seq= cache_size // len(input_list)
            print(f"cache = {cache_size} ", end="")
            model, input_pool = setup(mode, input_list[:num_decoding],
                                      np_weights_folder=args.path,
                                      batch_size=batch_size,
                                      cache_size=cache_size,
                                      max_cache_len_per_seq=max_cache_len_per_seq)
            output_pool = runner_1d(model, input_pool)
            mixed_pool = Jax1DInput(
                input_id_list[num_decoding:] + output_pool.input_tokens,
                [i+1 for i in range(num_decoding, len(input_id_list))] + output_pool.input_sentence_ids,
                output_pool.kv_caches,
                output_pool.kv_cache_ids,
                output_pool.num_prev_tokens
            )
            benchmark_1d(model, mixed_pool)


    print(f"Benchmark prompt-only on short sequences.")
    # 2D, short sequence, bs = 1
    benchmark_prompt(input_id_list, "2d", 1)
    # 2D, short sequence, bs = 8
    benchmark_prompt(input_id_list, "2d", 8)
    # 1D, short sequence
    batch_size_1d = sum(len(seq) for seq in input_id_list) + 50
    benchmark_prompt(input_id_list, "1d", batch_size_1d)
    # 1D, vary cache size
    max_seq_len = max(len(seq) for seq in input_id_list)
    num_seq = len(input_id_list)
    benchmark_prompt(input_id_list, "1d", batch_size_1d, cache_size=(max_seq_len + 2) * num_seq * 2)
    benchmark_prompt(input_id_list, "1d", batch_size_1d, cache_size=(max_seq_len + 2) * num_seq * 10)
    benchmark_prompt(input_id_list, "1d", batch_size_1d, cache_size=(max_seq_len + 2) * num_seq * 100)


    print(f"Benchmark prompt-only on long sequences.")
    extended_input_id_list = extend_input(input_id_list)
    # 2D, longer sequence, bs = 1
    benchmark_prompt(extended_input_id_list, "2d", 1)
    # 2D, longer sequence, bs = 8
    benchmark_prompt(extended_input_id_list, "2d", 8)
    # 1D, longer sequence
    batch_size_1d = sum(len(seq) for seq in extended_input_id_list) + 50
    max_seq_len = max(len(seq) for seq in extended_input_id_list)
    benchmark_prompt(extended_input_id_list, "1d", batch_size_1d, cache_size=(max_seq_len + 2) * len(extended_input_id_list))
    benchmark_prompt(extended_input_id_list, "1d", batch_size_1d, cache_size=(max_seq_len + 2) * len(extended_input_id_list) * 2)
    benchmark_prompt(extended_input_id_list, "1d", batch_size_1d, cache_size=(max_seq_len + 2) * len(extended_input_id_list) * 3)


    print(f"Benchmark mixed prompts + decoding on short sequences.")
    # 2D, short seq
    benchmark_mixed(input_id_list, "2d", batch_size=1)
    # 1D, short seq
    batch_size_1d = sum(len(seq) for seq in input_id_list) + 50
    max_seq_len = max(len(seq) for seq in input_id_list)
    benchmark_mixed(input_id_list, "1d", batch_size=batch_size_1d,
                    cache_size=(max_seq_len + 2) * len(input_id_list))
    benchmark_mixed(input_id_list, "1d", batch_size=batch_size_1d,
                    cache_size=(max_seq_len + 2) * len(input_id_list) * 2)
    benchmark_mixed(input_id_list, "1d", batch_size=batch_size_1d,
                    cache_size=(max_seq_len + 2) * len(input_id_list) * 10)
    benchmark_mixed(input_id_list, "1d", batch_size=batch_size_1d,
                    cache_size=(max_seq_len + 2) * len(input_id_list) * 100)

    print(f"Benchmark mixed prompts + decoding on long sequences.")
    # 1D, long seq
    benchmark_mixed(extended_input_id_list, "2d", batch_size=1)
    # 2D, longer seq
    max_seq_len = max(len(seq) for seq in extended_input_id_list)
    batch_size_1d = sum(len(seq) for seq in extended_input_id_list) + 50
    benchmark_mixed(extended_input_id_list, "1d", batch_size=batch_size_1d,
                    cache_size=(max_seq_len + 2) * len(extended_input_id_list))
    benchmark_mixed(extended_input_id_list, "1d", batch_size=batch_size_1d,
                    cache_size=(max_seq_len + 2) * len(extended_input_id_list) * 2)
    benchmark_mixed(extended_input_id_list, "1d", batch_size=batch_size_1d,
                    cache_size=(max_seq_len + 2) * len(extended_input_id_list) * 3)
