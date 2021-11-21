import argparse

import ray
import jax
from parax import global_config

from benchmark_gpt_bert import benchmark_gpt_internal

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# #head = num_heads, D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FD = force_data_parallel,
# RS = prefer_reduce_scatter, CK = use_checkpoint

gpt_specs = {
       # S，   H，    L,   #head, V,
"125M": (1024, 768,   12,  12,    51200),
"350M": (1024, 1024,  24,  16,    51200),
"760M": (1024, 1536,  24,  16,    51200),
"1.3B": (1024, 2048,  24,  32,    51200),
"2.7B": (1024, 2560,  32,  32,    51200),
"6.7B": (1024, 4096,  32,  32,    51200),
"13B":  (1024, 5120,  40,  40,    51200),
"39B":  (1024, 8192,  48,  64,    51200),
"76B":  (1024, 10240, 60,  80,    51200),
}

benchmark_suites = {
1: [
  # B,    S, H, L, #head, V,   D0, D1, NB, FD,    RS,    CK, 
   (128,  *gpt_specs["125M"],  1,  1,  1,  False, False, True),
   (256,  *gpt_specs["125M"],  1,  1,  1,  False, False, True),
   (512,  *gpt_specs["125M"],  1,  1,  1,  False, False, True),
]
}

def benchmark_one_case(case):
    # Launch physical mesh
    device_cluster = DeviceCluster()
    physical_mesh = device_cluster.get_physical_mesh()

    n_iter = 5

    # Run benchmark
    result = benchmark_gpt_internal(physical_mesh, case, n_iter)
    latencies, alloc_mem, tflops, param_count, ilp_objective = result

    # Log results
    heads = ["Model", "Model Config", "Parallel Config", "Param Count",
             "Alloc Mem", "ILP Objective", "Mean Latency", "Std Latency", "TFLOPS"]
    values = [args.model, case[:-6], case[-6:],
              f"{param_count/1e9:.3f}", f"{alloc_mem/GB:.3f}", f"{ilp_objective:.2f}",
              f"{np.mean(latencies):.3f}", f"{np.std(latencies):.3f}", f"{tflops:.2f}"]
    write_tsv(heads, values, f"result_{args.model}.tsv")

    physical_mesh.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ray.init(address="auto")
    num_gpus = int(ray.cluster_resources()["GPU"])

    jax.config.update('jax_platform_name', 'cpu')
    global_config.use_dummy_value_for_benchmarking = True

    # Get benchmark suite and run all cases
    try:
        suite = benchmark_suites[num_gpus]
    except KeyError:
        suite = None

    if not suite:
        print(f"No available benchmark suite for {num_gpus} GPUs")
        exit()

    for case in suite:
        benchmark_one_case(case)
