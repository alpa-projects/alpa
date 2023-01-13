"""Benchmark suites for moe with manual specifications."""
from collections import namedtuple
from benchmark_parallel_utils import BenchmarkCase, UniformParallelArgs

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# head = num_heads, S_ = expert_group_size, E = expert_number,
# NB = num_micro_batches, PM = parallel_mode
# 3D config = 3D parallel config (Data, Operator, Pipeline)
# RS = prefer_reduce_scatter, Remat = use_rematerialization,
# FM = force_batch_dim_mapping,

MoEModelConfig = namedtuple("MoEModelConfig", [
    "seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size",
    "num_experts", "expert_group_size"
])

moe_specs = {
    #                      S,    H,   L, head, V,   E,  S_
    "380M": MoEModelConfig(1024, 768, 8, 16, 32000, 8, 2048),
    "690M": MoEModelConfig(1024, 768, 8, 16, 32000, 16, 2048),
    "1.3B": MoEModelConfig(1024, 768, 16, 16, 32000, 16, 2048),
    "2.4B": MoEModelConfig(1024, 1024, 16, 16, 32000, 16, 2048),
    "7.1B": MoEModelConfig(1024, 1280, 16, 16, 32000, 32, 2048),
    "10B": MoEModelConfig(1024, 1536, 16, 16, 32000, 32, 2048),
    "27B": MoEModelConfig(1024, 2048, 16, 16, 32000, 48, 2048),
    "70B": MoEModelConfig(1024, 2048, 32, 16, 32000, 64, 2048),
    "140B": MoEModelConfig(1024, 2048, 32, 16, 32000, 128, 2048),
}

# Temporary debug suite
# key = the number of gpus, value = a list of cases
# B, model, NB, PM, RS, Remat, 3D Config, FM
tmp_suite = {
    1: [
        BenchmarkCase(8, moe_specs["380M"], 1, "uniform",
                      UniformParallelArgs(True, True, 1, 1, 1, False))
    ],
    8: [
        BenchmarkCase(16, moe_specs["1.3B"], 1, "uniform",
                      UniformParallelArgs(True, True, 1, 4, 2, False))
    ],
    16: [
        # verify cost model vs. profiling
        BenchmarkCase(1024, moe_specs["10B"], 32, "uniform",
                      UniformParallelArgs(True, True, 2, 8, 1, True))
    ],
}

# Fast performance test on models with fewer layers
# B, S, H, L,  #head, V, E, S_, NB, PM, Remat, RS, 3D Config, FM
perf_test_fast_2d_suite = {
    1: [
        BenchmarkCase(8, MoEModelConfig(1024, 1024, 8, 32, 25600, 8, 1024),
                      1, "uniform",
                      UniformParallelArgs(True, True, 1, 1, 1, True)),
    ],
    8: [
        BenchmarkCase(16, MoEModelConfig(1024, 1024, 4, 32, 25600, 32, 1024),
                      1, "uniform",
                      UniformParallelArgs(False, True, 8, 1, 1, False)),
        BenchmarkCase(16, MoEModelConfig(1024, 1024, 4, 32, 25600, 32, 1024),
                      1, "uniform",
                      UniformParallelArgs(False, True, 4, 2, 1, False)),
        BenchmarkCase(16, MoEModelConfig(1024, 1024, 4, 32, 25600, 32, 1024),
                      1, "uniform",
                      UniformParallelArgs(False, True, 2, 4, 1, False)),
    ],
}
