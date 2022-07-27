"""Benchmark suites for moe with auto parallelization."""
from suite_manual_moe import moe_specs
# Share parallel options with the GPT suite
from suite_auto_gpt import (get_search_cases, get_solution_case, force_dp_dict)

# Temporary debug suite
tmp_suite = {}

# Performance test with search solutions found for p3.16xlarge
perf_test_suite = {
    1:
        get_solution_case(moe_specs["380M"], 512, 1, [[0]], [(1, 1)], [(1, 1)],
                          [{}]),
    2:
        get_solution_case(moe_specs["690M"], 32, 8, [[0, 1, 2, 3, 4, 5, 6, 7]],
                          [(1, 2)], [(2, 1)], [force_dp_dict]),
    4:
        get_solution_case(moe_specs["1.3B"], 32, 8,
                          [[0, 1, 2, 3], [4, 5, 6, 7]], [(1, 2)] * 2,
                          [(2, 1)] * 2, [force_dp_dict] * 2),
    8:
        get_solution_case(moe_specs["2.4B"], 32, 8,
                          [[0, 1, 2, 3], [4, 5, 6, 7]], [(1, 4)] * 2,
                          [(4, 1)] * 2, [force_dp_dict] * 2),
    16:
        get_solution_case(moe_specs["10B"], 16, 8, [[0, 1, 2, 3], [4, 5, 6, 7]],
                          [(1, 8)] * 2, [(8, 1)] * 2, [{}] * 2),
    32:
        get_solution_case(moe_specs["27B"], 128, 8,
                          [[0], [1], [2], [3], [4], [5], [6], [7]],
                          [(1, 4)] * 8, [(4, 1)] * 8, [{}] * 8),
    64:
        get_solution_case(moe_specs["70B"], 64, 8,
                          [[0], [1], [2], [3], [4], [5], [6], [7]],
                          [(1, 8)] * 8, [(8, 1)] * 8, [{}] * 8),
}

# Grid search on hyperparameters
grid_search_suite = {
    2: (get_search_cases(moe_specs["690M"], [16, 32, 64], [8])),
    4: (get_search_cases(moe_specs["1.3B"], [16, 32, 64], [8])),
    8: (get_search_cases(moe_specs["2.4B"], [16, 32, 64], [8])),
    16: (get_search_cases(moe_specs["10B"], [16, 32, 64], [8])),
    32: (get_search_cases(moe_specs["27B"], [32, 64, 128], [4, 8, 16])),
    64: (get_search_cases(moe_specs["70B"], [64], [8, 16, 32])),
    # submesh_choices_mode: "small_power_of_two", max num_cpus = 20
}
