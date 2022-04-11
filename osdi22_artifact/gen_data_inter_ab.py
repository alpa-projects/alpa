from ..benchmark.alpa.benchmark_3d_one_case import benchmark_one_case


def run_equal_eqn_one_case(model, case, niter, num_hosts, num_devices_per_host):
    ablation_config = {"use_equal_eqn": True}
    return benchmark_one_case(model, case, niter, num_hosts,
                              num_devices_per_host, True, True, True,
                              ablation_config)


def run_equal_layer_one_case(model, case, niter, num_hosts,
                             num_devices_per_host):
    optimal_result = [0] * 6
    num_stages = 1
    if model == "moe" or model == "gpt":
        num_layers = case[3]
    elif model == "wresnet":
        num_layers = case[2]
    while num_layers % num_stages == 0:
        ablation_config = {"num_stages": num_stages}
        case[-1]["ablation_equal_layer"] = True
        result = benchmark_one_case(model, case, niter, num_hosts,
                                    num_devices_per_host, True, True, True,
                                    ablation_config)
        if result[5] > optimal_result[5]:
            optimal_result = result
        num_stages *= 2
    return optimal_result
