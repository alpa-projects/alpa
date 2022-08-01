import numpy as np
from timeit import default_timer as timer

from alpa.pipeline_parallel.stage_construction import get_submesh_choices, dp, dp_2 as dp_2


def test_stage_construction(
    submesh_choices, num_hosts, num_devices_per_host, num_layers, max_n_succ_stages
):
    num_devices = num_hosts * num_devices_per_host
    num_autosharding_configs = 1
    for i in range(1, num_devices + 1):
        if num_devices % i == 0:
            num_autosharding_configs += 1

    num_submesh_choices = len(submesh_choices)
    compute_cost = np.random.rand(
        num_layers, num_layers, num_submesh_choices, num_autosharding_configs
    )
    max_n_succ_stages = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        max_n_succ_stages,
    )
    return num_devices, num_autosharding_configs, compute_cost, max_n_succ_stages


powers_of_2 = [1, 2, 4, 8]

# num_layers = 16
# num_hosts = 4
# num_devices_per_host = 8
# num_micro_batches = 16
max_n_succ_stages_n = 4096
num_runs = 10

print("num_layers,num_hosts,num_devices_per_host,num_micro_batches,avg")

for num_layers in powers_of_2:
    for num_hosts in powers_of_2:
        for num_devices_per_host in powers_of_2:
            submesh_choices = get_submesh_choices(
                num_hosts, num_devices_per_host, "all"
            )
            for num_micro_batches in powers_of_2:
                tot = []
                for i in range(num_runs):
                    seed = np.random.randint(1, 1000000)
                    np.random.seed(seed)
                    (
                        num_devices,
                        num_autosharding_configs,
                        compute_cost,
                        max_n_succ_stages,
                    ) = test_stage_construction(
                        submesh_choices,
                        num_hosts,
                        num_devices_per_host,
                        num_layers,
                        max_n_succ_stages_n,
                    )

                    start = timer()
                    res_golden = dp(
                        num_layers,
                        num_devices,
                        num_micro_batches,
                        submesh_choices,
                        num_autosharding_configs,
                        compute_cost,
                        max_n_succ_stages,
                    )
                    end = timer()
                    golden_time = end - start

                    np.random.seed(seed)
                    start = timer()
                    res_mine = dp_2(
                        num_devices,
                        num_micro_batches,
                        submesh_choices,
                        compute_cost,
                        max_n_succ_stages,
                    )
                    end = timer()
                    mine_time = end - start

                    tot.append(golden_time / mine_time)

                    assert res_mine == res_golden, (res_mine, res_golden)

                print(
                    ",".join(
                        map(
                            str,
                            [
                                num_layers,
                                num_hosts,
                                num_devices_per_host,
                                num_micro_batches,
                                sum(tot[1:]) / num_runs,
                            ],
                        )
                    )
                )
