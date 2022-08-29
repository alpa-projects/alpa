import unittest
import numpy as np

from alpa.pipeline_parallel.stage_construction import (get_submesh_choices, dp,
                                                       dp_2)


def default_num_auto_sharding_configs(num_devices):
    num_autosharding_configs = 0
    for i in range(1, num_devices + 1):
        if num_devices % i == 0:
            num_autosharding_configs += 1
    return num_autosharding_configs


def generate_stage_construction_test_case(num_devices,
                                          submesh_choices,
                                          num_layers,
                                          num_autosharding_configs,
                                          compute_cost_factor=0.0,
                                          device_memory_size_factor=1.0,
                                          unlimited_memory=False):
    """
    Generate a test case for stage construction.
    Args:
        num_devices: number of total devices.
        submesh_choices: a list of submesh choices.
        num_layers: number of layers.
        num_autosharding_configs: number of auto sharding configs.
        compute_cost_factor: a factor to control the distributed compute cost.
            Take values in [-inf, inf].
        device_memory_size_factor: a factor to control the device memory size.
            Take values in [0, inf].
        unlimited_memory: ignore memory cost.
    """
    num_submesh_choices = len(submesh_choices)
    compute_cost = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        np.inf)
    max_n_succ_stages = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        -1)
    layer_base_cost = np.random.rand(num_layers)
    memory_base_cost = np.random.rand(num_layers)
    total_memory = memory_base_cost.sum()
    for start in range(num_layers):
        for end in range(start, num_layers):
            for s, submesh in enumerate(submesh_choices):
                submesh_size = np.prod(submesh)
                for l in range(num_autosharding_configs):
                    autosharding_factor = np.random.rand() + 1
                    compute_cost[start, end, s,
                                 l] = (layer_base_cost[start:end + 1].sum() *
                                       autosharding_factor *
                                       submesh_size**compute_cost_factor)
                    if unlimited_memory:
                        max_n_succ_stages[start, end, s, l] = 4096
                    else:
                        model_percentage = (
                            memory_base_cost[start:end + 1].sum() /
                            total_memory)
                        device_percentage = submesh_size / num_devices
                        max_n_succ_stages[start, end, s,
                                          l] = (device_memory_size_factor *
                                                num_layers * device_percentage /
                                                model_percentage /
                                                autosharding_factor)

    return compute_cost, max_n_succ_stages


class OldNewDPTest(unittest.TestCase):
    """Test the equivalence of old DP and new DP."""

    def test_dp(self):
        num_runs = 2
        np.random.seed(0)

        for num_layers in [4, 8]:
            for num_hosts in [1, 4]:
                for num_devices_per_host in [1, 4]:
                    submesh_choices = get_submesh_choices(
                        num_hosts, num_devices_per_host, "all")
                    for num_micro_batches in [1, 16, 512]:
                        for i in range(num_runs):
                            compute_cost_factor = np.random.rand() * 4 - 2
                            device_memory_size_factor = np.random.rand() * 4
                            num_devices = num_hosts * num_devices_per_host
                            num_autosharding_configs = np.random.randint(1, 5)
                            (compute_cost, max_n_succ_stages
                            ) = generate_stage_construction_test_case(
                                num_devices, submesh_choices, num_layers,
                                num_autosharding_configs, compute_cost_factor,
                                device_memory_size_factor)

                            res_old = dp(num_layers, num_devices,
                                         num_micro_batches, submesh_choices,
                                         num_autosharding_configs, compute_cost,
                                         max_n_succ_stages)

                            res_new = dp_2(num_devices, num_micro_batches,
                                           submesh_choices, compute_cost,
                                           max_n_succ_stages)
                            assert res_new == res_old


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(OldNewDPTest))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
