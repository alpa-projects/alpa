import unittest
import numpy as np

from alpa.pipeline_parallel.stage_construction import (get_submesh_choices, dp,
                                                       dp_2)


def test_stage_construction(submesh_choices, num_hosts, num_devices_per_host,
                            num_layers, max_n_succ_stages):
    num_devices = num_hosts * num_devices_per_host
    num_autosharding_configs = 1
    for i in range(1, num_devices + 1):
        if num_devices % i == 0:
            num_autosharding_configs += 1

    num_submesh_choices = len(submesh_choices)
    compute_cost = np.random.rand(num_layers, num_layers, num_submesh_choices,
                                  num_autosharding_configs)
    max_n_succ_stages = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        max_n_succ_stages)
    return (num_devices, num_autosharding_configs, compute_cost,
            max_n_succ_stages)


class OldNewDPTest(unittest.TestCase):
    """Test the equivalence of old DP and new DP."""

    def test_dp(self):
        cases = [1, 4]
        max_n_succ_stages_n = 4096
        num_runs = 10

        for num_layers in cases:
            for num_hosts in cases:
                for num_devices_per_host in cases:
                    submesh_choices = get_submesh_choices(
                        num_hosts, num_devices_per_host, "all")
                    for num_micro_batches in cases:
                        for i in range(num_runs):
                            (num_devices, num_autosharding_configs,
                             compute_cost,
                             max_n_succ_stages) = test_stage_construction(
                                 submesh_choices, num_hosts,
                                 num_devices_per_host, num_layers,
                                 max_n_succ_stages_n)

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
