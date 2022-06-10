"""Test OrderedSet."""

import numpy as np
import unittest

from alpa.pipeline_parallel.stage_construction import (
    dp as stage_construction_dp, get_submesh_choices)

class DynamicProgrammingTest(unittest.TestCase):
    """Test dynamic programming."""

    def test_stage_construction(self):
        """Test stage construction."""
        num_layers = 10
        num_hosts = 4
        num_devices_per_host = 8
        num_devices = num_hosts * num_devices_per_host
        num_micro_batches = 10
        num_autosharding_configs = 2
        submesh_choices = get_submesh_choices(num_hosts, num_devices_per_host, "all")
        num_submesh_choices = len(submesh_choices)
        compute_cost = np.full((num_layers, num_layers, num_submesh_choices,
                                num_autosharding_configs), np.inf)
        max_n_succ_stages = np.full(
            (num_layers, num_layers, num_submesh_choices,
             num_autosharding_configs), -1)
        stage_construction_dp(num_layers, num_devices, num_micro_batches,
                              submesh_choices, num_autosharding_configs, compute_cost, max_n_succ_stages)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(DynamicProgrammingTest))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
