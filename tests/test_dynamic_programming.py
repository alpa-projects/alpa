"""Test OrderedSet."""

import numpy as np
import unittest

import alpa
from alpa.pipeline_parallel.stage_construction import (dp as
                                                       stage_construction_dp,
                                                       get_submesh_choices)


class DynamicProgrammingTest(unittest.TestCase):
    """Test dynamic programming."""

    def test_stage_construction(self):
        """Test stage construction."""
        num_layers = 8
        num_hosts = 4
        num_devices_per_host = 8
        num_devices = num_hosts * num_devices_per_host
        num_micro_batches = 16
        num_autosharding_configs = 7
        submesh_choices = get_submesh_choices(num_hosts, num_devices_per_host,
                                              "all")
        num_submesh_choices = len(submesh_choices)
        np.random.seed(42)
        compute_cost = np.random.rand(num_layers, num_layers,
                                      num_submesh_choices,
                                      num_autosharding_configs)
        max_n_succ_stages = np.full(
            (num_layers, num_layers, num_submesh_choices,
             num_autosharding_configs), 4096)
        alpa.util._DISABLE_NUMBA = False
        numba_cost, numba_solution = stage_construction_dp(num_layers, num_devices, num_micro_batches,
                                               submesh_choices, num_autosharding_configs,
                                               compute_cost, max_n_succ_stages)
        alpa.util._DISABLE_NUMBA = True
        no_numba_cost, no_numba_solution = stage_construction_dp(num_layers, num_devices, num_micro_batches,
                                               submesh_choices, num_autosharding_configs,
                                               compute_cost, max_n_succ_stages)
        print(numba_cost)
        print(no_numba_cost)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(DynamicProgrammingTest))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
