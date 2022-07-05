"""
Test the numerical correctness of shard parallel with gradient accumulation.
"""
import os
import unittest

import numpy as np

from flax import linen as nn
import jax
import jax.numpy as jnp
import ray

from alpa import (init, shutdown, parallelize, grad, LocalPhysicalDeviceMesh,
                  ShardParallel)
from alpa.device_mesh import (get_global_cluster, get_global_physical_mesh,
                              set_global_physical_mesh)
from alpa.shard_parallel.auto_sharding import AutoShardingOption
from alpa.util import count_communication_primitives
from alpa.testing import assert_allclose
from alpa.test_install import get_mlp_train_state_and_step


class GradAccumulationTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        self.as_option = AutoShardingOption(allow_all_to_all=False)

    def run_gradient_accumulation(self, cluster, use_2d_mesh):
        if cluster == "ray":
            physical_mesh = get_global_physical_mesh()
            if physical_mesh is None:
                init(cluster="ray")
                physical_mesh = get_global_cluster().get_physical_mesh()
                set_global_physical_mesh(physical_mesh)
            logical_mesh = physical_mesh.get_logical_mesh()
        else:
            physical_mesh = LocalPhysicalDeviceMesh(jax.local_devices()[:4])
            if use_2d_mesh:
                logical_mesh = physical_mesh.get_logical_mesh([2, 2], [1, 1],
                                                              [1, 1])
            else:
                logical_mesh = physical_mesh.get_logical_mesh([1, 4], [1, 1],
                                                              [1, 1])

        state, batch, train_step = get_mlp_train_state_and_step(batch_size=256,
                                                                hidden_size=16,
                                                                num_layers=2)

        # Serial execution
        state_expected = train_step(state, batch)[0]

        # Parallel execution
        p_train_step = parallelize(train_step,
                                   method=ShardParallel(
                                       devices=logical_mesh,
                                       num_micro_batches=2,
                                       auto_sharding_option=self.as_option))
        state_actual = p_train_step(state, batch)[0]

        # Check results
        assert_allclose(state_expected.params,
                        state_actual.params,
                        atol=5e-4,
                        rtol=5e-4)

        # Check sharding strategy
        executable = p_train_step.get_last_executable()
        hlo_text = executable.get_hlo_text()
        if self.as_option.prefer_reduce_scatter:
            _, accumulate_grad, apply_grad = hlo_text.split("HloModule")

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
                count_communication_primitives(accumulate_grad))
            assert n_total == n_all_reduce + n_reduce_scatter == 1

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
                count_communication_primitives(apply_grad))
            assert n_total == n_all_gather == 1
        else:
            assert executable.grad_sync_channel_ids.count(".") == 2
            _, accumulate_grad, apply_grad = hlo_text.split("HloModule")

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
                count_communication_primitives(accumulate_grad))
            if use_2d_mesh:
                # TODO(lmzheng): investigate why n_total is 4 not 2
                assert n_total == n_all_reduce
            else:
                assert n_total == n_all_reduce == 1

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
                count_communication_primitives(apply_grad))
            assert n_total == 0

        executable.dump_debug_info("tmp")

        if cluster == "ray":
            shutdown()

    def test_gradient_accumulation_single_host(self):
        self.run_gradient_accumulation("local", use_2d_mesh=False)

    def test_gradient_accumulation_multi_host(self):
        self.run_gradient_accumulation("ray", use_2d_mesh=False)

    def test_gradient_accumulation_2d_mesh(self):
        self.run_gradient_accumulation("local", use_2d_mesh=True)

    def test_gradient_accumulation_reduce_scatter(self):
        self.as_option.prefer_reduce_scatter = True
        self.run_gradient_accumulation("local", use_2d_mesh=False)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        GradAccumulationTest("test_gradient_accumulation_single_host"))
    suite.addTest(GradAccumulationTest("test_gradient_accumulation_multi_host"))
    suite.addTest(GradAccumulationTest("test_gradient_accumulation_2d_mesh"))
    suite.addTest(
        GradAccumulationTest("test_gradient_accumulation_reduce_scatter"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
