"""
Test the numerical correctness of shard parallel with gradient accumulation.
"""
import os
import unittest

import numpy as np

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import ray

from alpa import (init, parallelize, grad,
                  LocalPhysicalDeviceMesh, ShardParallel)
from alpa.device_mesh import (get_global_cluster, get_global_physical_mesh,
                              set_global_physical_mesh)
from alpa.shard_parallel.auto_sharding import AutoShardingOption
from alpa.util import count_communication_primitives
from alpa.testing import assert_allclose


class GradAccumulationTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        self.as_option = AutoShardingOption()

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
        batch_size = 256
        num_micro_batches = 2
        hidden_size = 16
        use_bias = True

        self.as_option.allow_all_to_all = False

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(hidden_size, use_bias=use_bias)(x)
                x = nn.Dense(hidden_size, use_bias=use_bias)(x)
                return x

        batch = {
            "x": jnp.ones(
                (batch_size, hidden_size)) * jnp.arange(batch_size)[:, None],
            "y": jnp.ones((batch_size, hidden_size)),
        }

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, batch["x"])
        optimizer = optim.Momentum(1e-2).create(params)

        def train_step(optimizer, batch, apply_func):

            def loss_func(params):
                out = apply_func(params, batch['x'])
                return jnp.mean((out - batch['y'])**2)

            grads = grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grads)
            return new_optimizer

        # Serial execution
        optimizer_expected = train_step(optimizer, batch, model.apply)

        # Distributed execution
        p_train_step = parallelize(
            train_step,
            method=ShardParallel(devices=logical_mesh,
                                 num_micro_batches=num_micro_batches,
                                 auto_sharding_option=self.as_option))
        executable = p_train_step.get_executable(optimizer, batch, model.apply)
        optimizer_actual = p_train_step(optimizer, batch, model.apply)

        # Check results
        assert_allclose(optimizer_expected.target, optimizer_actual.target)

        # Check sharding strategy
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
                assert n_total == n_all_reduce == 2
            else:
                assert n_total == n_all_reduce == 1

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
                count_communication_primitives(apply_grad))
            assert n_total == 0

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
