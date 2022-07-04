"""Test whether there is any memory leak for distributed arrays and remote buffers."""
import unittest

import jax.numpy as jnp
import ray

from alpa import (init, shutdown, parallelize, grad, global_config,
                  ShardParallel, PipeshardParallel, AutoLayerOption)
from alpa.device_mesh import get_global_cluster
from alpa.test_install import create_train_state_and_batch


class MemoryLeakTest(unittest.TestCase):

    def setUp(self):
        init()
        global_config.delete_remote_arrays_threshold = 0

    def tearDown(self):
        shutdown()

    def test_shard_parallel(self):

        @parallelize(method=ShardParallel(num_micro_batches=2))
        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch['x'])
                return jnp.mean((out - batch['y'])**2)

            grads = grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        state, batch = create_train_state_and_batch(128, 128)

        for i in range(2):
            state = train_step(state, batch)
        del state

        # Assert all buffers are freed
        executable = train_step.get_last_executable()
        for w in executable.physical_mesh.workers:
            assert len(ray.get(w.get_live_buffer_uuids.remote())) == 0

        executable.physical_mesh.shutdown()

    def test_pipeline_parallel(self):
        layer_num = min(get_global_cluster().num_devices, 2)

        @parallelize(method=PipeshardParallel(
            num_micro_batches=2,
            layer_option=AutoLayerOption(layer_num=layer_num)))
        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch['x'])
                return jnp.mean((out - batch['y'])**2)

            grads = grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        state, batch = create_train_state_and_batch(128, 128)

        for i in range(2):
            state = train_step(state, batch)
        del state

        # Assert all buffers are freed
        executable = train_step.get_last_executable()
        for mesh in executable.mesh_group:
            for w in mesh.workers:
                assert len(ray.get(w.get_live_buffer_uuids.remote())) == 0


def suite():
    suite = unittest.TestSuite()
    suite.addTest(MemoryLeakTest("test_shard_parallel"))
    suite.addTest(MemoryLeakTest("test_pipeline_parallel"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
