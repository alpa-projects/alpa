"""Test whether there is any memory leak for distributed arrays and remote buffers."""

import unittest

import jax.numpy as jnp
import ray

from alpa import (parallelize, set_parallelize_options, grad, global_config,
                  automatic_layer_construction, DeviceCluster)

from test_install import create_train_state_and_batch

class MemoryLeakTest(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto", ignore_reinit_error=True)
        global_config.delete_remote_buffers_threshold = 0

    def tearDown(self):
        ray.shutdown()

    def test_shard_parallel(self):
        device_mesh = DeviceCluster().get_physical_mesh()
        set_parallelize_options(devices=device_mesh,
                                strategy="shard_parallel",
                                num_micro_batches=2)

        @parallelize
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
        for w in device_mesh.workers:
            assert len(ray.get(w.get_live_buffer_uuids.remote())) == 0

    def test_pipeline_parallel(self):
        device_mesh = DeviceCluster().get_virtual_physical_mesh()
        set_parallelize_options(devices=device_mesh,
                                strategy="pipeshard_parallel",
                                pipeline_stage_mode="uniform_stage",
                                num_micro_batches=2)

        layer_num = min(device_mesh.num_devices, 2)

        @parallelize
        def train_step(state, batch):

            @automatic_layer_construction(layer_num=layer_num)
            def loss_func(params):
                out = state.apply_fn(params, batch['x'])
                return jnp.mean((out - batch['y'])**2)

            grads = grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state


        state, batch = create_train_state_and_batch(128, 128)

        executable = train_step.get_executable(state, batch)
        for i in range(2):
            state = train_step(state, batch)
        del state

        # Assert all buffers are freed
        for mesh in executable.physical_meshes:
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
