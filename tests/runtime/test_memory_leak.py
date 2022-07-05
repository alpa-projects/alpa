"""Test whether there is any memory leak for distributed arrays and remote buffers."""
import unittest

import jax.numpy as jnp
import ray

from alpa import (init, shutdown, parallelize, global_config, ShardParallel,
                  PipeshardParallel)
from alpa.device_mesh import get_global_cluster
from alpa.test_install import get_mlp_train_state_and_step


class MemoryLeakTest(unittest.TestCase):

    def setUp(self):
        init()
        global_config.delete_remote_arrays_threshold = 0

    def tearDown(self):
        shutdown()

    def test_shard_parallel(self):
        state, batch, train_step = get_mlp_train_state_and_step(batch_size=128,
                                                                hidden_size=128)
        train_step = parallelize(train_step,
                                 method=ShardParallel(num_micro_batches=2))

        for i in range(2):
            state, loss = train_step(state, batch)
            del loss
        del state

        # Assert all buffers are freed
        executable = train_step.get_last_executable()
        for w in executable.physical_mesh.workers:
            # One loss array cannot be deleted due to python's GC behavior
            assert len(ray.get(w.get_live_buffer_uuids.remote())) <= 1

    def test_pipeline_parallel(self):
        state, batch, train_step = get_mlp_train_state_and_step(
            batch_size=128, hidden_size=128, add_manual_pipeline_marker=True)

        layer_num = min(get_global_cluster().num_devices, 2)
        train_step = parallelize(
            train_step,
            method=PipeshardParallel(num_micro_batches=2,
                                     layer_option="manual"))

        for i in range(2):
            state, loss = train_step(state, batch)
            del loss
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
