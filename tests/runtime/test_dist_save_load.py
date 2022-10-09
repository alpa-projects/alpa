"""Test distributed save and load."""

import subprocess
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from alpa import (init, shutdown, parallelize, DistributedArray,
                  PipeshardParallel, save_checkpoint, restore_checkpoint)
from alpa.device_mesh import get_global_cluster
from alpa.testing import (get_mlp_train_state_and_step,
                          get_bert_layer_train_state_and_step, assert_allclose)


class DistSaveLoadTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def tearDown(self):
        shutdown()

    def check_dist_array_eq(self, x, y):
        if isinstance(x, DistributedArray):
            x = np.array(
                x.device_mesh.get_remote_buffers(x.remote_ref, batching=True))
        if isinstance(y, DistributedArray):
            y = np.array(
                y.device_mesh.get_remote_buffers(y.remote_ref, batching=True))
        assert_allclose(x, y)

    def _get_efs_mount_point(self):
        # Hacky function to get the EFS mount point
        for line in subprocess.check_output("df -h",
                                            shell=True).decode().split('\n'):
            cols = line.split(' ')
            if "efs" in cols[0]:
                return cols[-1] + "/"
        return None

    def _get_save_prefix(self):
        device_cluster = get_global_cluster()
        if len(device_cluster.host_info) > 1:
            # Get EFS mount point for the multi-host test
            save_prefix = self._get_efs_mount_point()
            if save_prefix is None:
                self.skipTest("The multi-host test requires a mounted EFS! ")
        else:
            # Use tmp dir for the single-host test
            save_prefix = "/tmp/"
        return save_prefix

    def test_distributed_array_save_load(self):
        device_cluster = get_global_cluster()
        save_prefix = self._get_save_prefix()

        # Launch a device mesh contains four devices
        if device_cluster.num_devices < 4:
            self.skipTest(
                "This unit test requires a cluster with at least 4 devices! ")
        host_num = min(len(device_cluster.host_info), 4)
        device_per_host = 4 // host_num
        physical_mesh = device_cluster.get_physical_mesh(
            list(range(host_num)), device_per_host)
        logical_mesh = physical_mesh.get_logical_mesh([2, 2])

        global_input_shape = (4, 2)
        num = np.prod(np.array(global_input_shape))

        # Build DistributedArray to be saved
        # [[0,1],          [[0],  [[1],
        #  [2,3],  shard    [2]]   [3]]
        #  [4,5],  ====>   [[4],  [[5],
        #  [6,7]]           [6]]   [7]]
        global_input_data1 = jnp.arange(num).reshape(global_input_shape)
        input_sharding_spec = logical_mesh.make_tile_spec(
            global_input_data1, [0, 1], [0, 1])
        input_indices = input_sharding_spec.indices(
            global_input_data1.shape).flatten()
        (dist_input_data1,) = physical_mesh.shard_args_to_arrays(
            (jax.ShapedArray(global_input_data1.shape, jnp.int32),),
            (input_indices,), (input_sharding_spec,), (global_input_data1,))

        # Check the DistributedArray's remote buffers
        desired_buffers1 = np.array([[[0], [2]], [[1], [3]], [[4], [6]],
                                     [[5], [7]]])
        self.check_dist_array_eq(desired_buffers1, dist_input_data1)

        # cached save/load
        with tempfile.TemporaryDirectory(prefix=save_prefix) as ckpt_dir:
            with tempfile.TemporaryDirectory(prefix="/tmp/") as cache_dir:
                # Save the DistributedArray (one replica only)
                dist_input_data1.save(ckpt_dir, cache_dir)

                # Sync all the move workers
                physical_mesh.sync_move_workers()

                # Load previously saved DistributedArray with a different shardingSpec
                # [[0,1],          [[0,1],  [[0,1],
                #  [2,3],  shard    [2,3]]   [2,3]]
                #  [4,5],  ====>   [[4,5],  [[4,5],
                #  [6,7]]           [6,7]]   [6,7]]
                load_sharding_spec = logical_mesh.make_tile_spec(
                    global_input_data1, [0, 1], [0])
                dist_load_data1 = DistributedArray.load(
                    ckpt_dir,
                    jax.ShapedArray(global_input_data1.shape, jnp.int32),
                    physical_mesh, load_sharding_spec)

                # Check the DistributedArray's remote buffers
                desired_buffers2 = np.array([[[0, 1], [2, 3]], [[0, 1], [2, 3]],
                                             [[4, 5], [6, 7]], [[4, 5], [6,
                                                                         7]]])
                self.check_dist_array_eq(desired_buffers2, dist_load_data1)

        # Cleanup
        physical_mesh.shutdown()

    def test_jax_mlp_save_dist_load(self):
        save_prefix = self._get_save_prefix()

        # Init model
        jax_state, batch, train_step = get_mlp_train_state_and_step(
            batch_size=64,
            hidden_size=16,
            num_layers=4,
            add_manual_pipeline_marker=True)

        with tempfile.TemporaryDirectory(prefix=save_prefix) as ckpt_dir:
            # save normal jax model using tensorstore for distributed loading
            save_checkpoint(ckpt_dir, jax_state, 1)

            # Compile
            method = PipeshardParallel(num_micro_batches=2,
                                       layer_option="manual")
            serial_train_step = train_step
            parallel_train_step = parallelize(train_step, method=method)
            executable = parallel_train_step.get_executable(jax_state, batch)

            # Restore checkpoint
            state_ps, _ = executable.get_input_placement_specs()
            load_state = restore_checkpoint(ckpt_dir, 1, state_ps)

            # Run after load
            serial_state = serial_train_step(jax_state, batch)[0]
            load_state = parallel_train_step(load_state, batch)[0]

            # Check results
            assert_allclose(serial_state.params, load_state.params, 1e-3, 1e-3)

    def test_distributed_mlp_uncached_save_load(self):
        save_prefix = self._get_save_prefix()

        # Init model
        state, batch, train_step = get_mlp_train_state_and_step(
            batch_size=128,
            hidden_size=16,
            num_layers=4,
            add_manual_pipeline_marker=True)

        # Compile
        method = PipeshardParallel(num_micro_batches=1, layer_option="manual")
        serial_train_step = train_step
        parallel_train_step = parallelize(train_step, method=method)
        executable = parallel_train_step.get_executable(state, batch)

        # Run before save
        serial_state = state
        parallel_state = state
        serial_state = serial_train_step(serial_state, batch)[0]
        parallel_state = parallel_train_step(parallel_state, batch)[0]
        assert_allclose(serial_state.params, parallel_state.params, 1e-3, 1e-3)

        # uncached save/load
        with tempfile.TemporaryDirectory(prefix=save_prefix) as ckpt_dir:
            # Save checkpoint
            save_checkpoint(ckpt_dir, parallel_state, 1)

            # Restore checkpoint
            state_ps, _ = executable.get_input_placement_specs()
            load_state = restore_checkpoint(ckpt_dir, 1, state_ps)

            # Run after load
            serial_state = serial_train_step(serial_state, batch)[0]
            load_state = parallel_train_step(load_state, batch)[0]

            # Check results
            assert_allclose(serial_state.params, load_state.params, 1e-3, 1e-3)

    def test_distributed_bert_cached_save_load(self):
        save_prefix = self._get_save_prefix()

        # Init model
        state, batch, train_step = get_bert_layer_train_state_and_step(
            batch_size=16,
            seq_len=8,
            num_layers=4,
            hidden_size=128,
            num_heads=8,
            clip_by_global_norm=False,
            use_dynamic_scale=False,
            add_manual_pipeline_marker=True)

        # Compile
        method = PipeshardParallel(num_micro_batches=2, layer_option="manual")
        serial_train_step = train_step
        parallel_train_step = parallelize(train_step, method=method)
        executable = parallel_train_step.get_executable(state, batch)

        # Run before save
        serial_state = state
        parallel_state = state
        serial_state = serial_train_step(serial_state, batch)[0]
        parallel_state = parallel_train_step(parallel_state, batch)[0]
        assert_allclose(serial_state.params, parallel_state.params, 1e-3, 1e-3)

        # cached save/load
        with tempfile.TemporaryDirectory(prefix=save_prefix) as ckpt_dir:
            with tempfile.TemporaryDirectory(prefix="/tmp/") as cache_dir:
                # Save checkpoint
                save_checkpoint(ckpt_dir, parallel_state, 1, cache_dir)

                # Sync all the move workers
                executable.sync_move_workers()

                # Restore checkpoint
                state_ps, _ = executable.get_input_placement_specs()
                load_state = restore_checkpoint(ckpt_dir, 1, state_ps)

                # Run after load
                serial_state = serial_train_step(serial_state, batch)[0]
                load_state = parallel_train_step(load_state, batch)[0]

                # Check results
                assert_allclose(serial_state.params, load_state.params, 1e-3,
                                1e-3)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(DistSaveLoadTest("test_distributed_array_save_load"))
    suite.addTest(DistSaveLoadTest("test_jax_mlp_save_dist_load"))
    suite.addTest(DistSaveLoadTest("test_distributed_mlp_uncached_save_load"))
    suite.addTest(DistSaveLoadTest("test_distributed_bert_cached_save_load"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
