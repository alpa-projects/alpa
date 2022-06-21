"""Test distributed save and load."""

import subprocess
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from alpa import (init, shutdown, DistributedArray, PipeshardParallel,
                  save_checkpoint, restore_checkpoint)
from alpa.device_mesh import get_global_cluster
from alpa.model.bert_model import BertConfig
from alpa.model.model_util import TrainState
from alpa.testing import (MLPModel, BertLayerModel, create_train_state,
                          get_bert_layer_train_step, get_mlp_train_step,
                          assert_allclose)


class DistSaveLoadTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def tearDown(self):
        shutdown()

    def check_dist_array_eq(self, x, y):
        if isinstance(x, DistributedArray):
            x = np.array(
                x.device_mesh.get_remote_buffers(x.remote_buffers,
                                                 batching=True))
        if isinstance(y, DistributedArray):
            y = np.array(
                y.device_mesh.get_remote_buffers(y.remote_buffers,
                                                 batching=True))
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

        # Init model and optimizer
        batch_size = 64
        hidden_dim = 16
        input_dim = output_dim = hidden_dim
        model = MLPModel(hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         manual_pipeline_layer=True)

        # Init batch args
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, input_dim), jnp.float32)
        y = jax.random.normal(rngkey, (batch_size, output_dim), jnp.float32)
        batch = {'x': x, 'y': y}
        jax_state = create_train_state(rngkey, model, [x])

        with tempfile.TemporaryDirectory(prefix=save_prefix) as ckpt_dir:
            # save normal jax model using tensorstore for distributed loading
            save_checkpoint(ckpt_dir, jax_state, 1)

            # Compile
            method = PipeshardParallel(num_micro_batches=2)
            serial_train_step = get_mlp_train_step(None, None, None, False)
            parallel_train_step = get_mlp_train_step(method, True, False, False)
            executable = parallel_train_step.get_executable(jax_state, batch)

            # Restore checkpoint
            state_ss, _ = executable.get_load_info()
            load_state = restore_checkpoint(ckpt_dir, 1, state_ss)

            # Run after load
            serial_state = serial_train_step(jax_state, batch)[0]
            load_state = parallel_train_step(load_state, batch)[0]

        # Check results
        assert_allclose(serial_state.params, load_state.params, 1e-3, 1e-3)

    def test_distributed_mlp_uncached_save_load(self):
        save_prefix = self._get_save_prefix()

        # Init model and optimizer
        batch_size = 64
        hidden_dim = 16
        input_dim = output_dim = hidden_dim
        model = MLPModel(hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         manual_pipeline_layer=True)

        # Init batch args
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, input_dim), jnp.float32)
        y = jax.random.normal(rngkey, (batch_size, output_dim), jnp.float32)
        batch = {'x': x, 'y': y}
        state = create_train_state(rngkey, model, [x])

        # Compile
        method = PipeshardParallel(num_micro_batches=2)
        serial_train_step = get_mlp_train_step(None, None, None, False)
        parallel_train_step = get_mlp_train_step(method, True, False, False)
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
            state_ss, _ = executable.get_load_info()
            load_state = restore_checkpoint(ckpt_dir, 1, state_ss)

            # Run after load
            serial_state = serial_train_step(serial_state, batch)[0]
            load_state = parallel_train_step(load_state, batch)[0]
        # Check results
        assert_allclose(serial_state.params, load_state.params, 1e-3, 1e-3)

    def test_distributed_bert_cached_save_load(self):
        save_prefix = self._get_save_prefix()

        # Config
        batch_size = 16
        seq_len = 8
        hidden_size = 128
        num_heads = 8
        n_layers = 4
        dtype = jnp.float32

        # Init batch args
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=dtype)
        y = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=dtype)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=dtype)
        batch = {"x": x, "y": y, "attention_mask": attention_mask}

        # Init model and optimizer
        model = BertLayerModel(config=BertConfig(hidden_size=hidden_size,
                                                 intermediate_size=hidden_size *
                                                 4,
                                                 num_attention_heads=num_heads,
                                                 num_hidden_layers=n_layers))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x, attention_mask)
        tx = optax.sgd(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=tx,
                                  dynamic_scale=None)

        # Compile
        method = PipeshardParallel(num_micro_batches=2)
        serial_train_step = get_bert_layer_train_step(None, None, None,
                                                      n_layers, False)
        parallel_train_step = get_bert_layer_train_step(method, True, False,
                                                        n_layers, False)
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
                state_ss, _ = executable.get_load_info()
                load_state = restore_checkpoint(ckpt_dir, 1, state_ss)

                # Run after load
                serial_state = serial_train_step(serial_state, batch)[0]
                load_state = parallel_train_step(load_state, batch)[0]
        # Check results
        assert_allclose(serial_state.params, load_state.params, 1e-3, 1e-3)


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
