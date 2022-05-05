"""Test distributed save and load."""

from re import X
import subprocess
import tempfile
import unittest

from flax import linen as nn
from flax import optim
from flax.serialization import to_state_dict
import jax
import jax.numpy as jnp
import numpy as np
import ray

from alpa import DeviceCluster, DistributedArray, parallelize, set_parallelize_options, save_checkpoint
from alpa.global_env import set_parallelize_options, global_config
from alpa.pipeline_parallel.primitive_def import mark_pipeline
from alpa.testing import (MLPModel, create_train_state, get_mlp_train_step,
                          assert_allclose)


class DistSaveLoadTest(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto", ignore_reinit_error=True)

    def tearDown(self):
        ray.shutdown()

    def test_distributed_array_save_load(self):
        # Launch a device mesh contains four devices
        device_cluster = DeviceCluster()
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
        remote_buffers1 = dist_input_data1.device_mesh.get_remote_buffers(
            dist_input_data1.remote_buffers, batching=True)
        desired_buffers1 = [[[0], [2]], [[1], [3]], [[4], [6]], [[5], [7]]]
        assert_allclose(np.array(desired_buffers1), np.array(remote_buffers1))

        # Save the DistributedArray (one replica only)
        tmpdir = tempfile.TemporaryDirectory()
        subprocess.run(["rm", "-rf", tmpdir.name])
        dist_input_data1.save(tmpdir.name)

        # Load previously saved DistributedArray with a different shardingSpec
        # [[0,1],          [[0,1],  [[0,1],
        #  [2,3],  shard    [2,3]]   [2,3]]
        #  [4,5],  ====>   [[4,5],  [[4,5],
        #  [6,7]]           [6,7]]   [6,7]]
        load_sharding_spec = logical_mesh.make_tile_spec(
            global_input_data1, [0, 1], [0])
        dist_load_data1 = DistributedArray.load(
            tmpdir.name, jax.ShapedArray(global_input_data1.shape, jnp.int32),
            physical_mesh, load_sharding_spec)

        # Check the DistributedArray's remote buffers
        remote_buffers2 = dist_load_data1.device_mesh.get_remote_buffers(
            dist_load_data1.remote_buffers, batching=True)
        desired_buffers2 = [[[0, 1], [2, 3]], [[0, 1], [2, 3]], [[4, 5], [6,
                                                                          7]],
                            [[4, 5], [6, 7]]]
        assert_allclose(np.array(desired_buffers2), np.array(remote_buffers2))

        # Cleanup
        physical_mesh.shutdown()
    
    # def test_distributed_mlp_save_load(self):
    #     # Launch a multi-host device mesh
    #     virtual_mesh = DeviceCluster().get_virtual_physical_mesh()
    #     set_parallelize_options(devices=virtual_mesh,
    #                             strategy="pipeshard_parallel",
    #                             pipeline_stage_mode="uniform_stage")

    #     batch_size = 16
    #     input_dim = hidden_dim = output_dim = 32

    #     class Model(nn.Module):

    #         @nn.compact
    #         def __call__(self, x):
    #             x = nn.Dense(features=hidden_dim)(x)
    #             x = nn.relu(x)
    #             x = nn.Dense(features=output_dim)(x)
    #             return x

    #     def train_step(optimizer, batch, apply_fn):

    #         def loss_func(params):
    #             out = apply_fn(params, batch["x"])
    #             return jnp.mean((out - batch["y"])**2)

    #         grad = jax.grad(loss_func)(optimizer.target)
    #         new_optimizer = optimizer.apply_gradient(grad)
    #         return new_optimizer

    #     # One batch of data and label
    #     batch = {
    #         "x": np.random.randn(batch_size, input_dim),
    #         "y": np.random.randn(batch_size, output_dim),
    #     }

    #     # Init model and optimizer
    #     model = Model()
    #     rngkey = jax.random.PRNGKey(0)
    #     state = create_train_state(rngkey, model, [batch["x"]])

    #     # Compile
    #     global_config.num_micro_batches = 2
    #     serial_train_step = get_mlp_train_step(False, None, None, False)
    #     parallel_train_step = get_mlp_train_step(True, True, False, False)

    #     serial_state = state
    #     parallel_state = state
    #     serial_state = serial_train_step(serial_state, batch)[0]
    #     parallel_state = parallel_train_step(parallel_state, batch)[0]
    #     assert_allclose(serial_state.params, parallel_state.params, 1e-3, 1e-3)

    #     # Checkpoint (TODO: remove hard-coded path)
    #     # ckpt_dir1 = "/home/ubuntu/efs/ckpt1"
    #     # subprocess.run(["rm", "-rf", ckpt_dir1])
    #     # save_checkpoint(ckpt_dir1, parallel_state, 1)

    #     # Restore checkpoint
    #     # train_step_parallel_restore = parallelize(train_step)
    #     # executable = train_step_parallel_restore.get_executable(optimizer, batch, model.apply)
    #     # optimizer_parallel_load = restore_checkpoint(ckpt_dir1, 1)

    #     # Check results
    #     # assert_allclose(optimizer_serial.target, optimizer_parallel_load.target)

    #     # Cleanup
    #     # executable.shutdown()

    def test_distributed_mlp_save_load(self):
        virtual_mesh = DeviceCluster().get_virtual_physical_mesh()
        set_parallelize_options(devices=virtual_mesh,
                                strategy="pipeshard_parallel",
                                pipeline_stage_mode="uniform_stage")

        # Init model and optimizer
        batch_size = 64
        hidden_dim = 16
        input_dim = output_dim = hidden_dim
        model = MLPModel(hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         manual_pipeline_layer=True)
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, input_dim))
        y = jax.random.normal(rngkey, (batch_size, output_dim))
        batch = {'x': x, 'y': y}
        state = create_train_state(rngkey, model, [x])

        # Compile
        global_config.num_micro_batches = 2
        serial_train_step = get_mlp_train_step(False, None, None, False)
        parallel_train_step = get_mlp_train_step(True, True, False, False)
        executable = parallel_train_step.get_executable(state, batch)

        serial_state = state
        parallel_state = state
        serial_state = serial_train_step(serial_state, batch)[0]
        parallel_state = parallel_train_step(parallel_state, batch)[0]
        assert_allclose(serial_state.params, parallel_state.params, 1e-3, 1e-3)

        # Checkpoint (TODO: remove hard-coded path)
        ckpt_dir1 = "/home/ubuntu/efs/ckpt1"
        subprocess.run(["rm", "-rf", ckpt_dir1])
        save_checkpoint(ckpt_dir1, parallel_state, 1)

        # Restore checkpoint
        # train_step_parallel_restore = parallelize(train_step)
        # executable = train_step_parallel_restore.get_executable(optimizer, batch, model.apply)
        # optimizer_parallel_load = restore_checkpoint(ckpt_dir1, 1)

        # Check results
        # assert_allclose(optimizer_serial.target, optimizer_parallel_load.target)

        # Cleanup
        executable.shutdown()

def suite():
    suite = unittest.TestSuite()
    # suite.addTest(DistSaveLoadTest("test_distributed_array_save_load"))
    suite.addTest(DistSaveLoadTest("test_distributed_mlp_save_load"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
