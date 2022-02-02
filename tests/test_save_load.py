import unittest
import time
from tempfile import TemporaryFile

import ray
import jax
import jax.numpy as jnp
import numpy as np
import flax

from alpa.util import get_ray_namespace_str, tree_to_nparray
from alpa.device_mesh import DeviceCluster
from alpa.global_env import set_parallelize_options, global_config
from alpa.testing import (MLPModel, create_train_state, get_mlp_train_step, assert_allclose)

class SaveLoadTest:
    def setUp(self):
        ray.init(address="auto",
                 namespace=get_ray_namespace_str(
                     prefix=global_config.unittest_ray_namespace_prefix))
        jax.config.update('jax_platform_name', 'cpu')

    def tearDown(self):
        ray.shutdown()
        time.sleep(1)

    def mlp_state_load(self):
        virtual_mesh = DeviceCluster().get_virtual_physical_mesh()
        set_parallelize_options(devices=virtual_mesh,
                                strategy="3d_parallel",
                                pipeline_stage_mode="uniform_layer_gpipe")

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
        serial_train_step = get_mlp_train_step(False, None, None)
        parallel_train_step = get_mlp_train_step(True, True, False)
        executable = parallel_train_step.get_executable(state, batch)

        serial_state = state
        parallel_state = state
        serial_state = serial_train_step(serial_state, batch)
        parallel_state = parallel_train_step(parallel_state, batch)
        assert_allclose(serial_state.params, parallel_state.params, 1e-3, 1e-3)

        outfile = TemporaryFile()
        jnp.save(outfile, tree_to_nparray(
            flax.serialization.to_state_dict(parallel_state)))
        outfile.seek(0)
        loaded_state = flax.serialization.from_state_dict(state, jnp.load(outfile))
        outfile.close()

        assert_allclose(loaded_state.params, serial_state.params, 1e-3, 1e-3)
        assert_allclose(loaded_state.params, parallel_state.params, 1e-3, 1e-3)

        executable.shutdown()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SaveLoadTest('mlp_state_load'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
