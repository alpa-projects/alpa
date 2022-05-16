import unittest
import time
from tempfile import TemporaryFile

import ray
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import flax

from alpa import init, PipeshardParallel
from alpa.util import tree_to_nparray
from alpa.testing import (MLPModel, create_train_state, get_mlp_train_step,
                          assert_allclose)


class SaveLoadTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def test_mlp_state_load(self):
        # Init model and optimizer
        batch_size = 64
        hidden_dim = 16
        input_dim = output_dim = hidden_dim

        model = MLPModel(hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         manual_pipeline_layer=True)
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

        serial_state = state
        parallel_state = state
        serial_state = serial_train_step(serial_state, batch)[0]
        parallel_state = parallel_train_step(parallel_state, batch)[0]
        assert_allclose(serial_state.params, parallel_state.params, 1e-3, 1e-3)

        # Save model to a temporary file
        outfile = TemporaryFile()
        parallel_state_dict = flax.serialization.to_state_dict(parallel_state)
        pickle.dump(tree_to_nparray(parallel_state_dict), outfile)

        # Load model from the temporary file
        outfile.seek(0)
        loaded_state_dict = pickle.load(outfile)
        loaded_state = flax.serialization.from_state_dict(
            state, loaded_state_dict)
        outfile.close()

        # Compare the loaded state with the original state
        assert_allclose(loaded_state.params, serial_state.params, 1e-3, 1e-3)
        assert_allclose(loaded_state.params, parallel_state.params, 1e-3, 1e-3)

        # Take a step with the loaded state on both serial and parallel version
        serial_state = serial_train_step(serial_state, batch)[0]
        parallel_state = parallel_train_step(parallel_state, batch)[0]
        serial_loaded_state = serial_train_step(loaded_state, batch)[0]
        parallel_loaded_state = parallel_train_step(loaded_state, batch)[0]

        # All the states should be the same
        assert_allclose(serial_state.params, parallel_state.params, 1e-3, 1e-3)
        assert_allclose(serial_state.params, serial_loaded_state.params, 1e-3,
                        1e-3)
        assert_allclose(serial_state.params, parallel_loaded_state.params, 1e-3,
                        1e-3)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SaveLoadTest('test_mlp_state_load'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
