import unittest
import time
from tempfile import TemporaryFile

import ray
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import flax

from alpa import init, parallelize, PipeshardParallel, util
from alpa.testing import get_mlp_train_state_and_step, assert_allclose


class SaveLoadTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def test_mlp_state_load(self):
        # Init model
        state, batch, train_step = get_mlp_train_state_and_step(
            batch_size=128, hidden_size=128, add_manual_pipeline_marker=True)

        # Compile
        method = PipeshardParallel(num_micro_batches=2, layer_option="manual")
        serial_train_step = train_step
        parallel_train_step = parallelize(train_step, method=method)
        executable = parallel_train_step.get_executable(state, batch)

        serial_state = state
        parallel_state = state
        serial_state = serial_train_step(serial_state, batch)[0]
        parallel_state = parallel_train_step(parallel_state, batch)[0]
        assert_allclose(serial_state.params, parallel_state.params, 1e-3, 1e-3)

        # Save model to a temporary file
        outfile = TemporaryFile()
        parallel_state_dict = flax.serialization.to_state_dict(parallel_state)
        pickle.dump(util.map_to_nparray(parallel_state_dict), outfile)

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
