import jax
import jax.numpy as jnp
import numpy as np
import unittest

from alpa import init, parallelize, global_config, PipeshardParallel
from alpa.testing import assert_allclose, get_mlp_train_state_and_step


class MultipleGraphRuntimeTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def run_2_mlp(self, use_value_and_grad=False, stage_option="uniform"):

        def test_one_mlp(method, batch_size=64, hidden_size=16):
            state, batch, train_step = get_mlp_train_state_and_step(
                batch_size=batch_size,
                hidden_size=hidden_size,
                add_manual_pipeline_marker=True)

            # Compile
            serial_train_step = train_step
            parallel_train_step = parallelize(train_step, method=method)
            executable = parallel_train_step.get_executable(state, batch)

            # Run and check
            expected_new_state, expected_val = serial_train_step(state, batch)
            actual_new_state, actual_val = parallel_train_step(state, batch)

            assert_allclose(expected_new_state.params, actual_new_state.params,
                            1e-3, 1e-3)
            assert_allclose(expected_val, actual_val, 1e-3, 1e-3)

            return executable

        method = PipeshardParallel(num_micro_batches=2,
                                   stage_option=stage_option,
                                   layer_option="manual")
        executable = test_one_mlp(method)
        executable_2 = test_one_mlp(method)

        assert executable != executable_2

    def test_2_mlp(self):
        self.run_2_mlp()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(MultipleGraphRuntimeTest('test_2_mlp'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
