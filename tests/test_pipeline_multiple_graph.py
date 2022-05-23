import jax
import jax.numpy as jnp
import numpy as np
import unittest

from alpa import init, global_config, PipeshardParallel
from alpa.testing import (get_mlp_train_step, MLPModel,
                          create_train_state, assert_allclose)


class MultipleGraphRuntimeTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def run_2_mlp(self,
                  manual_pipeline_layer=True,
                  use_remat=False,
                  use_value_and_grad=False,
                  stage_mode="uniform",
                  do_numerical_test=True):

        def test_one_mlp(method, batch_size=64, hidden_dim=16):
            # Init model and optimizer
            input_dim = output_dim = hidden_dim

            model = MLPModel(hidden_dim=hidden_dim,
                             output_dim=output_dim,
                             manual_pipeline_layer=manual_pipeline_layer)
            rngkey = jax.random.PRNGKey(0)
            x = jax.random.normal(rngkey, (batch_size, input_dim), jnp.float32)
            y = jax.random.normal(rngkey, (batch_size, output_dim), jnp.float32)
            batch = {'x': x, 'y': y}
            state = create_train_state(rngkey, model, [x])

            # Compile
            serial_train_step = get_mlp_train_step(None,
                                                   None,
                                                   None,
                                                   use_value_and_grad)
            parallel_train_step = get_mlp_train_step(method,
                                                     manual_pipeline_layer,
                                                     use_remat,
                                                     use_value_and_grad)
            executable = parallel_train_step.get_executable(state, batch)

            expected_new_state, expected_val = serial_train_step(
                state, batch)
            actual_new_state, actual_val = parallel_train_step(
                state, batch)

            assert_allclose(expected_new_state.params,
                            actual_new_state.params, 1e-3, 1e-3)
            assert_allclose(expected_val, actual_val, 1e-3, 1e-3)

            return executable

        method = PipeshardParallel(stage_mode=stage_mode,
                                   num_micro_batches=2)
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
