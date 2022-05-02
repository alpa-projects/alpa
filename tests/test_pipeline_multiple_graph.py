import numpy as np
import jax
import jax.numpy as jnp
import unittest

from alpa.device_mesh import DeviceCluster
from alpa.global_env import set_parallelize_options, global_config
from alpa.testing import (get_mlp_train_step, MLPModel, PipelineBasicTest,
                          create_train_state, assert_allclose)


class MultipleGraphRuntimeTest(PipelineBasicTest):

    def run_2_mlp(self,
                  manual_pipeline_layer=True,
                  test_remat=False,
                  pipeline_stage_mode="uniform_stage",
                  do_numerical_test=True,
                  return_value=False):

        def test_one_mlp():
            # Init model and optimizer
            batch_size = 64
            hidden_dim = 16
            input_dim = output_dim = hidden_dim

            model = MLPModel(hidden_dim=hidden_dim,
                             output_dim=output_dim,
                             manual_pipeline_layer=manual_pipeline_layer)
            rngkey = jax.random.PRNGKey(0)
            x = jax.random.normal(rngkey, (batch_size, input_dim))
            y = jax.random.normal(rngkey, (batch_size, output_dim))
            batch = {'x': x, 'y': y}
            state = create_train_state(rngkey, model, [x])

            # Compile
            global_config.num_micro_batches = 4
            serial_train_step = get_mlp_train_step(False,
                                                   None,
                                                   None,
                                                   return_value=return_value)
            parallel_train_step = get_mlp_train_step(True,
                                                     manual_pipeline_layer,
                                                     test_remat,
                                                     return_value=return_value)
            executable = parallel_train_step.get_executable(state, batch)

            # Run correctnesss test
            if do_numerical_test:
                expected_new_state = None
                actual_new_state = None
                for i in range(3):
                    if i > 0:
                        state = expected_new_state
                    if return_value:
                        expected_new_state, expected_val = serial_train_step(
                            state, batch)
                    else:
                        expected_new_state, expected_val = serial_train_step(
                            state, batch), 0

                    if i > 0:
                        state = actual_new_state
                    if return_value:
                        actual_new_state, actual_val = parallel_train_step(
                            state, batch)
                    else:
                        actual_new_state, actual_val = parallel_train_step(
                            state, batch), 0

                    assert_allclose(expected_new_state.params,
                                    actual_new_state.params, 1e-3, 1e-3)
                    if return_value:
                        assert_allclose(expected_val, actual_val, 1e-3, 1e-3)

            return executable

        # TODO(zhuohan): Support distributed compile when there are multiple
        #  graphs
        global_config.pipeline_distributed_compile = False
        virtual_mesh = DeviceCluster().get_virtual_physical_mesh()
        set_parallelize_options(devices=virtual_mesh,
                                strategy="pipeshard_parallel",
                                pipeline_stage_mode=pipeline_stage_mode)
        executable = test_one_mlp()
        physical_meshes = executable.physical_meshes
        set_parallelize_options(devices=physical_meshes,
                                strategy="pipeshard_parallel",
                                pipeline_stage_mode=pipeline_stage_mode)
        executable_2 = test_one_mlp()
        executable_2.shutdown()

    def test_2_mlp(self):
        self.run_2_mlp()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(MultipleGraphRuntimeTest('test_2_mlp'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
