import unittest

import jax
import jax.numpy as jnp

from alpa import init, PipeshardParallel
from alpa.testing import (MLPModel, create_train_state, get_mlp_inference_step,
                          assert_allclose)


class PipelineInferenceTest(unittest.TestCase):
    def setUp(self):
        init(cluster="ray")

    def run_mlp_inference(self,
                          manual_pipeline_layer=True):
        method = PipeshardParallel(num_micro_batches=4,
                                   pipeline_schedule="inference")

        # Init model and optimizer
        batch_size = 64
        hidden_dim = 16
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
        serial_inference_step = get_mlp_inference_step(None, None)
        parallel_inference_step = get_mlp_inference_step(method, manual_pipeline_layer)
        executable = parallel_inference_step.get_executable(state, batch)

        # Run correctnesss test
        serial_out = serial_inference_step(state, batch)
        parallel_out = parallel_inference_step(state, batch)
        assert_allclose(serial_out, parallel_out, 1e-3, 1e-3)

        hlo_text = executable.get_hlo_text()
        return hlo_text

    def test_pipeline_inference_only(self):
        self.run_mlp_inference()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineInferenceTest("test_pipeline_inference_only"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
