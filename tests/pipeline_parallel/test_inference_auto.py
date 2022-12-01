import unittest

import jax
import jax.numpy as jnp
import numpy as np

from alpa import (init, shutdown, parallelize, PipeshardParallel,
                  mark_pipeline_boundary, AutoStageOption)
from alpa.model.bert_model import BertConfig, FlaxBertLayerCollection
from alpa.testing import (MLPModel, create_train_state, mlp_inference_step,
                          bert_layer_collection_inference_step, assert_allclose)


class PipelineInferenceAutoTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    # pylint: disable=no-self-use
    def tearDown(self):
        shutdown()

    def run_mlp_inference(self, manual_pipeline_layer):
        stage_option = AutoStageOption(
            submesh_physical_shape_space="manual",
            manually_specified_submeshes=((1, 2),),
            submesh_logical_shape_space="model_parallel_only")
        method = PipeshardParallel(num_micro_batches=1,
                                   pipeline_schedule="inference",
                                   layer_option="manual",
                                   stage_option=stage_option)

        # Init model and optimizer
        batch_size = 64
        hidden_size = 16

        model = MLPModel(hidden_size=hidden_size,
                         num_layers=4,
                         add_manual_pipeline_marker=manual_pipeline_layer)
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, hidden_size))
        y = jax.random.normal(rngkey, (batch_size, hidden_size))
        batch = {'x': x, 'y': y}
        state = create_train_state(rngkey, model, [x])

        # Compile
        serial_inference_step = mlp_inference_step

        parallel_inference_step = parallelize(mlp_inference_step,
                                              method=method,
                                              donate_argnums=())
        executable = parallel_inference_step.get_executable(state, batch)

        # Run correctnesss test
        serial_out = serial_inference_step(state, batch)
        parallel_out = parallel_inference_step(state, batch)
        assert_allclose(serial_out, parallel_out, 1e-3, 1e-3)

    def test_mlp(self):
        self.run_mlp_inference(True)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineInferenceAutoTest("test_mlp"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
