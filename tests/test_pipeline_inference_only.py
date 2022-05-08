import unittest
import time

import jax
import jax.numpy as jnp
import ray

from alpa.device_mesh import DeviceCluster
from alpa.global_env import set_parallelize_options, global_config
from alpa.model.bert_model import BertConfig
from alpa.util import get_ray_namespace_str
from alpa.testing import (MLPModel, create_train_state, get_mlp_inference_step,
                          assert_allclose, BertLayerModel, get_bert_layer_train_step, PipelineBasicTest)


class PipelineInferenceTest(PipelineBasicTest):
    def run_mlp_inference(self,
                          manual_pipeline_layer=True,
                          pipeline_stage_mode="uniform_stage"):
        virtual_mesh = DeviceCluster().get_virtual_physical_mesh()
        set_parallelize_options(devices=virtual_mesh,
                                strategy="pipeshard_parallel",
                                pipeline_parallel_schedule="inference",
                                pipeline_stage_mode=pipeline_stage_mode)

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
        serial_inference_step = get_mlp_inference_step(False, None)
        parallel_inference_step = get_mlp_inference_step(True, manual_pipeline_layer)
        executable = parallel_inference_step.get_executable(state, batch)
        serial_out = serial_inference_step(state, batch)
        parallel_out = parallel_inference_step(state, batch)
        assert_allclose(serial_out,
                        parallel_out, 1e-3, 1e-3)

        # Run correctnesss test
        hlo_text = executable.get_hlo_text()
        executable.shutdown()
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
