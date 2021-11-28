import jax
import jax.numpy as jnp
import unittest

from parax.api import parallelize
from parax.device_mesh import DeviceCluster
from parax.global_env import set_parallelize_options, global_config
from parax.model.bert_model import BertConfig
from parax.testing import (get_bert_layer_train_step, BertLayerModel, PipelineBasicTest, create_train_state)


class AutoStageClusteringOOMTest(PipelineBasicTest):

    def run_n_layer_bert(self,
                         n_layers,
                         manual_pipeline_layer=True,
                         test_remat=False,
                         pipeline_stage_mode="uniform_layer_gpipe",
                         cache_compute_cost=None,
                         forward_stage_layer_ids=None,
                         batch_size=16,
                         seq_len=256,
                         hidden_size=512,
                         num_heads=512 // 64,
                         submesh_shapes=None):
        virtual_mesh = DeviceCluster().get_virtual_mesh()
        set_parallelize_options(devices=virtual_mesh,
                                strategy="3d_parallel",
                                pipeline_stage_mode=pipeline_stage_mode,
                                cache_compute_cost=cache_compute_cost,
                                forward_stage_layer_ids=forward_stage_layer_ids,
                                sub_physical_mesh_shapes=submesh_shapes)

        train_step = get_bert_layer_train_step(manual_pipeline_layer,
                                               test_remat, n_layers)

        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=jnp.float32)
        y = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=jnp.float32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

        # Init model and optimizer
        model = BertLayerModel(config=BertConfig(hidden_size=hidden_size,
                                                 intermediate_size=hidden_size *
                                                 4,
                                                 num_attention_heads=num_heads,
                                                 num_hidden_layers=n_layers),
                               manual_pipeline_layer=manual_pipeline_layer)
        batch = {"x": x, "y": y, "attention_mask": attention_mask}
        state = create_train_state(rngkey, model, [x, attention_mask])

        global_config.num_micro_batches = 2
        parallel_train_step = parallelize(train_step)
        executable = parallel_train_step.get_executable(state, batch)
        new_state = parallel_train_step(state, batch)

        executable.shutdown()

    def test_GPT3_3B(self):
        self.run_n_layer_bert(n_layers=32,
                              hidden_size=2560,
                              batch_size=16,
                              seq_len=1024,
                              num_heads=32,
                              pipeline_stage_mode="auto_gpipe",
                              cache_compute_cost=None)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoStageClusteringOOMTest('test_GPT3_3B'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
