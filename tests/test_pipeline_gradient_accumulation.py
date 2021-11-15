import unittest
import time

import jax
import jax.numpy as jnp
import ray

import parax
from parax import (parallelize, global_config, set_parallelize_options,
                   DeviceCluster)
from parax.model.bert_model import BertConfig
from parax.pipeline_parallel.primitive_def import mark_pipeline
from parax.testing import (MLPModel, TwoLayerBertLayerModel, BertLayerModel,
                           assert_allclose, create_train_state,
                           decorate_loss_fn)


class AccumulateGradTest(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto")
        jax.config.update('jax_platform_name', 'cpu')

    def tearDown(self):
        ray.shutdown()
        time.sleep(1)

    def run_mlp(self,
                manual_pipeline_layer=True,
                test_remat=False,
                pipeline_stage_mode="uniform_layer_gpipe"):
        virtual_mesh = DeviceCluster().get_virtual_mesh()
        set_parallelize_options(devices=virtual_mesh,
                                strategy="3d_parallel",
                                pipeline_stage_mode=pipeline_stage_mode)
        batch_size = 256
        hidden_dim = 128
        input_dim = output_dim = hidden_dim

        model = MLPModel(hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         manual_pipeline_layer=manual_pipeline_layer)
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, input_dim))
        y = jax.random.normal(rngkey, (batch_size, output_dim))
        batch = {'x': x, 'y': y}
        state = create_train_state(rngkey, model, [x])

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                loss = jnp.mean((out - batch["y"])**2)
                if manual_pipeline_layer:
                    mark_pipeline(name='2', mark_type='end')
                return loss

            loss_func = decorate_loss_fn(loss_func, manual_pipeline_layer,
                                         test_remat, 2)

            param_grad = parax.grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=param_grad)
            return new_state

        global_config.num_micro_batches = 4

        nstep = 3
        parallel_train_step = parallelize(train_step)
        executable = parallel_train_step.get_executable(state, batch)
        expected_new_state = None
        actual_new_state = None
        for i in range(nstep):
            if i > 0:
                state = expected_new_state
            expected_new_state = train_step(state, batch)
            if i > 0:
                state = actual_new_state
            actual_new_state = parallel_train_step(state, batch)
            assert_allclose(expected_new_state.params, actual_new_state.params,
                            1e-3, 1e-3)

        executable.shutdown()

    def run_2_layer_bert(self,
                         manual_pipeline_layer=True,
                         test_remat=False,
                         pipeline_stage_mode="uniform_layer_gpipe"):
        virtual_mesh = DeviceCluster().get_virtual_mesh()
        set_parallelize_options(devices=virtual_mesh,
                                strategy="3d_parallel",
                                pipeline_stage_mode=pipeline_stage_mode)

        batch_size = 16
        seq_len = 8
        hidden_size = 512
        num_heads = 8

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"],
                                     batch["attention_mask"])
                loss = jnp.mean((out - batch["y"])**2)
                if manual_pipeline_layer:
                    mark_pipeline(name='2', mark_type='end')
                return loss

            loss_func = decorate_loss_fn(loss_func, manual_pipeline_layer,
                                         test_remat, 2)

            grad_param = parax.grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grad_param)
            return new_state

        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=jnp.float32)
        y = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=jnp.float32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

        # Init model and optimizer
        model = TwoLayerBertLayerModel(
            config=BertConfig(hidden_size=hidden_size,
                              intermediate_size=hidden_size * 4,
                              num_attention_heads=num_heads),
            manual_pipeline_layer=manual_pipeline_layer)
        batch = {"x": x, "y": y, "attention_mask": attention_mask}
        state = create_train_state(rngkey, model, [x, attention_mask])

        global_config.num_micro_batches = 2
        parallel_train_step = parallelize(train_step)
        executable = parallel_train_step.get_executable(state, batch)
        expected_new_state = None
        actual_new_state = None

        # Test ReplicatedDistributedArray correctness
        for i in range(3):
            if i > 0:
                state = expected_new_state
            expected_new_state = train_step(state, batch)
            if i > 0:
                state = actual_new_state
            actual_new_state = parallel_train_step(state, batch)
            assert_allclose(expected_new_state.params, actual_new_state.params,
                            1e-3, 1e-3)

        executable.shutdown()

    def run_n_layer_bert(self,
                         n_layers,
                         manual_pipeline_layer=True,
                         test_remat=False,
                         pipeline_stage_mode="uniform_layer_gpipe",
                         cache_compute_cost=None,
                         forward_stage_layer_ids=None,
                         submesh_shapes=None):
        virtual_mesh = DeviceCluster().get_virtual_mesh()
        set_parallelize_options(devices=virtual_mesh,
                                strategy="3d_parallel",
                                pipeline_stage_mode=pipeline_stage_mode,
                                cache_compute_cost=cache_compute_cost,
                                forward_stage_layer_ids=forward_stage_layer_ids,
                                sub_physical_mesh_shapes=submesh_shapes)

        batch_size = 16
        seq_len = 256
        hidden_size = 512
        num_heads = 512 // 64

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"],
                                     batch["attention_mask"])
                loss = jnp.mean((out - batch["y"])**2)
                if manual_pipeline_layer:
                    mark_pipeline(name=str(n_layers - 1), mark_type='end')
                return loss

            loss_func = decorate_loss_fn(loss_func, manual_pipeline_layer,
                                         test_remat, n_layers)

            grad_param = parax.grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grad_param)
            return new_state

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
        expected_new_state = None
        actual_new_state = None

        for i in range(3):
            if i > 0:
                state = expected_new_state
            expected_new_state = train_step(state, batch)
            if i > 0:
                state = actual_new_state
            actual_new_state = parallel_train_step(state, batch)
            assert_allclose(expected_new_state.params, actual_new_state.params,
                            1e-3, 1e-3)

        executable.shutdown()

    def test_mlp(self):
        self.run_mlp()

    def test_mlp_auto_layer_slicing(self):
        self.run_mlp(manual_pipeline_layer=False)

    def test_mlp_auto_stage_clustering(self):
        self.run_mlp(pipeline_stage_mode="auto_gpipe")

    def test_mlp_auto_layer_and_stage(self):
        self.run_mlp(manual_pipeline_layer=False,
                     pipeline_stage_mode="auto_gpipe")

    def test_mlp_remat(self):
        self.run_mlp(test_remat=True)

    def test_2_layer_bert(self):
        self.run_2_layer_bert()

    def test_2_layer_bert_auto_layer_slicing(self):
        self.run_2_layer_bert(manual_pipeline_layer=False)

    def test_2_layer_bert_auto_stage_clustering(self):
        self.run_2_layer_bert(pipeline_stage_mode="auto_gpipe")

    def test_2_layer_bert_auto_layer_and_stage(self):
        self.run_2_layer_bert(manual_pipeline_layer=False,
                              pipeline_stage_mode="auto_gpipe")

    def test_2_layer_bert_remat(self):
        self.run_2_layer_bert(test_remat=True)

    def test_2_layer_bert_auto_layer_slicing_remat(self):
        self.run_2_layer_bert(manual_pipeline_layer=False, test_remat=True)

    @unittest.skipIf(jax.device_count('gpu') < 8, "no enough device")
    def test_8_layer_bert(self):
        self.run_n_layer_bert(n_layers=8)

    @unittest.skipIf(jax.device_count('gpu') < 8, "no enough device")
    def test_8_layer_bert_manual_stage_assignment(self):
        self.run_n_layer_bert(n_layers=8,
                              pipeline_stage_mode="manual_gpipe",
                              forward_stage_layer_ids=[[0, 1, 2, 3],
                                                       [4, 5, 6, 7]],
                              submesh_shapes=[(1, 4), (1, 4)])

    @unittest.skipIf(jax.device_count('gpu') < 8, "no enough device")
    def test_8_layer_bert_auto_layer_slicing(self):
        self.run_n_layer_bert(n_layers=8, manual_pipeline_layer=False)

    def test_8_layer_bert_auto_stage_clustering(self):
        self.run_n_layer_bert(n_layers=8,
                              pipeline_stage_mode="auto_gpipe",
                              cache_compute_cost=None)

    def test_8_layer_bert_auto_layer_and_stage(self):
        self.run_n_layer_bert(n_layers=8,
                              manual_pipeline_layer=False,
                              pipeline_stage_mode="auto_gpipe",
                              cache_compute_cost=None)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AccumulateGradTest('test_mlp'))
    suite.addTest(AccumulateGradTest('test_mlp_auto_stage_clustering'))
    suite.addTest(AccumulateGradTest('test_mlp_remat'))
    # FIXME(zhuohan): The following 2 tests are failing because stage slicing
    #   in XLA will move the stages around and thus don't have correct order
    #   if stages on a same mesh doesn't have dependecies. Need to fix this
    #   in stage slicing in XLA.
    # suite.addTest(AccumulateGradTest('test_mlp_auto_layer_slicing'))
    # suite.addTest(AccumulateGradTest('test_mlp_auto_layer_and_stage'))
    suite.addTest(AccumulateGradTest('test_2_layer_bert'))
    suite.addTest(AccumulateGradTest('test_2_layer_bert_auto_layer_slicing'))
    suite.addTest(AccumulateGradTest('test_2_layer_bert_auto_stage_clustering'))
    suite.addTest(AccumulateGradTest('test_2_layer_bert_auto_layer_and_stage'))
    suite.addTest(AccumulateGradTest('test_2_layer_bert_remat'))
    suite.addTest(
        AccumulateGradTest('test_2_layer_bert_auto_layer_slicing_remat'))
    suite.addTest(AccumulateGradTest('test_8_layer_bert'))
    suite.addTest(
        AccumulateGradTest('test_8_layer_bert_manual_stage_assignment'))
    # suite.addTest(AccumulateGradTest('test_8_layer_bert_auto_layer_slicing'))
    # suite.addTest(AccumulateGradTest('test_8_layer_bert_auto_stage_clustering'))
    # suite.addTest(AccumulateGradTest('test_8_layer_bert_auto_layer_and_stage'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
