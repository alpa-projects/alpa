import unittest
from typing import Sequence

from jax._src.api import make_jaxpr
from jax.core import ClosedJaxpr, Var, gensym
import jax.numpy as jnp

from alpa import init, grad, parallelize, PipeshardParallel
from alpa.device_mesh import get_global_virtual_physical_mesh
from alpa.pipeline_parallel.stage_construction import (
    AutoStageOption, get_one_submesh_autosharding_config_choices)
from alpa.pipeline_parallel.compile_executable import (
    split_and_process_layers, slice_apply_grad_for_stage_construction)
from alpa.pipeline_parallel.layer_construction import ManualLayerOption
from alpa.pipeline_parallel.stage_profiling import (
    generate_stage_info, distributed_profile_on_mesh,
    get_merged_stages_memory_stats)
from alpa.shard_parallel.auto_sharding import AutoShardingOption
from alpa.testing import (get_bert_layer_train_state_and_step,
                          get_mlp_train_state_and_step)
from alpa.util import GradFuncTransformContext


def _aval_key(a):
    return (a.shape, repr(a.dtype))


def _assert_avals_allmatch(aval_seq_a, aval_seq_b):
    assert len(aval_seq_a) == len(
        aval_seq_b), f"{len(aval_seq_a)} != {len(aval_seq_b)}"
    aval_seq_a = sorted(aval_seq_a, key=_aval_key)
    aval_seq_b = sorted(aval_seq_b, key=_aval_key)
    for a, b in zip(aval_seq_a, aval_seq_b):
        assert a.shape == b.shape and a.dtype == b.dtype


class StageConstructUtilTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray", num_nodes=1, num_devices_per_node=1)

    def create_bert_layers(self, num_layers, num_microbatch):
        batch_size = 16
        state, batch, _ = get_bert_layer_train_state_and_step(
            batch_size=batch_size,
            seq_len=256,
            num_layers=num_layers,
            hidden_size=512,
            num_heads=512 // 64,
            clip_by_global_norm=False,
            use_dynamic_scale=False,
            add_manual_pipeline_marker=True,
        )

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"],
                                     batch["attention_mask"])
                loss = jnp.mean((out - batch["y"])**2)
                return loss

            grads = grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        microbatch_size = batch_size // num_microbatch
        micro_batch = {k: v[:microbatch_size] for k, v in batch.items()}
        return train_step, state, batch, micro_batch

    def create_mlp(self, num_microbatch, add_marker=True):
        batch_size = 16
        state, batch, train_step = get_mlp_train_state_and_step(
            batch_size=batch_size,
            hidden_size=512,
            num_layers=4,
            use_bias=False,
            add_manual_pipeline_marker=add_marker)

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"])**2)

            grads = grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        microbatch_size = batch_size // num_microbatch
        micro_batch = {k: v[:microbatch_size] for k, v in batch.items()}
        return train_step, state, batch, micro_batch

    def get_train_step_jaxpr(self,
                             train_step,
                             state,
                             batch,
                             micro_batch,
                             use_remat=False):
        # Compile
        with GradFuncTransformContext(ManualLayerOption(use_remat).transform):
            closed_jaxpr, output_tree = make_jaxpr(train_step,
                                                   return_shape=True)(
                                                       state, micro_batch)
            full_batch_closed_jaxpr, full_batch_output_tree = make_jaxpr(
                train_step, return_shape=True)(state, batch)

        num_params = len(closed_jaxpr.jaxpr.invars) - len(batch)
        donated_invars = [True] * num_params + [False] * len(batch)
        return closed_jaxpr, full_batch_closed_jaxpr, donated_invars

    def pre_process_jaxpr(self, closed_jaxpr: ClosedJaxpr,
                          full_batch_closed_jaxpr: ClosedJaxpr,
                          num_microbatch: int, donated_invars: Sequence[bool]):
        inference_mode = False
        gensym_func = gensym([closed_jaxpr.jaxpr])
        global_invars = closed_jaxpr.jaxpr.invars

        (closed_jaxpr, global_outvars, jax_pipeline_layers, apply_grad_jaxpr,
         microbatch_bound, reduction_vector, post_microbatch_bound,
         accumulator_mapping, acc_grad_invars,
         acc_grad_outvars) = (split_and_process_layers(closed_jaxpr,
                                                       full_batch_closed_jaxpr,
                                                       num_microbatch,
                                                       inference_mode,
                                                       gensym_func))

        (jax_apply_layers,
         apply_grad_global_info) = slice_apply_grad_for_stage_construction(
             jax_pipeline_layers, apply_grad_jaxpr, microbatch_bound,
             global_invars, global_outvars, donated_invars, accumulator_mapping,
             gensym_func, inference_mode)

        return (closed_jaxpr, global_outvars, jax_pipeline_layers,
                apply_grad_jaxpr, microbatch_bound, reduction_vector,
                post_microbatch_bound, accumulator_mapping, acc_grad_invars,
                acc_grad_outvars, jax_apply_layers, apply_grad_global_info)

    def generate_profile_result(self, jax_pipeline_layers, accumulator_mapping,
                                acc_grad_invars, acc_grad_outvars,
                                jax_apply_layers, apply_grad_global_info,
                                num_micro_batches, start_index, end_index):
        virtual_mesh = get_global_virtual_physical_mesh()
        submesh = (1, 1)
        virtual_submesh = virtual_mesh.slice_2d(tuple(range(
            submesh[0])), (tuple(range(submesh[1])),) * submesh[0])
        auto_sharding_config = get_one_submesh_autosharding_config_choices(
            virtual_submesh, "same_as_physical", batch_size=None)[0]

        assert len(jax_pipeline_layers) % 2 == 0
        num_layers = len(jax_pipeline_layers) // 2
        indices = list(range(2 * num_layers))

        forward_layer_indices = indices[start_index:end_index + 1]
        backward_layer_indices = indices[2 * num_layers - end_index -
                                         1:2 * num_layers - start_index]
        selected_apply_grad_layers = [
            jax_apply_layers[idx]
            for idx in forward_layer_indices
            if jax_apply_layers[idx] is not None
        ]

        stage_config = generate_stage_info(
            jax_pipeline_layers,
            [forward_layer_indices, backward_layer_indices],
            accumulator_mapping, acc_grad_invars, acc_grad_outvars,
            "test_stage", selected_apply_grad_layers, apply_grad_global_info)

        stage_index = 0
        stage = (stage_index, stage_config, auto_sharding_config)

        profile_results = {}
        default_as_option = AutoShardingOption(prefer_reduce_scatter=True)
        auto_stage_option = AutoStageOption()

        profile_results = distributed_profile_on_mesh(
            [stage], [virtual_submesh], num_micro_batches, default_as_option,
            auto_stage_option, profile_results)

        return profile_results[stage_index]

    def check_1d_2d_results_the_same(self, train_step, state, batch,
                                     micro_batch, num_layers, num_microbatch):
        (closed_jaxpr, full_batch_closed_jaxpr,
         donated_invars) = self.get_train_step_jaxpr(train_step, state, batch,
                                                     micro_batch)
        (closed_jaxpr, global_outvars, jax_pipeline_layers, apply_grad_jaxpr,
         microbatch_bound, reduction_vector, post_microbatch_bound,
         accumulator_mapping, acc_grad_invars, acc_grad_outvars,
         jax_apply_layers, apply_grad_global_info) = self.pre_process_jaxpr(
             closed_jaxpr, full_batch_closed_jaxpr, num_microbatch,
             donated_invars)
        # 2D
        profile_results_2d = self.generate_profile_result(
            jax_pipeline_layers, accumulator_mapping, acc_grad_invars,
            acc_grad_outvars, jax_apply_layers, apply_grad_global_info,
            num_microbatch, 0, num_layers - 1)

        # 1D
        profile_results_1d = []
        for layer_idx in range(num_layers):
            result = self.generate_profile_result(
                jax_pipeline_layers, accumulator_mapping, acc_grad_invars,
                acc_grad_outvars, jax_apply_layers, apply_grad_global_info,
                num_microbatch, layer_idx, layer_idx)
            profile_results_1d.append(result)

        # Compare
        (available_memory_2d, peak_memory_2d, initial_size_2d,
         intermediate_size_2d,
         max_stage_2d) = get_merged_stages_memory_stats([profile_results_2d])
        (available_memory_1d, peak_memory_1d, initial_size_1d,
         intermediate_size_1d,
         max_stage_1d) = get_merged_stages_memory_stats(profile_results_1d)

        assert available_memory_1d == available_memory_2d, (
            f"available_memory_1d: {available_memory_1d}, "
            f"available_memory_2d: {available_memory_2d}")
        assert initial_size_1d == initial_size_2d, (
            f"initial_size_1d: {initial_size_1d}, "
            f"initial_size_2d: {initial_size_2d}")
        assert intermediate_size_1d == intermediate_size_2d, (
            f"intermediate_size_1d: {intermediate_size_1d}, "
            f"intermediate_size_2d: {intermediate_size_2d}")
        # Note: peak_memory_1d is not equal to peak_memory_2d because
        # the greedy memory register allocation algorithm in XLA is not
        # optimal, and may behave different in 1D and 2D cases.

    def test_mlp_1d_2d_the_same(self):
        num_microbatch = 2
        num_layers = 2
        (train_step, state, batch,
         micro_batch) = self.create_mlp(num_microbatch)
        self.check_1d_2d_results_the_same(train_step, state, batch, micro_batch,
                                          num_layers, num_microbatch)

    def test_bert_1d_2d_the_same(self):
        num_microbatch = 2
        num_layers = 3
        (train_step, state, batch,
         micro_batch) = self.create_bert_layers(num_layers, num_microbatch)
        self.check_1d_2d_results_the_same(train_step, state, batch, micro_batch,
                                          num_layers, num_microbatch)

    def check_2d_real_the_same(self):
        num_microbatch = 2
        num_layers = 1
        (train_step, state, batch,
         micro_batch) = self.create_mlp(num_microbatch, add_marker=False)
        (closed_jaxpr, full_batch_closed_jaxpr,
         donated_invars) = self.get_train_step_jaxpr(train_step, state, batch,
                                                     micro_batch)

        (closed_jaxpr, global_outvars, jax_pipeline_layers, apply_grad_jaxpr,
         microbatch_bound, reduction_vector, post_microbatch_bound,
         accumulator_mapping, acc_grad_invars, acc_grad_outvars,
         jax_apply_layers, apply_grad_global_info) = self.pre_process_jaxpr(
             closed_jaxpr, full_batch_closed_jaxpr, num_microbatch,
             donated_invars)
        # 2D
        profile_results_2d = self.generate_profile_result(
            jax_pipeline_layers, accumulator_mapping, acc_grad_invars,
            acc_grad_outvars, jax_apply_layers, apply_grad_global_info,
            num_microbatch, 0, num_layers - 1)
        (available_memory_2d, peak_memory_2d, initial_size_2d,
         intermediate_size_2d,
         max_stage_2d) = get_merged_stages_memory_stats([profile_results_2d])

        # Real
        pipeshard_method = PipeshardParallel(
            num_micro_batches=num_microbatch,
            layer_option="manual",
            stage_option="uniform",
        )
        parallelized_train_step = parallelize(
            train_step,
            donate_argnums=(0,),
            method=pipeshard_method,
        )
        parallelized_train_step(state, batch)
        peak_memory = (parallelized_train_step.get_executable(
            state, batch).mesh_group.get_max_memory_allocated())
        print(f"2D peak_memory: {peak_memory_2d}")
        print(f"Real peak_memory: {peak_memory}")
        # Note: real peak_memory is not equal to peak_memory_2d because
        # of the same reason as above. In addition, our old profiling
        # method is also not accurate compared to the real peak memory.


def suite():
    suite = unittest.TestSuite()
    suite.addTest(StageConstructUtilTest("test_mlp_1d_2d_the_same"))
    suite.addTest(StageConstructUtilTest("test_bert_1d_2d_the_same"))
    # suite.addTest(StageConstructUtilTest("check_2d_real_the_same"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
