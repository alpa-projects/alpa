import unittest
import os

import jax
from jax._src.api import make_jaxpr
from jax.core import ClosedJaxpr, Var, gensym
import jax.numpy as jnp
import ray

from alpa import init
from alpa.model.bert_model import BertConfig
from alpa.pipeline_parallel.apply_grad import (compute_grad_to_accumulate_grad,
                                               process_apply_gradient,
                                               split_compute_grad_and_apply_grad
                                              )
from alpa.pipeline_parallel.computation import (
    create_donation_mapping,
    mark_missing_vars_in_backward_computation_pipeline_marks, offload_remat,
    pipeline_dce, slice_closed_jaxpr_by_full_pipeline_marks)
from alpa.pipeline_parallel.stage_profiling import (
    ApplyGradConfig, CompileConfig, ProfileConfig, generate_stage_info,
    compile_all, profile_all, compute_intermediate_size,
    compute_apply_grad_invar_size)
from alpa.testing import (BertLayerModel, create_train_state,
                          get_bert_layer_train_step)
from alpa.util import get_ray_namespace_str, OrderedSet


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
        init(cluster="ray")

    def _create_n_layer_jaxpr_with_donation(self,
                                            n_layers=2,
                                            batch_size=16,
                                            seq_len=256,
                                            hidden_size=512,
                                            num_heads=512 // 64,
                                            use_remat=True):
        manual_pipeline_layer = True
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=jnp.float32)
        y = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=jnp.float32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
        model = BertLayerModel(config=BertConfig(hidden_size=hidden_size,
                                                 intermediate_size=hidden_size *
                                                 4,
                                                 num_attention_heads=num_heads,
                                                 num_hidden_layers=n_layers),
                               manual_pipeline_layer=manual_pipeline_layer)
        batch = {"x": x, "y": y, "attention_mask": attention_mask}
        state = create_train_state(rngkey, model, [x, attention_mask])

        # Compile
        train_step = get_bert_layer_train_step(None,
                                               manual_pipeline_layer,
                                               use_remat,
                                               n_layers,
                                               False,
                                               decorate_loss=True)
        closed_jaxpr, output_tree = make_jaxpr(train_step,
                                               return_shape=True)(state, batch)
        num_params = len(closed_jaxpr.jaxpr.invars) - 3
        donated_invars = [True] * num_params + [False] * 3
        return closed_jaxpr, output_tree, donated_invars

    def _post_process_jaxpr(self,
                            closed_jaxpr: ClosedJaxpr,
                            donated_invars,
                            num_microbatch=2):
        gensym_func = gensym([closed_jaxpr.jaxpr])
        closed_jaxpr, compute_grad_jaxpr, apply_grad_jaxpr, barrier = (
            split_compute_grad_and_apply_grad(closed_jaxpr, gensym_func,
                                              num_microbatch))
        have_apply_grad = barrier is not None
        assert have_apply_grad
        reduction_vector = [True] * len(compute_grad_jaxpr.jaxpr.outvars)
        (acc_grad_jaxpr, acc_grad_dict,
         grad_in_to_out) = compute_grad_to_accumulate_grad(
             compute_grad_jaxpr, reduction_vector, gensym_func)
        acc_grad_invars = acc_grad_jaxpr.jaxpr.invars
        acc_grad_outvars = acc_grad_jaxpr.jaxpr.outvars

        jax_pipeline_layers = slice_closed_jaxpr_by_full_pipeline_marks(
            acc_grad_jaxpr)
        jax_pipeline_layers = mark_missing_vars_in_backward_computation_pipeline_marks(
            jax_pipeline_layers, acc_grad_invars, acc_grad_outvars, gensym_func)
        jax_pipeline_layers = pipeline_dce(jax_pipeline_layers,
                                           acc_grad_outvars)
        jax_pipeline_layers = offload_remat(jax_pipeline_layers, gensym_func)

        global_invars = closed_jaxpr.jaxpr.invars
        global_outvars = closed_jaxpr.jaxpr.outvars
        donation_mapping = dict(grad_in_to_out)

        num_forward_layers = len(jax_pipeline_layers) // 2
        layer_to_dummy_mesh = (list(range(num_forward_layers)) +
                               list(reversed(range(num_forward_layers))))
        reduce_invars = [True] * len(barrier.invars)

        (jax_apply_layers, _, _, _, _,
         dummy_donated_invars) = process_apply_gradient(
             apply_grad_jaxpr, barrier, acc_grad_dict, jax_pipeline_layers,
             layer_to_dummy_mesh, gensym_func, num_microbatch,
             len(jax_pipeline_layers) // 2, global_invars, global_outvars,
             donated_invars, reduce_invars)
        apply_grad_donation = create_donation_mapping(donation_mapping,
                                                      dummy_donated_invars,
                                                      global_invars,
                                                      global_outvars)
        return (jax_pipeline_layers, donation_mapping, acc_grad_outvars,
                jax_apply_layers, apply_grad_donation, global_outvars)

    def _generate_stage_info(self, info, start, end):
        (compute_layers, donation_mapping, compute_outvars, apply_grad_layers,
         apply_grad_donate_map, global_outvars) = info
        # create other configs
        indices = list(range(start, end + 1))
        compute_layer_indices = []
        for idx in indices:
            compute_layer_indices.append(idx)
            compute_layer_indices.append(len(compute_layers) - idx - 1)
        compute_layer_indices = sorted(compute_layer_indices)
        apply_grad_indices = indices
        apply_grad_selected = [apply_grad_layers[i] for i in apply_grad_indices]
        apply_grad_config = (apply_grad_donate_map, global_outvars)
        intermediate_vars, stage_config = generate_stage_info(
            compute_layers, compute_layer_indices, donation_mapping,
            compute_outvars, "tmp", end - start, apply_grad_selected,
            apply_grad_config)
        return intermediate_vars, stage_config

    def _test_generate_stage_config_indices(self, info, start, end):
        (compute_layers, donation_mapping, compute_outvars, apply_grad_layers,
         apply_grad_donate_map, global_outvars) = info
        # create other configs
        indices = list(range(start, end + 1))
        compute_layer_indices = []
        for idx in indices:
            compute_layer_indices.append(idx)
            compute_layer_indices.append(len(compute_layers) - idx - 1)
        compute_layer_indices = sorted(compute_layer_indices)
        apply_grad_indices = indices
        apply_grad_selected = [apply_grad_layers[i] for i in apply_grad_indices]
        apply_grad_info = (apply_grad_donate_map, global_outvars)
        intermediate_vars, stage_config = generate_stage_info(
            compute_layers, compute_layer_indices, donation_mapping,
            compute_outvars, "tmp", end - start, apply_grad_selected,
            apply_grad_info)
        aval = lambda x: [v.aval for v in x]
        # check intermediate vars. It is only to compute intermediate size, so
        # only check aval
        backward_input = OrderedSet()
        for idx in indices:
            backward_idx = len(compute_layers) - idx - 1
            active_vars = OrderedSet()
            for eqn in compute_layers[backward_idx].eqns[1:]:
                active_vars.update(
                    [v for v in eqn.invars if isinstance(v, Var)])
            start_marker = compute_layers[backward_idx].eqns[0]
            active_vars.intersection_update(start_marker.outvars)
            active_invars = OrderedSet()
            for invar, outvar in zip(start_marker.invars, start_marker.outvars):
                if outvar in active_vars:
                    active_invars.add(invar)
            backward_input.update(active_invars)
        forward_output = OrderedSet()
        for idx in indices:
            forward_output.update(compute_layers[idx].outvars)
        intermediate = forward_output.intersection(backward_input)
        _assert_avals_allmatch(aval(intermediate_vars), aval(intermediate))
        # check apply grad config
        config: ApplyGradConfig = stage_config.apply_grad_config
        apply_grad_invars = set()
        for layer in apply_grad_selected:
            apply_grad_invars.update(layer.invars)
        assert apply_grad_invars.issubset(config.invars)
        assert apply_grad_invars.issuperset(config.invars)

        apply_grad_only_invars = set(apply_grad_invars)
        for i in compute_layer_indices:
            apply_grad_only_invars.difference_update(compute_layers[i].invars)
            apply_grad_only_invars.difference_update(compute_layers[i].outvars)
        assert apply_grad_only_invars.issubset(config.apply_grad_only_invars)
        assert apply_grad_only_invars.issuperset(config.apply_grad_only_invars)
        # check compile config
        config: CompileConfig = stage_config.compile_config
        input_vars = OrderedSet()
        for idx in compute_layer_indices:
            input_vars.update(compute_layers[idx].invars)
        for idx in compute_layer_indices:
            input_vars.difference_update(compute_layers[idx].outvars)
        input_vars.update(apply_grad_only_invars)
        input_avals = [var.aval for var in input_vars]
        _assert_avals_allmatch(input_avals, config.input_avals)

        available_outvars = OrderedSet(compute_outvars)
        available_outvars.update(global_outvars)
        for idx in range(len(compute_layers)):
            if idx not in compute_layer_indices:
                available_outvars.update(compute_layers[idx].invars)
        output_vars = OrderedSet()
        for idx in compute_layer_indices:
            output_vars.update(compute_layers[idx].outvars)
        for layer in apply_grad_selected:
            output_vars.update(layer.outvars)
        output_vars.intersection_update(available_outvars)
        output_avals = [var.aval for var in output_vars]
        _assert_avals_allmatch(output_avals, config.output_avals)
        # Check of the two below is based on that donated inputs/outputs are
        # prior than others.
        backward_outs = OrderedSet()
        for idx in indices:
            backward_idx = len(compute_layers) - 1 - idx
            backward_outs.update(compute_layers[backward_idx].outvars)
        backward_outs.intersection_update(compute_outvars)
        assert config.output_acc_grad_indices == list(range(len(backward_outs)))

        assert len(config.donate_invars) == len(config.input_avals)
        inputs = OrderedSet()
        for idx in compute_layer_indices:
            inputs.update(compute_layers[idx].invars)
        for layer in apply_grad_selected:
            inputs.update(layer.invars)
        inputs.intersection_update(
            OrderedSet(donation_mapping.keys()).union(
                apply_grad_donate_map.keys()))
        donate_invars = len(inputs)
        for i, donate in enumerate(config.donate_invars):
            assert (i >= donate_invars) ^ donate
        # check profile config
        config: ProfileConfig = stage_config.profile_config
        input_vars = OrderedSet()
        for idx in compute_layer_indices:
            input_vars.update(compute_layers[idx].invars)
        for idx in compute_layer_indices:
            input_vars.difference_update(compute_layers[idx].outvars)
        input_avals = [var.aval for var in input_vars]
        _assert_avals_allmatch(input_avals, config.input_avals)

        output_vars = OrderedSet()
        for idx in compute_layer_indices:
            output_vars.update(compute_layers[idx].outvars)
        output_vars.intersection_update(available_outvars)
        output_avals = [var.aval for var in output_vars]
        _assert_avals_allmatch(output_avals, config.output_avals)
        config.donate_invars

    def test_generate_stage_config(self):
        (closed_jaxpr, output_tree,
         donated_invars) = self._create_n_layer_jaxpr_with_donation(n_layers=4)
        info = self._post_process_jaxpr(closed_jaxpr, donated_invars)
        self._test_generate_stage_config_indices(info, 0, 0)
        self._test_generate_stage_config_indices(info, 3, 3)
        self._test_generate_stage_config_indices(info, 0, 1)
        self._test_generate_stage_config_indices(info, 1, 2)
        self._test_generate_stage_config_indices(info, 2, 3)
        self._test_generate_stage_config_indices(info, 0, 3)

    def test_compile_all(self):
        (closed_jaxpr, output_tree,
         donated_invars) = self._create_n_layer_jaxpr_with_donation(n_layers=4)
        info = self._post_process_jaxpr(closed_jaxpr, donated_invars)
        stages = []
        test_intervals = [(0, 0), (3, 3), (0, 1), (1, 2), (2, 3), (0, 3)]
        # TODO(yonghao): compile all and check:
        # 1. number of all-reduce in each compiled model proto
        # 2. tensor's sharding protos when force batch dim to mesh dim


def suite():
    suite = unittest.TestSuite()
    suite.addTest(StageConstructUtilTest("test_generate_stage_config"))
    suite.addTest(StageConstructUtilTest("test_compile_all"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
