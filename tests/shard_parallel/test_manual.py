"""
Test the manual sharding spec.
"""
import unittest

import jax
from jax.experimental.pjit import PartitionSpec
from jax.tree_util import tree_map
import jax.numpy as jnp

import alpa
from alpa import (AutoShardingOption, LocalPhysicalDeviceMesh,
                  ManualShardingOption, ShardParallel, parallelize)
from alpa.testing import HloParser


class ManualShardingTest(unittest.TestCase):

    def setUp(self):
        self.as_option = AutoShardingOption(enable_auto_sharding=False)
        self.devices = LocalPhysicalDeviceMesh(jax.local_devices()[:4])
        self.devices = self.devices.get_logical_mesh((2, 2), (1, 1), (1, 1))
        self.mesh_axis_names = ("data", "model")

    def _get_fn_manual_sharding_with(self,
                                     fn,
                                     ms_option,
                                     *args,
                                     num_microbatches=None,
                                     batch_argnums=(1,)):
        method = ShardParallel(
            devices=self.devices,
            num_micro_batches=num_microbatches,
            auto_sharding_option=self.as_option,
            manual_sharding_option=ms_option,
        )
        parallelized = parallelize(fn,
                                   method=method,
                                   batch_argnums=batch_argnums)
        return parallelized.get_executable(*args).get_hlo_text()

    def test_set_input(self):

        def fn(a, b):
            return a + b

        a = jnp.ones((6, 4))
        b = jnp.ones((6, 4))
        in_axis_resources = (PartitionSpec(None, "model"),
                             PartitionSpec(None, "model"))
        ms_option = ManualShardingOption(self.mesh_axis_names,
                                         in_axis_resources=in_axis_resources)
        text = self._get_fn_manual_sharding_with(fn, ms_option, a, b)
        text = HloParser.get_param_line(text)
        assert "param: f32[6,2]" in text and "param.1: f32[6,2]" in text
        in_axis_resources = (PartitionSpec("data", None),
                             PartitionSpec("data", "model"))
        ms_option = ManualShardingOption(self.mesh_axis_names,
                                         in_axis_resources=in_axis_resources)
        text = self._get_fn_manual_sharding_with(fn, ms_option, a, b)
        text = HloParser.get_param_line(text)
        assert "param: f32[3,4]" in text and "param.1: f32[3,2]" in text
        in_axis_resources = (None, PartitionSpec("data", None))
        ms_option = ManualShardingOption(self.mesh_axis_names,
                                         in_axis_resources=in_axis_resources)
        text = self._get_fn_manual_sharding_with(fn, ms_option, a, b)
        text = HloParser.get_param_line(text)
        assert "param: f32[6,4]" in text and "param.1: f32[3,4]" in text

    def test_set_output(self):

        def fn(a):
            return a**2, a + 1, a * 2, a / 2

        a = jnp.ones((6, 4))
        out_axis_resources = (PartitionSpec("data", None), None,
                              PartitionSpec(None, "model"),
                              PartitionSpec("data", "model"))
        ms_option = ManualShardingOption(self.mesh_axis_names,
                                         out_axis_resources=out_axis_resources)
        text = self._get_fn_manual_sharding_with(fn, ms_option, a)
        text = HloParser.get_root_line(text)
        assert ("(f32[3,4]{1,0}, f32[6,4]{1,0}, f32[6,2]{1,0}, f32[3,2]{1,0}"
                in text)

    def test_grad_acc(self):

        def fn(params, batch):
            x, tgt = batch

            def loss_fn(params):
                w1, b1, w2, b2 = params
                y = jax.nn.relu(x @ w1 + b1)
                z = jax.nn.softmax(y @ w2 + b2)
                return jnp.mean((z - tgt)**2)

            grads = alpa.grad(loss_fn)(params)
            new_params = tree_map(lambda p, g: p - g, params, grads)
            return new_params

        batch_size = 64
        x = jnp.ones((batch_size, 6))
        tgt = jnp.ones((batch_size, 10))
        params = (jnp.ones((6, 8)), jnp.ones((8,)), jnp.ones(
            (8, 10)), jnp.ones((10,)))
        batch = (x, tgt)
        in_axis_resources = ((PartitionSpec(None,
                                            "model"), PartitionSpec("model"),
                              PartitionSpec("model",
                                            None), PartitionSpec(None)),
                             (PartitionSpec("data",
                                            None), PartitionSpec("data", None)))
        ms_option = ManualShardingOption(self.mesh_axis_names,
                                         in_axis_resources=in_axis_resources)
        text = self._get_fn_manual_sharding_with(fn,
                                                 ms_option,
                                                 params,
                                                 batch,
                                                 num_microbatches=2)
        apply_grad_start = text.find("HloModule", 1)
        acc_grad_text = text[:apply_grad_start]
        apply_grad_text = text[apply_grad_start:]
        # 1. Accumulate grad:
        acc_grad_params = HloParser.get_param_line(acc_grad_text)
        acc_grad_param_shapes = HloParser.parse_param_shapes(acc_grad_params)
        acc_grad_root = HloParser.get_root_line(acc_grad_text)
        acc_grad_root_shapes = HloParser.parse_root_shapes(acc_grad_root)

        param_shape = ("f32[6,4]", "f32[4]", "f32[4,10]", "f32[10]")
        # batch_size / num_microbatches / data_parallel
        batch_shape = ("f32[16,6]", "f32[16,10]")
        assert acc_grad_param_shapes == param_shape + batch_shape + param_shape
        assert acc_grad_root_shapes == param_shape
        # 2. Apply grad:
        apply_grad_params = HloParser.get_param_line(apply_grad_text)
        apply_grad_param_shapes = HloParser.parse_param_shapes(
            apply_grad_params)
        apply_grad_root = HloParser.get_root_line(apply_grad_text)
        apply_grad_root_shapes = HloParser.parse_root_shapes(apply_grad_root)
        assert apply_grad_param_shapes == param_shape + param_shape
        assert apply_grad_root_shapes == param_shape


def suite():
    suite = unittest.TestSuite()
    suite.addTest(ManualShardingTest("test_set_input"))
    suite.addTest(ManualShardingTest("test_set_output"))
    suite.addTest(ManualShardingTest("test_grad_acc"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
