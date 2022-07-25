import jax
import jax.numpy as jnp
import unittest

from alpa import init, parallelize, AutoShardingOption, PipeshardParallel
from alpa.testing import MLPModel


class SetInputShardSpecTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def run_set_input_shard_spec(self):
        hidden_size = 64

        rngkey = jax.random.PRNGKey(0)

        # Make a MLP model with 2 pipeline stages.
        model = MLPModel(num_layers=4,
                         hidden_size=hidden_size,
                         add_manual_pipeline_marker=True)
        data = jax.core.ShapedArray((1, hidden_size), jnp.float32)
        params = jax.eval_shape(model.init, rngkey, data)
        params = jax.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, jnp.float32), params)

        def infer_fn(params, batch):
            return model.apply(params, batch["x"])

        method = PipeshardParallel(
            num_micro_batches=1,
            pipeline_schedule="inference",
            layer_option="manual",
            default_auto_sharding_option=AutoShardingOption(
                force_batch_dim_to_mesh_dim=None,
                allow_all_to_all=False,
                allow_all_gather=False,
            ))

        # Compile with batch size 1
        executable_1 = parallelize(
            infer_fn, batch_argnums=(1,), method=method).get_executable(
                params,
                {"x": jax.core.ShapedArray((1, hidden_size), jnp.float32)})

        # Make another parallel method with the same input shard spec.
        method_with_input_shard = PipeshardParallel(
            num_micro_batches=1,
            pipeline_schedule="inference",
            layer_option="manual",
            default_auto_sharding_option=AutoShardingOption(
                force_batch_dim_to_mesh_dim=None,
                allow_all_to_all=False,
                allow_all_gather=False,
            ),
            stage_input_shardings=executable_1.stage_input_shard_specs)

        # Compile with a different batch size
        executable_2 = parallelize(
            infer_fn, batch_argnums=(1,), method=method).get_executable(
                params,
                {"x": jax.core.ShapedArray((8, hidden_size), jnp.float32)})

        # Compile with a different batch size but the same input shard specs
        executable_3 = parallelize(
            infer_fn, batch_argnums=(1,),
            method=method_with_input_shard).get_executable(
                params,
                {"x": jax.core.ShapedArray((8, hidden_size), jnp.float32)})

        assert executable_2.stage_input_shard_specs != executable_3.stage_input_shard_specs
        assert executable_1.stage_input_shard_specs == executable_3.stage_input_shard_specs

    def test_set_input_shard_spec(self):
        self.run_set_input_shard_spec()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SetInputShardSpecTest('test_set_input_shard_spec'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
