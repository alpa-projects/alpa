"""Test auto sharding with MLP."""

import unittest
from itertools import chain

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
from jax.interpreters.pxla import Chunked, NoSharding, Replicated, ShardedAxis
import optax

from alpa import (parallelize, LocalPhysicalDeviceMesh, AutoShardingOption,
                  ShardParallel, Zero2Parallel, Zero3Parallel)
from alpa.util import count_communication_primitives


def _get_sharding(x):
    sharding_spec = getattr(x, "sharding_spec", None)
    # The new ArrayImpl class does not have sharding_spec attribute.
    sharding_spec = sharding_spec or x._sharding.sharding_spec
    return sharding_spec


def assert_close(x, y, atol=0.01):
    assert abs((x + 1e-9) / (y + 1e-9) - 1) <= atol, f"{x} vs. {y}"


def assert_less_equal(x, y):
    assert abs((x + 1e-9) / (y + 1e-9)) <= 1.01, f"{x} vs. {y}"


def assert_column_partitioned(x, num_chunks, mesh_dim):
    assert _get_sharding(x).sharding == (NoSharding(), Chunked([num_chunks]))
    assert _get_sharding(x).mesh_mapping == (ShardedAxis(0),)


def assert_row_partitioned(x, num_chunks, mesh_dim):
    assert _get_sharding(x).sharding == (Chunked([num_chunks]), NoSharding())
    assert _get_sharding(x).mesh_mapping == (ShardedAxis(0),)


def assert_expert_partitioned(x, num_chunks, mesh_dim):
    assert _get_sharding(x).sharding == (Chunked([num_chunks]), NoSharding(),
                                        NoSharding())
    assert _get_sharding(x).mesh_mapping == (ShardedAxis(0),)


def assert_replicated_column_partitioned(x, mesh_shape):
    assert _get_sharding(x).sharding == (NoSharding(), Chunked([mesh_shape[1]]))
    assert _get_sharding(x).mesh_mapping[0] == Replicated(mesh_shape[0])
    assert _get_sharding(x).mesh_mapping[1] == ShardedAxis(0)


def assert_replicated_row_partitioned(x, mesh_shape):
    assert _get_sharding(x).sharding == (Chunked([mesh_shape[1]]), NoSharding())
    assert _get_sharding(x).mesh_mapping[0] == Replicated(mesh_shape[0])
    assert _get_sharding(x).mesh_mapping[1] == ShardedAxis(0)


def assert_all_replicated(x, num_replicas):
    for axis_shard in _get_sharding(x).sharding:
        assert axis_shard == NoSharding()
    assert _get_sharding(x).mesh_mapping[0] == Replicated(num_replicas)


def is_sharded(x):
    for axis in _get_sharding(x).mesh_mapping:
        if isinstance(axis, ShardedAxis):
            return True
    return False


def assert_sharded(x):
    assert is_sharded(x), f"Not sharded: {str(_get_sharding(x))}"


def is_fully_sharded(x):
    for axis in _get_sharding(x).mesh_mapping:
        if not isinstance(axis, ShardedAxis):
            return False
    return True


def assert_fully_sharded(x):
    assert is_fully_sharded(x), f"Not fully sharded: {str(_get_sharding(x))}"


def assert_sharding_zero_stage_3(state, allow_not_sharded_params=0):
    params = jax.tree_util.tree_leaves(state.params)
    opt_state = jax.tree_util.tree_leaves(state.opt_state)

    num_not_sharded = 0
    for weight in chain(params, opt_state):
        if not is_sharded(weight) and len(weight.shape) > 1:
            num_not_sharded += 1
    assert num_not_sharded <= allow_not_sharded_params


def assert_data_parallel_cost(state,
                              hlo_ir,
                              objective,
                              device_mesh,
                              as_option,
                              mesh_dim,
                              allow_not_sharded_params=0,
                              optimizer_type=None):
    params = jax.tree_util.tree_leaves(state.params)
    opt_state = jax.tree_util.tree_leaves(state.opt_state)

    # Check communication cost
    replicated_penalty = int(
        device_mesh.all_reduce_cost(1, 0) + device_mesh.all_reduce_cost(1, 1))
    expected = sum(
        device_mesh.all_reduce_cost(np.prod(x.shape) * 4, mesh_dim)
        for x in params)
    expected += replicated_penalty * (len(params) + len(opt_state))
    assert_close(objective, expected)

    # Check numbers of communication primitives
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
        count_communication_primitives(hlo_ir, ignore_scalar_all_reduce=True))

    # Special case 1 : adafactor
    if optimizer_type == "adafactor" and as_option.prefer_reduce_scatter:
        assert n_reduce_scatter == 1
        assert n_all_gather <= 2
        assert n_all_reduce <= 2
        return

    # Special case 2 : force zero stage 3
    if as_option.force_zero_stage_3:
        assert n_all_reduce == 0
        assert n_all_gather == 2
        assert n_reduce_scatter == 1
        assert_sharding_zero_stage_3(state)
        return

    # Normal case
    if as_option.prefer_reduce_scatter:
        assert n_reduce_scatter == 1
        assert n_all_gather == 1
        if allow_not_sharded_params:
            assert n_all_reduce == 1
        else:
            assert n_all_reduce == 0
        assert n_total == n_reduce_scatter + n_all_gather + n_all_reduce
    else:
        assert n_all_reduce == 1
        assert n_total == n_all_reduce

    # Check sharding specification
    if as_option.prefer_reduce_scatter:
        num_not_sharded = 0
        for weight in opt_state:
            if not is_sharded(weight) and len(weight.shape) > 0:
                num_not_sharded += 1
        assert num_not_sharded <= allow_not_sharded_params * 2
    else:
        for weight in params:
            assert_all_replicated(weight, np.prod(device_mesh.shape))


class AutoShardingMLPTest(unittest.TestCase):

    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.physical_mesh = LocalPhysicalDeviceMesh(jax.local_devices()[:4])
        self.method = ShardParallel(auto_sharding_option=AutoShardingOption())
        self.optimizer_type = "adam"

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        return self.physical_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_n_layer_mlp(self,
                        num_layers,
                        batch_size,
                        input_dim,
                        output_dim,
                        hidden_dim,
                        device_mesh,
                        use_bias=True):

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x):
                for i in range(num_layers - 1):
                    x = nn.Dense(features=hidden_dim, use_bias=use_bias)(x)
                    x = nn.relu(x)
                x = nn.Dense(features=output_dim, use_bias=use_bias)(x)
                return x

        self.method.devices = device_mesh

        @parallelize(method=self.method)
        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"])**2)

            grads = jax.grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

        # Init train state
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        if self.optimizer_type == "adam":
            tx = optax.adam(learning_rate=1e-2)
        elif self.optimizer_type == "adafactor":
            tx = optax.adafactor(learning_rate=1e-2, min_dim_size_to_factor=4)
        else:
            raise ValueError(f"Invalid optimizer_type: {self.optimizer_type}")
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        # JIT compile
        state = train_step(state, {"x": x, "y": y})

        # Get optimized HLO IR
        executable = train_step.get_last_executable()
        return (state, executable.get_hlo_text(),
                executable.auto_sharding_objective)

    def test_n_layer_mlp_data_parallel(self):
        num_layers = 6
        batch_size = 256
        hidden_dim = 32

        # Test on different device meshes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_n_layer_mlp(
                num_layers, batch_size, hidden_dim, hidden_dim, hidden_dim,
                device_mesh)

            assert_data_parallel_cost(state,
                                      hlo_ir,
                                      objective,
                                      device_mesh,
                                      self.method.as_option,
                                      i,
                                      optimizer_type=self.optimizer_type)

    def test_n_layer_mlp_model_parallel(self):
        num_layers = 6
        batch_size = 32
        hidden_dim = 256

        # Test on different device meshes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_n_layer_mlp(
                num_layers, batch_size, hidden_dim, hidden_dim, hidden_dim,
                device_mesh)

            # Check communication cost
            expected = (
                (num_layers - 1) *
                device_mesh.all_reduce_cost(batch_size * hidden_dim * 4, i))
            assert_close(objective, expected)

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
                count_communication_primitives(hlo_ir))
            if self.method.as_option.prefer_reduce_scatter:
                assert n_all_reduce + n_reduce_scatter == num_layers - 1
                assert n_reduce_scatter == n_all_gather
                assert n_total == n_all_reduce + n_reduce_scatter + n_all_gather
            else:
                assert n_all_reduce == num_layers - 1
                assert n_total == n_all_reduce

            # Check sharding specification
            for k in range(num_layers):
                weight = state.params["params"][f"Dense_{k}"]["kernel"]
                if k % 2 == 0:
                    assert_column_partitioned(weight, mesh_shape[i], i)
                else:
                    assert_row_partitioned(weight, mesh_shape[i], i)

    def test_n_layer_mlp_2d_mesh(self):
        num_layers = 6
        batch_size = 256
        hidden_dim = 32

        # Test on different device meshes
        mesh_shape = [2, 2]
        device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 0.1])
        state, hlo_ir, objective = self.run_n_layer_mlp(num_layers, batch_size,
                                                        hidden_dim, hidden_dim,
                                                        hidden_dim, device_mesh)

        # Check communication cost
        expected = (num_layers *
                    (device_mesh.all_reduce_cost(
                        hidden_dim * hidden_dim * 4 / mesh_shape[1], 0) +
                     device_mesh.all_reduce_cost(hidden_dim * 4, 0)) +
                    (num_layers - 1) * device_mesh.all_reduce_cost(
                        batch_size * hidden_dim * 4 / mesh_shape[0], 1))
        assert_close(objective, expected)

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_ir))
        if self.method.as_option.prefer_reduce_scatter:
            assert n_all_reduce == num_layers - 1
            # two reduce-scatter for two tensor dimensions
            assert n_reduce_scatter == 2
            # two for two tensor dimensions, although we can merge them
            assert n_all_gather <= 2
            assert n_total == n_all_reduce + n_all_gather + n_reduce_scatter
        else:
            assert n_all_reduce == num_layers
            assert n_total == n_all_reduce

        # Check sharding specification
        if self.method.as_option.prefer_reduce_scatter:
            for weight in jax.tree_util.tree_leaves(state.opt_state):
                if len(weight.shape) > 1:
                    assert_fully_sharded(weight)
        else:
            for k in range(num_layers):
                weight = state.params["params"][f"Dense_{k}"]["kernel"]
                if k % 2 == 0:
                    assert_replicated_column_partitioned(weight, mesh_shape)
                else:
                    assert_replicated_row_partitioned(weight, mesh_shape)

    def test_n_layer_mlp_force_data_parallel(self):
        num_layers = 6
        batch_size = 32
        hidden_dim = 256

        # Test on different device meshes
        for i, mesh_shape in enumerate([(4, 1), (2, 2)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            self.method.as_option.force_data_parallel = True
            state, hlo_ir, objective = self.run_n_layer_mlp(
                num_layers, batch_size, hidden_dim, hidden_dim, hidden_dim,
                device_mesh)

            assert_data_parallel_cost(state, hlo_ir, objective,
                                      device_mesh.flatten(),
                                      self.method.as_option, 0)

    def test_n_layer_mlp_force_batch_dim_mapping(self):
        num_layers = 6
        batch_size = 32
        hidden_dim = 256
        self.method.as_option.force_batch_dim_to_mesh_dim = 0

        # Data parallel
        device_mesh = self.get_device_mesh([4, 1], [1, 1], [1, 1])
        state, hlo_ir, objective = self.run_n_layer_mlp(num_layers, batch_size,
                                                        hidden_dim, hidden_dim,
                                                        hidden_dim, device_mesh)
        assert_data_parallel_cost(state, hlo_ir, objective, device_mesh,
                                  self.method.as_option, 0)

        # Model parallel
        device_mesh = self.get_device_mesh([1, 4], [1, 1], [1, 1])
        state, hlo_ir, objective = self.run_n_layer_mlp(num_layers, batch_size,
                                                        hidden_dim, hidden_dim,
                                                        hidden_dim, device_mesh)
        expected = ((num_layers - 1) *
                    device_mesh.all_reduce_cost(batch_size * hidden_dim * 4, 1))
        assert_close(objective, expected)

    def test_n_layer_mlp_data_parallel_reduce_scatter(self):
        self.method = Zero2Parallel()
        self.test_n_layer_mlp_data_parallel()

    def test_n_layer_mlp_model_parallel_reduce_scatter(self):
        self.method.as_option.prefer_reduce_scatter = True
        self.test_n_layer_mlp_model_parallel()

    def test_n_layer_mlp_2d_mesh_reduce_scatter(self):
        self.method.as_option.prefer_reduce_scatter = True
        self.test_n_layer_mlp_2d_mesh()

    def test_n_layer_mlp_data_parallel_reduce_scatter_adafactor(self):
        self.method.as_option.prefer_reduce_scatter = True
        self.optimizer_type = "adafactor"
        self.test_n_layer_mlp_data_parallel()

    def test_n_layer_mlp_data_parallel_reduce_scatter_zero_stage_3(self):
        self.method = Zero3Parallel()
        self.method.as_option.force_zero_stage_3_all_gather_threshold = (
            (32 * 32 + 32) * 6 * 4)
        self.test_n_layer_mlp_data_parallel()

    def test_weight_init(self):

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x, deterministic):
                x = nn.Dense(16)(x)
                x = nn.Dense(16)(x)
                return x

        x = jnp.ones((64, 16))
        y = jnp.ones((64, 16))

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)

        @parallelize(method=ShardParallel(devices=self.physical_mesh))
        def init_weight(rngkey):
            params = model.init(rngkey, x, True)
            tx = optax.adam(learning_rate=1e-2)
            state = TrainState.create(apply_fn=model.apply,
                                      params=params,
                                      tx=tx)
            return state

        state = init_weight(rngkey)

        # Check sharding specification
        assert_all_replicated(state.step, self.physical_mesh.num_devices)
        assert_sharded(state.params["params"]["Dense_0"]["kernel"])
        assert_sharded(state.params["params"]["Dense_1"]["kernel"])
        assert_sharded(state.opt_state[0].mu["params"]["Dense_0"]["kernel"])
        assert_sharded(state.opt_state[0].nu["params"]["Dense_1"]["kernel"])


def suite():
    suite = unittest.TestSuite()

    def add(name):
        suite.addTest(AutoShardingMLPTest(name))

    add("test_n_layer_mlp_data_parallel")
    add("test_n_layer_mlp_model_parallel")
    add("test_n_layer_mlp_2d_mesh")
    add("test_n_layer_mlp_force_data_parallel")
    add("test_n_layer_mlp_force_batch_dim_mapping")

    add("test_n_layer_mlp_data_parallel_reduce_scatter")
    add("test_n_layer_mlp_model_parallel_reduce_scatter")
    add("test_n_layer_mlp_2d_mesh_reduce_scatter")

    add("test_n_layer_mlp_data_parallel_reduce_scatter_adafactor")

    add("test_n_layer_mlp_data_parallel_reduce_scatter_zero_stage_3")

    add("test_weight_init")

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
