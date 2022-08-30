"""Test auto sharding with simple computational graphs."""
import unittest

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, ShardedAxis, NoSharding, Replicated
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

from alpa import parallelize, ShardParallel
from alpa.util import count_communication_primitives
from alpa.testing import assert_allclose

from tests.shard_parallel.test_mlp import assert_close

MB = 1024**2


class AutoShardingBasicTest(unittest.TestCase):

    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = jax.local_devices()[:4]
        self.method = ShardParallel(devices=self.devices)

    def test_donate_buffer(self):

        @parallelize(donate_argnums=(0,), method=self.method)
        def add_one(x):
            x = x + 1
            return x

        a = jnp.ones((128, 128))
        b = add_one(a)

        # Assert b is sharded
        assert (b.sharding_spec == pxla.ShardingSpec(
            sharding=(NoSharding(), Chunked([4])),
            mesh_mapping=(ShardedAxis(0),)) or b.sharding_spec
                == pxla.ShardingSpec(sharding=(Chunked([4]), NoSharding()),
                                     mesh_mapping=(ShardedAxis(0),)))

    def test_dot_reshape_transpose(self):
        dim_0 = 64
        dim_1 = 1024

        def func(a, b):
            a = jnp.transpose(a, [0, 2, 1])
            a = jnp.reshape(a, (dim_0, dim_1))
            b = jnp.reshape(b, (dim_1, dim_0))
            out = a @ b
            out = -out
            return out

        p_func = parallelize(func)

        a = jnp.ones((dim_0, dim_1 // 4, 4))
        b = jnp.ones((dim_1, dim_0 // 4, 4))

        # Check correctness
        expected = func(a, b)
        actual = p_func(a, b)
        assert_allclose(expected, actual)

    def test_one_by_one_mesh(self):

        @parallelize(method=ShardParallel(devices=self.devices[0:1]))
        def add_one(x):
            x = x + 1
            return x

        a = jnp.ones((128, 128))
        b = add_one(a)

        assert_allclose(b, a + 1)

    def test_dropout(self):

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x, deterministic):
                x = nn.Dense(16, use_bias=False)(x)
                x = nn.Dropout(0.1, deterministic=deterministic)(x)
                x = nn.Dense(16, use_bias=False)(x)
                return x

        x = jnp.ones((32, 32, 16))
        y = jnp.ones((32, 32, 16))

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x, True)
        tx = optax.sgd(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        @parallelize(method=self.method)
        def func(state, x, y, rngs):

            def loss_func(params):
                out = model.apply(params, x, False, rngs=rngs)
                return jnp.mean((out - y)**2)

            grad = jax.grad(loss_func)(state.params)
            return state.apply_gradients(grads=grad)

        # Check sharding strategy (data-parallel)
        executable = func.get_executable(state, x, y, {"dropout": rngkey})
        assert executable.auto_sharding_objective < 1e6

        hlo_ir = executable.get_hlo_text()
        assert "u64[1024]{0} iota()" in hlo_ir  # 1024 = 32 * 32 * 16 / 4 / 4
        n_total, n_allreduce, _, _, _ = count_communication_primitives(hlo_ir)
        assert n_total == n_allreduce == 1

    def test_gather(self):

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(32, use_bias=False)(x)
                idx = jnp.arange(16)
                x = x[:, idx]
                x = nn.Dense(16, use_bias=False)(x)
                return x

        x = jnp.ones((256, 32))
        y = jnp.ones((256, 16))

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        tx = optax.sgd(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        @parallelize(method=self.method)
        def func(state, x, y):

            def loss_func(params):
                out = model.apply(params, x)
                return jnp.mean((out - y)**2)

            grad = jax.grad(loss_func)(state.params)
            return state.apply_gradients(grads=grad)

        executable = func.get_executable(state, x, y)
        assert executable.auto_sharding_objective < 1e6

        hlo_ir = executable.get_hlo_text()
        assert "gather(f32[64,32]" in hlo_ir
        assert "scatter(f32[64,32]" in hlo_ir
        _, n_allreduce, _, _, _ = count_communication_primitives(hlo_ir)
        assert n_allreduce == 1

    def test_reshape_uneven_partition(self):
        # TODO(lmzheng): Support the uneven partition of reshape.
        # But this seems too complicated.

        @parallelize(method=self.method)
        def func(a):
            b = a.reshape((8, 18))
            #b = a.reshape((9, 16))
            return b

        a = jnp.ones(144)
        executable = func.get_executable(a)
        assert_close(executable.auto_sharding_objective, 0)

    def test_argmax(self):

        @parallelize(method=self.method)
        def func(a):
            b = jnp.argmax(a, axis=0)
            return b

        a = jnp.ones((144, 144))
        executable = func.get_executable(a)

        assert_close(executable.auto_sharding_objective, 0)
        hlo_ir = executable.get_hlo_text()
        assert "(param: f32[144,36])" in hlo_ir

    def test_sort(self):

        @parallelize(method=self.method)
        def func(a):
            b = jnp.argsort(a)
            return b

        a = jnp.ones((1024,), dtype=jnp.int32)
        executable = func.get_executable(a)

    def test_gemv(self):

        @parallelize(method=self.method)
        def func(a, b):
            return a @ b

        a = jnp.ones((128,), dtype=jnp.float32)
        b = jnp.ones((128, 256), dtype=jnp.float32)
        executable = func.get_executable(a, b)

        assert "f32[128,64]" in executable.get_hlo_text()

    def test_fast_call(self):

        @parallelize
        def add_one(x, y):
            return x + y

        a = jnp.ones((32, 32))
        b = jnp.ones((32, 32))
        executable = add_one.get_executable(a, b)
        c = executable(a, b)

        assert isinstance(c, pxla.ShardedDeviceArray)

        executable.dump_debug_info("tmp")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingBasicTest("test_donate_buffer"))
    suite.addTest(AutoShardingBasicTest("test_dot_reshape_transpose"))
    suite.addTest(AutoShardingBasicTest("test_one_by_one_mesh"))
    suite.addTest(AutoShardingBasicTest("test_dropout"))
    suite.addTest(AutoShardingBasicTest("test_gather"))
    suite.addTest(AutoShardingBasicTest("test_reshape_uneven_partition"))
    suite.addTest(AutoShardingBasicTest("test_argmax"))
    suite.addTest(AutoShardingBasicTest("test_sort"))
    suite.addTest(AutoShardingBasicTest("test_gemv"))
    suite.addTest(AutoShardingBasicTest("test_fast_call"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
