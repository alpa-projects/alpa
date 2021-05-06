"""
Usage:
python3 -m unittest -bv test_solver_mlp.py
"""
from collections import defaultdict
from enum import Enum
import unittest

import numpy as np

from hlo import *
from cluster_env import ClusterEnvironment
from solver import solve_auto_sharding, SolverOption

MB = 1024 ** 2


def assert_close(x, y):
    assert abs(x / y - 1) < 0.001, f"{x} vs. {y}"


def get_mlp_2_layer_computation(batch_size, input_dim, hidden_dim, output_dim):
    computation = HloComputation()
    with computation:
        x = HloParameter((batch_size, input_dim))
        y = HloParameter((batch_size, output_dim))
        w1 = HloParameter((input_dim, hidden_dim))
        w2 = HloParameter((hidden_dim, output_dim))

        ## forward
        h1 = HloDot(x, w1)
        h2 = HloDot(h1, w2)
        loss = HloSubtract(h2, y)

        ## backward
        coef = HloConstant(2 / batch_size / output_dim)
        coef = HloBroadcast(coef, (batch_size, output_dim))
        grad_loss = HloMutiply(loss, coef)

        grad_w2 = HloDot(h1, grad_loss,
                         lhs_contracting_dims=(0,),
                         rhs_contracting_dims=(0,),)
        new_w2 = HloSubtract(w2, grad_w2)
        grad_h1 = HloDot(grad_loss, w2,
                         lhs_contracting_dims=(1,),
                         rhs_contracting_dims=(1,),)

        grad_w1 = HloDot(x, grad_h1,
                         lhs_contracting_dims=(0,),
                         rhs_contracting_dims=(0,),)
        new_w1 = HloSubtract(w1, grad_w1)
        out = HloTuple((new_w1, new_w2))

        ## alias
        computation.set_alias([(w1, new_w1), (w2, new_w2)])

        """
         0: parameter.0 (128, 1024) = parameter()
         1: parameter.1 (128, 1024) = parameter()
         2: parameter.2 (1024, 1024) = parameter()
         3: parameter.3 (1024, 1024) = parameter()
         4: dot.0 (128, 1024) = dot(parameter.0, parameter.2)  lhs_con_dim=(1,), rhs_con_dim=(0,)
         5: dot.1 (128, 1024) = dot(dot.0, parameter.3)  lhs_con_dim=(1,), rhs_con_dim=(0,)
         6: subtract.0 (128, 1024) = subtract(dot.1, parameter.1)
         7: constant.0 () = constant(1.52587891e-05)
         8: broadcast.0 (128, 1024) = broadcast(constant.0)
         9: multiply.0 (128, 1024) = multiply(subtract.0, broadcast.0)
        10: dot.2 (1024, 1024) = dot(dot.0, multiply.0)  lhs_con_dim=(0,), rhs_con_dim=(0,)
        11: subtract.1 (1024, 1024) = subtract(parameter.2, dot.2)
        12: dot.3 (128, 1024) = dot(multiply.0, parameter.3)  lhs_con_dim=(1,), rhs_con_dim=(1,)
        13: dot.4 (1024, 1024) = dot(parameter.0, dot.3)  lhs_con_dim=(0,), rhs_con_dim=(0,)
        14: subtract.2 (1024, 1024) = subtract(parameter.2, dot.4)
        15: tuple.0 () = tuple('subtract.2', 'subtract.1') 
        """
    return computation


def get_mlp_2_layer_bias_computation(batch_size, input_dim, hidden_dim, output_dim):
    computation = HloComputation()
    with computation:
        x = HloParameter((batch_size, input_dim))
        y = HloParameter((batch_size, output_dim))
        w1 = HloParameter((input_dim, hidden_dim))
        w2 = HloParameter((hidden_dim, output_dim))
        b1 = HloParameter((hidden_dim,))
        b2 = HloParameter((output_dim,))

        ## forward
        h1 = HloDot(x, w1)
        bb1 = HloBroadcast(b1, (batch_size, hidden_dim), dimensions=(1,))
        h1_add = HloAdd(h1, bb1)

        h2 = HloDot(h1_add, w2)
        bb2 = HloBroadcast(b2, (batch_size, output_dim), dimensions=(1,))
        h2_add = HloAdd(h2, bb2)

        loss = HloSubtract(h2_add, y)

        ## backward
        coef = HloConstant(2 / batch_size / output_dim)
        coef = HloBroadcast(coef, (batch_size, output_dim))
        grad_loss = HloMutiply(loss, coef)

        grad_w2 = HloDot(h1_add, grad_loss,
                         lhs_contracting_dims=(0,),
                         rhs_contracting_dims=(0,),)
        new_w2 = HloSubtract(w2, grad_w2)

        grad_h1 = HloDot(grad_loss, w2,
                         lhs_contracting_dims=(1,),
                         rhs_contracting_dims=(1,),)

        grad_w1 = HloDot(x, grad_h1,
                         lhs_contracting_dims=(0,),
                         rhs_contracting_dims=(0,),)
        new_w1 = HloSubtract(w1, grad_w1)

        grad_b1 = HloReduce(grad_h1, dimensions=[0])
        new_b1 = HloSubtract(b1, grad_b1)

        grad_b2 = HloReduce(grad_loss, dimensions=[0])
        new_b2 = HloSubtract(b2, grad_b2)

        out = HloTuple((new_w1, new_w2, new_b1, new_b2))

        ## alias
        computation.set_alias([(w1, new_w1), (w2, new_w2)])

    return computation


def get_mlp_n_layer_computation(num_layers, batch_size, input_dim, hidden_dim, output_dim):
    computation = HloComputation()
    with computation:
        x = HloParameter((batch_size, input_dim))
        y = HloParameter((batch_size, output_dim))
        w_first = HloParameter((input_dim, hidden_dim))
        w_inter = []
        for i in range(num_layers - 2):
            manual_strategy = "S0" if i % 2 == 0 else "S1"
            w_inter.append(HloParameter((hidden_dim, hidden_dim)))
        w_last = HloParameter((hidden_dim, output_dim))

        # forward
        h_first = HloDot(x, w_first)
        h_now = h_first
        h_inter = []
        for i in range(num_layers - 2):
            h_now = HloDot(h_now, w_inter[i])
            h_inter.append(h_now)
        h_last = HloDot(h_now, w_last)

        loss = HloSubtract(h_last, y)

        # backward
        coef = HloConstant(2 / batch_size / output_dim)
        coef = HloBroadcast(coef, (batch_size, output_dim))
        grad_loss = HloMutiply(loss, coef)
        grad_h_now = grad_loss

        grad_w_last = HloDot(h_inter[-1], grad_h_now,
                             lhs_contracting_dims=(0,),
                             rhs_contracting_dims=(0,),)
        new_w_last = HloSubtract(w_last, grad_w_last)
        grad_h_now = HloDot(grad_h_now, w_last,
                             lhs_contracting_dims=(1,),
                             rhs_contracting_dims=(1,),)

        new_w_inter = []
        for i in range(num_layers - 3, -1, -1):
            grad_w = HloDot(h_inter[i-1], grad_h_now,
                            lhs_contracting_dims=(0,),
                            rhs_contracting_dims=(0,),)
            new_w = HloSubtract(w_inter[i], grad_w)
            grad_h_now = HloDot(grad_h_now, w_inter[i],
                                lhs_contracting_dims=(1,),
                                rhs_contracting_dims=(1,),)
            new_w_inter.append(new_w)

        grad_w_first = HloDot(x, grad_h_now,
                              lhs_contracting_dims=(0,),
                              rhs_contracting_dims=(0,),)
        new_w_first = HloSubtract(w_first, grad_w_first)

        out = HloTuple([new_w_first] + new_w_inter + [new_w_last])

        # alias
        alias_list = [(w_first, new_w_first), (w_last, new_w_last)] +\
            [(w_old, w_new) for w_old, w_new in zip(w_inter, reversed(new_w_inter))]
        computation.set_alias(alias_list)
    return computation


class MLPSolverTest(unittest.TestCase):
    def test_mlp_2_layer_data_parallel(self):
        # Build Hlo Computation
        batch_size = 1024
        hidden_dim = 128

        computation = get_mlp_2_layer_computation(batch_size, hidden_dim,
            hidden_dim, hidden_dim)

        # Test different device meshes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
            cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 1],
                                             memory_per_device=1000 * MB)
            objective = solve_auto_sharding(computation, cluster_env)

            # The expecte cost is always two all-reduce on weights
            expected = 2 * cluster_env.all_reduce_cost(hidden_dim * hidden_dim * 4, i)
            assert_close(objective, expected)

    def test_mlp_2_layer_model_parallel(self):
        # Build Hlo Computation
        batch_size = 128
        hidden_dim = 1024

        computation = get_mlp_2_layer_computation(batch_size, hidden_dim,
            hidden_dim, hidden_dim)

        # Test different device meshes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
            cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 1],
                                             memory_per_device=1000 * MB)
            objective = solve_auto_sharding(computation, cluster_env)

            # The expecte cost is always one all-reduce on activations
            expected = cluster_env.all_reduce_cost(batch_size * hidden_dim * 4, i)
            assert_close(objective, expected)

    def test_mlp_n_layer_data_parallel(self):
        # Build Hlo Computation
        num_layers = 12
        batch_size = 1024
        hidden_dim = 128

        computation = get_mlp_n_layer_computation(num_layers, batch_size, hidden_dim,
            hidden_dim, hidden_dim)

        # Test different device meshes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
            cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 1],
                                             memory_per_device=1000 * MB)
            objective = solve_auto_sharding(computation, cluster_env)

            expected = num_layers *\
                       cluster_env.all_reduce_cost(hidden_dim * hidden_dim * 4, i)
            assert_close(objective, expected)

    def test_mlp_n_layer_model_parallel(self):
        # Build Hlo Computation
        num_layers = 12
        batch_size = 128
        hidden_dim = 1024

        computation = get_mlp_n_layer_computation(num_layers, batch_size, hidden_dim,
            hidden_dim, hidden_dim)

        # Test different device meshes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
            cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 1],
                                             memory_per_device=1000 * MB)
            objective = solve_auto_sharding(computation, cluster_env)

            expected = (num_layers - 1) *\
                       cluster_env.all_reduce_cost(batch_size * hidden_dim * 4, i)
            assert_close(objective, expected)

    def test_mlp_2_layer_2d_mesh(self):
        # Build Hlo Computation
        batch_size = 1024
        hidden_dim = 128

        computation = get_mlp_2_layer_computation(batch_size, hidden_dim,
            hidden_dim, hidden_dim)

        # Test different device meshes
        for mesh_shape in [(4, 8), (8, 4), (3, 4)]:
            device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
            cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 0.01],
                                             memory_per_device=1000 * MB)
            objective = solve_auto_sharding(computation, cluster_env)

            expected =\
                2 * cluster_env.all_reduce_cost(
                    hidden_dim * hidden_dim * 4 / mesh_shape[1], 0) +\
               cluster_env.all_reduce_cost(batch_size * hidden_dim * 4 / mesh_shape[0], 1)
            assert_close(objective, expected)

    def test_mlp_n_layer_2d_mesh(self):
        # Build Hlo Computation
        num_layers = 12
        batch_size = 1024
        hidden_dim = 128

        computation = get_mlp_n_layer_computation(num_layers, batch_size, hidden_dim,
            hidden_dim, hidden_dim)

        for mesh_shape in [(4, 8), (8, 4), (3, 4)]:
            device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
            cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 0.01],
                                             memory_per_device=1000 * MB)
            objective = solve_auto_sharding(computation, cluster_env)

            expected = \
                num_layers * cluster_env.all_reduce_cost(
                    hidden_dim * hidden_dim * 4 / mesh_shape[1], 0) +\
                (num_layers - 1)  * cluster_env.all_reduce_cost(
                   batch_size * hidden_dim * 4 / mesh_shape[0], 1)
            assert_close(objective, expected)

    def test_mlp_2_layer_bias_data_parallel(self):
        # Build Hlo Computation
        batch_size = 1024
        hidden_dim = 128

        computation = get_mlp_2_layer_bias_computation(batch_size, hidden_dim,
            hidden_dim, hidden_dim)

        # Test different device meshes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
            cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 1],
                                             memory_per_device=1000 * MB)
            objective = solve_auto_sharding(computation, cluster_env)

            expected = \
                cluster_env.all_reduce_cost(hidden_dim * hidden_dim * 4, i) * 2 +\
                cluster_env.all_reduce_cost(hidden_dim * 4, i) * 2
            assert_close(objective, expected)

    def test_mlp_2_layer_bias_model_parallel(self):
        # Build Hlo Computation
        batch_size = 128
        hidden_dim = 1024

        computation = get_mlp_2_layer_bias_computation(batch_size, hidden_dim,
            hidden_dim, hidden_dim)

        # Test different device meshes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
            cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 1],
                                             memory_per_device=1000 * MB)
            objective = solve_auto_sharding(computation, cluster_env)

            expected = cluster_env.all_reduce_cost(batch_size * hidden_dim * 4, i)
            assert_close(objective, expected)

    def test_mlp_2_layer_bias_2d_mesh(self):
        # Build Hlo Computation
        batch_size = 1024
        hidden_dim = 128

        computation = get_mlp_2_layer_bias_computation(batch_size, hidden_dim,
            hidden_dim, hidden_dim)

        # Test different device meshes
        for mesh_shape in [(4, 8), (8, 4), (3, 4)]:
            device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
            cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 0.01],
                                             memory_per_device=1000 * MB)
            objective = solve_auto_sharding(computation, cluster_env)

            expected = \
                cluster_env.all_reduce_cost(batch_size * hidden_dim * 4 / mesh_shape[0], 1) +\
                cluster_env.all_reduce_cost(hidden_dim * hidden_dim * 4 / mesh_shape[1], 0) * 2 +\
                cluster_env.all_reduce_cost(hidden_dim * 4, 0) +\
                cluster_env.all_reduce_cost(hidden_dim * 4 / mesh_shape[1], 0)
            assert_close(objective, expected)


    def test_mlp_2_layer_force_data_parallel(self):
        # Build Hlo Computation
        batch_size = 128
        hidden_dim = 1024

        computation = get_mlp_2_layer_computation(batch_size, hidden_dim,
            hidden_dim, hidden_dim)

        # Test different device meshes
        mesh_shape = [4, 1]
        device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
        solver_option = SolverOption()
        solver_option.force_batch_dim_to_mesh_dim = 0
        solver_option.force_all_gather_cost = 1e10
        cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 1],
                                         memory_per_device=1000 * MB,
                                         solver_option=solver_option)
        objective = solve_auto_sharding(computation, cluster_env, solver_option)

        # The expecte cost is always one all-reduce on activations
        expected = 2 * cluster_env.all_reduce_cost(hidden_dim * hidden_dim * 4, 0)
        assert_close(objective, expected)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(MLPSolverTest('test_mlp_2_layer_data_parallel'))
    suite.addTest(MLPSolverTest('test_mlp_2_layer_model_parallel'))
    suite.addTest(MLPSolverTest('test_mlp_n_layer_data_parallel'))
    suite.addTest(MLPSolverTest('test_mlp_n_layer_model_parallel'))

    suite.addTest(MLPSolverTest('test_mlp_2_layer_2d_mesh'))
    suite.addTest(MLPSolverTest('test_mlp_n_layer_2d_mesh'))

    suite.addTest(MLPSolverTest('test_mlp_2_layer_bias_data_parallel'))
    suite.addTest(MLPSolverTest('test_mlp_2_layer_bias_model_parallel'))
    suite.addTest(MLPSolverTest('test_mlp_2_layer_bias_2d_mesh'))

    suite.addTest(MLPSolverTest('test_mlp_2_layer_force_data_parallel'))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

