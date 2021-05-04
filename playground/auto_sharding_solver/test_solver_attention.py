"""
Usage:
python3 -m unittest -bv test_solver_attention.py
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


def solve_without_all_gather(computation, mesh_shape):
    device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
    solver_option = SolverOption()
    solver_option.force_all_gather_cost = 1e8
    cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 1],
                                     memory_per_device=1000 * MB,
                                     solver_option=solver_option)
    objective = solve_auto_sharding(computation, cluster_env, solver_option)
    return objective, cluster_env


def get_attention_forward_computation(batch_size, seq_len, hidden_dim, num_head, force_replicated_output):
    per_head = hidden_dim // num_head
    computation = HloComputation()

    with computation:
        # hidden states
        hidden_states = HloParameter((batch_size, seq_len, hidden_dim))
        hidden_states = HloReshape(hidden_states, (batch_size * seq_len, hidden_dim))

        # query matmul
        weight_query_dense = HloParameter((hidden_dim, num_head, per_head))
        weight_query_dense_ = HloReshape(weight_query_dense, (hidden_dim, hidden_dim))
        query = HloDot(hidden_states, weight_query_dense_)
        query = HloReshape(query, (batch_size, seq_len, num_head, per_head))

        # query bias_add
        bias_query_dense = HloParameter((num_head, per_head))
        bias_query_dense_ = HloBroadcast(bias_query_dense, (batch_size, seq_len, num_head, per_head), dimensions=(2, 3))
        query = HloAdd(query, bias_query_dense_)

        # query normalization
        c = HloConstant(0.125)
        c = HloBroadcast(c, (batch_size, seq_len, num_head, per_head))
        query = HloMutiply(c, query)
        # query transpose
        query = HloTranspose(query, [0, 2, 1, 3])

        # key matmul
        weight_key_dense = HloParameter((hidden_dim, num_head, per_head))
        weight_key_dense_ = HloReshape(weight_key_dense, (hidden_dim, hidden_dim))
        key = HloDot(hidden_states, weight_key_dense_)
        key = HloReshape(key, (batch_size, seq_len, num_head, per_head))

        # key bias_add
        bias_key_dense = HloParameter((num_head, per_head))
        bias_key_dense_ = HloBroadcast(bias_key_dense, (batch_size, seq_len, num_head, per_head), dimensions=(2, 3))
        key = HloAdd(key, bias_key_dense_)

        # key transpose
        key = HloTranspose(key, [0, 2, 3, 1])

        # att_weight
        att_weight = HloDot(query, key,
                            lhs_batch_dims=(0,1), lhs_contracting_dims=(3,),
                            rhs_batch_dims=(0,1), rhs_contracting_dims=(2,))

        # mask
        mask = HloParameter((batch_size, seq_len))

        # attention_bias_pred
        zero = HloConstant(0)
        zero = HloBroadcast(zero, (batch_size, seq_len))
        pred = HloCompare(mask, zero)

        # all zero
        zero = HloConstant(0)
        zero = HloBroadcast(zero, (batch_size, seq_len))

        # all neg-infinity
        neg_inf = HloConstant(-1e10)
        neg_inf = HloBroadcast(neg_inf, (batch_size, seq_len))

        # attention bias
        select = HloSelect(pred, zero, neg_inf)

        # attention bias_add
        att_bias = HloBroadcast(select, (batch_size, num_head, seq_len, seq_len), dimensions=(0, 3))
        att_weight = HloAdd(att_weight, att_bias)

        # softmax_max
        max_reduce = HloReduce(att_weight, dimensions=(3,))
        max_reduce = HloBroadcast(max_reduce, (batch_size, num_head, seq_len, seq_len), dimensions=(0, 1, 2))
        diff = HloSubtract(att_weight, max_reduce)
        exp = HloExp(diff)
        # softmax_sum
        sum_reduce = HloReduce(exp, dimensions=(3,))
        sum_reduce = HloBroadcast(sum_reduce, (batch_size, num_head, seq_len, seq_len), dimensions=(0, 1, 2))
        # softmax_norm
        softmax = HloDiv(exp, sum_reduce)

        # value matmul
        weight_value_dense = HloParameter((hidden_dim, num_head, per_head))
        weight_value_dense_ = HloReshape(weight_value_dense, (hidden_dim, hidden_dim))
        value = HloDot(hidden_states, weight_value_dense_)
        value = HloReshape(value, (batch_size, seq_len, num_head, per_head))

        # value bias_add
        bias_value_dense = HloParameter((num_head, per_head))
        bias_value_dense_ = HloBroadcast(bias_value_dense, (batch_size, seq_len, num_head, per_head), dimensions=(2, 3))
        value = HloAdd(value, bias_value_dense_)

        # value transpose
        value = HloTranspose(value, [0, 2, 3, 1])

        # self attention
        self_att = HloDot(value, softmax,
                          lhs_batch_dims=(0, 1), lhs_contracting_dims=(3,),
                          rhs_batch_dims=(0, 1), rhs_contracting_dims=(3,))
        self_att = HloTranspose(self_att, [0, 3, 1, 2])
        self_att = HloReshape(self_att, [batch_size * seq_len, hidden_dim])

        # out matmul
        weight_out_dense = HloParameter((hidden_dim, num_head, per_head))
        weight_out_dense_ = HloReshape(weight_out_dense, (hidden_dim, hidden_dim))
        out = HloDot(self_att, weight_out_dense_)
        out = HloReshape(out, (batch_size, seq_len, hidden_dim))

        # out bias_add
        bias_out_dense = HloParameter((hidden_dim,))
        bias_out_dense_ = HloBroadcast(bias_out_dense, (batch_size, seq_len, hidden_dim), dimensions=(2,))
        out = HloAdd(out, bias_out_dense_)

        if force_replicated_output:
            out = HloForceReplicated(out)

        out = HloTuple([out,
                        weight_value_dense, bias_value_dense, 
                        weight_query_dense, bias_query_dense,
                        weight_key_dense, bias_key_dense,
                        weight_out_dense, bias_out_dense,
        ])

    return computation

class AttentionSolverTest(unittest.TestCase):
    def test_tranpose(self):
        # Build Hlo Computation
        computation = HloComputation()
        dim_0 = 128
        dim_1 = 2048

        with computation:
            x = HloParameter((dim_1, dim_0))
            y = HloParameter((dim_0, dim_1))
            x = HloTranspose(x, [1, 0])
            y = HloTranspose(y, [1, 0])
            out = HloDot(x, y)
            out = HloTranspose(out, [1, 0])
            out = HloForceReplicated(out)
            out = HloTuple((out,))

        # Solve
        mesh_shape = [1, 4]
        objective, cluster_env = solve_without_all_gather(computation, mesh_shape)

        expected = cluster_env.all_reduce_cost(dim_0 * dim_0 * 4, 1)
        print("Objective:", objective)
        print("Expected:", expected)
        assert_close(objective, expected)

    def test_mulit_tranpose(self):
        # Build Hlo Computation
        computation = HloComputation()
        dim_0 = 128
        dim_1 = 2048

        with computation:
            x = HloParameter((dim_1, dim_0))
            y = HloParameter((dim_0, dim_1))
            x = HloTranspose(x, [1, 0])
            y = HloTranspose(y, [1, 0])
            x = HloTranspose(x, [1, 0])
            y = HloTranspose(y, [1, 0])
            x = HloTranspose(x, [1, 0])
            y = HloTranspose(y, [1, 0])
            out = HloDot(x, y)
            out = HloTranspose(out, [1, 0])
            out = HloTranspose(out, [1, 0])
            out = HloForceReplicated(out)
            out = HloTuple((out,))

        # Solve
        mesh_shape = [4, 1]
        objective, cluster_env = solve_without_all_gather(computation, mesh_shape)

        expected = cluster_env.all_reduce_cost(dim_0 * dim_0 * 4, 0)
        print("Objective:", objective)
        print("Expected:", expected)
        assert_close(objective, expected)


    def test_reshape(self):
        # Build Hlo Computation
        computation = HloComputation()
        dim_0 = 128
        dim_1 = 2048

        with computation:
            x = HloParameter((dim_0, dim_1 // 2, 2))
            y = HloParameter((dim_1 // 2, 2, dim_0))
            x = HloReshape(x, (dim_0, dim_1))
            y = HloReshape(y, (dim_1, dim_0))
            out = HloDot(x, y)
            out = HloForceReplicated(out)
            out = HloTuple((out,))

        # Solve
        mesh_shape = [1, 4]
        objective, cluster_env = solve_without_all_gather(computation, mesh_shape)

        expected = cluster_env.all_reduce_cost(dim_0 * dim_0 * 4, 1)
        print("Objective:", objective)
        print("Expected:", expected)
        assert_close(objective, expected)

    def test_mulit_reshape(self):
        # Build Hlo Computation
        computation = HloComputation()
        dim_0 = 128
        dim_1 = 2048

        with computation:
            x = HloParameter((dim_0, dim_1 // 2, 2))
            y = HloParameter((dim_1 // 2, 2, dim_0))
            x = HloReshape(x, (dim_0, dim_1))
            y = HloReshape(y, (dim_1, dim_0))
            x = HloReshape(x, (dim_0 // 4, 4, dim_1))
            y = HloReshape(y, (dim_1 // 4, 4, dim_0))
            x = HloReshape(x, (dim_0, dim_1))
            y = HloReshape(y, (dim_1, dim_0))
            out = HloDot(x, y)
            out = HloReshape(out, (dim_0, 2, dim_0 // 2))
            out = HloForceReplicated(out)
            out = HloTuple((out,))

        # Solve
        mesh_shape = [4, 1]
        objective, cluster_env = solve_without_all_gather(computation, mesh_shape)

        expected = cluster_env.all_reduce_cost(dim_0 * dim_0 * 4, 0)
        print("Objective:", objective)
        print("Expected:", expected)
        assert_close(objective, expected)

    def test_allreduce_simplification(self):
        # Build Hlo Computation
        computation = HloComputation()
        dim_0 = 128
        dim_1 = 2048

        with computation:
            x = HloParameter((dim_0, dim_1))
            y = HloParameter((dim_1, dim_0))
            h1 = HloDot(x, y)
            h2 = HloDot(x, y)
            out = HloAdd(h1, h2)
            out = HloForceReplicated(out)
            out = HloTuple((out,))

        # Solve
        mesh_shape = [1, 4]
        objective, cluster_env = solve_without_all_gather(computation, mesh_shape)

        expected = 2 * cluster_env.all_reduce_cost(dim_0 * dim_0 * 4, 1)
        print("Objective:", objective)
        print("Expected:", expected)
        assert_close(objective, expected)

    def test_allreduce_simplification_out_reuse(self):
        # Build Hlo Computation
        computation = HloComputation()
        dim_0 = 128
        dim_1 = 2048

        with computation:
            x = HloParameter((dim_0, dim_1))
            y = HloParameter((dim_1, dim_0))
            z = HloParameter((dim_0 // 4, 4, dim_0))
            h1 = HloDot(x, y)
            h2 = HloDot(x, y)
            h3 = HloDot(x, y)
            h1 = HloReshape(h1, (dim_0 // 4, 4, dim_0))
            h2 = HloReshape(h2, (dim_0 // 4, 4, dim_0))
            h3 = HloReshape(h3, (dim_0 // 4, 4, dim_0))
            out = z
            out = HloAdd(out, h1)
            out = HloAdd(out, h2)
            out = HloAdd(out, h3)
            b1 = HloExp(out)
            b2 = HloExp(out)
            b3 = HloExp(out)
            b4 = HloExp(out)
            b5 = HloExp(out)
            b6 = HloExp(out)
            b7 = HloForceReplicated(b6)
            out = HloTuple((b1, b2, b3, b4, b5, b6, b7))

        # Solve
        mesh_shape = [1, 4]
        objective, cluster_env = solve_without_all_gather(computation, mesh_shape)

        expected = 3 * cluster_env.all_reduce_cost(dim_0 * dim_0 * 4, 1)
        print("Objective:", objective)
        print("Expected:", expected)
        assert_close(objective, expected)

    def test_attention_forward(self):
        # Build Hlo Computation
        batch_size = 4
        seq_len = 128
        hidden_dim = 512
        num_head = 16

        computation = get_attention_forward_computation(
            batch_size, seq_len, hidden_dim, num_head, True)

        # Solve
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            objective, cluster_env = solve_without_all_gather(computation, mesh_shape)

            expected = cluster_env.all_reduce_cost(batch_size * seq_len * hidden_dim * 4, i)
            print("Objective:", objective)
            print("Expected:", expected)
            assert_close(objective, expected)

    def test_attention_forward_2d_mesh(self):
        # Build Hlo Computation
        batch_size = 4
        seq_len = 128
        hidden_dim = 2048
        num_head = 16

        computation = get_attention_forward_computation(
            batch_size, seq_len, hidden_dim, num_head, False)

        # Solve
        mesh_shape = [4, 4]
        objective, cluster_env = solve_without_all_gather(computation, mesh_shape)

        expected = cluster_env.all_reduce_cost(
            batch_size * seq_len * hidden_dim * 4 / mesh_shape[0], 1)
        print("Objective:", objective)
        print("Expected:", expected)
        assert_close(objective, expected)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AttentionSolverTest('test_tranpose'))
    suite.addTest(AttentionSolverTest('test_mulit_tranpose'))
    suite.addTest(AttentionSolverTest('test_reshape'))
    suite.addTest(AttentionSolverTest('test_mulit_reshape'))
    suite.addTest(AttentionSolverTest('test_allreduce_simplification'))
    suite.addTest(AttentionSolverTest('test_allreduce_simplification_out_reuse'))
    suite.addTest(AttentionSolverTest('test_attention_forward'))
    suite.addTest(AttentionSolverTest('test_attention_forward_2d_mesh'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

