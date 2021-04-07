from collections import defaultdict
from enum import Enum

import numpy as np

from hlo import *
from cluster_env import ClusterEnvironment
from solver import solve_auto_sharding


def test_mlp_2_layer_forward():
    # Build Hlo Computation
    batch_size = 128
    input_dim = hidden_dim = output_dim = 1024

    computation = HloComputation()

    with computation:
        x = HloParameter((batch_size, input_dim))
        w1 = HloParameter((input_dim, hidden_dim))
        w2 = HloParameter((hidden_dim, output_dim))
        h1 = HloDot(x, w1)
        h2 = HloDot(h1, w2)
        out = HloTuple((h2, w1, w2))

    cluster_env = ClusterEnvironment(num_devices=4, memory_per_device=4 * 1024**2)
    solve_auto_sharding(computation, cluster_env)


def test_mlp_2_layer_forward_backward():
    # Build Hlo Computation
    batch_size = 128
    input_dim = hidden_dim = output_dim = 1024


    computation = HloComputation()
    with computation:
        x = HloParameter((batch_size, input_dim))
        y = HloParameter((batch_size, output_dim))
        w1 = HloParameter((input_dim, hidden_dim))
        w2 = HloParameter((hidden_dim, output_dim))

        h1 = HloDot(x, w1)
        h2 = HloDot(h1, w2)
        loss = HloSubtract(h2, y)

        coef = HloConstant(1.52587891e-05)
        coef = HloBroadcast(coef, (batch_size, output_dim))

        grad_loss = HloMutiply(loss, coef)
        grad_w2 = HloDot(h1, grad_loss,
                         lhs_contracting_dims=(0,),
                         rhs_contracting_dims=(0,),)
        new_w2 = HloSubtract(w2, grad_w2)
        grad_h2 = HloDot(grad_loss, w2,
                         lhs_contracting_dims=(1,),
                         rhs_contracting_dims=(1,),)
        grad_w1 = HloDot(x, grad_h2,
                         lhs_contracting_dims=(0,),
                         rhs_contracting_dims=(0,),)
        new_w1 = HloSubtract(w1, grad_w1)
        out = HloTuple((new_w1, new_w2))

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

    # | method                        | peak mem | communication cost          |
    # | ----------------------------- | -------- | --------------------------- |
    # | data-parallel                 | 16.3 M   | 2 all-reduce                |
    # | megatron-LM                   | 5.00 M   | 1 all-reduce                |
    # | megatron-LM + naive partition | 4.62 M   | 1 all-reduce + 2 all-gather |
    # | ours new finding              | 4.25 M   | 2 all-reduce                |
    cluster_env = ClusterEnvironment(num_devices=4, memory_per_device=5 * 1024**2)
    solve_auto_sharding(computation, cluster_env)


def test_mlp_n_layer_forward():
    # Build Hlo Computation
    batch_size = 128
    input_dim = hidden_dim = output_dim = 1024
    num_layers = 6

    computation = HloComputation()
    with computation:
        x = HloParameter((batch_size, input_dim))
        w_first = HloParameter((input_dim, hidden_dim))
        w_inter = []
        for i in range(num_layers - 2):
            w_inter.append(HloParameter((hidden_dim, hidden_dim)))
        w_last = HloParameter((hidden_dim, output_dim))

        h_first = HloDot(x, w_first)
        h_now = h_first
        for i in range(num_layers - 2):
            h_now = HloDot(h_now, w_inter[i])
        h_last = HloDot(h_now, w_last)
        out = HloTuple([h_last, w_first, w_last] + w_inter)

    cluster_env = ClusterEnvironment(num_devices=4, memory_per_device=8 * 1024**2)
    solve_auto_sharding(computation, cluster_env)


if __name__ == "__main__":
    #test_mlp_2_layer_forward()
    test_mlp_2_layer_forward_backward()
    #test_mlp_n_layer_forward()

