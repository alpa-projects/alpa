"""ILP Solver"""
import numpy as np
import pulp
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, lpDot, LpStatus

from parax.auto_sharding import call_solver_serialized_args


def call_solver(N, M, s_len, E, L, c, d, m, r):
    s_len_np = np.empty((N,), dtype=np.int32)
    s_len_np[:] = s_len

    # serialize edge set 
    len_edges = len(E)
    E_np = np.empty((len_edges, 2), dtype=np.int32)
    for (no, (i, j)) in enumerate(E):
        E_np[no][:] = [i, j]

    # serialize liveness set 
    len_liveness_set = N + sum(len(v) for v in L)
    L_np = np.empty((len_liveness_set,), dtype=np.int32)
    L_np[0:N] = [len(v) for v in L]
    L_np[N:] = [x for v in L for x in v]

    # serialize node costs 
    len_node_costs = sum(len(v) for v in c)
    c_np = np.empty((len_node_costs,), dtype=np.float32)
    d_np = np.empty((len_node_costs,), dtype=np.float32)
    m_np = np.empty((len_node_costs,), dtype=np.float32)
    c_np[:] = [x for v in c for x in v]
    d_np[:] = [x for v in d for x in v]
    m_np[:] = [x for v in m for x in v]

    # serialize edge costs 
    len_edge_costs = sum(len(r[i][j]) for (i, j) in E)
    r_np = np.empty((len_edge_costs,), dtype=np.float32)
    r_np[:] = [x for (i, j) in E for x in r[i][j]]

    return call_solver_serialized_args(
        N, M, s_len_np, E_np, L_np, c_np, d_np, m_np, r_np)


def solve_auto_sharding(computation, cluster_env):
    print("===== Hlo Computation =====")
    print(computation, "\n")

    print("===== Liveness Analysis =====")
    liveness_dict = computation.liveness_analysis()
    for i in range(len(computation.instructions)):
        names = [ins.name for ins in liveness_dict[i]]
        names.sort()
        print(f"Time: {i}, Live set: {names}")

    # Build strategies and costs
    computation.build_strategy_and_cost(cluster_env)

    # Build all constants for ILP
    N = len(computation.instructions)
    M = cluster_env.memory_per_device

    s_len = []
    E = []
    L = []
    c = []
    d = []
    m = []
    r = [[None for i in range(N)] for j in range(N)]
    for i in range(N):
        ins = computation.instructions[i]
        s_len.append(len(ins.strategies))
        L.append([ins.index for ins in liveness_dict[i]])
        c.append(ins.compute_costs)
        d.append(ins.communication_costs)
        m.append(ins.memory_costs)

        for op_idx, operand in enumerate(ins.operands):
            E.append((operand.index, i))

            src = operand.index
            dst = i

            #ins.resharding_costs  # [s_i, operand_idx, s_operand]
            cost = []
            for p in range(len(computation.instructions[src].strategies)):
                for q in range(len(computation.instructions[dst].strategies)):
                    cost.append(ins.resharding_costs[q][op_idx][p])
            r[src][dst] = cost

    for ((ins_a, ins_b), cost_vector) in zip(computation.alias_list,
                                             computation.alias_cost_vector):
        E.append((ins_a.index, ins_b.index))
        r[ins_a.index][ins_b.index] = cost_vector

    s_val, e_val = call_solver(N, M, s_len, E, L, c, d, m, r)

    # Print sharding spec
    instructions = computation.instructions
    for i in range(N):
        name = instructions[i].strategies[s_val[i]].name
        print(f"Time {i:2d}: {computation.instructions[i]}  Strategy: {name}")

    # Print edge cost
    for (no, (i, j)) in enumerate(E):
        if r[i][j][e_val[no]] > 0:
            print("Edge cost", i, j)

    # Print peak memory
    for t in range(N):
        mem = 0
        for i in L[t]:
            mem += m[i][s_val[i]]
        print(f"Time {t}, memory: {mem / 1024**2: .2f} MB")

