"""ILP Solver"""

import pulp
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, lpDot, LpStatus


def get_non_zero_index(binary_vector):
    """Get the index of non-zero item in a vector"""
    ct = 0
    ret = None
    for i in range(len(binary_vector)):
        if pulp.value(binary_vector[i]):
            ret = i
            ct += 1

    assert ct == 1
    return ret


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

    # Build an ILP problem
    # 0. Constant
    N = len(computation.instructions)
    M = cluster_env.memory_per_device

    # 1. Strategy vector
    E = []
    L = []
    c = []
    d = []
    m = []
    r = [[None for i in range(N)] for j in range(N)]
    s = []
    e = [[None for i in range(N)] for j in range(N)]
    for i in range(N):
        ins = computation.instructions[i]
        c.append(ins.compute_costs)
        d.append(ins.communication_costs)
        m.append(ins.memory_costs)
        s.append(LpVariable.matrix(f"s[{i}]",
            (range(len(ins.strategies)),), cat="Binary"))
        L.append([ins.index for ins in liveness_dict[i]])
        for op_idx, operand in enumerate(ins.operands):
            E.append((operand.index, i))

            src = operand.index
            dst = i

            #ins.resharding_costs  # [s_i, operand_idx, s_operand]
            cost = []
            for p in range(len(s[src])):
                for q in range(len(s[dst])):
                    cost.append(ins.resharding_costs[q][op_idx][p])
            r[src][dst] = cost

    for ((ins_a, ins_b), cost_vector) in zip(computation.alias_list,
                                             computation.alias_cost_vector):
        E.append((ins_a.index, ins_b.index))
        #assert r[ins_a.index][ins_b.index] is None
        r[ins_a.index][ins_b.index] = cost_vector

    for (i, j) in E:
        e[i][j] = LpVariable.matrix(f"e[{i},{j}]",
            (range(len(s[i]) * len(s[j])),), cat="Binary")
        assert len(r[i][j]) == len(e[i][j])


    # 2. Objective
    prob = LpProblem("myProblem", LpMinimize)
    # (a). compute cost
    obj = 0
    for i in range(N):
        obj += lpDot(s[i], c[i]) + lpDot(s[i], d[i])

    # (b). communication cost
    for (i, j) in E:
        obj += lpDot(e[i][j], r[i][j])

    prob += obj

    # 3. Constraints
    # (a). specified by `cat="Binary"`

    # (b)
    for i in range(N):
        prob += lpSum(s[i]) == 1

    # (c)
    for t in range(N):
        mem = 0
        for i in L[t]:
            mem += lpSum(s[i][j] * m[i][j] for j in range(len(s[i])))
        prob += mem <= M

    # (d). specified by `cat="Binary"`

    # (e)
    for (i, j) in E:
        prob += lpSum(e[i][j]) == 1

    # (f)
    for (i, j) in E:
        for row in range(len(s[i])):
            C = len(s[j])
            prob += lpSum(e[i][j][row * C + col] for col in range(0, C)) <= s[i][row]

    # (g)
    for (i, j) in E:
        for col in range(len(s[j])):
            R = len(s[i])
            C = len(s[j])
            prob += lpSum(e[i][j][row * C + col] for row in range(0, R)) <= s[j][col]

    #print(prob)

    prob.solve()
    print("Status:", LpStatus[prob.status])

    # Print sharding spec
    instructions = computation.instructions
    for i in range(N):
        spec_index = get_non_zero_index(s[i])
        print(f"Instruction {i}, Strategy: "
              f"{instructions[i].strategies[spec_index].name}")

    # Check edges
    for (i, j) in E:
        spec_index = get_non_zero_index(e[i][j])
        i_spec_index = spec_index // len(s[j])
        j_spec_index = spec_index % len(s[j])
        assert i_spec_index == get_non_zero_index(s[i])
        assert j_spec_index == get_non_zero_index(s[j])

        if r[i][j][spec_index] > 0.0:
            print("Edge cost", i, j)

    # Print peak memory
    for t in range(N):
        mem = 0
        for i in L[t]:
            mem += sum(pulp.value(s[i][j]) * m[i][j] for j in range(len(s[i])))
        print(f"Time {t}, memory: {mem / 1024**2: .2f} MB")

