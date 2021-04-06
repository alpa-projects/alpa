from collections import defaultdict
from enum import Enum

import numpy as np
import pulp
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, lpDot, LpStatus


def compute_bytes(shape):
    return np.prod(shape) * 4


class ShardingSpec(Enum):
    ONE = 0
    SPLIT_0 = 1
    SPLIT_1 = 2
    REPLICATE = 3
    TUPLE = 4


class ClusterEnvironment:
    def __init__(self, num_devices, memory_per_device):
        self.num_devices = num_devices
        self.memory_per_device = memory_per_device

    def all_reduce_cost(self, num_bytes):
        return 2 * (self.num_devices - 1) / self.num_devices * num_bytes

    def all_gather_cost(self, num_bytes):
        return (self.num_devices - 1) / self.num_devices * num_bytes

    def resharding_cost(self, shape, src_spec, dst_spec):
        if src_spec == ShardingSpec.SPLIT_0:
            if dst_spec == ShardingSpec.SPLIT_0:
                return 0
            elif dst_spec == ShardingSpec.SPLIT_1:
                return self.all_gather_cost(compute_bytes(shape))
            elif dst_spec == ShardingSpec.REPLICATE:
                return self.all_gather_cost(compute_bytes(shape))
            else:
                raise ValueError(f"Invalid sharding spec: {dst_spec}")
        elif src_spec == ShardingSpec.SPLIT_1:
            if dst_spec == ShardingSpec.SPLIT_0:
                return self.all_gather_cost(compute_bytes(shape))
            elif dst_spec == ShardingSpec.SPLIT_1:
                return 0
            elif dst_spec == ShardingSpec.REPLICATE:
                return self.all_gather_cost(compute_bytes(shape))
            else:
                raise ValueError(f"Invalid sharding spec: {dst_spec}")
        elif src_spec == ShardingSpec.REPLICATE:
            return 0
        else:
            raise ValueError(f"Invalid sharding spec: {src_spec}")


def resharding_cost_vector(source_ins, required_spec, cluster_env):
    cost_vector = []
    for strategy in source_ins.strategies:
        cost_vector.append(cluster_env.resharding_cost(source_ins.shape,
            strategy.output_spec, required_spec))
    return cost_vector


class OpCode(Enum):
    PARAMETER = 0
    MATMUL = 1
    TUPLE = 2

op_code_ct = defaultdict(int)


class InstructionStrategy:
    def __init__(self, name, output_spec):
        self.name = name
        self.output_spec = output_spec


class HloInstruction:
    def __init__(self, op_code, shape, operands=[], attrs=None):
        # Attributes
        self.op_code = op_code
        self.shape = shape
        self.operands = operands
        self.attrs = attrs
        self.name = f"{str(op_code)[7:].lower()}.{op_code_ct[op_code]}"
        op_code_ct[op_code] += 1

        # Cost
        self.strategies = []
        self.compute_costs = []
        self.communication_costs = []
        self.memory_costs = []
        self.resharding_costs = []

        # The index in HloComputation
        self.index = None


    def build_strategy_and_cost(self, cluster_env):
        if self.op_code == OpCode.PARAMETER:
            assert len(self.shape) == 2

            self.strategies.append(InstructionStrategy("S0", ShardingSpec.SPLIT_0))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)

            self.strategies.append(InstructionStrategy("S1", ShardingSpec.SPLIT_1))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)

            self.strategies.append(InstructionStrategy("R", ShardingSpec.REPLICATE))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape))
        elif self.op_code == OpCode.MATMUL:
            assert len(self.shape) == 2

            # SPLIT_0 = SPLIT_0 * REPLICATE
            self.strategies.append(InstructionStrategy("S0 = S0 x R", ShardingSpec.SPLIT_0))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
            self.resharding_costs.append([
                resharding_cost_vector(self.operands[0], ShardingSpec.SPLIT_0, cluster_env),
                resharding_cost_vector(self.operands[1], ShardingSpec.REPLICATE, cluster_env),
            ])

            # SPLIT_1 = REPLICATE * SPLIT_1
            self.strategies.append(InstructionStrategy("S1 = R x S1", ShardingSpec.SPLIT_1))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
            self.resharding_costs.append([
                resharding_cost_vector(self.operands[0], ShardingSpec.REPLICATE, cluster_env),
                resharding_cost_vector(self.operands[1], ShardingSpec.SPLIT_1, cluster_env),
            ])

            # REPLICATE = SPLIT_1 * SPLIT_0
            self.strategies.append(InstructionStrategy("R = S1 x S0", ShardingSpec.REPLICATE))
            self.compute_costs.append(0)
            self.communication_costs.append(cluster_env.all_reduce_cost(compute_bytes(self.shape)))
            self.memory_costs.append(compute_bytes(self.shape))
            self.resharding_costs.append([
                resharding_cost_vector(self.operands[0], ShardingSpec.SPLIT_1, cluster_env),
                resharding_cost_vector(self.operands[1], ShardingSpec.SPLIT_0, cluster_env),
            ])

            # REPLICATE = REPLICATE * REPLICATE
            self.strategies.append(InstructionStrategy("R = R x R", ShardingSpec.REPLICATE))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape))
            self.resharding_costs.append([
                resharding_cost_vector(self.operands[0], ShardingSpec.REPLICATE, cluster_env),
                resharding_cost_vector(self.operands[1], ShardingSpec.REPLICATE, cluster_env),
            ])
        elif self.op_code == OpCode.TUPLE:
            self.strategies.append(InstructionStrategy("tuple", ShardingSpec.TUPLE))
            self.memory_costs.append(0)
            self.compute_costs.append(0)
            self.resharding_costs.append([np.zeros(len(operand.strategies))
                for operand in self.operands])
        else:
            raise ValueError(f"Invalid op code: {self.op_code}")

    def __str__(self):
        if self.op_code == OpCode.PARAMETER:
            return f"{self.name} {self.shape} = parameter()"
        elif self.op_code == OpCode.MATMUL:
            lhs = self.operands[0].name
            rhs = self.operands[1].name
            return f"{self.name} {self.shape} = matmul({lhs}, {rhs})"
        elif self.op_code == OpCode.TUPLE:
            names = tuple(x.name for x in self.operands)
            return f"{self.name} {self.shape} = tuple{names}"
        else:
            raise ValueError(f"Invalid op code: {self.op_code}")


class HloComputation:
    def __init__(self, instructions):
        self.instructions = instructions

        for i in range(len(instructions)):
            instructions[i].index = i

    def liveness_analysis(self):
        liveness_dict = dict()

        live_set = set()

        for t in range(len(self.instructions)-1, -1, -1):
            inst = self.instructions[t]

            live_set.add(inst)
            for operand in inst.operands:
                live_set.add(operand)

            liveness_dict[t] = set(live_set)

            live_set.remove(inst)

        return liveness_dict

    def __str__(self):
        strs = []
        for ins in self.instructions:
            strs.append(str(ins))
        return "\n".join(strs)
    

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
    for ins in computation.instructions:
        ins.build_strategy_and_cost(cluster_env)

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
        c.append([int(x) for x in ins.compute_costs])
        d.append([int(x) for x in ins.communication_costs])
        m.append([int(x) for x in ins.memory_costs])
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
                    cost.append(int(ins.resharding_costs[q][op_idx][p]))
            r[src][dst] = cost

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
        for r in range(len(s[i])):
            C = len(s[j])
            prob += lpSum(e[i][j][r * C + c] for c in range(0, C)) <= s[i][r]

    # (g)
    for (i, j) in E:
        for c in range(len(s[j])):
            R = len(s[i])
            C = len(s[j])
            prob += lpSum(e[i][j][r * C + c] for r in range(0, R)) <= s[j][c]

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

    # Print peak memory
    for t in range(N):
        mem = 0
        for i in L[t]:
            mem += sum(pulp.value(s[i][j]) * m[i][j] for j in range(len(s[i])))
        print(f"Time {i}, memory: {mem / 1024**2: .2f} MB")


def test_mlp_2_layer_forward():
    # Build Hlo Computation
    batch_size = 128
    input_dim = hidden_dim = output_dim = 1024

    x = HloInstruction(OpCode.PARAMETER, (batch_size, input_dim))
    w1 = HloInstruction(OpCode.PARAMETER, (input_dim, hidden_dim))
    w2 = HloInstruction(OpCode.PARAMETER, (hidden_dim, output_dim))

    h1 = HloInstruction(OpCode.MATMUL, (batch_size, hidden_dim), [x, w1])
    h2 = HloInstruction(OpCode.MATMUL, (batch_size, output_dim), [h1, w2]) 
    out = HloInstruction(OpCode.TUPLE, (), [h2, w1, w2])

    computation = HloComputation([x, w1, w2, h1, h2, out])
    cluster_env = ClusterEnvironment(num_devices=4, memory_per_device=4 * 1024**2)

    solve_auto_sharding(computation, cluster_env)


def test_mlp_n_layer_forward():
    # Build Hlo Computation
    batch_size = 128
    input_dim = hidden_dim = output_dim = 1024
    num_layers = 6

    x = HloInstruction(OpCode.PARAMETER, (batch_size, input_dim))
    w_first = HloInstruction(OpCode.PARAMETER, (input_dim, hidden_dim))
    w_intermidiate = []
    for i in range(num_layers - 2):
        w_intermidiate.append(HloInstruction(OpCode.PARAMETER, (hidden_dim, hidden_dim)))
    w_last = HloInstruction(OpCode.PARAMETER, (hidden_dim, output_dim))

    h_first = HloInstruction(OpCode.MATMUL, (batch_size, hidden_dim), [x, w_first])
    h_now = h_first
    h_intermidiate = []
    for i in range(num_layers - 2):
        h_now = HloInstruction(OpCode.MATMUL, (batch_size, hidden_dim),
                               [h_now, w_intermidiate[i]])
        h_intermidiate.append(h_now)
    h_last = HloInstruction(OpCode.MATMUL, (batch_size, output_dim), [h_now, w_last])
    output = HloInstruction(OpCode.TUPLE, (), [h_last, w_first, w_last] + w_intermidiate)

    computation = HloComputation([x, w_first] + w_intermidiate + [w_last, h_first] +
                                 h_intermidiate + [h_last, output])
    cluster_env = ClusterEnvironment(num_devices=4, memory_per_device=8 * 1024**2)

    solve_auto_sharding(computation, cluster_env)


if __name__ == "__main__":
    test_mlp_2_layer_forward()
    #test_mlp_n_layer_forward()

