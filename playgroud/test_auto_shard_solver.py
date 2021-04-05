from collections import defaultdict
from enum import Enum

import numpy as np
import pulp
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, LpStatus

def compute_bytes(shape):
    return np.prod(shape) * 4

class OpCode(Enum):
    PARAMETER = 0
    MATMUL = 1
    TUPLE = 2

op_code_ct = defaultdict(int)


class ShardingSpec(Enum):
    ONE = 0
    SPLIT_0 = 1
    SPLIT_1 = 2
    REPLICATE = 3
    TUPLE = 4

def comm_cost(shape, src_spec, dst_spec, num_devices):
    if src_spec == ShardingSpec.SPLIT_0:
        if dst_spec == ShardingSpec.SPLIT_0:
            return 0
        elif dst_spec == ShardingSpec.SPLIT_1:
            # all-gather
            return (num_devices - 1) / num_devices * compute_bytes(shape)
        elif dst_spec == ShardingSpec.REPLICATE:
            # all-gather
            return (num_devices - 1) / num_devices * compute_bytes(shape)
        else:
            raise ValueError(f"Invalid sharding spec: {dst_spec}")
    elif src_spec == ShardingSpec.SPLIT_1:
        if dst_spec == ShardingSpec.SPLIT_0:
            # all-gather
            return (num_devices - 1) / num_devices * compute_bytes(shape)
        elif dst_spec == ShardingSpec.SPLIT_1:
            return 0
        elif dst_spec == ShardingSpec.REPLICATE:
            # all-gather
            return (num_devices - 1) / num_devices * compute_bytes(shape)
        else:
            raise ValueError(f"Invalid sharding spec: {dst_spec}")
    elif src_spec == ShardingSpec.REPLICATE:
        return 0
    else:
        raise ValueError(f"Invalid sharding spec: {src_spec}")


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
        self.sharding_specs = []
        self.memory_costs = []
        self.compute_costs = []
        self.communication_costs = []

        # Id in HloComputation
        self.index = None

    def build_intra_inst_cost(self, num_devices):
        if self.op_code == OpCode.PARAMETER:
            assert len(self.shape) == 2

            self.sharding_specs.append(ShardingSpec.SPLIT_0)
            self.memory_costs.append(compute_bytes(self.shape) / num_devices)
            self.compute_costs.append(0)

            self.sharding_specs.append(ShardingSpec.SPLIT_1)
            self.memory_costs.append(compute_bytes(self.shape) / num_devices)
            self.compute_costs.append(0)

            self.sharding_specs.append(ShardingSpec.REPLICATE)
            self.memory_costs.append(compute_bytes(self.shape))
            self.compute_costs.append(0)
        elif self.op_code == OpCode.MATMUL:
            assert len(self.shape) == 2

            # SPLIT_0 = SPLIT_0 * R
            self.sharding_specs.append(ShardingSpec.SPLIT_0)
            self.memory_costs.append(compute_bytes(self.shape) / num_devices +
                                     compute_bytes(self.operands[0].shape) / num_devices +
                                     compute_bytes(self.operands[1].shape))
            self.compute_costs.append(0)

            # SPLIT_1 = R * SPLIT_1
            self.sharding_specs.append(ShardingSpec.SPLIT_1)
            self.memory_costs.append(compute_bytes(self.shape) / num_devices +
                                     compute_bytes(self.operands[0].shape) +
                                     compute_bytes(self.operands[1].shape) / num_devices)
            self.compute_costs.append(0)

            # R = R * R
            self.sharding_specs.append(ShardingSpec.REPLICATE)
            self.memory_costs.append(compute_bytes(self.shape) +
                                     compute_bytes(self.operands[0].shape) +
                                     compute_bytes(self.operands[1].shape))
            self.compute_costs.append(0)
        elif self.op_code == OpCode.TUPLE:
            self.sharding_specs = [ShardingSpec.TUPLE]
            self.memory_costs.append(0)
            self.compute_costs.append(0)
        else:
            raise ValueError(f"Invalid op code: {self.op_code}")


    def build_inter_inst_cost(self, num_devices):
        if self.op_code == OpCode.PARAMETER:
            assert len(self.operands) == 0
        elif self.op_code == OpCode.MATMUL:
            for i in range(len(self.sharding_specs)):
                lhs, rhs = self.operands[0], self.operands[1]
                lhs_cost, rhs_cost = [], []
                if self.sharding_specs[i] == ShardingSpec.SPLIT_0:
                    # SPLIT_0 = SPLIT_0 * R
                    for j in range(len(lhs.sharding_specs)):
                        cost = comm_cost(lhs.shape, lhs.sharding_specs[j],
                                         ShardingSpec.SPLIT_0, num_devices)
                        lhs_cost.append(cost)
                    for j in range(len(rhs.sharding_specs)):
                        cost = comm_cost(rhs.shape, rhs.sharding_specs[j],
                                         ShardingSpec.REPLICATE, num_devices)
                        rhs_cost.append(cost)
                elif self.sharding_specs[i] == ShardingSpec.SPLIT_1:
                    # SPLIT_1 = R * SPLIT_1
                    for j in range(len(lhs.sharding_specs)):
                        cost = comm_cost(lhs.shape, lhs.sharding_specs[j],
                                         ShardingSpec.REPLICATE, num_devices)
                        lhs_cost.append(cost)
                    for j in range(len(rhs.sharding_specs)):
                        cost = comm_cost(rhs.shape, rhs.sharding_specs[j],
                                         ShardingSpec.SPLIT_1, num_devices)
                        rhs_cost.append(cost)
                elif self.sharding_specs[i] == ShardingSpec.REPLICATE:
                    # R = R * R
                    for j in range(len(lhs.sharding_specs)):
                        cost = comm_cost(lhs.shape, lhs.sharding_specs[j],
                                         ShardingSpec.REPLICATE, num_devices)
                        lhs_cost.append(cost)
                    for j in range(len(rhs.sharding_specs)):
                        cost = comm_cost(rhs.shape, rhs.sharding_specs[j],
                                         ShardingSpec.REPLICATE, num_devices)
                        rhs_cost.append(cost)
                else:
                    raise ValueError(f"Invalid sharding spec: {self.sharding_specs[i]}")
                self.communication_costs.append([lhs_cost, rhs_cost])
        elif self.op_code == OpCode.TUPLE:
            self.communication_costs = [[np.zeros(len(operand.sharding_specs))
                for operand in self.operands]]
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
    

def test_mlp_forward():
    # Build Hlo Computation
    batch_size = 64
    input_dim = hidden_dim = output_dim = 1024

    x = HloInstruction(OpCode.PARAMETER, (batch_size, input_dim))
    w1 = HloInstruction(OpCode.PARAMETER, (input_dim, hidden_dim))
    w2 = HloInstruction(OpCode.PARAMETER, (hidden_dim, output_dim))

    h1 = HloInstruction(OpCode.MATMUL, (batch_size, hidden_dim), [x, w1])
    h2 = HloInstruction(OpCode.MATMUL, (batch_size, output_dim), [h1, w2]) 
    out = HloInstruction(OpCode.TUPLE, (), [h2, w1, w2])

    computation = HloComputation([x, w1, w2, h1, h2, out])
    print("===== Hlo Computation =====")
    print(computation, "\n")

    print("===== Liveness Analysis =====")
    liveness_dict = computation.liveness_analysis()
    for i in range(len(computation.instructions)):
        names = [ins.name for ins in liveness_dict[i]]
        names.sort()
        print(i, names)

    # Build Strategy Vector
    num_devices = 4
    for ins in computation.instructions:
        ins.build_intra_inst_cost(num_devices)

    for ins in computation.instructions:
        ins.build_inter_inst_cost(num_devices)

    # Build ILP Problem
    # 0. Constant
    N = len(computation.instructions)
    M = 1 << 30

    # 1. Strategy vector
    s = []
    c = []
    m = []
    P = []
    L = []
    D = [[None for i in range(N)] for j in range(N)]
    for i in range(N):
        ins = computation.instructions[i]
        c.append(ins.compute_costs)
        m.append(ins.memory_costs)
        s.append(LpVariable.dicts(f"s{i}",
            (range(len(ins.sharding_specs)),), cat="Binary"))
        P.append([operand.index for operand in ins.operands])
        L.append([ins.index for ins in liveness_dict[i]])
        ins.communication_costs  # [s_i, operand, s_j]
        for j_idx, j_val in enumerate(P[i]):
            cost = np.empty((len(s[i]), len(s[j_val])))
            for ii in range(len(s[i])):
                for jj in range(len(s[j_val])):
                    cost[ii][jj] = ins.communication_costs[ii][j_idx][jj]
            D[i][j_val] = cost

    # 2. Objective
    prob = LpProblem("myProblem", LpMinimize)
    # (a). compute cost
    obj = 0
    for i in range(N):
        for j in range(len(s[i])):
            obj += s[i][j] * c[i][j]

    # (b). memory cost
    for i in range(N):
        for j in P[i]:
            for ii in range(len(s[i])):
                for jj in range(len(s[j])):
                    obj += s[i][ii] * int(D[i][j][ii][jj]) * s[j][jj]

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

    print(prob)

    prob.solve()
    print("Status:", LpStatus[prob.status])


if __name__ == "__main__":
    test_mlp_forward()

