"""Definition of HLO Instructions"""

from collections import defaultdict
from enum import Enum, auto

import numpy as np

from common import compute_bytes


class ShardingSpec(Enum):
    ONE = 0
    SPLIT_0 = 1
    SPLIT_1 = 2
    REPLICATE = 3
    TUPLE = 4


def resharding_cost_vector(source_ins, required_spec, cluster_env):
    cost_vector = []
    for strategy in source_ins.strategies:
        cost_vector.append(cluster_env.resharding_cost(source_ins.shape,
            strategy.output_spec, required_spec))
    return cost_vector


class InstructionStrategy:
    def __init__(self, name, output_spec):
        self.name = name
        self.output_spec = output_spec


class OpCode(Enum):
    PARAMETER = auto()
    CONSTANT = auto()
    DOT = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    BROADCAST = auto()
    TUPLE = auto()

op_code_ct = defaultdict(int)


class HloInstruction:
    def __init__(self, op_code, shape, operands=[]):
        # Attributes
        self.op_code = op_code
        self.shape = shape
        self.operands = operands
        self.name = f"{str(op_code)[7:].lower()}.{op_code_ct[op_code]}"
        op_code_ct[op_code] += 1

        # Cost
        self.strategies = []
        self.compute_costs = []
        self.communication_costs = []
        self.memory_costs = []
        self.resharding_costs = []

        # The index in HloComputation
        self.index = HloComputation.cur_env.append(self)

    def build_strategy_and_cost(self, cluster_env):
        raise NotImplementedError(f"{self.op_code}")


class HloParameter(HloInstruction):
    def __init__(self, shape):
        super().__init__(OpCode.PARAMETER, shape, [])

    def build_strategy_and_cost(self, cluster_env):
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

    def __str__(self):
        return f"{self.name} {self.shape} = parameter()"


class HloConstant(HloInstruction):
    def __init__(self, value):
        super().__init__(OpCode.CONSTANT, (), [])
        self.value = value

    def build_strategy_and_cost(self, cluster_env):
        self.strategies.append(InstructionStrategy("R", ShardingSpec.REPLICATE))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape))

    def __str__(self):
        return f"{self.name} {self.shape} = constant({self.value})"


class HloBroadcast(HloInstruction):
    def __init__(self, operand, shape):
        super().__init__(OpCode.BROADCAST, shape, [operand])

    def build_strategy_and_cost(self, cluster_env):
        self.strategies.append(InstructionStrategy("S0", ShardingSpec.SPLIT_0))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
        self.resharding_costs.append([
            resharding_cost_vector(self.operands[0], ShardingSpec.SPLIT_0, cluster_env),
        ])

        self.strategies.append(InstructionStrategy("S1", ShardingSpec.SPLIT_1))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
        self.resharding_costs.append([
            resharding_cost_vector(self.operands[0], ShardingSpec.SPLIT_1, cluster_env),
        ])

        self.strategies.append(InstructionStrategy("R", ShardingSpec.REPLICATE))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape))
        self.resharding_costs.append([
            resharding_cost_vector(self.operands[0], ShardingSpec.REPLICATE, cluster_env),
        ])

    def __str__(self):
        return f"{self.name} {self.shape} = broadcast({self.operands[0].name})"


class HloBinary(HloInstruction):
    def __init__(self, op_code, lhs, rhs):
        assert lhs.shape == rhs.shape
        super().__init__(op_code, lhs.shape, [lhs, rhs])
        self.lhs = lhs
        self.rhs = rhs

    def build_strategy_and_cost(self, cluster_env):
        assert len(self.shape) == 2

        self.strategies.append(InstructionStrategy("S0", ShardingSpec.SPLIT_0))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
        self.resharding_costs.append([
            resharding_cost_vector(self.lhs, ShardingSpec.SPLIT_0, cluster_env),
            resharding_cost_vector(self.rhs, ShardingSpec.SPLIT_0, cluster_env),
        ])

        self.strategies.append(InstructionStrategy("S1", ShardingSpec.SPLIT_1))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
        self.resharding_costs.append([
            resharding_cost_vector(self.lhs, ShardingSpec.SPLIT_1, cluster_env),
            resharding_cost_vector(self.rhs, ShardingSpec.SPLIT_1, cluster_env),
        ])

        self.strategies.append(InstructionStrategy("R", ShardingSpec.REPLICATE))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape))
        self.resharding_costs.append([
            resharding_cost_vector(self.lhs, ShardingSpec.REPLICATE, cluster_env),
            resharding_cost_vector(self.rhs, ShardingSpec.REPLICATE, cluster_env),
        ])


    def __str__(self):
        fun_name = str(self.op_code)[7:].lower()
        return f"{self.name} {self.shape} = {fun_name}({self.lhs.name}, {self.rhs.name})"


class HloSubtract(HloBinary):
    def __init__(self, lhs, rhs):
        super().__init__(OpCode.SUBTRACT, lhs, rhs)


class HloMutiply(HloBinary):
    def __init__(self, lhs, rhs):
        super().__init__(OpCode.MULTIPLY, lhs, rhs)


class HloDot(HloInstruction):
    def __init__(self, lhs, rhs,
                 lhs_contracting_dims=(1,), rhs_contracting_dims=(0,)):
        # shape inference
        shape = \
            tuple(lhs.shape[i] for i in range(len(lhs.shape)) if i not in lhs_contracting_dims) +\
            tuple(rhs.shape[i] for i in range(len(rhs.shape)) if i not in rhs_contracting_dims)

        for i, j in zip(lhs_contracting_dims, rhs_contracting_dims):
            assert lhs.shape[i] == rhs.shape[j]

        super().__init__(OpCode.DOT, shape, [lhs, rhs])
        self.lhs = lhs
        self.rhs = rhs
        self.lhs_contracting_dims = lhs_contracting_dims
        self.rhs_contracting_dims = rhs_contracting_dims

    def build_strategy_and_cost(self, cluster_env):
        assert len(self.shape) == 2
        assert len(self.lhs_contracting_dims) == 1
        assert len(self.rhs_contracting_dims) == 1

        # SPLIT_0 = SPLIT_0 * REPLICATE
        self.strategies.append(InstructionStrategy("S0 = S0 x R", ShardingSpec.SPLIT_0))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
        lhs_spec = ShardingSpec.SPLIT_0 if self.lhs_contracting_dims == (1,) else ShardingSpec.SPLIT_1
        self.resharding_costs.append([
            resharding_cost_vector(self.lhs, lhs_spec, cluster_env),
            resharding_cost_vector(self.rhs, ShardingSpec.REPLICATE, cluster_env),
        ])

        # SPLIT_1 = REPLICATE * SPLIT_1
        self.strategies.append(InstructionStrategy("S1 = R x S1", ShardingSpec.SPLIT_1))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
        rhs_spec = ShardingSpec.SPLIT_1 if self.rhs_contracting_dims == (0,) else ShardingSpec.SPLIT_0
        self.resharding_costs.append([
            resharding_cost_vector(self.lhs, ShardingSpec.REPLICATE, cluster_env),
            resharding_cost_vector(self.rhs, rhs_spec, cluster_env),
        ])

        # REPLICATE = SPLIT_1 * SPLIT_0
        self.strategies.append(InstructionStrategy("R = S1 x S0", ShardingSpec.REPLICATE))
        self.compute_costs.append(0)
        self.communication_costs.append(cluster_env.all_reduce_cost(compute_bytes(self.shape)))
        self.memory_costs.append(compute_bytes(self.shape))
        lhs_spec = ShardingSpec.SPLIT_1 if self.lhs_contracting_dims == (1,) else ShardingSpec.SPLIT_0
        rhs_spec = ShardingSpec.SPLIT_0 if self.rhs_contracting_dims == (0,) else ShardingSpec.SPLIT_1
        self.resharding_costs.append([
            resharding_cost_vector(self.lhs, lhs_spec, cluster_env),
            resharding_cost_vector(self.rhs, rhs_spec, cluster_env),
        ])

        # REPLICATE = REPLICATE * REPLICATE
        self.strategies.append(InstructionStrategy("R = R x R", ShardingSpec.REPLICATE))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape))
        self.resharding_costs.append([
            resharding_cost_vector(self.lhs, ShardingSpec.REPLICATE, cluster_env),
            resharding_cost_vector(self.rhs, ShardingSpec.REPLICATE, cluster_env),
        ])

    def __str__(self):
        return f"{self.name} {self.shape} = dot({self.lhs.name}, {self.rhs.name}) "\
               f" lhs_con_dim={self.lhs_contracting_dims},"\
               f" rhs_con_dim={self.rhs_contracting_dims}"


class HloTuple(HloInstruction):
    def __init__(self, operands):
        super().__init__(OpCode.TUPLE, (), operands)

    def build_strategy_and_cost(self, cluster_env):
        self.strategies.append(InstructionStrategy("tuple", ShardingSpec.TUPLE))
        self.memory_costs.append(0)
        self.compute_costs.append(0)
        self.resharding_costs.append([np.zeros(len(operand.strategies))
            for operand in self.operands])

    def __str__(self):
        names = tuple(x.name for x in self.operands)
        return f"{self.name} {self.shape} = tuple{names}"


class HloComputation:
    cur_env = None

    def __init__(self):
        self.ct = 0
        self.instructions = []
        self.alias_list = []
        self.alias_cost_vector = []

    def append(self, instruction):
        ct = len(self.instructions)
        self.instructions.append(instruction)

        return ct

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

    def set_alias(self, alias_list):
        self.alias_list = alias_list

    def build_strategy_and_cost(self, cluster_env):
        # build strategies and costs for each instruction
        for ins in self.instructions:
            ins.build_strategy_and_cost(cluster_env)

        # build alias costs
        for (ins_a, ins_b) in self.alias_list:
            assert ins_a.shape == ins_b.shape
            cost_vector = []
            for stra_a in ins_a.strategies:
                for stra_b in ins_b.strategies:
                    #cost_vector.append(cluster_env.resharding_cost(ins_a.shape,
                    #    stra_a.output_spec, stra_b.output_spec))
                    if stra_a.output_spec == stra_b.output_spec:
                        cost_vector.append(0)
                    else:
                        cost_vector.append(1 << 30)
            self.alias_cost_vector.append(cost_vector)

    def __enter__(self):
        assert HloComputation.cur_env is None
        HloComputation.cur_env = self

    def __exit__(self, *args, **kwargs):
        HloComputation.cur_env = None

    def __str__(self):
        strs = []
        for i, ins in enumerate(self.instructions):
            strs.append(f"{i:2d}: " + str(ins))
        return "\n".join(strs)
 
