"""Definition of HLO Instructions"""

from collections import defaultdict
from enum import Enum, auto

import numpy as np

from common import compute_bytes

class ShardingSpecType(Enum):
    REPLICATED = auto()
    MAXIMAL = auto()
    OTHER = auto()
    TUPLE = auto()


class ShardingSpec:
    def __init__(self, type_, tile_assignment_dimensions, tile_assignment_devices):
        self.type = type_
        self.tile_assignment_dimensions = tile_assignment_dimensions
        self.tile_assignment_devices = tile_assignment_devices

    @staticmethod
    def split(shape, dim, cluster_env):
        tile_assignment_dimensions = tuple(
            cluster_env.num_devices if i == dim else 0 for i in range(len(shape)))
        tile_assignment_devices = tuple(range(cluster_env.num_devices))
        return ShardingSpec(ShardingSpecType.OTHER,
                            tile_assignment_dimensions, tile_assignment_devices)

    @staticmethod
    def replicated(cluster_env):
        tile_assignment_devices = tuple(range(cluster_env.num_devices))
        return ShardingSpec(ShardingSpecType.REPLICATED, None, tile_assignment_devices)

    @staticmethod
    def tuple():
        return ShardingSpec(ShardingSpecType.TUPLE, None, None)

    def __eq__(self, other):
        return (self.type == other.type and self.tile_assignment_dimensions == other.tile_assignment_dimensions
                and self.tile_assignment_devices == other.tile_assignment_devices)


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
    BROADCAST = auto()
    RESHAPE = auto()
    TRANSPOSE = auto()
    EXP = auto()
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIV = auto()
    COMPARE = auto()
    SELECT = auto()
    REDUCE = auto()
    DOT = auto()
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
    def __init__(self, shape, fix_strategy=None):
        super().__init__(OpCode.PARAMETER, shape, [])
        self.fix_strategy = fix_strategy

    def build_strategy_and_cost(self, cluster_env):
        for i in range(len(self.shape)):
            name = f"S{i}"
            self.strategies.append(InstructionStrategy(name, ShardingSpec.split(self.shape, i, cluster_env)))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)

        self.strategies.append(InstructionStrategy("R", ShardingSpec.replicated(cluster_env)))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape))

        if self.fix_strategy:
            new_strategies = []
            new_compute_costs = []
            new_communication_costs = []
            new_memory_costs = []

            # filter strategies
            for i in range(len(self.strategies)):
                if self.strategies[i].name == self.fix_strategy:
                    new_strategies.append(self.strategies[i])
                    new_compute_costs.append(self.compute_costs[i])
                    new_communication_costs.append(self.communication_costs[i])
                    new_memory_costs.append(self.memory_costs[i])

            self.strategies = new_strategies
            self.compute_costs = new_compute_costs
            self.communication_costs = new_communication_costs
            self.memory_costs = new_memory_costs

    def __str__(self):
        return f"{self.name} {self.shape} = parameter()"


class HloConstant(HloInstruction):
    def __init__(self, value):
        super().__init__(OpCode.CONSTANT, (), [])
        self.value = value

    def build_strategy_and_cost(self, cluster_env):
        self.strategies.append(InstructionStrategy("R", ShardingSpec.replicated(cluster_env)))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape))

    def __str__(self):
        return f"{self.name} {self.shape} = constant({self.value})"


class HloBroadcast(HloInstruction):
    def __init__(self, operand, shape, dimensions=()):
        for i in dimensions:
            assert shape[i] == operand.shape[dimensions.index(i)]
        super().__init__(OpCode.BROADCAST, shape, [operand])
        self.dimensions = dimensions

    def build_strategy_and_cost(self, cluster_env):
        for i in range(len(self.shape)):
            name = f"S{i}"
            self.strategies.append(InstructionStrategy(name, ShardingSpec.split(self.shape, i, cluster_env)))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)

            if i in self.dimensions:
                orignal_dim = self.dimensions.index(i)
                self.resharding_costs.append([
                    resharding_cost_vector(self.operands[0],
                        ShardingSpec.split(self.operands[0].shape, orignal_dim, cluster_env), cluster_env),
                ])
            else:
                self.resharding_costs.append([
                    resharding_cost_vector(self.operands[0], ShardingSpec.replicated(cluster_env), cluster_env),
                ])

        self.strategies.append(InstructionStrategy("R", ShardingSpec.replicated(cluster_env)))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape))
        self.resharding_costs.append([
            resharding_cost_vector(self.operands[0], ShardingSpec.replicated(cluster_env), cluster_env),
        ])

    def __str__(self):
        return f"{self.name} {self.shape} = broadcast({self.operands[0].name})"


class HloReshape(HloInstruction):
    def __init__(self, operand, new_shape):
        # todo: mark this as inplace
        assert np.prod(operand.shape) == np.prod(new_shape)
        super().__init__(OpCode.RESHAPE, new_shape, [operand])
        self.new_shape = new_shape

    def build_strategy_and_cost(self, cluster_env):
        old_shape = self.operands[0].shape
        new_shape = self.new_shape

        if len(old_shape) - 1 == len(new_shape) and np.prod(old_shape[:-2]) == new_shape[-1]:
            # [768, 12, 64] -> [768, 768]
            dim_mapping = {0: 0, 1: 1}
        elif len(old_shape) - 1 == len(new_shape) and np.prod(old_shape[:2]) == new_shape[0]:
            # [4, 512, 768] -> [2048, 768]
            dim_mapping = {0: 0, 1: 2}
        elif len(old_shape) + 1 == len(new_shape) and np.prod(new_shape[:2]) == old_shape[0]:
            # [2048, 768] -> [4, 512, 768]
            dim_mapping = {0: 0, 2: 1}
        elif (len(old_shape) + 2 == len(new_shape) and np.prod(new_shape[:2]) == old_shape[0] and 
                                                       np.prod(new_shape[2:]) == old_shape[1]):
            # [768, 768] -> [4, 512, 12, 64]
            dim_mapping = {0: 0, 2: 1}
        elif (len(old_shape) - 2 == len(new_shape) and np.prod(old_shape[:2]) == new_shape[0] and 
                                                       np.prod(old_shape[2:]) == new_shape[1]):
            # [4, 512, 12, 64] -> [768, 768]
            dim_mapping = {0: 0, 1: 2}
        else:
            print(old_shape, new_shape)
            raise NotImplementedError

        for i in range(len(new_shape)):
            name = f"S{i}"
            self.strategies.append(InstructionStrategy(name, ShardingSpec.split(self.shape, i, cluster_env)))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
            if i in dim_mapping:
                before_spec = ShardingSpec.split(self.operands[0].shape, dim_mapping[i], cluster_env)
            else:
                before_spec = ShardingSpec.replicated(cluster_env)
            self.resharding_costs.append([
                resharding_cost_vector(self.operands[0], before_spec, cluster_env),
            ])

        self.strategies.append(InstructionStrategy("R", ShardingSpec.replicated(cluster_env)))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape))
        self.resharding_costs.append([
            resharding_cost_vector(self.operands[0], ShardingSpec.replicated(cluster_env), cluster_env),
        ])

    def __str__(self):
        return f"{self.name} {self.shape} = reshape({self.operands[0].name})"


class HloTranspose(HloInstruction):
    def __init__(self, operand, dimensions):
        assert len(dimensions) == len(operand.shape)
        new_shape = [operand.shape[i] for i in dimensions]
        super().__init__(OpCode.TRANSPOSE, new_shape, [operand])
        self.dimensions = dimensions

    def build_strategy_and_cost(self, cluster_env):
        for i in range(len(self.shape)):
            name = f"S{i}"
            self.strategies.append(InstructionStrategy(name, ShardingSpec.split(self.shape, i, cluster_env)))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)

            orignal_dim = self.dimensions[i]
            self.resharding_costs.append([
                resharding_cost_vector(self.operands[0],
                    ShardingSpec.split(self.operands[0].shape, orignal_dim, cluster_env), cluster_env),
            ])

        self.strategies.append(InstructionStrategy("R", ShardingSpec.replicated(cluster_env)))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape))
        self.resharding_costs.append([
            resharding_cost_vector(self.operands[0], ShardingSpec.replicated(cluster_env), cluster_env),
        ])

    def __str__(self):
        return f"{self.name} {self.shape} = transpose({self.operands[0].name}) " +\
               f"dimensions={self.dimensions}"


class HloElementwise(HloInstruction):
    def __init__(self, op_code, operands):
        for i in range(0, len(operands)):
            assert operands[0].shape == operands[i].shape
        super().__init__(op_code, operands[0].shape, operands)

    def build_strategy_and_cost(self, cluster_env):
        for i in range(len(self.shape)):
            name = f"S{i}"
            self.strategies.append(InstructionStrategy(name, ShardingSpec.split(self.shape, i, cluster_env)))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
            self.resharding_costs.append([
                resharding_cost_vector(self.operands[j],
                                       ShardingSpec.split(self.operands[j].shape, i, cluster_env), cluster_env)
                for j in range(len(self.operands))
            ])

        self.strategies.append(InstructionStrategy("R", ShardingSpec.replicated(cluster_env)))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape))
        self.resharding_costs.append([
            resharding_cost_vector(self.operands[j], ShardingSpec.replicated(cluster_env), cluster_env)
            for j in range(len(self.operands))
        ])

    def __str__(self):
        fun_name = str(self.op_code)[7:].lower()
        args = ", ".join(f"{self.operands[i].name}" for i in range(len(self.operands)))
        return f"{self.name} {self.shape} = {fun_name}({args})"


class HloExp(HloElementwise):
    def __init__(self, operand):
        super().__init__(OpCode.EXP, [operand])


class HloAdd(HloElementwise):
    def __init__(self, lhs, rhs):
        super().__init__(OpCode.ADD, [lhs, rhs])


class HloSubtract(HloElementwise):
    def __init__(self, lhs, rhs):
        super().__init__(OpCode.SUBTRACT, [lhs, rhs])


class HloMutiply(HloElementwise):
    def __init__(self, lhs, rhs):
        super().__init__(OpCode.MULTIPLY, [lhs, rhs])


class HloDiv(HloElementwise):
    def __init__(self, lhs, rhs):
        super().__init__(OpCode.DIV, [lhs, rhs])


class HloCompare(HloElementwise):
    def __init__(self, lhs, rhs):
        super().__init__(OpCode.COMPARE, [lhs, rhs])


class HloSelect(HloElementwise):
    def __init__(self, pred, true_value, false_value):
        super().__init__(OpCode.SELECT, [pred, true_value, false_value])


class HloReduce(HloInstruction):
    def __init__(self, operand, dimensions):
        new_shape = [operand.shape[i] for i in range(len(operand.shape)) if i not in dimensions]
        super().__init__(OpCode.REDUCE, new_shape, [operand])
        self.dimensions = dimensions

    def build_strategy_and_cost(self, cluster_env):
        dim_mapping = {}
        ct = 0
        for i in range(len(self.operands[0].shape)):
            if i in self.dimensions:
                continue
            dim_mapping[ct] = i
            ct += 1

        for i in range(len(self.shape)):
            name = f"S{i}"
            self.strategies.append(InstructionStrategy(name, ShardingSpec.split(self.shape, i, cluster_env)))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)

            orignal_dim = dim_mapping[i]
            self.resharding_costs.append([
                resharding_cost_vector(self.operands[0],
                    ShardingSpec.split(self.operands[0].shape, orignal_dim, cluster_env), cluster_env),
            ])

        self.strategies.append(InstructionStrategy("R", ShardingSpec.replicated(cluster_env)))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape))
        self.resharding_costs.append([
            resharding_cost_vector(self.operands[0], ShardingSpec.replicated(cluster_env), cluster_env),
        ])

    def __str__(self):
        return f"{self.name} {self.shape} = transpose({self.operands[0].name}) " +\
               f"dimensions={self.dimensions}"


class HloDot(HloInstruction):
    def __init__(self, lhs, rhs,
                 lhs_batch_dims=(), lhs_contracting_dims=(1,),
                 rhs_batch_dims=(), rhs_contracting_dims=(0,)):
        # shape inference
        lhs_space_shape = \
            tuple(lhs.shape[i] for i in range(len(lhs.shape))
                  if i not in lhs_contracting_dims and i not in lhs_batch_dims)
        rhs_space_shape = \
            tuple(rhs.shape[i] for i in range(len(rhs.shape))
                  if i not in rhs_contracting_dims and i not in rhs_batch_dims)
        lhs_batch_shape = tuple(lhs.shape[i] for i in lhs_batch_dims)

        shape = lhs_batch_shape + lhs_space_shape + rhs_space_shape

        for i, j in zip(lhs_contracting_dims, rhs_contracting_dims):
            assert lhs.shape[i] == rhs.shape[j]
        for i, j in zip(lhs_batch_dims, rhs_batch_dims):
            assert lhs.shape[i] == rhs.shape[j]

        super().__init__(OpCode.DOT, shape, [lhs, rhs])
        self.lhs = lhs
        self.lhs_batch_dims = lhs_batch_dims
        self.lhs_contracting_dims = lhs_contracting_dims
        self.lhs_space_dims = tuple(set(range(len(lhs.shape))) - set(self.lhs_batch_dims) - set(self.lhs_contracting_dims))
        assert len(self.lhs_contracting_dims) == 1
        assert len(self.lhs_space_dims) == 1
        self.rhs = rhs
        self.rhs_batch_dims = rhs_batch_dims
        self.rhs_contracting_dims = rhs_contracting_dims
        self.rhs_space_dims = tuple(set(range(len(rhs.shape))) - set(self.rhs_batch_dims) - set(self.rhs_contracting_dims))
        assert len(self.rhs_contracting_dims) == 1
        assert len(self.rhs_space_dims) == 1

    def build_strategy_and_cost(self, cluster_env):
        lhs = self.lhs
        lhs_batch_dims = self.lhs_batch_dims
        lhs_space_dim = self.lhs_space_dims[0]
        lhs_con_dim = self.lhs_contracting_dims[0]

        rhs = self.rhs
        rhs_batch_dims = self.rhs_batch_dims
        rhs_space_dim = self.rhs_space_dims[0]
        rhs_con_dim = self.rhs_contracting_dims[0]

        space_base_dim = len(self.lhs_batch_dims)

        # split the space dim of lhs
        self.strategies.append(InstructionStrategy("Sl = Sl x R", ShardingSpec.split(self.shape, space_base_dim, cluster_env)))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
        self.resharding_costs.append([
            resharding_cost_vector(lhs, ShardingSpec.split(lhs.shape, lhs_space_dim, cluster_env), cluster_env),
            resharding_cost_vector(rhs, ShardingSpec.replicated(cluster_env), cluster_env),
        ])

        # split the space dim of rhs
        self.strategies.append(InstructionStrategy("Sr = R x Sr", ShardingSpec.split(self.shape, space_base_dim+1, cluster_env)))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
        self.resharding_costs.append([
            resharding_cost_vector(lhs, ShardingSpec.replicated(cluster_env), cluster_env),
            resharding_cost_vector(rhs, ShardingSpec.split(rhs.shape, rhs_space_dim, cluster_env), cluster_env)
        ])

        # split the contracting dim
        self.strategies.append(InstructionStrategy("R = Sk x Sk", ShardingSpec.replicated(cluster_env)))
        self.compute_costs.append(0)
        self.communication_costs.append(cluster_env.all_reduce_cost(compute_bytes(self.shape)))
        self.memory_costs.append(compute_bytes(self.shape))
        self.resharding_costs.append([
            resharding_cost_vector(lhs, ShardingSpec.split(lhs.shape, lhs_con_dim, cluster_env), cluster_env),
            resharding_cost_vector(rhs, ShardingSpec.split(rhs.shape, rhs_con_dim, cluster_env), cluster_env),
        ])

        # split the batch dim
        for i in range(len(self.lhs_batch_dims)):
            name = f"Sb = Sb x Sb {i}"
            self.strategies.append(InstructionStrategy(name, ShardingSpec.split(self.shape, i, cluster_env)))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / cluster_env.num_devices)
            self.resharding_costs.append([
                resharding_cost_vector(lhs, ShardingSpec.split(lhs.shape, lhs_batch_dims[i], cluster_env), cluster_env),
                resharding_cost_vector(rhs, ShardingSpec.split(rhs.shape, rhs_batch_dims[i], cluster_env), cluster_env),
            ])

        # replicated
        self.strategies.append(InstructionStrategy("R = R x R", ShardingSpec.replicated(cluster_env)))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape))
        self.resharding_costs.append([
            resharding_cost_vector(self.lhs, ShardingSpec.replicated(cluster_env), cluster_env),
            resharding_cost_vector(self.rhs, ShardingSpec.replicated(cluster_env), cluster_env),
        ])

    def __str__(self):
        return f"{self.name} {self.shape} = dot({self.lhs.name}, {self.rhs.name}) "\
               f" lhs_con_dim={self.lhs_contracting_dims},"\
               f" rhs_con_dim={self.rhs_contracting_dims}"


class HloTuple(HloInstruction):
    def __init__(self, operands):
        super().__init__(OpCode.TUPLE, (), operands)

    def build_strategy_and_cost(self, cluster_env):
        self.strategies.append(InstructionStrategy("tuple", ShardingSpec.tuple()))
        self.memory_costs.append(0)
        self.compute_costs.append(0)
        self.communication_costs.append(0)
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
            # todo: some cases can be reduced to an equality constraint
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
 
