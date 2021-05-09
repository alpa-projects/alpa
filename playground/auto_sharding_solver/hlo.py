"""Definition of HLO Instructions"""

from collections import defaultdict
from enum import Enum, auto

import numpy as np

from common import compute_bytes, append_flatten_elements, transpose_flatten, reshape_flatten


class ShardingSpecType(Enum):
    REPLICATED = auto()
    MAXIMAL = auto()
    OTHER = auto()
    TUPLE = auto()
    PARTIAL_REDUCTION = auto()


INF_COST = 1e10  # infinity cost


class ShardingSpec:
    def __init__(self, type_, tile_assignment_dimensions, tile_assignment_devices,
                 replicate_on_last_tile_dim, partial_reduce_replication):
        self.type = type_
        self.tile_assignment_dimensions = tuple(tile_assignment_dimensions)
        self.tile_assignment_devices = tuple(tile_assignment_devices)
        self.replicate_on_last_tile_dim = replicate_on_last_tile_dim
        self.partial_reduce_replication = partial_reduce_replication

    def num_tile_devices(self):
        if self.type == ShardingSpecType.REPLICATED:
            return 1

        assert self.type == ShardingSpecType.OTHER
        ret = np.prod(self.tile_assignment_dimensions)
        if self.replicate_on_last_tile_dim:
            ret /= self.tile_assignment_dimensions[-1]
        return ret

    def transpose(self, dimensions):
        if self.type == ShardingSpecType.REPLICATED:
            return self

        assert self.type == ShardingSpecType.OTHER

        spec_trans_dims = list(dimensions)
        if self.replicate_on_last_tile_dim:
            spec_trans_dims.append(len(dimensions))

        tile_assignment_dimensions = [self.tile_assignment_dimensions[i]
            for i in spec_trans_dims]
        tile_assignment_devices = transpose_flatten(self.tile_assignment_devices,
            self.tile_assignment_dimensions, spec_trans_dims)

        ret = ShardingSpec(self.type,
                           tile_assignment_dimensions,
                           tile_assignment_devices,
                           self.replicate_on_last_tile_dim,
                           self.partial_reduce_replication)
        return ret

    def broadcast(self, new_shape, dimensions):
        if self.type == ShardingSpecType.REPLICATED:
            return self

        assert self.type == ShardingSpecType.OTHER

        tile_assignment_dimensions = []
        for i in range(len(new_shape)):
            if i in dimensions:
                tile_assignment_dimensions.append(
                    self.tile_assignment_dimensions[dimensions.index(i)])
            else:
                tile_assignment_dimensions.append(1)

        if self.replicate_on_last_tile_dim:
            tile_assignment_dimensions.append(self.tile_assignment_dimensions[-1])

        output_spec = ShardingSpec(self.type,
                                   tile_assignment_dimensions,
                                   self.tile_assignment_devices,
                                   self.replicate_on_last_tile_dim,
                                   self.partial_reduce_replication)
        return output_spec

    def reshape(self, old_shape, new_shape):
        if self.type == ShardingSpecType.REPLICATED:
            return self

        assert self.type == ShardingSpecType.OTHER

        # Construct a map that maps an old dimension to its corresponding new dimension
        dim_mapping = {}
        new_pt = -1
        old_pt = -1
        old_prod = 1
        new_prod = 1
        while True:
            move_new = False
            move_old = False

            if new_prod == old_prod:
                dim_mapping[old_pt + 1] = new_pt + 1
                move_new = move_old = True
            elif new_prod < old_prod:
                move_new = True
            else:
                move_old = True

            if move_new:
                new_pt += 1
                if new_pt < len(new_shape):
                    new_prod *= new_shape[new_pt]
                else:
                    break
            if move_old:
                old_pt += 1
                if old_pt < len(old_shape):
                    old_prod *= old_shape[old_pt]
                else:
                    break

        tile_assignment_dimensions = []
        cur_prod = 1
        state = 1  # 0: start  1: middle
        i = 0

        failed = False
        while i < len(old_shape) and not failed:
            if state == 0:
                assert i in dim_mapping
                while len(tile_assignment_dimensions) < dim_mapping[i]:
                    tile_assignment_dimensions.append(1)
                tile_assignment_dimensions.append(
                    self.tile_assignment_dimensions[i])
                state = 1
                i += 1
            elif state == 1:
                if i in dim_mapping:
                    state = 0
                else:
                    if self.tile_assignment_dimensions[i] == 1:
                        i += 1
                    else:
                        failed = True

        if failed:
            return None

        while len(tile_assignment_dimensions) < len(new_shape):
            tile_assignment_dimensions.append(1)

        if self.replicate_on_last_tile_dim:
            tile_assignment_dimensions.append(self.tile_assignment_dimensions[-1])
        output_spec = ShardingSpec(self.type,
                                   tile_assignment_dimensions,
                                   self.tile_assignment_devices,
                                   self.replicate_on_last_tile_dim,
                                   self.partial_reduce_replication)
        return output_spec

    @staticmethod
    def tile_internal(shape, tensor_dims, mesh_dims, cluster_env, partial_reduce_replication):
        assert len(tensor_dims) == len(mesh_dims)

        tile_assignment_dimensions = [1] * len(shape)

        # Split on certain mesh dimensions
        split_prod = 1
        for tensor_dim, mesh_dim in zip(tensor_dims, mesh_dims):
            tile_assignment_dimensions[tensor_dim] = cluster_env.device_mesh.shape[mesh_dim]
            split_prod *= cluster_env.device_mesh.shape[mesh_dim]

        if split_prod == 1:
            return ShardingSpec.replicated(cluster_env)

        # Replicate on reminding mesh dimensions
        if split_prod < cluster_env.num_devices:
            tile_assignment_dimensions.append(cluster_env.num_devices // split_prod)
            replicate_on_last_tile_dim = True
        else:
            replicate_on_last_tile_dim = False

        # Map device ids from device_mesh to tile_assignment_devices
        tile_assignment_devices = []
        tmp_indices = [None] * len(cluster_env.device_mesh.shape)
        def generate_tile_assignment_devices(tensor_dim, mesh_indices):
            if tensor_dim == len(shape) - 1:
                append_flatten_elements(tile_assignment_devices, cluster_env.device_mesh,
                                        mesh_indices, -1, tmp_indices)
            else:
                next_tensor_dim = tensor_dim + 1
                next_mesh_dim = -1

                if next_tensor_dim in tensor_dims:
                    next_mesh_dim = mesh_dims[tensor_dims.index(next_tensor_dim)]

                for i in range(tile_assignment_dimensions[next_tensor_dim]):
                    if next_mesh_dim != -1:
                        mesh_indices[next_mesh_dim] = i
                    generate_tile_assignment_devices(next_tensor_dim, mesh_indices)

        generate_tile_assignment_devices(-1, [-1] * len(cluster_env.device_mesh.shape))

        return ShardingSpec(ShardingSpecType.OTHER,
                            tile_assignment_dimensions, tile_assignment_devices,
                            replicate_on_last_tile_dim,
                            False)

    @staticmethod
    def tile(shape, tensor_dims, mesh_dims, cluster_env):
        return ShardingSpec.tile_internal(shape, tensor_dims, mesh_dims, cluster_env, False)

    @staticmethod
    def tile_partial_reduce(shape, tensor_dims, mesh_dims, cluster_env):
        return ShardingSpec.tile_internal(shape, tensor_dims, mesh_dims, cluster_env, True)

    @staticmethod
    def replicated(cluster_env):
        tile_assignment_devices = range(cluster_env.num_devices)
        return ShardingSpec(ShardingSpecType.REPLICATED, (), tile_assignment_devices,
                            False, False)

    @staticmethod
    def split(shape, dim, cluster_env):
        tile_assignment_dimensions = [1] * len(shape)
        tile_assignment_dimensions[dim] = cluster_env.num_devices
        tile_assignment_devices = range(cluster_env.num_devices)
        return ShardingSpec(ShardingSpecType.OTHER,
                            tile_assignment_dimensions, tile_assignment_devices,
                            False, False)

    @staticmethod
    def tuple():
        return ShardingSpec(ShardingSpecType.TUPLE, (), (), False, False)

    def __str__(self):
        return f"{self.tile_assignment_dimensions}"\
               f"{list(self.tile_assignment_devices)}"

    def __eq__(self, other):
        return (self.type == other.type and
                self.tile_assignment_dimensions == other.tile_assignment_dimensions and
                self.tile_assignment_devices == other.tile_assignment_devices and
                self.replicate_on_last_tile_dim == other.replicate_on_last_tile_dim and
                self.partial_reduce_replication == other.partial_reduce_replication)


def resharding_cost_vector(cluster_env, source_ins, required_spec):
    cost_vector = []
    for strategy in source_ins.strategies:
        cost_vector.append(cluster_env.resharding_cost(source_ins.shape,
            strategy.output_spec, required_spec))
    return cost_vector


def follow_ins_cost_vector(source_ins, index):
    ret = [INF_COST] * len(source_ins.strategies)
    ret[index] = 0
    return ret


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
    IDENTITY = auto()
    EXP = auto()
    FORCE_REPLICATED = auto()
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
        self.follow_ins = None
        self.depth = None

        # The index in HloComputation
        self.index = HloComputation.cur_env.append(self)
        self.batch_dim = None

    def build_strategy_and_cost(self, cluster_env, solver_option):
        raise NotImplementedError(f"{self.op_code}")

    def propagate_batch_dim(self, operand):
        raise NotImplementedError(f"{self.op_code}")


class HloParameter(HloInstruction):
    def __init__(self, shape, fix_strategy=None):
        super().__init__(OpCode.PARAMETER, shape, [])
        self.fix_strategy = fix_strategy

    def build_strategy_and_cost(self, cluster_env, solver_option):
        for i in range(len(self.shape)):
            for j in range(len(cluster_env.device_mesh.shape)):
                if (cluster_env.device_mesh.shape[j] == 1 or
                    self.shape[i] < cluster_env.device_mesh.shape[j]):
                    continue

                name = f"S{i} @ {j}"
                output_spec = ShardingSpec.tile(self.shape, [i], [j], cluster_env)
                self.strategies.append(InstructionStrategy(name, output_spec))
                self.compute_costs.append(0)
                self.communication_costs.append(0)
                self.memory_costs.append(compute_bytes(self.shape) / output_spec.num_tile_devices())

        self.strategies.append(InstructionStrategy("R", ShardingSpec.replicated(cluster_env)))
        self.compute_costs.append(2)
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

    def build_strategy_and_cost(self, cluster_env, solver_option):
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

    def build_strategy_and_cost(self, cluster_env, solver_option):
        follow = self.operands[0]
        self.follow_ins = follow

        for sid in range(len(follow.strategies)):
            output_spec = follow.strategies[sid].output_spec.broadcast(
                    self.shape, self.dimensions)
            name = f"{output_spec.tile_assignment_dimensions}"
            self.strategies.append(InstructionStrategy(name, output_spec))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / output_spec.num_tile_devices())
            self.resharding_costs.append([follow_ins_cost_vector(follow, sid)])

    def __str__(self):
        return f"{self.name} {self.shape} = broadcast({self.operands[0].name})"


class HloReshape(HloInstruction):
    def __init__(self, operand, new_shape):
        # todo: mark this as inplace
        assert np.prod(operand.shape) == np.prod(new_shape)
        super().__init__(OpCode.RESHAPE, new_shape, [operand])
        self.new_shape = new_shape

    def build_strategy_and_cost(self, cluster_env, solver_option):
        follow = self.operands[0]
        self.follow_ins = follow
        old_shape = self.operands[0].shape
        new_shape = self.new_shape

        for sid in range(len(follow.strategies)):
            output_spec = follow.strategies[sid].output_spec.reshape(
                    follow.shape, self.shape)
            if output_spec is None:
                continue

            name = f"{output_spec.tile_assignment_dimensions}"
            self.strategies.append(InstructionStrategy(name, output_spec))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / output_spec.num_tile_devices())
            self.resharding_costs.append([follow_ins_cost_vector(follow, sid)])

    def __str__(self):
        return f"{self.name} {self.shape} = reshape({self.operands[0].name})"


class HloTranspose(HloInstruction):
    def __init__(self, operand, dimensions):
        assert len(dimensions) == len(operand.shape)
        new_shape = tuple(operand.shape[i] for i in dimensions)
        super().__init__(OpCode.TRANSPOSE, new_shape, [operand])
        self.dimensions = dimensions

    def build_strategy_and_cost(self, cluster_env, solver_option):
        follow = self.operands[0]
        self.follow_ins = follow

        for sid in range(len(follow.strategies)):
            output_spec = follow.strategies[sid].output_spec.transpose(self.dimensions)
            name = f"{output_spec.tile_assignment_dimensions}"
            self.strategies.append(InstructionStrategy(name, output_spec))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / output_spec.num_tile_devices())
            self.resharding_costs.append([follow_ins_cost_vector(follow, sid)])

    def __str__(self):
        return f"{self.name} {self.shape} = transpose({self.operands[0].name}) " +\
               f"dimensions={self.dimensions}"


class HloElementwise(HloInstruction):
    def __init__(self, op_code, operands):
        for i in range(0, len(operands)):
            assert operands[0].shape == operands[i].shape
        super().__init__(op_code, operands[0].shape, operands)

    def build_strategy_and_cost(self, cluster_env, solver_option):
        depths = [operand.depth for operand in self.operands]
        follow_idx = np.argmax(depths)

        follow = self.operands[follow_idx]
        self.follow_ins = follow

        for sid in range(len(follow.strategies)):
            output_spec = follow.strategies[sid].output_spec

            name = f"{output_spec.tile_assignment_dimensions}"
            self.strategies.append(InstructionStrategy(name, output_spec))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / output_spec.num_tile_devices())

            resharding_costs = []
            for k in range(len(self.operands)):
                if k == follow_idx:
                    resharding_costs.append(
                        follow_ins_cost_vector(follow, sid))
                else:
                    resharding_costs.append(
                    resharding_cost_vector(cluster_env, self.operands[k], output_spec))
            self.resharding_costs.append(resharding_costs)

    def propagate_batch_dim(self, ins):
        self.batch_dim = ins.batch_dim
        return True

    def __str__(self):
        fun_name = str(self.op_code)[7:].lower()
        args = ", ".join(f"{self.operands[i].name}" for i in range(len(self.operands)))
        return f"{self.name} {self.shape} = {fun_name}({args})"


class HloIdentity(HloElementwise):
    def __init__(self, operand):
        super().__init__(OpCode.IDENTITY, [operand])


class HloExp(HloElementwise):
    def __init__(self, operand):
        super().__init__(OpCode.EXP, [operand])


class HloForceReplicated(HloElementwise):
    def __init__(self, operand):
        super().__init__(OpCode.FORCE_REPLICATED, [operand])

    def build_strategy_and_cost(self, cluster_env, solver_option):
        self.strategies.append(InstructionStrategy("R",
            ShardingSpec.replicated(cluster_env)))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(0)
        self.resharding_costs.append([
            resharding_cost_vector(cluster_env, self.operands[0],
                ShardingSpec.replicated(cluster_env))
        ])


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
        new_shape = tuple(operand.shape[i] for i in range(len(operand.shape)) if i not in dimensions)
        super().__init__(OpCode.REDUCE, new_shape, [operand])
        self.dimensions = dimensions

    def build_strategy_and_cost(self, cluster_env, solver_option):
        operand = self.operands[0]
        self.follow_ins = operand

        # Map old dims to new dim
        old_dim_to_new_dim = []
        pt = 0
        for old_dim in range(len(operand.shape)):
            if old_dim in self.dimensions:
                old_dim_to_new_dim.append(-1)
            else:
                old_dim_to_new_dim.append(pt)
                pt += 1
        assert pt == len(self.shape)

        # Create follow strategies
        for sid in range(len(operand.strategies)):
            tensor_dim_to_mesh = cluster_env.get_tensor_dim_to_mesh_dim(
                operand.shape, operand.strategies[sid].output_spec)

            tile_tensor_dims = []
            tile_mesh_dims = []
            all_reduce_dims = []

            for tensor_dim in range(len(operand.shape)):
                mesh_dim = tensor_dim_to_mesh[tensor_dim]
                if tensor_dim in self.dimensions:
                    if mesh_dim == -1:  # reduce on a replicated dim
                        continue
                    else:               # reduce on a split dim
                        all_reduce_dims.append(mesh_dim)
                else:
                    if mesh_dim == -1: # follow replicated dim
                        pass
                    else:              # follow split dim
                        tile_tensor_dims.append(old_dim_to_new_dim[tensor_dim])
                        tile_mesh_dims.append(mesh_dim)

            output_spec = ShardingSpec.tile(self.shape, tile_tensor_dims, tile_mesh_dims, cluster_env)

            mem_cost = compute_bytes(self.shape) / output_spec.num_tile_devices()
            comm_cost = 0
            for mesh_dim in all_reduce_dims:
                comm_cost += cluster_env.all_reduce_cost(mem_cost, mesh_dim)

            reduce_dims_str = "".join([str(x) for x in all_reduce_dims])
            if reduce_dims_str:
                name = f"follow (allreduce @ {reduce_dims_str})"
            else:
                name = f"{output_spec.tile_assignment_dimensions}"

            self.strategies.append(InstructionStrategy(name, output_spec))
            self.compute_costs.append(0)
            self.communication_costs.append(comm_cost)
            self.memory_costs.append(mem_cost)
            self.resharding_costs.append([follow_ins_cost_vector(operand, sid)])

    def __str__(self):
        return f"{self.name} {self.shape} = reduce({self.operands[0].name}) " +\
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

    def build_strategy_and_cost(self, cluster_env, solver_option):
        lhs = self.lhs
        lhs_batch_dims = self.lhs_batch_dims
        lhs_space_dim = self.lhs_space_dims[0]
        lhs_con_dim = self.lhs_contracting_dims[0]

        rhs = self.rhs
        rhs_batch_dims = self.rhs_batch_dims
        rhs_space_dim = self.rhs_space_dims[0]
        rhs_con_dim = self.rhs_contracting_dims[0]

        space_base_dim = len(self.lhs_batch_dims)

        assert len(cluster_env.device_mesh.shape) == 2

        # Split lhs space dim + rhs space dim
        # @ {0, 1}
        output_spec =\
            ShardingSpec.tile(self.shape, [space_base_dim, space_base_dim + 1], [0, 1], cluster_env)
        self.strategies.append(InstructionStrategy("SS = SR x RS @ {0,1}", output_spec))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape) / output_spec.num_tile_devices())
        self.resharding_costs.append([
            resharding_cost_vector(cluster_env, lhs,
                ShardingSpec.tile(lhs.shape, [lhs_space_dim], [0], cluster_env)),
            resharding_cost_vector(cluster_env, rhs,
                ShardingSpec.tile(rhs.shape, [rhs_space_dim], [1], cluster_env))
        ])

        # @ {1, 0}
        output_spec =\
            ShardingSpec.tile(self.shape, [space_base_dim, space_base_dim + 1], [1, 0], cluster_env)
        self.strategies.append(InstructionStrategy("SS = SR x RS @ {1,0}", output_spec))
        self.compute_costs.append(0)
        self.communication_costs.append(0)
        self.memory_costs.append(compute_bytes(self.shape) / output_spec.num_tile_devices())
        self.resharding_costs.append([
            resharding_cost_vector(cluster_env, lhs,
                ShardingSpec.tile(lhs.shape, [lhs_space_dim], [1], cluster_env)),
            resharding_cost_vector(cluster_env, rhs,
                ShardingSpec.tile(rhs.shape, [rhs_space_dim], [0], cluster_env))
        ])

        # Split lhs space dim + contracting dim
        # @ {0, 1}
        if cluster_env.device_mesh.shape[1] > 1:
            output_spec = ShardingSpec.tile(self.shape, [space_base_dim], [0], cluster_env)
            memory_cost = compute_bytes(self.shape) / output_spec.num_tile_devices()
            self.strategies.append(
                InstructionStrategy("SR = SS x SR @ {0,1} (allreduce @ 1)", output_spec))
            self.compute_costs.append(0)
            self.communication_costs.append(cluster_env.all_reduce_cost(memory_cost, 1))
            self.memory_costs.append(memory_cost)
            self.resharding_costs.append([
                resharding_cost_vector(cluster_env, lhs,
                    ShardingSpec.tile(lhs.shape, [lhs_space_dim, lhs_con_dim], [0, 1], cluster_env)),
                resharding_cost_vector(cluster_env, rhs,
                    ShardingSpec.tile(rhs.shape, [rhs_con_dim], [1], cluster_env))
            ])

        # @ {1, 0}
        if cluster_env.device_mesh.shape[0] > 1:
            output_spec = ShardingSpec.tile(self.shape, [space_base_dim], [1], cluster_env)
            memory_cost = compute_bytes(self.shape) / output_spec.num_tile_devices()
            self.strategies.append(
                InstructionStrategy("SR = SS x SR @ {1,0} (allreduce @ 0)", output_spec))
            self.compute_costs.append(0)
            self.communication_costs.append(cluster_env.all_reduce_cost(memory_cost, 0))
            self.memory_costs.append(memory_cost)
            self.resharding_costs.append([
                resharding_cost_vector(cluster_env, lhs,
                    ShardingSpec.tile(lhs.shape, [lhs_space_dim, lhs_con_dim], [1, 0], cluster_env)),
                resharding_cost_vector(cluster_env, rhs,
                    ShardingSpec.tile(rhs.shape, [rhs_con_dim], [0], cluster_env))
            ])

        # Split rhs space dim + contracting dim
        # @ {0, 1}
        if cluster_env.device_mesh.shape[0] > 1 and cluster_env.device_mesh.shape[1] > 1:
            output_spec = ShardingSpec.tile(self.shape, [space_base_dim+1], [1], cluster_env)
            memory_cost = compute_bytes(self.shape) / output_spec.num_tile_devices()
            self.strategies.append(
                InstructionStrategy("RS = RS x SS @ {0,1} (allreduce @ 0)", output_spec))
            self.compute_costs.append(0)
            self.communication_costs.append(cluster_env.all_reduce_cost(memory_cost, 0))
            self.memory_costs.append(memory_cost)
            self.resharding_costs.append([
                resharding_cost_vector(cluster_env, lhs,
                    ShardingSpec.tile(lhs.shape, [lhs_con_dim], [0], cluster_env)),
                resharding_cost_vector(cluster_env, rhs,
                    ShardingSpec.tile(rhs.shape, [rhs_con_dim, rhs_space_dim], [0, 1], cluster_env))
            ])

        # @ {1, 0}
        if cluster_env.device_mesh.shape[0] > 1 and cluster_env.device_mesh.shape[1] > 1:
            output_spec = ShardingSpec.tile(self.shape, [space_base_dim+1], [0], cluster_env)
            memory_cost = compute_bytes(self.shape) / output_spec.num_tile_devices()
            self.strategies.append(
                InstructionStrategy("RS = RS x SS @ {1,0} (allreduce @ 1)", output_spec))
            self.compute_costs.append(0)
            self.communication_costs.append(cluster_env.all_reduce_cost(memory_cost, 1))
            self.memory_costs.append(memory_cost)
            self.resharding_costs.append([
                resharding_cost_vector(cluster_env, lhs,
                    ShardingSpec.tile(lhs.shape, [lhs_con_dim], [1], cluster_env)),
                resharding_cost_vector(cluster_env, rhs,
                    ShardingSpec.tile(rhs.shape, [rhs_con_dim, rhs_space_dim], [1, 0], cluster_env))
            ])

        # Split one batch dim
        for i in range(len(self.lhs_batch_dims)):
            for j in range(len(cluster_env.device_mesh.shape)):
                if (cluster_env.device_mesh.shape[j] == 1 or
                    self.shape[i] < cluster_env.device_mesh.shape[j]):
                    continue

                output_spec = ShardingSpec.tile(self.shape, [i], [j], cluster_env)
                self.strategies.append(InstructionStrategy(f"Sb_{i} = Sb x Sb @ {j}", output_spec))
                self.compute_costs.append(0)
                self.communication_costs.append(0)
                self.memory_costs.append(compute_bytes(self.shape) / output_spec.num_tile_devices())
                self.resharding_costs.append([
                    resharding_cost_vector(cluster_env, lhs,
                        ShardingSpec.tile(lhs.shape, [lhs_batch_dims[i]], [j], cluster_env)),
                    resharding_cost_vector(cluster_env, rhs,
                        ShardingSpec.tile(rhs.shape, [rhs_batch_dims[i]], [j], cluster_env))
                ])

        # Split two batch dims
        if len(self.lhs_batch_dims) == 2 and cluster_env.device_mesh.shape[0] > 1\
                and cluster_env.device_mesh.shape[1] > 1:

            self.strategies = []
            self.compute_costs = []
            self.communication_costs = []
            self.memory_costs = []
            self.resharding_costs = []

            # Split two batch dims
            output_spec = ShardingSpec.tile(self.shape, [0, 1], [0, 1], cluster_env)
            self.strategies.append(InstructionStrategy("Sb = Sb x Sb @ {0,1}", output_spec))
            self.compute_costs.append(0)
            self.communication_costs.append(0)
            self.memory_costs.append(compute_bytes(self.shape) / output_spec.num_tile_devices())
            self.resharding_costs.append([
                resharding_cost_vector(cluster_env, lhs,
                    ShardingSpec.tile(lhs.shape, [lhs_batch_dims[0], lhs_batch_dims[1]], [0, 1], cluster_env)),
                resharding_cost_vector(cluster_env, rhs,
                    ShardingSpec.tile(rhs.shape, [rhs_batch_dims[0], rhs_batch_dims[1]], [0, 1], cluster_env))
            ])

        # If force batch dim to a mesh dim, filter out invalid strategies
        if solver_option.force_batch_dim_to_mesh_dim is not None and self.batch_dim is not None:
            filter_indices = []
            for i in range(len(self.strategies)):
                tensor_dim_to_mesh_dim = cluster_env.get_tensor_dim_to_mesh_dim(
                    self.shape, self.strategies[i].output_spec)
                if tensor_dim_to_mesh_dim[self.batch_dim] == solver_option.force_batch_dim_to_mesh_dim:
                    filter_indices.append(i)

            self.strategies = [self.strategies[i] for i in filter_indices]
            self.compute_costs = [self.compute_costs[i] for i in filter_indices]
            self.communication_costs = [self.communication_costs[i] for i in filter_indices]
            self.memory_costs = [self.memory_costs[i] for i in filter_indices]
            self.resharding_costs = [self.resharding_costs[i] for i in filter_indices]

    def propagate_batch_dim(self, operand):
        index = self.operands.index(operand)

        if index == 0:
            for i in range(len(self.lhs_batch_dims)):
                if operand.batch_dim == self.lhs_batch_dims[i]:
                    self.batch_dim = i
                    return True
            if operand.batch_dim == self.lhs_space_dims[0]:
                self.batch_dim = len(self.lhs_batch_dims)
                return True
            if operand.batch_dim in self.lhs_contracting_dims:
                return False
        else:
            for i in range(len(self.rhs_batch_dims)):
                if operand.batch_dim == self.rhs_batch_dims[i]:
                    self.batch_dim = i
                    return True
            if operand.batch_dim == self.rhs_space_dims[0]:
                self.batch_dim = len(self.rhs_batch_dims)
                return True
            if operand.batch_dim in self.rhs_contracting_dims:
                return False

    def __str__(self):
        return f"{self.name} {self.shape} = dot({self.lhs.name}, {self.rhs.name}) "\
               f" lhs_con_dim={self.lhs_contracting_dims},"\
               f" rhs_con_dim={self.rhs_contracting_dims}"


class HloTuple(HloInstruction):
    def __init__(self, operands):
        super().__init__(OpCode.TUPLE, (), operands)

    def build_strategy_and_cost(self, cluster_env, solver_option):
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

        self.parameters = []

        self.strategy_built = False

    def append(self, instruction):
        ct = len(self.instructions)
        self.instructions.append(instruction)

        if instruction.op_code == OpCode.PARAMETER:
            self.parameters.append(instruction)

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

    def concurrency_analysis(self):
        frontier_list = []
        edge_dict = defaultdict(list)

        # Build degree dict
        #out_degree = defaultdict(lambda : 0)
        #for ins in self.instructions:
        #    for operand in ins.operands:
        #        out_degree[operand] += 1

        degree = defaultdict(lambda : 0)
        for ins in self.instructions:
            for operand in ins.operands:
                degree[ins] += 1
                edge_dict[operand].append(ins)

        # Init frontier
        collected = 0
        current_frontier = []
        for ins in self.instructions:
            if degree[ins] == 0:
                current_frontier.append(ins)
                collected += 1
        frontier_list.append(current_frontier)

        # Push forward frontier
        while collected < len(self.instructions):
            current_frontier = frontier_list[-1]
            next_frontier = []
            for ins in current_frontier:
                for node in edge_dict[ins]:
                    degree[node] -= 1
                    if degree[node] == 0:
                        next_frontier.append(node)
                        collected += 1
            frontier_list.append(next_frontier)

        for i, frontier in enumerate(frontier_list):
            print(i)
            for ins in frontier:
                print(ins)

    def forward_backward_analysis(self):
        used_by = defaultdict(list)
        for ins in self.instructions:
            for operand in ins.operands:
                used_by[operand].append(ins.index)

        sep_id = 0
        for param in self.parameters:
            if len(used_by[param]) > 2:
                backward_id = used_by[param][0]
                sep_id = max(sep_id, backward_id + 1)

        return sep_id

    def batch_dim_analysis(self):
        # Build used by dict
        used_by = defaultdict(list)
        for ins in self.instructions:
            for operand in ins.operands:
                used_by[operand].append(ins)

        # Find source.
        # Rule: The first dim of parameters that are only used once
        #possible_inputs = []
        #for param in self.parameters:
        #    if len(used_by[param]) == 1:
        #        possible_inputs.append(param)
        #source = possible_inputs[0]
        source = self.instructions[0]
        source.batch_dim = 0

        # Dim propagation
        queue = [source]
        visited = set([source])

        while len(queue) > 0:
            ins = queue.pop(0)

            # Propagate to operand

            # Propagate to used_by
            for consumer in used_by[ins]:
                #print(f"Propagate from {ins} to {consumer}")
                success = consumer.propagate_batch_dim(ins)
                if not success:
                    continue
                if consumer.index not in visited:
                    visited.add(consumer)
                    queue.append(consumer)

    def depth_analysis(self):
        edge_dict = defaultdict(list)

        degree = defaultdict(lambda : 0)
        for ins in self.instructions:
            for operand in ins.operands:
                degree[ins] += 1
                edge_dict[operand].append(ins)

        # Init frontier
        collected = 0
        current_frontier = []
        for ins in self.instructions:
            if degree[ins] == 0:
                ins.depth = 0
                current_frontier.append(ins)
                collected += 1

        # Push forward frontier
        depth = 0
        while collected < len(self.instructions):
            next_frontier = []
            for ins in current_frontier:
                for node in edge_dict[ins]:
                    degree[node] -= 1
                    if degree[node] == 0:
                        next_frontier.append(node)
                        collected += 1

            depth += 1
            current_frontier = next_frontier
            for ins in current_frontier:
                ins.depth = depth

    def build_strategy_and_cost(self, cluster_env, solver_option):
        if self.strategy_built:
            for ins in self.instructions:
                ins.strategies = []
                ins.compute_costs = []
                ins.communication_costs = []
                ins.memory_costs = []
                ins.resharding_costs = []
                ins.follow_ins = None

            self.alias_cost_vector = []

        # Analyze depth for all instructions
        self.depth_analysis()

        # Analyze batch dim
        if solver_option.force_batch_dim_to_mesh_dim is not None:
            batch_dim = self.batch_dim_analysis()
            print("===== Batch Dim Analysis =====")
            for i in range(len(self.instructions)):
                print(f"Time {i:2d}: {self.instructions[i]}  Batch: {self.instructions[i].batch_dim}")

        # Build strategies and costs for each instruction
        for ins in self.instructions:
            ins.build_strategy_and_cost(cluster_env, solver_option)

        # Build alias costs
        for (ins_a, ins_b) in self.alias_list:
            assert ins_a.shape == ins_b.shape
            cost_vector = []
            for stra_a in ins_a.strategies:
                for stra_b in ins_b.strategies:
                    if stra_a.output_spec == stra_b.output_spec:
                        cost_vector.append(0)
                    else:
                        cost_vector.append(1)
            self.alias_cost_vector.append(cost_vector)

        self.strategy_built = True

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
 
