import copy
import numba
import numpy as np
from typing import List, Sequence, Tuple

import jax.numpy as jnp
from jax.core import ClosedJaxpr, Jaxpr, jaxpr_as_fun
from jax.interpreters import pxla

from parax.api import parallelize
from parax.device_mesh import DistributedArray, PhysicalDeviceMesh, VirtualMesh, _shard_device_array
from parax.global_env import global_config
from parax.pipeline_parallel.cross_mesh_resharding import (
    CollectiveGroup, ReshardingTask, ReshardingTaskSpec, VirtualDistributedArray
    as VDA)
from parax.pipeline_parallel.stage import JaxPipelineStage


def split_sharding_specs(layers: Sequence[JaxPipelineStage],
                         mixed_jaxpr: ClosedJaxpr, in_sharding_specs,
                         out_sharding_specs):
    '''
    split sharding specs of layers. Some intermediate sharding specs are missed,
    but they do not cross mesh so this does not matter.
    '''

    def map_or_none(var, ref):
        return ref[var] if var in ref else None

    in_sharding_dict = dict(zip(mixed_jaxpr.jaxpr.invars, in_sharding_specs))
    out_sharding_dict = dict(zip(mixed_jaxpr.jaxpr.outvars, out_sharding_specs))
    layer_in_sharding_specs = []
    layer_out_sharding_specs = []
    for layer in layers:
        layer_in_sharding_specs.append(
            [map_or_none(var, in_sharding_dict) for var in layer.invars])
        layer_out_sharding_specs.append(
            [map_or_none(var, out_sharding_dict) for var in layer.outvars])
    return layer_in_sharding_specs, layer_out_sharding_specs


def compile_and_profile_layer_cost_c(layers: Sequence[JaxPipelineStage],
                                     mesh: PhysicalDeviceMesh):
    """
    Args:
        layers (Sequence[JaxPipelineStage]): forward and corresponding backward
        mesh (PhysicalDeviceMesh): the assigned mesh
    """
    import jax.numpy as jnp
    backup_config = global_config.backup()

    global_config.num_micro_batches = None
    global_config.devices = mesh
    global_config.strategy = "shard_parallel"
    global_config.use_dummy_value_for_benchmarking = True

    invars = set()
    outvars = set()
    eqns = []
    consts_dir = {}
    for stage in layers:
        consts_dir.update(stage.consts_dir)
        # Do not add local invars into the invars
        invars.update([var for var in stage.invars if var not in outvars])
        outvars.update(stage.outvars)
        eqns += stage.eqns
    jaxpr = Jaxpr(
        constvars=list(consts_dir.keys()),
        invars=list(invars),
        outvars=list(outvars),
        eqns=eqns,
    )
    mixed_jaxpr = ClosedJaxpr(jaxpr, consts_dir.values())
    fn = jaxpr_as_fun(mixed_jaxpr)
    compiled_fn = parallelize(fn)
    args = [
        jnp.zeros(v.aval.shape, v.aval.dtype) for v in mixed_jaxpr.jaxpr.invars
    ]
    executable = compiled_fn(*args, __return_value_mode='get_executable')
    ret = executable.profile_with_dummy_inputs()

    global_config.restore(backup_config)
    split_in_specs, split_out_specs = split_sharding_specs(
        layers, mixed_jaxpr, executable.input_sharding_specs,
        executable.output_sharding_specs)
    return ret, split_in_specs, split_out_specs


def create_collective_group(src_mesh: PhysicalDeviceMesh,
                            dst_mesh: PhysicalDeviceMesh) -> CollectiveGroup:
    cg = CollectiveGroup(set(src_mesh.device_strs + dst_mesh.device_strs),
                           src_mesh, dst_mesh)
    cg.instantiate()
    return cg


def dummy_resharding_strategy(spec: ReshardingTaskSpec):
    strategy = []
    _sender_loads = {sender: 0 for sender in spec.src.device_mesh.device_strs}
    for dst_tile, src_tileslices, _ in spec.dst_tile_to_src_tiles_map:
        # plan is a 2D array
        per_spec_plan = np.empty(
            (len(dst_tile.replica_device_strs), len(src_tileslices)),
            dtype=object)
        for receiver_idx, _ in enumerate(dst_tile.replica_device_strs):
            for src_tileslice_idx, src_tileslice in enumerate(src_tileslices):
                loads = {
                    sender: _sender_loads[sender]
                    for sender in src_tileslice.replica_device_strs
                }
                sender = min(loads, key=loads.get)
                per_spec_plan[receiver_idx][src_tileslice_idx] = sender
                # upload load on-the-fly
                _sender_loads[sender] += src_tileslice.slice_size
        strategy.append(per_spec_plan)
    spec.set_resharding_strategy(strategy)
    return strategy


def profile_layer_cost_e(src: JaxPipelineStage, dest: JaxPipelineStage,
                         src_outvar_sharding_spec, dst_invar_sharding_spec,
                         src_mesh: VirtualMesh, dst_mesh: VirtualMesh,
                         collective_group: CollectiveGroup):
    src_outvars = {v: idx for idx, v in enumerate(src.outvars)}
    tot_cost = 0
    backup_use_dummy_value = global_config.use_dummy_value_for_benchmarking
    global_config.use_dummy_value_for_benchmarking = True
    tasks = []
    src_phy_mesh = collective_group.src_mesh
    for idx, invar in enumerate(dest.invars):
        if invar in src_outvars:
            out_sharding_spec = src_outvar_sharding_spec[src_outvars[invar]]
            in_sharding_spec = dst_invar_sharding_spec[idx]
            src_array = VDA(device_mesh=src_mesh,
                            aval=invar.aval,
                            sharding_spec=out_sharding_spec)
            dst_array = VDA(device_mesh=dst_mesh,
                            aval=invar.aval,
                            sharding_spec=in_sharding_spec)
            task_spec = ReshardingTaskSpec(src_array, dst_array)
            # create resharding strategy, ignore global load balance
            dummy_resharding_strategy(task_spec)
            # create distributed array as dummy inputs
            input_indices = pxla.spec_to_indices(invar.aval.shape,
                                                 out_sharding_spec)
            remote_buffers = _shard_device_array(jnp.zeros_like(invar.aval),
                                                 src_phy_mesh, input_indices)
            val = DistributedArray(src_phy_mesh, invar.aval, in_sharding_spec,
                                   remote_buffers, input_indices)
            task = ReshardingTask(task_spec, collective_group, val)
            tasks.append(task)

    for task in tasks:
        task.do()

    global_config.use_dummy_value_for_benchmarking = backup_use_dummy_value
    return tot_cost


def draw_fill(puzzle, tileLength, tileWidth, start, count, solList):
    """
    DFS to find solutions filling a puzzle with given size tiles
    
    Args:
        puzzle: current puzzle status. 1 if filled and 0 otherwise.
        tileLength, tileWidth: the length and width of the tile.
        start: current most top-left empty coordinate.
        count: number of already used tiles
        solList: the solution list
    """
    count += 1
    puzzleLength, puzzleWidth = puzzle.shape
    patternNum = (puzzleWidth * puzzleLength) / (tileWidth * tileLength)

    horizontal = False
    if (start[0] + tileLength <= puzzleLength and
            start[1] + tileWidth <= puzzleWidth):
        horizontal = True
        for i in range(start[0], start[0] + tileLength):
            for j in range(start[1], start[1] + tileWidth):
                if puzzle[i][j] != 0:
                    horizontal = False
    if horizontal:
        newPuzzle = copy.deepcopy(puzzle)
        for i in range(start[0], start[0] + tileLength):
            for j in range(start[1], start[1] + tileWidth):
                newPuzzle[i][j] = count
        if count == patternNum:
            solList.append(newPuzzle)
            return
        for i in range(start[0], puzzleLength):
            for j in range(0, puzzleWidth):
                if newPuzzle[i][j] == 0:
                    newStart = (i, j)
                    break
            else:
                continue
            break
        draw_fill(newPuzzle, tileLength, tileWidth, newStart, count, solList)

    vertical = False
    if tileLength != tileWidth and start[
            0] + tileWidth <= puzzleLength and start[
                1] + tileLength <= puzzleWidth:
        vertical = True
        for i in range(start[0], start[0] + tileWidth):
            for j in range(start[1], start[1] + tileLength):
                if puzzle[i][j] != 0:
                    vertical = False
    if vertical:
        newPuzzle = copy.deepcopy(puzzle)
        for i in range(start[0], start[0] + tileWidth):
            for j in range(start[1], start[1] + tileLength):
                newPuzzle[i][j] = count
        if count == patternNum:
            solList.append(newPuzzle)
            return
        for i in range(start[0], puzzleLength):
            for j in range(0, puzzleWidth):
                if newPuzzle[i][j] == 0:
                    newStart = (i, j)
                    break
            else:
                continue
            break
        draw_fill(newPuzzle, tileLength, tileWidth, newStart, count, solList)


def backtrack(puzzleLength, puzzleWidth, tileLength, tileWidth):
    patternNum = (puzzleWidth * puzzleLength) / (tileWidth * tileLength)
    solList = []
    if patternNum % 1 == 0:
        inputPuzzle = np.zeros((puzzleLength, puzzleWidth))
        draw_fill(inputPuzzle, tileLength, tileWidth, (0, 0), 0, solList)
    return solList


def generate_initial(M, N):
    h_w_list = []

    h_w_list.append((M, 1))
    h_w_list.append((1, N))
    known = {}

    configs = []
    for (h, w) in h_w_list:
        solution = backtrack(M, N, h, w)

        assert len(solution) > 0
        config_idx = np.random.choice(len(solution), size=1)[0]
        config = solution[config_idx]
        configs.append(config)

        solution.pop(config_idx)

        known[(h, w)] = solution

    return h_w_list, configs, known


def get_costs(config, grid, layers):
    """
    Get the cost of both communication and computation
    
    Args:
        config: a 2-dimension array indicating the mesh allocation info
        layers: layers to be allocated
    Returns:
        cost_e: computation cost
        cost_c: communication cost
    """
    # TODO
    pass


@numba.jit()
def pipeline_dp(cost_e, cost_c, k, B):
    """
    Assign L (forward) layers into k-meshes
    
    Args:
        cost_e(Sequence[float]): computation cost
        cost_c(Sequence[Sequence[float]]): communication cost
        k(int): number of sliced stages
        B(int): number of micro batches
    Returns:
        cost(float): the minimal computation and communication cost
        plan(List[int]): plan[i] is the number of layers at mesh[i]
    """
    possible = set()
    L = len(cost_e)
    presum = np.array([0] + list(cost_e))
    for i in range(1, L + 1):
        presum[i] += presum[i - 1]
    for i in range(L):
        for j in range(i + 1, L + 1):
            possible.add(presum[j] - presum[i])
    M = len(possible)
    possible = sorted(list(possible))
    cost2idx = {cost: i for i, cost in enumerate(possible)}
    possible = np.array(sorted(list(possible)))

    last_cut = np.zeros((L, k, M), dtype=np.int32)
    last_idx = np.zeros((L, k, M), dtype=np.int32)
    costs = np.zeros((L, k, M))
    for i in range(L):
        for j in range(k):
            if i + 1 <= j:
                continue
            elif j == 0:
                e_sum = presum[i + 1]
                for m in range(M):
                    costs[i][j][m] = (B - 1) * max(0, e_sum - possible[m])
                    last_cut[i][j][m] = -1
            else:
                for m in range(M):
                    best_cost = np.infty
                    best_slice = -1
                    best_idx = -1
                    for cut in range(j - 1, i):
                        cur_sum = presum[i + 1] - presum[cut]
                        m_idx = cost2idx[max(cur_sum, possible[m])]
                        cost_ = costs[cut][j - 1][m_idx]
                        cost_ += (B - 1) * max(0, cur_sum - possible[m])
                        cost_ += cost_c[cut][j - 1]
                        if cost_ < best_cost:
                            best_cost = cost_
                            best_slice = cut
                            best_idx = m_idx
                    costs[i][j][m] = best_cost
                    last_cut[i][j][m] = best_slice
                    last_idx[i][j][m] = best_idx
    # trace back to get the solution
    l, s, m = L - 1, k - 1, 0
    solution = []
    while (l >= 0):
        new_m = last_idx[l][s][m]
        new_l = last_cut[l][s][m]
        solution.append(l - new_l)
        l, s, m = new_l, s - 1, new_m
    solution = list(reversed(solution))

    return costs[L - 1][k - 1][0], solution


def cool_down(iter, max_iter, init_temp):
    return init_temp * (1 - iter / max_iter)


def neighbor(cur, known, M, N, maximum_try=10):
    """
    Get neighbor of current status
    
    Args:
        cur: current tile
        known: cache of known status
        M, N: grid shape
        maximum_try: threshold to stop tries
    Returns:
        new_h, new_w: new tile shape
        config: an arbitrary config for the new tile
    """
    import time
    h, w = cur

    time_s = time.time()
    while time.time() - time_s < maximum_try:
        index = np.random.choice([0, 1], size=1)[0]
        if index == 0:
            valid = []
            upper = min(M, N)
            upper = min((M * N) // w, upper) + 1

            for i in range(1, upper):
                if (i, w) in known.keys():
                    solution = known[(i, w)]
                else:
                    solution = backtrack(M, N, i, w)
                    known[(i, w)] = solution

                if len(solution) > 0:
                    valid.append(i)

            if len(valid) == 0:
                continue

            new_h = np.random.choice(valid, size=1)[0]

            new_config_idx = np.random.choice(len(known[(new_h, w)]), size=1)[0]
            ret = known[(new_h, w)].pop(new_config_idx)
            return new_h, w, ret

        else:
            valid = []
            upper = min(M, N)
            upper = min((M * N) // h, upper) + 1
            for i in range(1, upper):
                if (h, i) in known.keys():
                    solution = known[(h, i)]
                else:
                    solution = backtrack(M, N, h, i)
                    known[(h, i)] = solution

                if len(solution) > 0:
                    valid.append(i)

            if len(valid) == 0:
                continue

            new_w = np.random.choice(valid, size=1)[0]
            new_config_idx = np.random.choice(len(known[(h, new_w)]), size=1)[0]
            ret = known[(h, new_w)].pop(new_config_idx)
            return h, new_w, ret
    return None


def predict(configs, grid, layers, B) -> Tuple[np.ndarray, List[List[int]]]:
    costs = []
    solutions = []
    for i in range(len(configs)):
        config = configs[i]
        config = np.asarray(config)
        cost_e, cost_c = get_costs(config, grid, layers)
        k = int(np.max(config))

        # refer to pipeling slicing
        cost, solution = pipeline_dp(cost_e, cost_c, k, B)
        costs.append(cost)
        solutions.append(solution)
    return np.asarray(costs), solutions


def get_mesh_slicing_configs(
        grid: VirtualMesh,
        layers,
        B,
        num_iter=500,
        init_t=1) -> Tuple[List[np.ndarray], np.ndarray, List[List[int]]]:
    """
    Simulated annealing to get a set of mesh slicing algorithm.
    Return multiple plans for auto-tuner

    Args:
        grid: the grid to create meshes
        layers: (forward) layers of the computation
        B: number of micro-batches
        num_iter, init_t: parameters for simulated annealing
    Returns:
        configs: a list of np.arrays, each indicating a mesh slicing plan
        costs: the estimated cost for each plan
        solutions: the solution of layer-mesh map for each plan
    """
    assert isinstance(grid, VirtualMesh)
    M, N = len(grid.host_ids), grid.num_devices_per_host

    h_w, configs, known = generate_initial(M, N)
    costs, solutions = predict(configs, grid, layers, B)

    for i in range(num_iter):
        cur_t = cool_down(i, num_iter, init_t)

        new_configs = []
        new_h_w = []

        for (h, w) in h_w:
            step = neighbor((h, w), known, M, N)
            if step is None:
                new_h, new_w, new_config = (h, w, configs[h_w.index((h, w))])
            else:
                new_h, new_w, new_config = step
            new_h_w.append((new_h, new_w))
            new_configs.append(new_config)

        new_costs, new_solutions = predict(new_configs, grid, layers, B)
        acc_prob = np.exp(np.minimum((costs - new_costs) / (cur_t + 1e-5), 0))
        acc_index = (np.random.random(len(acc_prob)) < acc_prob)

        for j in range(len(configs)):
            if acc_index[j]:
                configs[j] = new_configs[j]
                costs[j] = new_costs[j]
                solutions[j] = new_solutions[j]

    return configs, costs, solutions


def config_to_logical_meshes(raw_mesh: VirtualMesh, config: np.ndarray):
    """
    Translate a config array into logical meshes
    """
    mesh_info = []
    M = config.shape[0]
    N = config.shape[1]

    visited = set()
    max_num = -1
    for i in range(M):
        for j in range(N):
            if config[i][j] not in visited:
                mesh_num = config[i][j]
                visited.add(mesh_num)
                start = (i, j)
                for p in range(j, N):
                    if config[i][p] != mesh_num:
                        p -= 1
                        break
                for q in range(i, M):
                    if config[q][j] != mesh_num:
                        q -= 1
                        break
                end = (q, p)
                mesh_info.append((mesh_num, start, end))
                max_num = max(max_num, mesh_num)
    assert max_num >= 0
    meshes = (None for _ in range(max_num))
    for info in mesh_info:
        id, start, end = info
        meshes[id] = raw_mesh.slice(0, range(start[0], end[0] + 1)).slice(
            1, range(start[1], end[1] + 1))
    return meshes


def slice_mesh(layers, **kwargs):
    raw_mesh = global_config.devices
    B = global_config.num_micro_batches
    configs, costs, solutions = get_mesh_slicing_configs(raw_mesh, layers, B)
    best_idx = costs.argmax()[0]
    best_config = configs[best_idx]
    best_solution = solutions[best_idx]
    sliced_meshes = config_to_logical_meshes(raw_mesh, best_config)
    global_config.sliced_meshes = sliced_meshes
    return best_solution