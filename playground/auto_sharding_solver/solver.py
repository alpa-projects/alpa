"""ILP Solver"""
import numpy as np

from alpa.shard_parallel.auto_sharding import _call_solver_serialized_args


def call_solver(N, M, s_len, s_follow, E, A, L, c, d, m, r, v, s_init):
    """Serialize python lists to flatten numpy arraies and call solver"""
    # Serialize strategy lengths
    s_len_np = np.array(s_len, dtype=np.int32)
    s_follow_np = np.array(s_follow, dtype=np.int32)

    # Serialize edge set
    len_edges = len(E)
    E_np = np.empty((len_edges, 2), dtype=np.int32)
    for (idx, (i, j)) in enumerate(E):
        E_np[idx][:] = [i, j]

    # Serialize alias set
    len_aliases = len(A)
    A_np = np.empty((len_aliases, 2), dtype=np.int32)
    for (idx, (i, j)) in enumerate(A):
        A_np[idx][:] = [i, j]

    # Serialize liveness set
    len_liveness_set = N + sum(len(v) for v in L)
    L_np = np.empty((len_liveness_set,), dtype=np.int32)
    L_np[0:N] = [len(v) for v in L]
    L_np[N:] = [x for v in L for x in v]

    # Serialize node costs
    len_node_costs = sum(len(v) for v in c)
    c_np = np.empty((len_node_costs,), dtype=np.float32)
    d_np = np.empty((len_node_costs,), dtype=np.float32)
    m_np = np.empty((len_node_costs,), dtype=np.float32)
    c_np[:] = [x for v in c for x in v]
    d_np[:] = [x for v in d for x in v]
    m_np[:] = [x for v in m for x in v]

    # Serialize edge costs
    len_edge_costs = sum(len(vec) for vec in r)
    r_np = np.empty((len_edge_costs,), dtype=np.float32)
    r_np[:] = [x for vec in r for x in vec]

    # Serialize alias costs
    len_alias_costs = sum(len(vec) for vec in v)
    v_np = np.empty((len_alias_costs,), dtype=np.float32)
    v_np[:] = [x for vec in v for x in vec]

    # Serialize init value
    s_init_np = None

    return _call_solver_serialized_args(
        N, M, s_len_np, s_follow_np, E_np, A_np, L_np,
        c_np, d_np, m_np, r_np, v_np, s_init_np)


class CostGraph:
    def __init__(self, node_lens, edges, edge_costs, to_merge_pair):
        self.node_lens = node_lens
        self.adjacency = dict()   # map a node to its neighbors
        self.edge_costs = dict()  # map an edge to its cost matrix
        self.reindexing_vector = dict()  # map a node to its reindexing vector
        self.merged_to = dict()   # map an merged node to its destination
        self.to_merge_pair = to_merge_pair  # the input follow pairs

        for i in range(len(node_lens)):
            self.adjacency[i] = set()

        # For redundant edges, we will overwrite the results with
        # the last value
        for ((i, j), cost) in zip(edges, edge_costs):
            cost = np.reshape(cost, (self.node_lens[i], self.node_lens[j]))

            self.add_edge_cost(i, j, cost)

    def get_edge_cost(self, i, j):
        if i <= j:
            return self.edge_costs[(i, j)]
        else:
            return self.edge_costs[(j, i)].transpose()

    def add_edge_cost(self, i, j, cost):
        if i > j:
            i, j = j, i
            cost = cost.transpose()

        if (i, j) in self.edge_costs:
            assert i in self.adjacency[j]
            assert j in self.adjacency[i]
            self.edge_costs[(i, j)] += cost
        else:
            self.adjacency[i].add(j)
            self.adjacency[j].add(i)
            self.edge_costs[(i, j)] = cost

    def remove_edge(self, i, j):
        if i > j:
            i, j = j, i

        assert j in self.adjacency[i]
        assert i in self.adjacency[j]
        assert (i, j) in self.edge_costs

        self.adjacency[i].remove(j)
        self.adjacency[j].remove(i)
        del self.edge_costs[(i, j)]

    def merge_node(self, src, dst):
        """Merge node src to node dst"""
        print(f"merge {src} to {dst}")
        assert dst in self.adjacency[src]
        assert src in self.adjacency[dst]
        assert dst not in self.merged_to
        assert src != dst

        edge_cost = self.get_edge_cost(dst, src)

        # Find the strategy to follow greedily
        reindexing = []
        candidates = list(range(self.node_lens[src]))
        for i in range(self.node_lens[dst]):
            # Pick the strategy with the lowest cost to follow.
            # If there are multiple strategies with the same lowest costs,
            # prefer to follow "replicated", which has the largest index.
            keys = [(edge_cost[i][j], -j) for j in range(self.node_lens[src])]
            candidates.sort(key=lambda j: keys[j])
            reindexing.append(candidates[0])

        self.merged_to[src] = dst
        self.reindexing_vector[src] = reindexing

        # Merge edge cost matrix
        adj_list = list(self.adjacency[src])
        for adj in adj_list:
            if adj == dst:
                continue
            added_edge_cost = np.empty((self.node_lens[dst], self.node_lens[adj]))
            for i in range(self.node_lens[dst]):
                j = reindexing[i]
                edge_cost_src_adj = self.get_edge_cost(src, adj)
                for k in range(self.node_lens[adj]):
                    added_edge_cost[i][k] = edge_cost_src_adj[j][k] + edge_cost[i][j]

            self.add_edge_cost(dst, adj, added_edge_cost)

        # Remove edges
        for adj in adj_list:
            self.remove_edge(src, adj)

    def query_destination(self, node):
        if node in self.merged_to:
            old_dst = self.merged_to[node]
            new_dst = self.query_destination(old_dst)
            if old_dst != new_dst:
                # Compress path
                old_reindexing_vector = self.reindexing_vector[node]
                new_reindexing_vector = []
                for i in range(self.node_lens[new_dst]):
                    new_reindexing_vector.append(
                        old_reindexing_vector[self.reindexing_vector[old_dst][i]])

                self.reindexing_vector[node] = new_reindexing_vector
                self.merged_to[node] = new_dst
            return new_dst
        else:
            return node

    def simplify(self):
        for (src, dst) in self.to_merge_pair:
            assert src not in self.merged_to
            dst = self.query_destination(dst)
            if src != dst:
                self.merge_node(src, dst)

    def export_result(self):
        E = []
        r = []
        s_follow = []

        for i in range(len(self.node_lens)):
            if i in self.merged_to:
                s_follow.append(self.query_destination(i))
            else:
                s_follow.append(-1)

        for ((i, j), v) in self.edge_costs.items():
            v = v.reshape(-1)
            E.append((i, j))
            r.append(v)

            assert len(v) == self.node_lens[i] * self.node_lens[j]

        return s_follow, E, r, self.reindexing_vector

    def __str__(self):
        ret = ""
        for i in range(len(self.node_lens)):
            ret += f"Node {i}: {self.node_lens[i]}\n"

        edges = list(self.edge_costs.keys())
        edges.sort()

        for (i, j) in edges:
            ret += f"Edge {(i, j)}:\n"
            ret += str(self.edge_costs[(i, j)]) + "\n"

        return ret


class SolverOption:
    def __init__(self):
        self.force_batch_dim_to_mesh_dim = None

        self.forward_backward_sep_id = None
        self.force_all_reduce_cost = None
        self.force_all_gather_cost = None
        self.force_reduce_scatter_cost = None


def solve_auto_sharding(computation, cluster_env, solver_option=None):
    print("===== Hlo Computation =====")
    print(computation, "\n")

    print("===== Liveness Analysis =====")
    liveness_dict = computation.liveness_analysis()
    for i in range(len(computation.instructions)):
        names = [ins.name for ins in liveness_dict[i]]
        names.sort()
        print(f"Time: {i}, Live set: {names}")

    if solver_option is None:
        solver_option = SolverOption()

    # Build strategies and costs
    computation.build_strategy_and_cost(cluster_env, solver_option)

    # Build all constants for ILP
    N = len(computation.instructions)
    M = cluster_env.memory_per_device

    s_len = []
    follow_pair = []
    E = []
    A = []
    L = []
    c = []
    d = []
    m = []
    r = []
    v = []
    for i in range(N):
        ins = computation.instructions[i]
        s_len.append(len(ins.strategies))
        L.append([ins.index for ins in liveness_dict[i]])
        c.append(ins.compute_costs)
        d.append(ins.communication_costs)
        m.append(ins.memory_costs)

        if ins.follow_ins is not None:
            follow_pair.append((ins.index, ins.follow_ins.index))

        for op_idx, operand in enumerate(ins.operands):
            E.append((operand.index, i))

            src = operand.index
            dst = i

            #ins.resharding_costs  # [s_i, operand_idx, s_operand]
            cost = []
            for p in range(len(computation.instructions[src].strategies)):
                for q in range(len(computation.instructions[dst].strategies)):
                    cost.append(ins.resharding_costs[q][op_idx][p])
            r.append(cost)

    # Simplify the graph by merging nodes
    cost_graph = CostGraph(s_len, E, r, follow_pair)
    cost_graph.simplify()
    s_follow, E, r, reindexing_vector = cost_graph.export_result()

    for src, dst in enumerate(s_follow):
        if dst >= 0:
            s_len[src] = len(reindexing_vector[src])
            c[src] = np.array(c[src])[reindexing_vector[src]]
            d[src] = np.array(d[src])[reindexing_vector[src]]
            m[src] = np.array(m[src])[reindexing_vector[src]]

    # Deal with alias
    for ((ins_a, ins_b), cost_vector) in zip(computation.alias_list,
                                             computation.alias_cost_vector):

        idx_a, idx_b = ins_a.index, ins_b.index
        cost_vector = np.array(cost_vector).reshape(
            len(ins_a.strategies), len(ins_b.strategies))

        if s_follow[idx_a] >= 0:
            reindexing_a = reindexing_vector[idx_a]
            idx_a = s_follow[idx_a]
        else:
            reindexing_a = range(len(ins_a.strategies))

        if s_follow[idx_b] >= 0:
            reindexing_b = reindexing_vector[idx_b]
            idx_b = s_follow[idx_b]
        else:
            reindexing_b = range(len(ins_b.strategies))

        if idx_a != idx_b:
            A.append((idx_a, idx_b))
            new_cost_vector = []
            for i in reindexing_a:
                for j in reindexing_b:
                    new_cost_vector.append(cost_vector[i, j])
            v.append(new_cost_vector)

    s_val, e_val, objective, status = call_solver(N, M, s_len, s_follow, E, A, L,
                                                  c, d, m, r, v, s_init=None)

    if True:
        # Print sharding spec
        instructions = computation.instructions
        print("===== Sharding Strategy =====")
        for i in range(N):
            if s_follow[i] < 0:
                stra_idx = s_val[i]
                name = instructions[i].strategies[stra_idx].name
                follow_map = ""
                spec = instructions[i].strategies[stra_idx].output_spec
            else:
                dst = s_follow[i]
                stra_idx = reindexing_vector[i][s_val[i]]
                name = instructions[i].strategies[stra_idx].name + f" follow {dst}"
                spec = instructions[i].strategies[stra_idx].output_spec

                follow_map = ""
                for idx in range(len(reindexing_vector[i])):
                    stra_idx = reindexing_vector[i][idx]
                    follow_map += f"[{instructions[dst].strategies[idx].name} -> "\
                            f"{instructions[i].strategies[stra_idx].name}] "
            #print(f"Time {i:2d}: {computation.instructions[i]}  Strategy: {name} Spec: {spec}")
            print(f"Time {i:2d}: {computation.instructions[i]}  Strategy: {name}")
            #if follow_map:
            #    print(follow_map)

        # Print edge cost
        for (idx, (i, j)) in enumerate(E):
            if r[idx][e_val[idx]] > 0:
                print(f"Edge cost {(i, j)} : {r[idx][e_val[idx]]}")

        # Print peak memory
        print("===== Memory Usage =====")
        for t in range(N):
            mem = 0
            for i in L[t]:
                mem += m[i][s_val[i]]
            print(f"Time {t}, memory: {mem / 1024**2: .2f} MB")

    return objective
