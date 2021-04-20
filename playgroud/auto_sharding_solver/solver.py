"""ILP Solver"""
import numpy as np

from parax.auto_sharding import call_solver_serialized_args


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

    return call_solver_serialized_args(
        N, M, s_len_np, s_follow_np, E_np, A_np, L_np,
        c_np, d_np, m_np, r_np, v_np, s_init_np)


class CostGraph:
    def __init__(self, node_lens, edges, edge_costs, to_merge_pair):
        self.node_lens = node_lens
        self.adjacency = dict()   # map a node to its neighbors
        self.edge_costs = dict()  # map an edge to its cost matrix
        self.merged_strategy_mapping = dict()  # map a node to its strategy mapping during merge
        self.merged_to = dict()   # map an merged node to its destination
        self.to_merge_pair = to_merge_pair  # the input follow pairs

        for i in range(len(node_lens)):
            self.adjacency[i] = set()

        # for redundant edges, we will overwrite the results with
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
        row_sum = np.sum(edge_cost, axis=0)
        greedy_map = []
        for i in range(self.node_lens[dst]):
            # Pick the strategy with the lowest cost to follow.
            # If there are multiple strategies with the same lowest costs,
            # prefer to follow "replicated", which has the largest index.
            candidates = list(range(self.node_lens[src]))
            candidates.sort(key=lambda j: (edge_cost[i][j], -j))
            greedy_map.append(candidates[0])

        self.merged_to[src] = dst
        self.merged_strategy_mapping[src] = greedy_map

        # Merge edge cost matrix
        for adj in self.adjacency[src]:
            if adj == dst:
                continue
            new_edge_cost = []
            for i in range(self.node_lens[dst]):
                j = greedy_map[i]
                edge_cost_src_adj = self.get_edge_cost(src, adj)
                new_edge_cost.append(edge_cost_src_adj[j] + edge_cost[i][j])

            new_edge_cost = np.array(new_edge_cost)
            self.add_edge_cost(dst, adj, new_edge_cost)

        # Remove edges
        to_remove = [(src, x) for x in self.adjacency[src]]
        for i, j in to_remove:
            self.remove_edge(i, j)

    def query_destination(self, node):
        if node in self.merged_to:
            old_dst = self.merged_to[node]
            new_dst = self.query_destination(old_dst)
            if old_dst != new_dst:
                # Compress path
                old_strategy_mapping = self.merged_strategy_mapping[node]
                new_strategy_mapping = []
                for i in range(self.node_lens[new_dst]):
                    new_strategy_mapping.append(
                        old_strategy_mapping[self.merged_strategy_mapping[old_dst][i]])

                self.merged_strategy_mapping[node] = new_strategy_mapping
                self.merged_to[node] = new_dst
            return new_dst
        else:
            return node

    def simplify(self):
        for (src, dst) in self.to_merge_pair:
            assert src not in self.merged_to
            dst = self.query_destination(dst)
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

        return s_follow, E, r, self.merged_strategy_mapping

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
    s_follow, E, r, strategy_mapping = cost_graph.export_result()

    for src, dst in enumerate(s_follow):
        if dst >= 0:
            s_len[src] = len(strategy_mapping[src])
            c[src] = np.array(c[src])[strategy_mapping[src]]
            d[src] = np.array(d[src])[strategy_mapping[src]]
            m[src] = np.array(m[src])[strategy_mapping[src]]

    # Deal with alias
    for ((ins_a, ins_b), cost_vector) in zip(computation.alias_list,
                                             computation.alias_cost_vector):
        A.append((ins_a.index, ins_b.index))
        v.append(cost_vector)

    s_val, e_val, objective, status = call_solver(N, M, s_len, s_follow, E, A, L,
                                                  c, d, m, r, v, s_init=None)

    if True:
        # Print sharding spec
        instructions = computation.instructions
        for i in range(N):
            if s_follow[i] < 0:
                ins_idx = s_val[i]
                name = instructions[i].strategies[ins_idx].name
            else:
                ins_idx = strategy_mapping[i][s_val[i]]
                name = instructions[i].strategies[ins_idx].name + "_follow"
            print(f"Time {i:2d}: {computation.instructions[i]}  Strategy: {name}")

        # Print edge cost
        for (idx, (i, j)) in enumerate(E):
            if r[idx][e_val[idx]] > 0:
                print("Edge cost", i, j)

        # Print peak memory
        for t in range(N):
            mem = 0
            for i in L[t]:
                mem += m[i][s_val[i]]
            print(f"Time {t}, memory: {mem / 1024**2: .2f} MB")

    return objective
