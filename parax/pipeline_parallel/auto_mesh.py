import numba
import numpy as np
from parax.global_env import global_config


def get_costs():
    """get the cost of both communication and computation"""
    # TODO
    pass


@numba.jit()
def pipeline_dp(L, cost_e, cost_c, k, B):
    """Assign L (forward) layers into k-meshes
    Args:
    L: number of layers

    cost_c: communication(edge) cost

    cost_e: computation cost

    k: number of sliced stages

    B: batch size
    """
    possible = set()
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
    while(l >= 0):
        new_m = last_idx[l][s][m]
        new_l = last_cut[l][s][m]
        solution.append(l - new_l)
        l, s, m = new_l, s - 1, new_m
    solution = list(reversed(solution))

    return costs[L - 1][k - 1][0], solution


def init(M, N):
    pass


def search_logical_mesh(M, N):
    assert global_config.search_logical_mesh_shape
    init(M, N)
