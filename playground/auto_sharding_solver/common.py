"""Common Utilities"""

import numpy as np


def append_elements(result, array, indices, cur_depth, cur_indices):
    if cur_depth == len(array.shape) - 1:
        result.append(array[tuple(cur_indices)])
    else:
        next_depth = cur_depth + 1
        index = indices[next_depth]

        if index == -1:
            for i in range(array.shape[next_depth]):
                cur_indices[next_depth] = i
                append_elements(result, array, indices, next_depth, cur_indices)
        else:
            cur_indices[next_depth] = index
            append_elements(result, array, indices, next_depth, cur_indices)

def get_flatten_elements(array, indices):
    result = []
    cur_indices = [None] * len(array.shape)
    append_elements(result, array, indices, -1, cur_indices)
    return result

def get_dim_last_value(array, dim):
    indices = [0] * len(array.shape)
    indices[dim] = -1
    return array[tuple(indices)]


def compute_bytes(shape):
    return np.prod(shape) * 4

