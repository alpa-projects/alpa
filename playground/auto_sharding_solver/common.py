"""Common Utilities"""

import numpy as np


def append_flatten_elements(result, array, indices, cur_depth, cur_indices):
    """Append elements of `array` to `result`. The `indices` is a generalized
       multi-dimensional index that can index a whole row (use -1 to indicate this)"""
    if cur_depth == len(array.shape) - 1:
        result.append(array[tuple(cur_indices)])
    else:
        next_depth = cur_depth + 1
        index = indices[next_depth]

        if index == -1:
            for i in range(array.shape[next_depth]):
                cur_indices[next_depth] = i
                append_flatten_elements(result, array, indices, next_depth, cur_indices)
        else:
            cur_indices[next_depth] = index
            append_flatten_elements(result, array, indices, next_depth, cur_indices)


def get_dim_last_value(array, shape, dim):
    """Get the value of the last element in a dimension"""
    array = np.array(array).reshape(shape)
    indices = [0] * len(shape)
    indices[dim] = shape[dim] - 1
    return array[tuple(indices)]


def transpose_flatten(array, shape, dimensions):
    """Transpose a flatten array"""
    array = np.array(array)
    return np.array(np.transpose(array.reshape(shape), dimensions)).flatten()


def reshape_flatten(array, shape, new_shape):
    """Reshape a flatten array"""
    array = np.array(array)
    return np.array(array.reshape(shape)).flatten()


def compute_bytes(shape):
    return np.prod(shape) * 4

