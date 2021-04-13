"""Common Utilities"""

import numpy as np


def compute_bytes(shape):
    return np.prod(shape) * 4

