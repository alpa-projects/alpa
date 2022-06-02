import numpy as np


meta_weight_path = "/tmp/part5_tensors/decoder.layers.15.fc1.weight"
alpa_weight_path = "/tmp/part5_tensors/decoder.layers.15.fc1.weight"


meta_weight = np.load(meta_weight_path)
alpa_weight = np.load(alpa_weight_path)

weight_shape = meta_weight.shape
assert weight_shape[0] == 6144
assert weight_shape[1] == 12288

num_diff = 0

for i in range(weight_shape[0]):
    for j in range(weight_shape[1]):
        meta_val = meta_weight[i][j]
        alpa_val = alpa_weight[i][j]
        if not np.isclose(meta_val, alpa_val):
            num_diff += 1
            print(f"({i}, {j})")

print(f"Num diff: {num_diff}")
print(f"L2 distance: {np.linalg.norm(meta_weight - alpa_weight)}")
