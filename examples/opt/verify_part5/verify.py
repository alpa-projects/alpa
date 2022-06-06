import numpy as np
import csv

official_md5 = "/tmp/opt175b_md5sum_shards.csv"


official_md5_dict = {}
with open(official_md5, "r") as csvfile:
    for row in csvfile:
        shard_name, md5_val = row.split(",")
        shard_name = shard_name.strip()
        md5_val = md5_val.strip()
        if shard_name.startswith("checkpoint"):
            official_md5_dict[shard_name] = md5_val
print(official_md5_dict)


my_md5 = "/tmp/my_md5"
my_md5_dict = {}
with open(my_md5, "r") as my_md5:
    for row in my_md5:
        md5_val, shard_name = row.split("  ")
        shard_name = shard_name.strip()
        md5_val = md5_val.strip()
        if shard_name.startswith("checkpoint"):
            my_md5_dict[shard_name] = md5_val

# do comparison
assert len(official_md5_dict) == len(my_md5_dict)

for shard_name in official_md5_dict:
    assert shard_name in my_md5_dict
    print(f"{shard_name}: official md5 {official_md5_dict[shard_name]}, my md5: {my_md5_dict[shard_name]}")
    assert official_md5_dict[shard_name] == my_md5_dict[shard_name], "wrong!"
