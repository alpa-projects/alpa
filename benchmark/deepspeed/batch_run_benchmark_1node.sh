#!/bin/bash
python benchmark_moe.py --nproc_per_node $1 --exp_name $2 --suite paper_moe 2>&1 | tee /tmp/$2
