#!/bin/bash
python benchmark_moe.py --nproc_per_node 8 --nnodes 8 --exp_name $1 --suite paper_moe 2>&1 | tee /tmp/$1
