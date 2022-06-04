#!/bin/bash
python benchmark_moe.py --nproc_per_node 8 --nnodes 4 --exp_name none --suite test_moe 2>&1 | tee /tmp/none
