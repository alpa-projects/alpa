#!/bin/bash
python benchmark_moe_test.py --nproc_per_node $1 --exp_name none --suite test_moe 2>&1 | tee /tmp/none
