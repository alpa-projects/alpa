#!/bin/bash
python benchmark_gpt_bert.py --nproc_per_node $1 --exp_name $2 2>&1 | tee /tmp/$2
