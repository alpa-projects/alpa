#!/bin/bash
ROOT_DIR=/home/ubuntu/parax-efs/pycharm/parax
python benchmark_gpt_bert.py --nproc_per_node 2 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt
