#!/bin/bash
# Check the correctness of scripts.

python3 benchmark_gpt_bert_2d.py --suite default
python3 benchmark_gpt_bert_3d.py --suite default
python3 benchmark_moe_2d.py --suite default
python3 benchmark_moe_3d.py --suite default
