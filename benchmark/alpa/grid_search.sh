#!/bin/bash --login

python3 benchmark.py --suite gpt.grid_search_auto --num-hosts 1  --num-devices-per-host=2 | tee -a log.txt
sleep 10m 
python3 keep.py --gpus 16