#!/bin/bash

CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')

NUM_HOSTS=8
NUM_DEVICES_PER_HOST=8
NUM_GPUS=$((NUM_HOSTS * NUM_DEVICES_PER_HOST))
echo "--- Running experiment with $NUM_HOSTS hosts and $NUM_DEVICES_PER_HOST devices per host ---"
python3 -u benchmark_3d.py --suite gpt.test_auto \
  --exp_name auto_${NUM_GPUS}_gpus \
  --num-hosts ${NUM_HOSTS} \
  --num-devices-per-host ${NUM_DEVICES_PER_HOST} \
  --disable-tqdm \
  |& tee auto_gpt_${NUM_GPUS}_gpus_${CURRENT_TIME}.log
