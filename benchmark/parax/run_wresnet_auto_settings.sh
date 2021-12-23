#!/bin/bash

CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')

run_experiment () {
  NUM_HOSTS=$1
  NUM_DEVICES_PER_HOST=$2
  NUM_GPUS=$((NUM_HOSTS * NUM_DEVICES_PER_HOST))
  echo "--- Running experiment with $NUM_HOSTS hosts and $NUM_DEVICES_PER_HOST devices per host ---"
  python3 -u benchmark_wide_resnet_3d.py --suite 16gb \
    --exp_name auto_${NUM_GPUS}_gpus \
    --num-hosts ${NUM_HOSTS} \
    --num-devices-per-host ${NUM_DEVICES_PER_HOST} \
    --logical_mesh_search_space only_dp \
    |& tee auto_wresnet_${NUM_GPUS}_gpus_${CURRENT_TIME}.log
  sleep 0.1 # for ctrl+c to work
}

run_experiment 1 2
run_experiment 2 8
run_experiment 1 8
run_experiment 1 4
