#!/bin/bash

CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')

run_experiment () {
  NUM_HOSTS=$1
  NUM_DEVICES_PER_HOST=$2
  NUM_GPUS=$((NUM_HOSTS * NUM_DEVICES_PER_HOST))
  echo "--- Running experiment with $NUM_HOSTS hosts and $NUM_DEVICES_PER_HOST devices per host ---"
  python3 -u benchmark_3d_infer.py --suite gpt.infer_perf_test \
    --exp_name infer_${NUM_HOSTS}x${NUM_GPUS}_gpus \
    --result_name "result_gpt_26B_preshard" \
    --num-hosts ${NUM_HOSTS} \
    --num-devices-per-host ${NUM_DEVICES_PER_HOST} \
    --disable-tqdm \
    |& tee auto_gpt_${NUM_HOSTS}x${NUM_GPUS}_gpus_${CURRENT_TIME}.log
  sleep 0.1 # for ctrl+c to work
}

#run_experiment 1 1
#run_experiment 1 2
run_experiment 1 4
#run_experiment 1 8
# run_experiment 2 1
# run_experiment 2 2
# run_experiment 2 4
# run_experiment 2 8
