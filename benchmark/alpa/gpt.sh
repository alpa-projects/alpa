#!/bin/bash --login

python benchmark.py --suite gpt.perf_test_fast_2d \
   --shard-only --num-hosts 1 --num-devices-per-host 4 \
   --num_gpt_layer 1 --num_batch_size 12 --num_micro_batches 3


# python benchmark.py --suite gpt.perf_test_fast_2d \
#    --shard-only --num-hosts 1 --num-devices-per-host 4 \
#    --num_gpt_layer 1 --num_batch_size 8 --num_micro_batches 2