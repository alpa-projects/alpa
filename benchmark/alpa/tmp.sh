#!/bin/bash --login
python benchmark.py --suite gpt.perf_test_fast_2d \
        --shard-only --local --num-hosts 1 --num-devices-per-host 1 \
        --num_gpt_layer 1 --num_batch_size 4 --num_micro_batches 1 \
        --dp 1 --op 1
mv tmp Layer1_BatchSize4_MicroBS4

python benchmark.py --suite gpt.perf_test_fast_2d \
        --shard-only --local --num-hosts 1 --num-devices-per-host 1 \
        --num_gpt_layer 1 --num_batch_size 1 --num_micro_batches 1 \
        --dp 1 --op 1
mv tmp Layer1_BatchSize1_MicroBS1

python benchmark.py --suite gpt.perf_test_fast_2d \
        --shard-only --local --num-hosts 1 --num-devices-per-host 1 \
        --num_gpt_layer 1 --num_batch_size 16 --num_micro_batches 4 \
        --dp 1 --op 1

mv tmp Layer1_BatchSize16_MicroBS4

python benchmark.py --suite gpt.perf_test_fast_2d \
        --shard-only --local --num-hosts 1 --num-devices-per-host 1 \
        --num_gpt_layer 2 --num_batch_size 4 --num_micro_batches 1 \
        --dp 1 --op 1
mv tmp Layer2_BatchSize4_MicroBS4

python benchmark.py --suite gpt.perf_test_fast_2d \
        --shard-only --local --num-hosts 1 --num-devices-per-host 1 \
        --num_gpt_layer 4 --num_batch_size 4 --num_micro_batches 1 \
        --dp 1 --op 1
mv tmp Layer4_BatchSize4_MicroBS4