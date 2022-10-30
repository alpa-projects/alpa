#!/bin/bash --login
start_time=$(date +%s)
dp=(1 2 4 8)
op=(8 4 2 1)


for ((k=0; k<${#dp[*]}; k=k+1)); do
        python benchmark.py --suite gpt.perf_test_fast_2d \
                    --shard-only --num-hosts 1 --num-devices-per-host 8 \
                    --num_gpt_layer 4 --num_batch_size 32 --num_micro_batches 1 \
                    --dp ${dp[k]} --op ${op[k]} --reduce_scatter
done  

python benchmark.py --suite gpt.perf_test_fast_2d \
        --shard-only --local --num-hosts 1 --num-devices-per-host 1 \
        --num_gpt_layer 4 --num_batch_size 4 --num_micro_batches 1 \
        --dp 1 --op 1

end_time=$(date +%s)
cost_time=$[ $end_time - $start_time]
echo "running spends $(($cost_time/60))min $(($cost_time%60))s"