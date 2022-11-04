#!/bin/bash --login
start_time=$(date +%s)
dp=(1)
op=(4)
layer=(8 12 16)

for ((i=0; i<${#layer[*]}; i=i+1)); do
        for ((k=0; k<${#dp[*]}; k=k+1)); do
                python benchmark.py --suite gpt.perf_test_fast_2d \
                        --shard-only --num-hosts 1 --num-devices-per-host 4 \
                        --num_gpt_layer ${layer[i]} --num_batch_size 4 --num_micro_batches 1 \
                        --dp ${dp[k]} --op ${op[k]} --recomputation
                
        done  
done 

for ((i=0; i<${#layer[*]}; i=i+1)); do
        for ((k=0; k<${#dp[*]}; k=k+1)); do
                python benchmark.py --suite gpt.perf_test_fast_2d \
                        --shard-only --num-hosts 1 --num-devices-per-host 4 \
                        --num_gpt_layer ${layer[i]} --num_batch_size 4 --num_micro_batches 1 \
                        --dp ${dp[k]} --op ${op[k]}
                
        done  
done 


end_time=$(date +%s)
cost_time=$[ $end_time - $start_time]
echo "running spends $(($cost_time/60))min $(($cost_time%60))s"