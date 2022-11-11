#!/bin/bash --login
start_time=$(date +%s)


dp=(1)
op=(1)
#mb=(1 2 4 8 16 32 64 128 256)
mb=(32)

for ((k=0; k<${#dp[*]}; k=k+1)); do
    for ((j=0; j<${#mb[*]}; j=j+1)); do 
        python benchmark.py --suite gpt.perf_test_fast_2d \
                    --shard-only --num-hosts 1 --num-devices-per-host 1 \
                    --num_batch_size 1024 --num_micro_batches ${mb[j]} \
                    --dp ${dp[k]} --op ${op[k]} \
                    --recomputation --reduce_scatter

        mv tmp 2d_gpu1_dp${dp[k]}_op${op[k]}_mb${mb[j]}

    done     #mv tmp dp${dp[k]}_op${op[k]}_BatchSize32_MicroB1_Layer4
done  

dp=(1)
op=(2)
#mb=(1 2 4 8 16 32 64 128 256)
mb=(32)

for ((k=0; k<${#dp[*]}; k=k+1)); do
    for ((j=0; j<${#mb[*]}; j=j+1)); do 
        python benchmark.py --suite gpt.perf_test_fast_2d \
                    --shard-only --num-hosts 1 --num-devices-per-host 2 \
                    --num_batch_size 1024 --num_micro_batches ${mb[j]} \
                    --dp ${dp[k]} --op ${op[k]} \
                    --recomputation --reduce_scatter

        mv tmp 2d_gpu2_dp${dp[k]}_op${op[k]}_mb${mb[j]}

    done     #mv tmp dp${dp[k]}_op${op[k]}_BatchSize32_MicroB1_Layer4
done  




end_time=$(date +%s)
cost_time=$[ $end_time - $start_time]
echo "running spends $(($cost_time/60))min $(($cost_time%60))s"