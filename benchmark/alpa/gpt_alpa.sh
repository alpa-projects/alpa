#!/bin/bash --login
start_time=$(date +%s)
layers=(1,2,4,8,16,24)
gpus=(4,8,16)
reduce_scatter=(0,1)
dp=(1,2,4,8)
op=(8,4,2,1)

# for ((j=0; j<${#gpus[*]}; j=j+1)); do
#     for ((k=0; k<${#layers[*]}; k=k+1)); do
#             python benchmark.py --suite gpt.perf_test_fast_2d \
#             --shard-only --num-hosts 1 --num-devices-per-host ${layers[k]} \
#             --num_gpt_layer ${layers[k]} --num_batch_size 12 --num_micro_batches 3 \
#             --dp 4 --op 1 --reduce_scatter
#     done  
# done

for ((k=0; k<${#dp[*]}; k=k+1)); do
        python benchmark.py --suite gpt.perf_test_fast_2d \
                    --shard-only --num-hosts 1 --num-devices-per-host 8 \
                    --num_gpt_layer 4 --num_batch_size 32 --num_micro_batches 1 \
                    --dp ${dp[k]} --op ${op[k]} --reduce_scatter

        python benchmark.py --suite gpt.perf_test_fast_2d \
                    --shard-only --num-hosts 1 --num-devices-per-host 8 \
                    --num_gpt_layer 4 --num_batch_size 32 --num_micro_batches 1 \
                     --dp ${dp[k]} --op ${op[k]}      
done  

# python benchmark.py --suite gpt.perf_test_fast_2d \
#             --shard-only --num-hosts 1 --num-devices-per-host 8 \
#             --num_gpt_layer 4 --num_batch_size 32 --num_micro_batches 1 \
#             --dp 4 --op 1 --reduce_scatter



# python benchmark.py --suite gpt.perf_test_fast_2d \
#    --shard-only --num-hosts 1 --num-devices-per-host 4 \
#    --num_gpt_layer 1 --num_batch_size 8 --num_micro_batches 2

end_time=$(date +%s)
cost_time=$[ $end_time - $start_time]
echo "running spends $(($cost_time/60))min $(($cost_time%60))s"