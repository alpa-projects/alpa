#!/bin/bash --login
start_time=$(date +%s)
gpus=(1 2 4 8 8 16)
device=(1 1 1 1 2 1)



for ((k=0; k<${#gpus[*]}; k=k+1)); do
        python benchmark.py --suite gpt.perf_test_auto \
                     --num-hosts ${device[k]} --num-devices-per-host ${gpus[k]} 

done  

end_time=$(date +%s)
cost_time=$[ $end_time - $start_time]
echo "running spends $(($cost_time/60))min $(($cost_time%60))s"