#!/bin/bash

for i in 1 2 4 8
do
  echo i
  ray stop
  ray start --head --num-gpus ${i}
  sleep 1
  python benchmark_gpt_bert_3d.py --suite auto_gpt --exp_name auto_${i}_gpus &> auto_gpt_${i}_gpus.log
  sleep 1
done