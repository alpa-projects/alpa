#!/bin/bash

for i in 1 2 4 8
do
  echo i
  ray stop
  ray start --head --num-gpus ${i}
  python benchmark_gpt_bert_3d.py --suite auto_gpt &> auto_gpt_${i}_gpus.txt
done