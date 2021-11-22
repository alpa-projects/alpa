#!/bin/bash
ROOT_DIR=/home/ubuntu/parax-efs/pycharm/parax
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.44.28 "source $ROOT_DIR/../megatron-env/bin/activate; cd /home/ubuntu/parax-efs/pycharm/parax/benchmark/megatron; python benchmark_gpt_bert.py --nproc_per_node 8 --nnodes 4 --node_rank 1 --master_port 41000 --master_addr 172.31.41.194 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt" &
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.38.88 "source $ROOT_DIR/../megatron-env/bin/activate; cd /home/ubuntu/parax-efs/pycharm/parax/benchmark/megatron; python benchmark_gpt_bert.py --nproc_per_node 8 --nnodes 4 --node_rank 2 --master_port 41000 --master_addr 172.31.41.194 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt" &
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.43.214 "source $ROOT_DIR/../megatron-env/bin/activate; cd /home/ubuntu/parax-efs/pycharm/parax/benchmark/megatron; python benchmark_gpt_bert.py --nproc_per_node 8 --nnodes 4 --node_rank 3 --master_port 41000 --master_addr 172.31.41.194 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt" &

python benchmark_gpt_bert.py --nproc_per_node 8 --nnodes 4 --node_rank 0 --master_port 41000 --master_addr 172.31.41.194 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt
