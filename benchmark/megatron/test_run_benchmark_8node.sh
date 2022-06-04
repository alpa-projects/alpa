#!/bin/bash
ROOT_DIR=/home/ubuntu/parax-efs/pycharm/alpa
# Node 2
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.44.28 "source $ROOT_DIR/../megatron-env/bin/activate; cd /home/ubuntu/parax-efs/pycharm/alpa/benchmark/megatron; python benchmark_gpt_bert.py --nproc_per_node 8 --nnodes 8 --node_rank 1 --master_port 41000 --master_addr 172.31.41.194 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt" &
# Node 3
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.38.88 "source $ROOT_DIR/../megatron-env/bin/activate; cd /home/ubuntu/parax-efs/pycharm/alpa/benchmark/megatron; python benchmark_gpt_bert.py --nproc_per_node 8 --nnodes 8 --node_rank 2 --master_port 41000 --master_addr 172.31.41.194 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt" &
# Node 4
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.43.214 "source $ROOT_DIR/../megatron-env/bin/activate; cd /home/ubuntu/parax-efs/pycharm/alpa/benchmark/megatron; python benchmark_gpt_bert.py --nproc_per_node 8 --nnodes 8 --node_rank 3 --master_port 41000 --master_addr 172.31.41.194 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt" &
# Node 5
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.41.200 "source $ROOT_DIR/../megatron-env/bin/activate; cd /home/ubuntu/parax-efs/pycharm/alpa/benchmark/megatron; python benchmark_gpt_bert.py --nproc_per_node 8 --nnodes 8 --node_rank 4 --master_port 41000 --master_addr 172.31.41.194 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt" &
# Node 6
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.39.125 "source $ROOT_DIR/../megatron-env/bin/activate; cd /home/ubuntu/parax-efs/pycharm/alpa/benchmark/megatron; python benchmark_gpt_bert.py --nproc_per_node 8 --nnodes 8 --node_rank 5 --master_port 41000 --master_addr 172.31.41.194 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt" &
# Node 7
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.44.213 "source $ROOT_DIR/../megatron-env/bin/activate; cd /home/ubuntu/parax-efs/pycharm/alpa/benchmark/megatron; python benchmark_gpt_bert.py --nproc_per_node 8 --nnodes 8 --node_rank 6 --master_port 41000 --master_addr 172.31.41.194 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt" &
# Node 8
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.38.129 "source $ROOT_DIR/../megatron-env/bin/activate; cd /home/ubuntu/parax-efs/pycharm/alpa/benchmark/megatron; python benchmark_gpt_bert.py --nproc_per_node 8 --nnodes 8 --node_rank 7 --master_port 41000 --master_addr 172.31.41.194 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt" &

python benchmark_gpt_bert.py --nproc_per_node 8 --nnodes 8 --node_rank 0 --master_port 41000 --master_addr 172.31.41.194 --exp_name none --suite test_gpt 2>&1 | tee /tmp/test_gpt
