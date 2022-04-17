#!/bin/bash

ROOT_DIR=/home/ubuntu/

usage() {
  echo "usage: num_gpus: [1|4|8|16|32]"
  exit 1
}

NUM_GPUS=$1

case $1 in
  1)
    python benchmark_gpt.py --nproc_per_node 1
    ;;
  4)
    python benchmark_gpt.py --nproc_per_node 4
    ;;
  8)
    python benchmark_gpt.py --nproc_per_node 8
    ;;
  16)
    ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.44.28 "source $ROOT_DIR/megatron-env/bin/activate; cd /home/ubuntu/alpa/osdi22_artifact/megatron; python benchmark_gpt.py --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_port 41000 --master_addr 172.31.41.194" &
    python benchmark_gpt.py --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_port 41000 --master_addr 172.31.41.194
    ;;
  32)
    ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.44.28 "source $ROOT_DIR/megatron-env/bin/activate; cd /home/ubuntu/alpa/osdi22_artifact/megatron; python benchmark_gpt.py --nproc_per_node 8 --nnodes 4 --node_rank 1 --master_port 41000 --master_addr 172.31.41.194" &
    ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.38.88 "source $ROOT_DIR/megatron-env/bin/activate; cd /home/ubuntu/alpa/osdi22_artifact/megatron; python benchmark_gpt.py --nproc_per_node 8 --nnodes 4 --node_rank 2 --master_port 41000 --master_addr 172.31.41.194" &
    ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.43.214 "source $ROOT_DIR/megatron-env/bin/activate; cd /home/ubuntu/alpa/osdi22_artifact/megatron; python benchmark_gpt.py --nproc_per_node 8 --nnodes 4 --node_rank 3 --master_port 41000 --master_addr 172.31.41.194" &
    python benchmark_gpt.py --nproc_per_node 8 --nnodes 4 --node_rank 0 --master_port 41000 --master_addr 172.31.41.194
    ;;
  *)
    usage
esac