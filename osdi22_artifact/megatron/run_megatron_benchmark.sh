#!/bin/bash

ROOT_DIR=/home/ubuntu/efs

usage() {
  echo "usage: num_gpus: [1|4|8|16|32]"
  exit 1
}

NUM_GPUS=$1

unset -v ip0 ip1 ip2 ip3
for var in ip0 ip1 ip2 ip3; do
  IFS= read -r "$var" || break
done < $ROOT_DIR/alpa/osdi22_artifact/ips

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
    ssh -o StrictHostKeyChecking=no ubuntu@$ip1 "source $ROOT_DIR/megatron-env/bin/activate; cd $ROOT_DIR/alpa/osdi22_artifact/megatron; python benchmark_gpt.py --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_port 41000 --master_addr $ip0" &
    python benchmark_gpt.py --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_port 41000 --master_addr $ip0
    ;;
  32)
    ssh -o StrictHostKeyChecking=no ubuntu@$ip1 "source $ROOT_DIR/megatron-env/bin/activate; cd $ROOT_DIR/alpa/osdi22_artifact/megatron; python benchmark_gpt.py --nproc_per_node 8 --nnodes 4 --node_rank 1 --master_port 41000 --master_addr $ip0" &
    ssh -o StrictHostKeyChecking=no ubuntu@$ip2 "source $ROOT_DIR/megatron-env/bin/activate; cd $ROOT_DIR/alpa/osdi22_artifact/megatron; python benchmark_gpt.py --nproc_per_node 8 --nnodes 4 --node_rank 2 --master_port 41000 --master_addr $ip0" &
    ssh -o StrictHostKeyChecking=no ubuntu@$ip3 "source $ROOT_DIR/megatron-env/bin/activate; cd $ROOT_DIR/alpa/osdi22_artifact/megatron; python benchmark_gpt.py --nproc_per_node 8 --nnodes 4 --node_rank 3 --master_port 41000 --master_addr $ip0" &
    python benchmark_gpt.py --nproc_per_node 8 --nnodes 4 --node_rank 0 --master_port 41000 --master_addr $ip0
    ;;
  *)
    usage
esac