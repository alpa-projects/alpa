#!/bin/bash

ROOT_DIR=/home/ubuntu/

usage() {
  echo "usage: num_gpus: [1|4|8|16|32]"
  exit 1
}

NUM_GPUS=$1

case $1 in
  1)
    python benchmark_moe.py --nproc_per_node 1
    ;;
  4)
    python benchmark_moe.py --nproc_per_node 4
    ;;
  8)
    python benchmark_moe.py --nproc_per_node 8
    ;;
  16)
    python benchmark_moe.py --nproc_per_node 8 --nnodes 2
    ;;
  32)
    python benchmark_moe.py --nproc_per_node 8 --nnodes 4
    ;;
  *)
    usage
esac