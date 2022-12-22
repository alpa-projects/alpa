#!/bin/bash
#SBATCH --job-name=test_alpa_prerequisites
#SBATCH -p GPU-shared
#SBATCH -t 1:00
#SBATCH --gpus=v100-16:1

echo "hello world"

module load cuda
module load cudnn
module load nvhpc

nvcc --version
