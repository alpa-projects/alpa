#!/bin/bash
#SBATCH --job-name=test_cuda
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1:00
#SBATCH --gpus=v100-16:1

#import modules
module purge
module load cuda
module load nvhpc

#check environments
echo $CUDA_HOME
nvcc --version

#exit
