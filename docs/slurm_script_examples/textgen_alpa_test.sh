#!/bin/bash
#SBATCH --job-name=ray_singlenode_test
# load modules
module purge
module load cuda
module load nvhpc
conda init bash
source ~/.bashrc
# test nvcc
nvcc --version
# start environment using conda
conda activate dedong_test_p39
# start ray on head
ray start --head
# start alpa textgen.py
python3 alpa/examples/llm_serving/textgen.py --model alpa/bloom-560m --n-prompts 1 --path $PROJECT/alpa_weights
# end ray
ray stop
# exit environment
conda deactivate
exit
