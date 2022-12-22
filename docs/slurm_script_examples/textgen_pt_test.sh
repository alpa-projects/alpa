#!/bin/bash
#SBATCH --job-name=ray_singlenode_test
# load modules
module purge
module load cuda
module load nvhpc
conda init bash
source ~/.bashrc
# test nvcc
echo "test cuda version"
nvcc --version
echo "-----------------"
# start conda
echo "simple test start"
conda activate dedong_test_p39
echo "entered conda env"
# test nccl
echo "test import nccl"
python3 -c "from cupy.cuda import nccl"
echo "test nccl done"
# environment activated, check environment
echo "python version:"
python3 -V
echo "---------------"
# start ray on head
echo "start ray"
ray start --head
echo"----------------"
# start alpa textgen.py
echo "textgen with pytorch version"
python3 alpa/examples/llm_serving/textgen.py --model facebook/opt-125m --n-prompts 1 --path $PROJECT/alpa_weights
# end ray
echo "end ray"
ray stop
echo "ray stopped"
# exit environment
conda deactivate
echo "---Finished successfully---"
exit
