# Benchmark Deepspeed

## Requirements
1. Install dependencies
```
# torch
pip3 install torch==1.8.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip3 install nltk pandas sentencepiece boto3 pybind11 python-config

# Adafactor optimizer
pip3 install torch-optimizer

# pdsh
sudo apt-get update
sudo apt-get install pdsh

# Apex
git clone https://github.com/NVIDIA/apex
cd apex
# Comment out the raised RuntimeError in setup.py if you get errors running the following command.
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

2. Install deepspeed and deepspeed examples
```
pip3 install deepspeed==0.5.4
git clone --recursive https://github.com/microsoft/DeepSpeed.git
echo 'export DEEPSPEED_PATH=~/efs/DeepSpeed' >> ~/.bashrc   # use your own path
source ~/.bashrc

# Replace source files (use your own path)
cp alpa/benchmark/deepspeed/patch/training.py DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/megatron/training.py
cp alpa/benchmark/deepspeed/patch/gpt2_model.py DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/megatron/model/gpt2_model.py
cp alpa/benchmark/deepspeed/patch/transformer.py DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/megatron/model/transformer.py
```

3. Download dataset
```
wget deepspeed_dataset.zip  # ask Lianmin to get the file
tar xzf deepspeed_dataset.zip
cd deepspeed_dataset/
ln -s $(pwd) ~/efs/alpa/benchmark/deepspeed/data   # use your own path
```

## Run
### Single Node
```
# GPT
python3 benchmark_gpt2.py --nproc_per_node 8
# MOE
python3 benchmark_gpt2_moe.py --nproc_per_node 8
```

### Multiple Node
- Modify the [hostfile](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) and setup the ssh connections.
```
python3 benchmark_gpt2.py --nnodes 2 --nproc_per_node 8
```
