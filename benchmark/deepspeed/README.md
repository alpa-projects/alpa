# Benchmark Deepspeed

## Requirements
1. Install dependencies
```
torch==1.8.0
```

2. Install deepspeed and download examples
```
pip3 install deepspeed
git clone --recursive https://github.com/microsoft/DeepSpeed.git
echo 'export DEEPSPEED_PATH=~/efs/DeepSpeed' >> ~/.bashrc   # use your own path
source ~/.bashrc
```

3. Download dataset
```
mkdir dataset
cd dataset
wget https://deepspeed.blob.core.windows.net/megatron-data/webtext.tgz
tar xzf webtext.tgz
ln -s $(pwd) ~/efs/parax/benchmark/deepspeed/data  # use your own path
```

## Run
```
# Single Node
python3 benchmark_gpt2.py --nproc_per_node 8
```
