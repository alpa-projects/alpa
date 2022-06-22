# Examples: Serving OPT-175B using Alpa
As a serving system, Alpa provides the following unique advantages:
- **Support commodity hardware**: With Alpa, you can serve OPT-175B using your in-house GPU cluster, without needing the latest generations of A100 80GB GPUs nor fancy InfiniBand connections -- no hardware constraints!
- **Flexible parallelism strategies**: Alpa will automatically figure out the appropriate model-parallelism strategies based on your cluster setup.
- **Serve with arbitrary numbers of GPUs, from 0 - 100s**: No matter how many GPUs you have, you can serve OPT as long as your total memory is sufficient.

In this example, we use Alpa to serve the open-source OPT model, supporting all sizes ranging from 125M to 175B. 

Specifically, Alpa provides:
- A backend to perform model-parallel distributed inference for the large OPT models;
- A web frontend to collect and batch inference requests from users.

**Note**: the trained OPT model weights can be obtained from [Metaseq](https://github.com/facebookresearch/metaseq), subject to their license.

## Requirements
1. Install Alpa following the [installation guide](https://alpa-projects.github.io/install.html).
2. Install additional requirements for serving:
```shell
cd examples/opt_serving && pip3 install -r requirements.txt
```
3. Compile several cython files for faster data processing:
```shell
cd examples/opt_serving && bash build.sh
```

## Get OPT weights
There are two ways you can obtain the pretrained OPT weights.

1. You can download the original OPT weights released by [Metaseq](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT), 
then use our script [convert_to_numpy_weight.py](scripts/convert_to_numpy_weights.py) to convert it into Alpa-compatible formats. 

2. We provide downloads for the processed 125M and 2.7B model here. For other sizes of OPT, please join [Alpa slack](https://forms.gle/YEZTCrtZD6EAVNBQ7) to request a copy from the Alpa developer team. 
   - [OPT-125M weights](https://drive.google.com/file/d/1Ps7DFD80wNO7u2t39YCYcBX-9XwypGzl/view?usp=sharing)
   - [OPT-2.7B weights](https://drive.google.com/file/d/1ayIaKRhxF9osZWgcFG-3vSkjcepSWdQd/view?usp=sharing) 


## Run and benchmark generation in command line
```shell
cd benchmark

# Run generation on a GPU with pytorch/huggingface backend
python3 benchmark_text_gen.py --model facebook/opt-125m

# Run distributed generation using the 2.7B model with Alpa
python3 benchmark_text_gen.py --model alpa/opt-2.7b --path [PATH_TO_WEIGHT]

# Run distributed generation using the 175B model with Alpa
python3 benchmark_text_gen.py --model alpa/opt-175b --path [PATH_TO_WEIGHT]
```

## Start a web to serve the OPT models
``shell
# Serve the OPT-175B model
python3 interative_hosted.py --model alpa/opt-175b --port 10001
``

## Code structure

- [examples/opt_serving/benchmark](benchmark): Benchmark scripts for generation via command line.
- [examples/opt_serving/dataset](dataset): Data loaders for serving. 
- [examples/opt_serving/service](service): Model serving web server.
- [examples/opt_serving/generator.py](generator.py): Backend for web server.
- [examples/opt_serving/interactive_hosted.py](interactive_hosted.py): Web server entry point.
