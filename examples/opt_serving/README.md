# Examples: Serving OPT-175B using Alpa

In this example, Alpa provides backend and a web frontend for serving the open-source OPT model, supporting all sizes ranging from 125M to 175B. 
The train model weights can be obtained from [Metaseq](https://github.com/facebookresearch/metaseq), subject to their open source license.

As a serving system, Alpa provides the following unique features:
- **Support commodity GPUs**: you can serve OPT-175B using your on-premise GPU cluster without needing latest A100 80GB GPUs, no hardware constraints!
- **Flexible parallelism strategies**: Alpa automatically figure out the appropriate model-parallelism strategies based on your cluster setup.
- **Serve with an arbitrary number of GPUs, from 0 - 100s**: No matter how many GPUs you have, you can serve OPT as long as your total memory is sufficient.

## Requirements
1. Install Alpa following the [installation guide](https://alpa-projects.github.io/install.html) 
2. Install additional requirements for serving:
```shell
cd examples/opt_serving && pip3 install -r requirements.txt
```
3. Compile several cython files for faster data processing:
```shell
cd examples/opt_serving && bash build.sh
```

## Get OPT weights
There are two weights you can obtain the pretrained OPT weights.

1. You can download the original OPT weights released by [Metaseq](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT), then use the scripts
[convert_to_numpy_weight.py](scripts/convert_to_numpy_weights.py) to convert it into Alpa-compatible formats. 

2. You can download the 125M and 2.7B model here. For other sizes of OPT, please please join [Alpa slack](https://forms.gle/YEZTCrtZD6EAVNBQ7) to request a copy from the Alpa developer team. 
   - [OPT-125M weights](https://drive.google.com/file/d/1Ps7DFD80wNO7u2t39YCYcBX-9XwypGzl/view?usp=sharing)
   - [OPT-2.7B weights](https://drive.google.com/file/d/1ayIaKRhxF9osZWgcFG-3vSkjcepSWdQd/view?usp=sharing) 

Put the weights under `~/opt_weights/`.

## Benchmark generation in command line
```
cd benchmark

# with pytorch and huggingface backend
python3 benchmark_text_gen.py --model facebook/opt-125m

# with alpa backend on AWS cluster
python benchmark_text_gen.py --model alpa/opt-2.7b --cluster aws

# with alpa backend and the 175B model
python3 
```

## Code structure

- [examples/opt_serving/benchmark](benchmark): Benchmark scripts for generation via command line.
- [examples/opt_serving/dataset](dataset): Data loaders for serving. 
- [examples/opt_serving/service](service): Model serving web server.
- [examples/opt_serving/generator.py](generator.py): Backend for web server.
- [examples/opt_serving/interactive_hosted.py](interactive_hosted.py): Web server entry point.
