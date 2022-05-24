Adopted from https://github.com/google-research/scenic for Unet and 
https://github.com/baudcode/tf-semantic-segmentation for COCO dataset

## COCO Segmentation Task

Train a Unet model ([Ronneberger et al., 2015]) on the COCO Segmentation task ([Lin et al., 2015]).

In this example, we allow to change the number of convolution layers at each height which was 2 for default unet configuration. This modification allows us to scale up model size more easily and then could better demonstrate parallelization speedup by Alpa. 

Table of contents:

- [Requirements](#requirements)
- [Running Locally](#running-locally)
  - [Overriding parameters on the command line](#overriding-parameters-on-the-command-line)
  - [Preparing the dataset](#preparing-the-dataset)
- [Scalability Analysis](#scalability-analysis)
  - [Statistics](#statistics)
  - [Run benchmark](#run-benchmark)

### Requirements
* For data preprocess: `Tensorflow`, `imageio`, `opencv-python`

### Running Locally

```shell
python main.py --workdir=./coco --config=configs/default.py
```

#### Overriding parameters on the command line

Specify a hyperparameter configuration by the means of setting `--config` flag. Configuration flag is defined using config_flags. config_flags allows overriding configuration fields. This can be done as follows:

```shell
python main.py --workdir=./coco --config=configs/default.py --config.data_path="./COCO_dataset"
```

#### Preparing the dataset

Users need to specify save path of dataset, then dataset will be automatically downloaded at the first time of running this code. The dataset is large, users need to reserve at least 35G available storage for unzipped COCO dataset. 


### Scalability Analysis

We benchmark unet on various number of GPUs to see scalability of Alpa's parallelization strategy. 

#### Statistics

Experiments are carried out on AWS p3 series instances. 
For experiments with 2 or 4 gpus, I use single p3.8xlarge in which there are 4 V100 for each host and each V100 has 16 GB memory. For experiments with 8 and more gpus, I use p3.16xlarge in which there are 8 V100 for each host and each V100 has 16 GB memory. 


| # of GPU   | 2 | 4 | 8 | 16 | 32 |
| :------ | -----: | -------: | -------------: | ----------: | :---------------------------------------- |
| tflops | 2.69 | 4.96 | 7.25 | 10.16 | 10.64 |
| param_count | 0.4143M | 0.8567M | 1.5918M | 3.1771M | 6.5202M |

#### Run benchmark

Users could feel free to run the benchmark code with command below. Arguments `--num-hosts` and `--num-devices-per-host` are to specify the amount of resources to use, their product is the exactly the number of devices for parallelization. `--num-devices-per-host` should be less than the number of gpus avaible in each host. Users could also set hyperparameters in `./config/benchmark_unet_suite.py`. Here is an example for 2 gpu training case:

```shell
python3 -u benchmark_main.py --num-hosts 1 --num-devices-per-host 2
```

One thing to mention, users may need to increase `profile_timeout` in global variables if the profiling time is too long on some platforms. 

