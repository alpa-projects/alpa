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

Experiments are carried out on Nvidia A100 GPU clusters in which each host has 4 A100 GPU and each A100 GPU has 40GB memory. 

*TODO: Finish all experiments and upload the results*

| # of GPU   | 2 | 4 | 8 | 16 | 32 |
| :------ | -----: | -------: | -------------: | ----------: | :---------------------------------------- |
| A | | | | | |
| B | | | | | |
| ... | | | | | |

#### Run benchmark

Users could feel free to run the benchmark code with command below. Arguments `--num-hosts` and `--num-devices-per-host` are to specify the amount of resources to use, their product is the exactly the number of devices for parallelization. Users could also set hyperparameters in `./config/benchmark_unet_suite.py`. Here is an example for 2 gpu training case:

```shell
python3 -u benchmark_main.py --num-hosts 1 --num-devices-per-host 2
```



