<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
Adopted from https://github.com/huggingface/transformers/tree/main/examples/flax/vision

Use `alpa.parallelize` to parallelize the training loop.

# Image Classification training examples

The following example showcases how to train/fine-tune `ViT` for image-classification using the JAX/Flax backend.

JAX/Flax allows you to trace pure functions and compile them into efficient, fused accelerator code on both GPU and TPU.
Models written in JAX/Flax are **immutable** and updated in a purely functional
way which enables simple and efficient model parallelism.


In this example we will train/fine-tune the model on the [imagenette](https://github.com/fastai/imagenette) dataset.

## Prepare the dataset

We will use the [imagenette](https://github.com/fastai/imagenette) dataset to train/fine-tune our model. Imagenette is a subset of 10 easily classified classes from Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).


### Download and extract the data.

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -xvzf imagenette2.tgz
```

This will create a `imagenette2` dir with two subdirectories `train` and `val` each with multiple subdirectories per class. The training script expects the following directory structure

```bash
root/dog/xxx.png
root/dog/xxy.png
root/dog/[...]/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/[...]/asd932_.png
```

### Train the model

Finally, we can run the example script to pretrain the model:

#### Launch a Ray cluster
1. Use the command below to launch ray on a head node  
  ```ray start --head```
2. (Optional) If you have more nodes, connect them to the head node. The command should look like this, but with the ip address and password printed by the previous command.   
  ```ray start --address='172.31.34.216:6379' --redis-password='5241590000000000'```

##### Run
```bash
python run_image_classification.py \
    --output_dir ./vit-base-patch16-imagenette \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --train_dir="imagenette2/train" \
    --validation_dir="imagenette2/val" \
    --num_train_epochs 5 \
    --num_micro_batches 2 \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --overwrite_output_dir \
    --preprocessing_num_workers 32 \
```
Training should converge at a loss of 0.0614 and validation accuracy of ~98% after 5 epochs. This should take ~7 minutes on a single machine with 2 P100 GPUs. Training statistics can be accessed on https://tensorboard.dev/experiment/3Vz06C4xQKaqaHENFeIrGg/