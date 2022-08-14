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

--------------------------------------------------------------------------------

Adopted from https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling

Use `alpa.parallelize` to parallelize the training loop.

--------------------------------------------------------------------------------

# Language model training examples

The following example showcases how to train a language model from scratch 
using the JAX/Flax backend.

JAX/Flax allows you to trace pure functions and compile them into efficient, fused accelerator code on both GPU and TPU.
Models written in JAX/Flax are **immutable** and updated in a purely functional
way which enables simple and efficient model parallelism.

## Causal language modeling

In the following, we demonstrate how to train an auto-regressive causal transformer model 
in JAX/Flax.
More specifically, we pretrain a randomely initialized [**`gpt2`**](https://huggingface.co/gpt2) model in Norwegian
to pre-train 124M [**`gpt2`**](https://huggingface.co/gpt2)
in Norwegian.

The example script uses the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.


To setup all relevant files for training, let's create a directory.

```bash
mkdir ./norwegian-gpt2
```

### Train tokenizer

In the first step, we train a tokenizer to efficiently process the text input for the model. Similar to how it is shown in [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train), we use a **`ByteLevelBPETokenizer`**.
The tokenizer is trained on the complete Norwegian dataset of OSCAR
and consequently saved in the cloned model directory.
This can take up to 10 minutes depending on your hardware ☕.

```python
from datasets import load_dataset
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer

# load dataset
dataset = load_dataset("oscar", "unshuffled_deduplicated_no", split="train")

# Instantiate tokenizer
tokenizer = ByteLevelBPETokenizer()

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]

# Customized training
tokenizer.train_from_iterator(batch_iterator(), vocab_size=50256, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save("./norwegian-gpt2/tokenizer.json")
```

### Create configuration

Next, we create the model's configuration file. This is as simple 
as loading and storing [`**gpt2**`](https://huggingface.co/gpt2)
in the local model folder:

```python
from transformers import GPT2Config

config = GPT2Config.from_pretrained("gpt2", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, vocab_size=50256)
config.save_pretrained("./norwegian-gpt2")
```

Great, we have set up our model repository. During training, we will now automatically
push the training logs and model weights to the repo.

### Train model

Finally, we can run the example script to pretrain the model:

#### Launch a Ray cluster
1. Use the command below to launch ray on a head node  
  ```ray start --head```
2. (Optional) If you have more nodes, connect them to the head node. The command should look like this, but with the ip address and password printed by the previous command.   
  ```ray start --address='172.31.34.216:6379' --redis-password='5241590000000000'```

##### Run
```bash
python3 run_clm_flax.py \
    --output_dir="./norwegian-gpt2" \
    --model_type="gpt2" \
    --config_name="./norwegian-gpt2" \
    --tokenizer_name="./norwegian-gpt2" \
    --dataset_name="oscar" \
    --dataset_config_name="unshuffled_deduplicated_no" \
    --do_train --do_eval \
    --block_size="512" \
    --per_device_train_batch_size="96" \
    --per_device_eval_batch_size="96" \
    --num_micro_batches="4" \
    --dtype="float16" \
    --learning_rate="1e-3" --warmup_steps="1000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="20" \
    --logging_steps="100" \
    --save_steps="2500" \
    --eval_steps="2500"
```

Training should converge at a loss and perplexity 
of 3.24 and 25.72 respectively after 20 epochs
This should take less than ~21 hours on a single TPUv3-8 or a machine with 8 V100 GPUs.
Training statistics can be accessed on [tfhub.de](https://tensorboard.dev/experiment/2zEhLwJ0Qp2FAkI3WVH9qA).

For a step-by-step walkthrough of how to do causal language modeling in Flax, please have a 
look at [this](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/causal_language_modeling_flax.ipynb) google colab.
