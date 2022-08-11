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

# Fine-tuning OPT Language Models

Quick run
```
ray start --head
python3 run_clm_flax.py \
    --output_dir="./output" \
    --model_name_or_path="facebook/opt-125m" \
    --dataset_name="wikitext" \
    --dataset_config_name="wikitext-2-raw-v1" \
    --do_train --do_eval \
    --block_size="512" \
    --per_device_train_batch_size="96" \
    --per_device_eval_batch_size="96" \
    --dtype="float16" \
    --learning_rate="5e-4" --warmup_steps="2000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="20" \
    --logging_steps="100" \
    --save_steps="2500" \
    --eval_steps="2500"
```

More documentation coming soon.


# Acknowledgement
Adopted from https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling
