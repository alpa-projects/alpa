# TODO:
# 1. Import Llama Model Definition(done);
# 2. Import Manual partition spec(done);
# 3. Import Fastchat dataset(done);
# 4. Weight Conversion(done);
# 5. Distributed load/store.
# 6. wandb support

#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pre-training/Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
import functools
from pathlib import Path
from typing import Callable, Optional

import datasets
import numpy as np
from datasets import Dataset
from tqdm import tqdm

import alpa
from alpa.model.model_util import DynamicScale, TrainState
from alpa import ManualShardingOption
import jax
from jax.experimental.pjit import PartitionSpec
import jax.numpy as jnp
import optax
import transformers
from transformers.utils import get_full_repo_name, send_example_telemetry
import tensorflow as tf
from flax import traverse_util
from optax import tree_map_params
from huggingface_hub import Repository
from transformers import (
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    is_tensorboard_available,
    set_seed,
)

alpa.init(cluster="ray")

tf.config.experimental.set_visible_devices([], 'GPU')

from EasyLM.models.llama.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLM
)

from hf_datasets import make_supervised_data_module
from hf_jax_conversion import hf_to_jax_weight
from monkey_patch import do_monkey_patch

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
IGNORE_TOKEN_ID = -100


@dataclass
class TrainingArguments:
    """A subset of Huggingface's training arguments"""
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    num_micro_batches: int = field(default=1, metadata={"help": "The number of micro batches for gradient accumulation."})
    operator_parallel: int = field(default=1, metadata={"help": "The degree of operator model parallelism."})
    pipeline_parallel: int = field(default=1, metadata={"help": "The degree of pipeline model parallelism."})
    use_remat: bool = field(default=True, metadata={"help": "Whether or not to use gradient rematerilization/gradient checkpointing."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_ratio: float = field(default=0.0, metadata={"help": "Linear warmup over a ratio of overall steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int,
                min_batch_size: int, shuffle: bool = False):
    """
    Returns batches of size `batch_size` from truncated `dataset`, sharded over all local devices.
    Shuffle batches if `shuffle` is `True`.
    """
    if len(dataset) < batch_size:
        assert len(dataset) >= min_batch_size
        batch_size = len(dataset) // min_batch_size * min_batch_size

    data_collator = transformers.DefaultDataCollator("np")
    tf_dataset = dataset.to_tf_dataset(batch_size=batch_size,
                                       columns=dataset.column_names,
                                       collate_fn=data_collator,
                                       shuffle=shuffle,
                                       drop_remainder=True)

    for batch in tf_dataset:
        batch = {k: v._numpy() for k, v in batch.items()}
        yield batch


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = alpa.util.get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, warmup_ratio: float, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def llama_manual_sharding(num_layers, state: TrainState):
    # TODO: when rebased to jax 0.4.6, use the tree_map_with_path
    param_partition = {
        'transformer': {
            'wte': {'embedding': PartitionSpec("mp", None)},
            'ln_f': {'kernel': PartitionSpec(None)},
            'h': {
                '%d' % (layer): {
                    'attention': {
                        # TODO: check whether we need the transpose or not
                        'wq': {'kernel': PartitionSpec(None, "mp")},
                        'wk': {'kernel': PartitionSpec(None, "mp")},
                        'wv': {'kernel': PartitionSpec(None, "mp")},
                        'wo': {'kernel': PartitionSpec("mp", None)},
                    },
                    'feed_forward': {
                        'w1': {'kernel': PartitionSpec(None, "mp")},
                        'w2': {'kernel': PartitionSpec("mp", None)},
                        'w3': {'kernel': PartitionSpec(None, "mp")},
                    },
                    'attention_norm': {'kernel': PartitionSpec(None)},
                    'ffn_norm': {'kernel': PartitionSpec(None)},
                }
            for layer in range(num_layers)},
        },
        'lm_head': {'kernel': PartitionSpec(None, "mp")},
    }
    replicate = lambda x : jax.tree_util.tree_map(lambda _: PartitionSpec(None), x)
    opt_state = tree_map_params(state.tx, lambda _, spec: spec, state.opt_state,
                                param_partition, transform_non_params=lambda _: PartitionSpec(None))
    manual_partition = TrainState(
        step=PartitionSpec(None),
        params=param_partition,
        master_copy=param_partition if state.master_copy else None,
        dynamic_scale=replicate(state.dynamic_scale),
        tx=state.tx,
        apply_fn=state.apply_fn,
        opt_state=opt_state)
    return manual_partition


# TODO: smoothing factor
def loss_fn(logits, labels, ignore_indices):
    # Shift logits
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    # Handle the ignore index: compute the valid first
    valid = jnp.full(shift_labels.shape, True)
    for ignore_index in ignore_indices:
        new_valid = jnp.not_equal(shift_labels, ignore_index)
        valid = jnp.logical_and(valid, new_valid)
    valid_len = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    # OneHot and mask the ignore index. For ignore_index(-100), the whole line
    # in the output would be 0.
    one_hot_labels = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
    # Compute the softmax loss
    log_p = jax.nn.log_softmax(shift_logits, axis=-1)
    # (bs, seq_len, vocab) -> (bs, seq_len)
    cross_entropy = jnp.sum(one_hot_labels * log_p, axis=-1)
    loss = -jnp.mean(jnp.sum(cross_entropy, axis=-1) / valid_len)
    return loss


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args, framework="flax")

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Handle the repository creation
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name, token=training_args.hub_token
            )
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(training_args.output_dir, clone_from=repo_name)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = LLaMAConfig.load_config('7b')
    if model_args.dtype == "float16":
        dtype = jnp.float16
    elif model_args.dtype == "float32":
        dtype = jnp.float32
    elif model_args.dtype == "bfloat16":
        dtype = jnp.bfloat16
    else:
        raise ValueError(f"{model_args.dtype} unsupported")
    # TODO: set the correct remat policy.

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=config.max_sequence_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    config.update(dict(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ))

    # TODO(yonghao): don't init weight when loaded somewhere
    dummy_input_shape = (4, config.max_sequence_length)
    # Monkey patch the model's init to init_dummy
    do_monkey_patch()
    model = FlaxLLaMAForCausalLM(config, dummy_input_shape, dtype=dtype)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )
    loaded_params = hf_to_jax_weight(hf_model)

    #  Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    data_module = make_supervised_data_module(tokenizer, data_args.dataset_name, IGNORE_TOKEN_ID)


    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    logger.info("***** Build dataset *****")

    if training_args.do_train:
        if "train_dataset" not in data_module:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = data_module["train_dataset"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "eval_dataset" not in data_module:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = data_module["eval_dataset"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Adjust batch size and num_micro_batches for small datasets
    num_devices = alpa.get_global_num_devices()
    train_min_batch_size = (num_devices // training_args.operator_parallel //
                            training_args.pipeline_parallel * training_args.num_micro_batches)
    eval_num_micro_batches = training_args.num_micro_batches
    eval_min_batch_size = (num_devices // training_args.operator_parallel //
                           training_args.pipeline_parallel * eval_num_micro_batches)
    while training_args.do_eval and (len(eval_dataset) < eval_min_batch_size):
        eval_num_micro_batches //= 2
        eval_min_batch_size = (num_devices // training_args.operator_parallel //
                               training_args.pipeline_parallel * eval_num_micro_batches)

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * num_devices
    eval_batch_size = int(training_args.per_device_eval_batch_size) * num_devices
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule
    cosine_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_ratio,
        training_args.learning_rate,
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxGPT2.
    # For other models, one should correct the layer norm parameter naming
    # accordingly.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in [("ln_1", "scale"), ("ln_2", "scale"), ("ln_f", "scale")])
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    if training_args.adafactor:
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=cosine_decay_lr_schedule_fn,
        )
    else:
        # A tmp hack for llama finetune. Remove it either:
        # 1) rebase to jax 0.4 and use tree_util's mask with path for partition spec;
        # 2) optax fixes the issue of symbolic exec with decay mask fn.
        if training_args.weight_decay == 0.0:
            decay_mask_fn = None
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=cosine_decay_lr_schedule_fn,
                b1=training_args.adam_beta1,
                b2=training_args.adam_beta2,
                eps=training_args.adam_epsilon,
                weight_decay=training_args.weight_decay,
                mask=decay_mask_fn)
        )

    # Setup train state
    if model_args.dtype == "float16":
        use_master_copy = True
        dynamic_scale = DynamicScale()
        # Fix a bug in huggingface's implementation (https://github.com/huggingface/transformers/pull/18462)
        alpa.global_config.flax_always_use_fp16_embedding = True
    else:
        use_master_copy = dynamic_scale = None
    state = TrainState.create(apply_fn=model.__call__, params=loaded_params, tx=optimizer,
                              dynamic_scale=dynamic_scale, use_master_copy=use_master_copy)

    # Manual partition spec
    state_manual_sharding = llama_manual_sharding(config.num_hidden_layers, state)
    ms_option = ManualShardingOption(
        ("dp", "mp"), in_axis_resources=(state_manual_sharding, PartitionSpec("dp", None)))
    ignore_ids = (IGNORE_TOKEN_ID, tokenizer.pad_token_id)

    # Define gradient update step fn
    def train_step(state, batch):

        def compute_loss(params):
            labels = batch.pop("labels")
            # Currently we don't support non-deterministic training with remat,
            # so train=False. This arg has no other impact.
            logits = state.apply_fn(**batch, params=params, train=False)[0]
            loss = loss_fn(logits, labels, ignore_ids)
            return loss

        dynamic_scale = state.dynamic_scale
        if dynamic_scale:
            grad_fn = dynamic_scale.value_and_grad(compute_loss)
            dynamic_scale, is_fin, loss, grads = grad_fn(state.params)
        else:
            grad_fn = alpa.value_and_grad(compute_loss)
            loss, grads = grad_fn(state.params)

        new_state = state.apply_gradients(grads=grads)

        if dynamic_scale:
            new_state = new_state.replace(
                opt_state=jax.tree_map(
                    functools.partial(jnp.where, is_fin),
                    new_state.opt_state, state.opt_state),
                params=jax.tree_map(
                    functools.partial(jnp.where, is_fin),
                    new_state.params, state.params),
                master_copy=jax.tree_map(
                    functools.partial(jnp.where, is_fin),
                    new_state.master_copy, state.master_copy),
                dynamic_scale=dynamic_scale)

        metrics = {"loss": loss, "learning_rate": cosine_decay_lr_schedule_fn(state.step)}

        return new_state, metrics

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, deterministic=True)[0]
        loss = loss_fn(logits, labels, IGNORE_TOKEN_ID)

        # summarize metrics
        metrics = {"loss": loss}
        return metrics

    # Create parallel version of the train and eval step
    method = alpa.get_3d_parallel_method(
            num_micro_batches=training_args.num_micro_batches,
            data_parallel=-1,
            operator_parallel=training_args.operator_parallel,
            pipeline_parallel=training_args.pipeline_parallel,
            manual_layer_num=config.num_hidden_layers,
            manual_sharding_option=ms_option)

    p_train_step = alpa.parallelize(train_step,
                                    method=method,
                                    donate_argnums=(0,))
    p_eval_step = alpa.parallelize(eval_step,
                                   method=alpa.FollowParallel(
                                       p_train_step, num_micro_batches=eval_num_micro_batches))

    dump_debug_info_train_step = dump_debug_info_eval_step = True

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Batch size per device (w. accumulation) = {training_args.per_device_train_batch_size}")
    logger.info(f"  Global train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    train_metrics = []
    epochs = tqdm(range(num_epochs), desc="Epoch ... ", position=0)

    step_ct = 0
    last_time = time.time()

    epochs.write("Initial compilation. This might take some minutes...")

    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        train_loader = data_loader(input_rng, train_dataset, train_batch_size,
                                   train_min_batch_size, shuffle=True)
        steps_per_epoch = len(train_dataset) // train_batch_size
        # train
        for step in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
            batch = next(train_loader)
            batch["position_ids"] = (batch["attention_mask"].cumsum(axis=1) *
                                     batch["attention_mask"]) - 1
            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)

            cur_step = epoch * (len(train_dataset) // train_batch_size) + step

            if dump_debug_info_train_step:
                dump_debug_info_train_step = False
                executable = p_train_step.get_last_executable()
                executable.sync()
                executable.dump_debug_info("alpa_debug_info")
                epochs.write(f"Initial compilation completed. "
                             f"Time elapsed: {time.time() - train_start:.2f} s")

            step_ct += 1
            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                executable.sync()
                latency = (time.time() - last_time) / step_ct
                throughput_tokens = np.prod(batch["input_ids"].shape) / latency
                throughput_tflops = alpa.util.compute_gpt_tflops(
                    batch_size=batch["input_ids"].shape[0],
                    seq_len=batch["input_ids"].shape[1],
                    num_layers=config.num_hidden_layers,
                    hidden_size=config.hidden_size,
                    vocab_size=config.vocab_size,
                    num_gpus=alpa.get_global_num_devices(),
                    latency=latency,
                    checkpoint_activations=True,
                    intermediate_size=config.intermediate_size)
                step_ct = 0

                # Save metrics
                train_time += time.time() - train_start
                if has_tensorboard:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                train_metric = jax.tree_map(np.mean, train_metric)

                epochs.write(
                    f"Step... {cur_step} | "
                    f"Loss: {train_metric['loss'].mean():.4f}, "
                    f"Learning Rate: {train_metric['learning_rate'].mean()}, "
                    f"Throughput: {throughput_tokens:.2f} token/s, "
                    f"{throughput_tflops:.2f} TFLOP/s"
                )

                train_metrics = []
                last_time = time.time()

            if training_args.do_eval and cur_step % training_args.eval_steps == 0 and cur_step > 0:
                # ======================== Evaluating ==============================
                eval_metrics = []
                eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size,
                                          eval_min_batch_size)
                eval_steps = max(len(eval_dataset) // eval_batch_size, 1)
                for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
                    # Model forward
                    batch = next(eval_loader)
                    batch["position_ids"] = (batch["attention_mask"].cumsum(axis=1) *
                                             batch["attention_mask"]) - 1
                    metrics = p_eval_step(state.params, batch)
                    eval_metrics.append(metrics)

                    if dump_debug_info_eval_step:
                        dump_debug_info_eval_step = False
                        executable = p_eval_step.get_last_executable()
                        executable.dump_debug_info("alpa_debug_info")

                # normalize eval metrics
                eval_metrics = alpa.util.get_metrics(eval_metrics)
                eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

                try:
                    eval_metrics["perplexity"] = math.exp(eval_metrics["loss"])
                except OverflowError:
                    eval_metrics["perplexity"] = float("inf")

                # Print metrics and update progress bar
                desc = (
                    f"Step... ({cur_step} | Eval Loss: {eval_metrics['loss']} | Eval Perplexity:"
                    f" {eval_metrics['perplexity']})"
                )
                epochs.write(desc)

                # Save metrics
                if has_tensorboard:
                    write_eval_metric(summary_writer, eval_metrics, cur_step)

            if cur_step % training_args.save_steps == 0 and cur_step > 0:
                # save checkpoint after each epoch and push checkpoint to the hub
                epochs.write("\nSave checkpoint...")
                alpa.prefetch(state.params)
                params = alpa.util.map_to_nparray(state.params)
                model.save_pretrained(training_args.output_dir, params=params)
                tokenizer.save_pretrained(training_args.output_dir)
                if training_args.push_to_hub:
                    repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}", blocking=False)

    # Eval after training
    if training_args.do_eval:
        eval_metrics = []
        eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size,
                                  eval_min_batch_size)
        eval_steps = max(len(eval_dataset) // eval_batch_size, 1)
        for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
            # Model forward
            batch = next(eval_loader)
            batch["position_ids"] = (batch["attention_mask"].cumsum(axis=1) *
                                     batch["attention_mask"]) - 1
            metrics = p_eval_step(state.params, batch)
            eval_metrics.append(metrics)

        # normalize eval metrics
        eval_metrics = alpa.util.get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(lambda x: jnp.mean(x).item(), eval_metrics)

        try:
            eval_metrics["perplexity"] = math.exp(eval_metrics["loss"])
        except OverflowError:
            eval_metrics["perplexity"] = float("inf")

        eval_metrics = {f"eval_{metric_name}": value for metric_name, value in eval_metrics.items()}
        path = os.path.join(training_args.output_dir, "eval_results.json")
        with open(path, "w") as f:
            json.dump(eval_metrics, f, indent=4, sort_keys=True)

    # Save the final model
    epochs.write("\nSave the final model...")
    alpa.prefetch(state.params)
    params = alpa.util.map_to_nparray(state.params)
    model.save_pretrained(training_args.output_dir, params=params)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
