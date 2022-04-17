# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Pretrain GPT2"""

import json
import os
import torch

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from megatron import get_args
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron import print_rank_0
from megatron.model import GPT2Model
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import reduce_losses, get_parameters_in_billions
from megatron.data.gpt2_dataset import build_train_valid_test_datasets


def moe_parser(parser):
    #data
    # cuda
    # parser.add_argument('--with_cuda',
    #                     default=False,
    #                     action='store_true',
    #                     help='use CPU in case there\'s no GPU support')
    # parser.add_argument('--use_ema',
    #                     default=False,
    #                     action='store_true',
    #                     help='whether use exponential moving average')

    # train
    # parser.add_argument('-b',
    #                     '--batch_size',
    #                     default=32,
    #                     type=int,
    #                     help='mini-batch size (default: 32)')
    # parser.add_argument('-e',
    #                     '--epochs',
    #                     default=30,
    #                     type=int,
    #                     help='number of total epochs (default: 30)')
    # parser.add_argument('--local_rank',
    #                     type=int,
    #                     default=-1,
    #                     help='local rank passed from distributed launcher')
    #
    # parser.add_argument('--log-interval',
    #                     type=int,
    #                     default=2000,
    #                     help="output logging information at a given interval")
    group = parser.add_argument_group(title='MOE')
    group.add_argument("--vocab-size",
                       default=51200,
                       type=int,
                       help="vocabulary size")
    group.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')
    group.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')
    group.add_argument('--num-experts',
                        default=1,
                        type=int,
                        help='(moe) number of total experts')
    group.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')
    group.add_argument(
        '--min-capacity',
        default=0,
        type=int,
        help=
        '(moe) minimum capacity of an expert regardless of the capacity_factor'
    )
    group.add_argument(
        '--noisy-gate-policy',
        default=None,
        type=str,
        help=
        '(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'
    )
    group.add_argument(
        '--moe-param-group',
        default=False,
        action='store_true',
        help=
        '(moe) create separate moe param groups, required when using ZeRO w. MoE'
    )
    group.add_argument(
        '--output_name',
        default="none",
        help="where to save results."
    )
    return parser


def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    see_memory_usage(f"Before Building Model", force=True)
    args = get_args()

    args.padded_vocab_size = int(os.environ.get("PYTHON_VOCAB_SIZE", 25600))

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device=='none' else args.remote_device,
                             config=args.deepspeed_config,
                             enabled=args.zero_stage==3):
        model = GPT2Model(num_tokentypes=0, parallel_output=True)
    see_memory_usage(f"After Building Model", force=True)

    if mpu.get_data_parallel_rank() == 0:
        billion_params = get_parameters_in_billions(model)
        print(f' > number of parameters on model parallel rank {mpu.get_model_parallel_rank()}\
            {round(billion_params, 3)} Billion',
            flush=True)

    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()

    # Hack for our vocab_size modification
    tokens_ = (tokens_.float() / args.padded_vocab_size).long()
    tokenizer_eod = args.padded_vocab_size - 1

    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer_eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, model, curriculum_learning=False):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch generator').stop()
    # Forward model.
    losses = model(tokens, position_ids, attention_mask, labels=labels)
    if curriculum_learning and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    reduced_loss = reduce_losses([loss])

    return loss, {'lm loss': reduced_loss[0]}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT2 ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT2 datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             extra_args_provider=moe_parser)
    args = get_args()
    rank = torch.distributed.get_rank()
    if rank == 0:
        import numpy as np
        from util import compute_moe_parameter_count, compute_moe_tflops, write_tsv
        from megatron.training import step_latencies
        GB = 1 << 30

        args = get_args()
        seq_len = args.seq_length
        num_layers = args.num_layers
        hidden_size = args.hidden_size
        num_heads = args.num_attention_heads
        num_experts = args.num_experts
        vocab_size = args.padded_vocab_size
        mlp_factor = 8
        if args.deepspeed:
            num_micro_batches = json.load(open(
                args.deepspeed_config))["gradient_accumulation_steps"]
        else:
            num_micro_batches = 1
        batch_size = args.batch_size * mpu.get_data_parallel_world_size() * num_micro_batches
        warmup_iter = 2

        alloc_mem = torch.cuda.max_memory_allocated(0)
        latencies = np.array(step_latencies[warmup_iter * num_micro_batches:])\
                    .reshape((-1, num_micro_batches)).sum(axis=-1)

        param_count = compute_moe_parameter_count(
            num_layers, hidden_size, vocab_size, num_experts, mlp_factor=mlp_factor)

        expert_group_size = batch_size * seq_len // num_micro_batches \
                            // mpu.get_data_parallel_world_size()

        tflops = compute_moe_tflops(batch_size, seq_len, num_layers,
                                    hidden_size, expert_group_size,
                                    vocab_size, num_experts,
                                    torch.distributed.get_world_size(),
                                    np.mean(latencies), mlp_factor=mlp_factor)
        tflops_ckpt = compute_moe_tflops(batch_size, seq_len, num_layers,
                                         hidden_size, expert_group_size ,
                                         vocab_size, num_experts, torch.distributed.get_world_size(),
                                         np.mean(latencies), mlp_factor=mlp_factor,
                                         checkpoint_activations=True)
        model_config = (batch_size, seq_len, hidden_size, num_layers, num_heads, num_experts)
        parallel_config = (mpu.get_data_parallel_world_size(),
                           mpu.get_model_parallel_world_size(),
                           1,
                           args.ep_world_size)

        # Log results
        heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape", "#Microbatch",
                 "Force DP", "Remat", "Mean Time", "Std Time", "#Params", "TFLOPs", "TFLOPs (ckpt)",
                 "Peak Mem"]
        values = ["MOE", str(model_config), str(parallel_config),
                  "N/A", str(num_micro_batches), "N/A",
                  str(args.checkpoint_activations), f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                  f"{param_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                  f"{alloc_mem/GB:5.3f}G"]
        write_tsv(heads, values,f"moe_deepspeed_{args.output_name}_rank{rank}.tsv")
