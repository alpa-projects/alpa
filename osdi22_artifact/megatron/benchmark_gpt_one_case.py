import sys
sys.path.append("../../benchmark/")
import time
from functools import partial

import numpy as np
import torch
from benchmark.util import write_tsv, benchmark_func, \
    compute_gpt_tflops, compute_gpt_parameter_count

from megatron import mpu, initialize_megatron, get_args, get_timers
from megatron.model import GPTModel
from megatron.model import ModelType
from megatron.training import train_step, setup_model_and_optimizer

GB = 1024 ** 3


def get_gpt_functions():
    args = get_args()
    micro_batch_size = args.micro_batch_size
    seq_len = args.encoder_seq_length

    def model_provider(pre_process=True, post_process=True):
        model = GPTModel(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )
        return model

    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # Reduce loss for logging.
        #averaged_loss = average_losses_across_data_parallel_group([loss])
        averaged_loss = [0]
        return loss, {'lm loss': averaged_loss[0]}

    tokens = torch.ones((micro_batch_size, seq_len)).cuda().long()
    labels = torch.ones((micro_batch_size, seq_len)).cuda().long()
    loss_mask = torch.ones((micro_batch_size, seq_len)).cuda().int()
    attention_mask = \
        torch.ones(micro_batch_size, 1, seq_len, seq_len).cuda().bool()
    position_ids = torch.ones((micro_batch_size, seq_len)).cuda().long()

    def forward_step(data_iterator, model):
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)
        return output_tensor, partial(loss_func, loss_mask)

    return model_provider, loss_func, forward_step


def benchmark_gpt_bert_one_case(benchmark_case, output_file_name):
    # Model configs
    model_type, global_batch_size, seq_len, hidden_size, num_layers, num_heads, \
    vocab_size, dp_size, tensor_mp_size, p_dim0, p_dim1, pipeline_mp_size, \
    num_micro_batches, force_dp,  checkpoint_activations, _, _, _ \
        = benchmark_case

    num_gpus = dp_size * tensor_mp_size * pipeline_mp_size
    assert global_batch_size % (dp_size * num_micro_batches) == 0
    micro_batch_size = global_batch_size // dp_size // num_micro_batches

    # always use local DDP
    ddp_impl = True

    # Parallel configs
    # Initialize megatron
    sys.argv += ["--micro-batch-size", str(micro_batch_size)]
    sys.argv += ["--tensor-model-parallel-size", str(tensor_mp_size)]
    sys.argv += ["--pipeline-model-parallel-size", str(pipeline_mp_size)]
    sys.argv += ["--global-batch-size", str(global_batch_size)]
    sys.argv += ["--num-layers", str(num_layers)]
    sys.argv += ["--hidden-size", str(hidden_size)]
    sys.argv += ["--num-attention-heads", str(num_heads)]
    sys.argv += ["--seq-length", str(seq_len)]
    sys.argv += ["--max-position-embeddings", str(seq_len)]
    sys.argv += ["--optimizer", "adam"]
    sys.argv += ["--train-iters", "100"]
    sys.argv += ["--lr", "0.00015"]
    sys.argv += ["--bert-no-binary-head"]
    sys.argv += ["--DDP-impl", "local" if ddp_impl else "torch"]
    sys.argv += ["--fp16"]
    sys.argv += ["--loss-scale", "8"]
    if checkpoint_activations:
        sys.argv += ["--checkpoint-activations"]

    initialize_megatron()
    args = get_args()
    args.padded_vocab_size = vocab_size
    rank = torch.distributed.get_rank()

    # Check initialization
    assert dp_size == mpu.get_data_parallel_world_size()
    assert tensor_mp_size == mpu.get_tensor_model_parallel_world_size()
    assert pipeline_mp_size == mpu.get_pipeline_model_parallel_world_size()

    # Build model
    if model_type == "gpt":
        model_provider, loss_func, forward_step = get_gpt_functions()
    else:
        raise ValueError("This benchmark code can only support gpt.")

    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider,
                                                               model_type=ModelType.encoder_or_decoder)

    parameter_count = compute_gpt_parameter_count(
        num_layers, hidden_size, vocab_size)

    def run_func():
        train_step(forward_step, None, model, optimizer, lr_scheduler)

    # Warmup and reset timers
    run_func()
    timers = get_timers()
    names = list(timers.timers.keys())
    for name in names:
        timers(name).reset()

    # Benchmark step time
    repeat = 2
    number = 1
    costs = benchmark_func(run_func, sync_func=None,
                           warmup=0, repeat=repeat, number=number)
    timers.log(names, normalizer=repeat * number)

    # Print results
    if rank == 0:
        tflops_ckpt = compute_gpt_tflops(global_batch_size, seq_len, num_layers,
                                         hidden_size, vocab_size,
                                         torch.distributed.get_world_size(),
                                         np.mean(costs), True)
        num_hosts = num_gpus // 8 + 1
        num_devices_per_host = num_gpus % 8 if num_gpus <= 8 else 8
        heads = ["exp_name", "instance", "num_hosts", "num_devices_per_host", "model_name", "method", "value", "time_stamp"]
        values = ["e2e", "p3.16", num_hosts, num_devices_per_host, "gpt", "megatron",
                  str({"tflops": tflops_ckpt, "parameter_count": parameter_count / (10 ** 9)}), time.time()]
        write_tsv(heads, values, f"{output_file_name}.tsv")


if __name__ == "__main__":
    case = eval(sys.argv[-2])
    output_file_name = sys.argv[-1]
    del sys.argv[-1]
    del sys.argv[-1]
    benchmark_gpt_bert_one_case(case, output_file_name)
