import os
import tempfile

import ray

import alpa
from alpa import (init, shutdown, parallelize, DistributedArray,
                  PipeshardParallel, save_checkpoint, restore_checkpoint)
from alpa.device_mesh import get_global_cluster
from alpa.testing import get_bert_layer_train_state_and_step, assert_allclose
from alpa.parallel_method import get_3d_parallel_method


def _get_save_prefix():
    device_cluster = get_global_cluster()
    if len(device_cluster.host_info) > 1:
        raise RuntimeError("The multi-host test requires a mounted EFS! ")
    else:
        # Use tmp dir for the single-host test
        save_prefix = "/tmp/"
    return save_prefix


alpa.init()
ckpt_dir = "/mnt/alpa-opt/alpa/examples/opt_finetune/test_ckpt"
state, batch, train_step = get_bert_layer_train_state_and_step(
    batch_size=16,
    seq_len=8,
    num_layers=2,
    hidden_size=128,
    num_heads=8,
    clip_by_global_norm=False,
    use_dynamic_scale=False,
    add_manual_pipeline_marker=True)

method = PipeshardParallel(num_micro_batches=2, layer_option="manual")

serial_train_step = train_step
parallel_train_step = parallelize(train_step, method=method)
executable = parallel_train_step.get_executable(state, batch)

serial_state = state
parallel_state = state
serial_state = serial_train_step(serial_state, batch)[0]
parallel_state = parallel_train_step(parallel_state, batch)[0]
# assert_allclose(serial_state.params, parallel_state.params, 1e-3, 1e-3)

with tempfile.TemporaryDirectory(prefix="/tmp/") as cache_dir:
    # Save checkpoint
    save_checkpoint(ckpt_dir, parallel_state, 1, cache_dir)

    # Sync all the move workers
    executable.sync_move_workers()

    # Restore checkpoint
    state_ps, _ = executable.get_input_placement_specs()
    load_state = restore_checkpoint(ckpt_dir, 1, state_ps)

    # Run after load
    serial_state = serial_train_step(serial_state, batch)[0]
    load_state = parallel_train_step(load_state, batch)[0]

    # Check results
    assert_allclose(serial_state.params, load_state.params, 1e-3,
                    1e-3)
