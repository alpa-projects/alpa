# pylint: disable=line-too-long
"""Example trainer that runs an SGD training loop"""
import functools
from collections import namedtuple

import alpa
import alpa.torch as atorch
# pylint: disable=pointless-string-statement
"""
FAQ: When to use atorch vs. torch?

Answer:
- All `atorch` usage is contained within the trainer code (i.e. this file),
no `atorch` mentions in user code (e.g. test_torch_simple.py).
- No `torch` usage in trainer code. e.g. PyTorch dataloader will be
encapsulated in alpa.torch dataloader (TBD), where we will add features
related to dist dataloading.
"""

TrainState = namedtuple("TrainState", ["params", "bufs", "optim_state"])


def train_torch_module(pt_module_gen, weight_init_func, dataloader, loss_func,
                       optim_gen, parallel_method):
    for mode in ["dist"]:
        # "local": pure PT eager mode on single GPU,
        #     allows print in middle of graph, no dist training
        # "dist": graph mode by lowering PT program to JAX,
        #     doesn't allow print, supports dist training
        # NOTE: as we see below, the two modes can share most of the code.
        atorch.set_mode(mode)
        # Prints verbose log for debugging.
        atorch.debug = True

        if atorch.mode() == "dist":
            alpa.init(cluster="ray")

        # Functionalize the PyTorch model and optimizer
        pt_module = atorch.meta_init(pt_module_gen)
        module_func, params, bufs, name_map = atorch.functionalize(pt_module)
        optim_func, optim_state, optim_state_init_func = optim_gen(params)
        state = TrainState(params, bufs, optim_state)

        # Enable dist mode for all user-defined functions
        if atorch.mode() == "dist":
            (module_func, weight_init_func, loss_func, optim_func,
             optim_state_init_func) = atorch.enable_dist_for_funcs(
                 module_func,
                 weight_init_func,
                 loss_func,
                 optim_func,
                 optim_state_init_func,
             )

        # Define the training loop
        def sgd_train_step(module_func, loss_func, optim_func,
                           state, batch):
            inputs = batch[0]
            targets = batch[1]

            # wrap forward pass + loss computation in a function
            def compute_loss(params, bufs, inputs, targets):
                # do forward pass
                bufs, out = module_func(params, bufs, inputs)

                # do loss computation
                loss_value = loss_func(out, targets)
                return loss_value, bufs

            if atorch.mode() == "dist" and isinstance(parallel_method,
                                                      alpa.PipeshardParallel):
                compute_loss = alpa.automatic_layer_construction(
                    layer_num=2)(compute_loss)

            # do model forward + backward pass
            (loss_value, bufs), params_grad = atorch.value_and_grad(
                compute_loss, has_aux=True)(state.params,
                    state.bufs, inputs, targets)

            # do optimizer step
            params, optim_state = optim_func(state.params, state.optim_state, params_grad)

            return TrainState(params, bufs, optim_state), loss_value

        train_step = functools.partial(sgd_train_step, module_func, loss_func,
                                       optim_func)

        if atorch.mode() == "dist":
            train_step = alpa.parallelize(
                train_step,
                method=parallel_method,
                # NOTE: assumes the 4th argument is input batch
                batch_argnums=(1,),
                # NOTE: preserves mem addr and sharding spec for first argument
                donate_argnums=(0,),
                static_argnums=(),
            )

        if atorch.mode() == "dist":
            # Assume we have a dataloader that supports `peek` function
            # (i.e. look at next batch but don't advance the pointer).
            # Create shape-only version of inputs
            pt_batch = atorch.to_format(atorch.mode(), atorch.meta_like(*dataloader[0]))  # dataloader.peek()
            alpa.global_env.global_config.use_dummy_value_for_benchmarking = True
            state, loss_value = train_step(state, pt_batch)
            alpa.global_env.global_config.use_dummy_value_for_benchmarking = False

        # Materialize and initialize the weights and optimizer state
        params, bufs, optim_state = atorch.materialize(*state)
        params, bufs = weight_init_func(pt_module, name_map, params, bufs)
        optim_state = optim_state_init_func(optim_state)
        state = TrainState(params, bufs, optim_state)

        # Run training loops
        for i, pt_batch in enumerate(dataloader):
            pt_batch = atorch.to_format(atorch.mode(), pt_batch)
            state, loss_value = train_step(state, pt_batch)

            # do whatever with the loss value, e.g. plot it on a graph
            print(f"Iter: {i}, Loss: {float(loss_value):.6f}")

        if atorch.mode() == "dist":
            alpa.shutdown()
