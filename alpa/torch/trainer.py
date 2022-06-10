"""Example trainer that runs an SGD training loop
"""
import functools

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


def train_torch_module(pt_module_gen, weight_init_func, dataloader, loss_func,
                       optim_gen, parallel_method):
    for mode in ["local", "dist"]:
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

        # This sets both `torch.manual_seed`
        # and
        # `alpa.manual_seed` (if dist mode) under the hood.
        atorch.manual_seed(seed=123)

        # Assume we have a dataloader that supports `peek` function
        # (i.e. look at next batch but don't advance the pointer)
        pt_inputs, pt_targets = dataloader[0]  # dataloader.peek()

        # Create shape-only version of input
        pt_inputs, pt_targets = atorch.meta_like(pt_inputs, pt_targets)

        # Functionalize the PyTorch model
        pt_module = atorch.meta_init(pt_module_gen)
        module_func, params, bufs, name_map = atorch.functionalize(pt_module)

        optim_func, optim_state, optim_state_init_func = optim_gen(params)

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
        def sgd_train_func(module_func, loss_func, optim_func, params, bufs,
                           optim_state, batch):
            inputs = batch[0]
            targets = batch[1]

            # wrap forward pass + loss computation in a function
            def compute_loss(params, bufs, inputs, targets):
                # do forward pass
                bufs, out = module_func(params, bufs, inputs)

                # do some debugging when in local mode
                if atorch.mode() == "local":
                    print("out: ", out)

                # do loss computation
                loss_value = loss_func(out, targets)
                return loss_value, bufs

            if atorch.mode() == "dist" and isinstance(parallel_method,
                                                      alpa.PipeshardParallel):
                compute_loss = alpa.automatic_layer_construction(
                    layer_num=2)(compute_loss)

            # do model forward + backward pass
            (loss_value, bufs), params_grad = atorch.value_and_grad(
                compute_loss, has_aux=True)(params, bufs, inputs, targets)

            # do optimizer step
            params, optim_state = optim_func(params, params_grad, optim_state)

            return params, bufs, optim_state, loss_value

        # pylint: disable=cell-var-from-loop
        train_func = functools.partial(sgd_train_func, module_func, loss_func,
                                       optim_func)

        if atorch.mode() == "dist":
            train_func = alpa.parallelize(
                train_func,
                method=parallel_method,
                # NOTE: assumes the 4th argument is input batch
                batch_argnums=(3,),
                # NOTE: preserves mem addr and sharding spec for first 3 args
                donate_argnums=(0, 1, 2),
            )

        def iter_func(params, bufs, optim_state, pt_batch):
            params, bufs, optim_state, loss_value = train_func(
                params,
                bufs,
                optim_state,
                atorch.to_format(atorch.mode(), pt_batch),
            )
            # Show that outputs are in Alpa format
            # (no need in actual training code)
            atorch.assert_format(atorch.mode(), params, bufs, optim_state,
                                 loss_value)
            return params, bufs, optim_state, loss_value

        # Generate sharding plan based on shape-only tensors.
        # Only needed for dist training.
        # TODO: improve after https://github.com/alpa-projects/alpa/pull/489
        if atorch.mode() == "dist":
            # pylint: disable=line-too-long
            alpa.global_env.global_config.use_dummy_value_for_benchmarking = True
            params, bufs, optim_state, _ = iter_func(params, bufs, optim_state,
                                                     (pt_inputs, pt_targets))
            # pylint: disable=line-too-long
            alpa.global_env.global_config.use_dummy_value_for_benchmarking = False

        # Materialize and initialize the weights and optimizer state
        params, bufs, optim_state = atorch.materialize(params, bufs,
                                                       optim_state)
        params, bufs = weight_init_func(pt_module, name_map, params, bufs)
        optim_state = optim_state_init_func(optim_state)

        # Run training loops
        for _, (pt_inputs, pt_targets) in enumerate(dataloader):
            params, bufs, optim_state, loss_value = iter_func(
                params, bufs, optim_state, (pt_inputs, pt_targets))
            # do whatever with the loss value, e.g. plot it on a graph
            print("loss value: ", loss_value)

        if atorch.mode() == "dist":
            alpa.shutdown()
