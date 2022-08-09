"""Adam optimizer"""
import copy

import torch


def adam(lr=1e-4):
    """torchoptim.adam(**adam_config)(params)
        Factory that generates functional version of Adam optimizer.
        Implementation has no in-place op and no data-dependent control flow.

        Returns:
            - `optim_func`: a function that:
                - takes (`params`, `optim_state`, `params_grad`) as input
                - returns (`params`, `optim_state`)
                  after applying Adam algorithm
            - `optim_state_init_func`: a function that:
                - takes `optim_state` as input
                - returns `optim_state` which is Adam optimizer state
            - `optim_state`: tracked state (shape-only) of Adam optimizer.
    """

    # TODO FIXME: properly implement Adam optimizer

    def optim_gen(params):

        def optim_func(params, optim_state, params_grad):
            for k in params:
                params[k] = params[k] + params_grad[k] * lr
                optim_state[k] = optim_state[k] + params_grad[k]
            return params, optim_state

        optim_state = copy.deepcopy(params)

        def optim_state_init_func(optim_state):
            new_state = {}
            for k, v in optim_state.items():
                new_state[k] = torch.full_like(v, 0.0)
            return new_state

        return optim_func, optim_state_init_func, optim_state

    return optim_gen
