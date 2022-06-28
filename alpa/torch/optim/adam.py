"""Adam optimizer"""
import copy

import torch


def adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
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

    # TODO: this Adam impl might not be 100% correct

    def optim_gen(params):

        def optim_func(params, optim_state, params_grad):
            beta1, beta2 = betas
            step = optim_state["step"]
            for k in params:
                param = params[k]
                grad = params_grad[k]
                exp_avg = optim_state["exp_avgs"][k]
                exp_avg_sq = optim_state["exp_avg_sqs"][k]

                bias_correction1 = 1 - torch.pow(beta1, step)
                bias_correction2 = 1 - torch.pow(beta2, step)

                if weight_decay != 0:
                    grad = grad + weight_decay * param

                # Decay the first and second moment running average coefficient
                exp_avg = exp_avg * beta1 + (1 - beta1) * grad
                exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * grad * grad.conj()
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)) + eps

                step_size = lr / bias_correction1
                param = param + (-step_size * exp_avg / denom)

                params[k] = param
                optim_state["exp_avgs"][k] = exp_avg
                optim_state["exp_avg_sqs"][k] = exp_avg_sq
            optim_state["step"] = step + 1
            return params, optim_state

        optim_state = {
            "exp_avgs": {k: torch.empty(v.shape, device="meta") for k, v in params.items()},
            "exp_avg_sqs": {k: torch.empty(v.shape, device="meta") for k, v in params.items()},
            "step": torch.empty(1, device="meta"),
        }

        def optim_state_init_func(optim_state):
            new_state = {}
            new_state["exp_avgs"] = {k: torch.zeros_like(v) for k, v in optim_state["exp_avgs"].items()}
            new_state["exp_avg_sqs"] = {k: torch.zeros_like(v) for k, v in optim_state["exp_avg_sqs"].items()}
            new_state["step"] = torch.full_like(optim_state["step"], 1),
            return new_state

        return optim_func, optim_state_init_func, optim_state

    return optim_gen
