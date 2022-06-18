import unittest

import torch
import alpa.torch.optim as torchoptim
import alpa
from alpa.torch.trainer import train_torch_module


class MyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16)
        self.linear2 = torch.nn.Linear(16, 16)
        self.linear3 = torch.nn.Linear(16, 16)
        self.linear4 = torch.nn.Linear(16, 16)

    def forward(self, input_dict):
        x = input_dict["x"]
        y = input_dict["dict2"]["y"]
        x = self.linear1(x) + y
        # do some debugging when in local mode
        if getattr(torch, "local_mode", True):
            print(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x


def weight_init_func(pt_module, name_map, params, bufs):
    for k, m in pt_module.named_modules():
        if isinstance(m, torch.nn.Linear):
            params[name_map[f"{k}.weight"]] = torch.nn.init.xavier_uniform(
                params[name_map[f"{k}.weight"]])
            params[name_map[f"{k}.bias"]] = torch.nn.init.normal(
                params[name_map[f"{k}.bias"]], std=1e-6)
    return params, bufs


class TorchDictInputTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(123)
        alpa.set_seed(123)

    def test_dict_input(self):
        pt_module_gen = lambda: MyModule()

        dataloader = [
            ({
                "x": torch.randn(8, 16),
                "dict2": {
                    "y": torch.randn(8, 16)
                }
            }, torch.randn(8, 16)),
            ({
                "x": torch.randn(8, 16),
                "dict2": {
                    "y": torch.randn(8, 16)
                }
            }, torch.randn(8, 16)),
        ]
        loss_func = lambda *args, **kwargs: torch.nn.functional.mse_loss(
            *args, **kwargs)
        optim_gen = torchoptim.adam(lr=1e-3)
        parallel_method = alpa.ShardParallel()

        train_torch_module(pt_module_gen, weight_init_func, dataloader,
                           loss_func, optim_gen, parallel_method)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TorchDictInputTest("test_dict_input"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
