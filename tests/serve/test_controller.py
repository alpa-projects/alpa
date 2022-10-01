"""Test alpa.serve controller."""
import unittest

import numpy as np
import ray
import requests
from tokenizers import Tokenizer

from alpa.api import parallelize
from alpa.serve.controller import run_controller


class EchoModel:

    async def handle_request(self, request):
        obj = await request.json()
        return obj


class AddOneModel:

    def __init__(self):

        def func(x):
            return x + 1

        self.add_one = parallelize(func)

    async def handle_request(self, request):
        obj = await request.json()
        x = np.array(obj["x"])
        y = self.add_one(x)
        return await y.to_np_async()


class TokenizerModel:

    def __init__(self):
        self.tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

    async def handle_request(self, request):
        obj = await request.json()
        x = obj["input"]
        y = self.tokenizer.encode(x)
        return y.ids


class ControllerTest(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto", namespace="alpa_serve")

    def tearDown(self):
        ray.shutdown()

    def test_query(self):
        controller = run_controller("localhost")

        info = ray.get(controller.get_info.remote())
        host, port, root_path = info["host"], info["port"], info["root_path"]

        controller.register_model.remote("echo", EchoModel)
        controller.register_model.remote("add_one", AddOneModel)
        controller.register_model.remote("tokenizer", TokenizerModel)
        group_id = 0
        controller.launch_mesh_group_manager.remote(group_id, [1, 4])
        a = controller.create_replica.remote("echo", group_id)
        b = controller.create_replica.remote("add_one", group_id)
        c = controller.create_replica.remote("tokenizer", group_id)

        ray.get([a, b, c])
        url = f"http://{host}:{port}{root_path}"

        json = {
            "model": "echo",
            "task": "completions",
            "input": "Paris is the capital city of",
        }
        resp = requests.post(url=url, json=json)
        assert resp.json() == json

        resp = requests.post(url=url,
                             json={
                                 "model": "add_one",
                                 "x": list(range(16)),
                             })
        assert resp.text == str(list(range(1, 17)))

        src = "Paris is the capital city of"
        resp = requests.post(url=url, json={"model": "tokenizer", "input": src})
        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        assert resp.text == str(tokenizer.encode(src).ids)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(ControllerTest("test_query"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
