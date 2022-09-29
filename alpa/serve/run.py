"""Run a controller."""
import argparse

import ray

from alpa.serve.controller import run_controller

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int)
    parser.add_argument("--root-path", type=str, default="/")
    args = parser.parse_args()

    ray.init(address="auto", namespace="alpa_serve")
    controller = run_controller(args.host, args.port, args.root_path)

    while True:
        pass
