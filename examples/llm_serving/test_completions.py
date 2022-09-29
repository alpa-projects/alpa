"""
Usage:

python3 test_completions.py --url http://localhost:20001
python3 test_completions.py --url https://opt.alpa.ai --api-key YOUR_KEY
"""
import argparse

from client import Client


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str)
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--model", type=str, default="default")
    args = parser.parse_args()

    client = Client(args.url, api_key=args.api_key, default_model=args.model)
    ret = client.completions(
        ["Paris is the capital city of",
         "Computer science is the study of"]
    )
    print(ret)
