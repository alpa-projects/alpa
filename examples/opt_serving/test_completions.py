import argparse

from client import Client


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str)
    args = parser.parse_args()

    client = Client(args.url)
    ret = client.completions(
        ["Paris is the capital city of",
         "Computer science is the study of"]
    )
    print(ret)
