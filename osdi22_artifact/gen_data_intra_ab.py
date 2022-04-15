import argparse

from alpa.util import run_cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    if args.model is None:
        cmd = "python3 exp_intra_ablation.py"
    else:
        cmd = f"python3 exp_intra_ablation.py --model {args.model}"

    run_cmd("cd ../benchmark/alpa && " + cmd)

    run_cmd("cp ../benchmark/alpa/results_intra_ablation.tsv .")
