import argparse

from alpa_serve.profiling import ProfilingDatabase

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="benchmark_results.tsv")
    parser.add_argument("--output", type=str, default="profiling_result.pkl")
    args = parser.parse_args()

    database = ProfilingDatabase(args.output, True)
    database.update_from_csv(args.input)
    database.materialize()
