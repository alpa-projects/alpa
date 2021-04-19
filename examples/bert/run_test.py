import argparse
import os

from parax.util import run_cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="run_mlm_flax.py")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    args = parser.parse_args()

    cmd = f"python3 {args.script} "\
          f"--model_name_or_path {args.model} "\
           "--dataset_name wikitext "\
           "--dataset_config_name wikitext-2-raw-v1 "\
           "--do_train "\
           "--do_eval "\
           "--output_dir /tmp/test-mlm "\
           "--logging_dir /tmp/test-mlm "\
           "--logging_steps 5 "\
           "--overwrite_output_dir "\
           "--per_device_train_batch_size 8 "\
           "--per_device_eval_batch_size 16 "\

    run_cmd(cmd)

