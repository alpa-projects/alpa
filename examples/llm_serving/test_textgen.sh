# Test the correctness of textgen.py
set -x

python3 textgen.py --model bigscience/bloom-560m
python3 textgen.py --model jax/bloom-560m
python3 textgen.py --model alpa/bloom-560m

python3 textgen.py --model facebook/opt-1.3b
python3 textgen.py --model jax/opt-1.3b
python3 textgen.py --model alpa/opt-1.3b
