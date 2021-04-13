# BERT Language Model Pretraining
Modified from https://github.com/huggingface/transformers/tree/master/examples/language-modeling

## Dependency
```
pip3 install datasets transformers torch
```

## Commands

- Run flax
```
python3 run_test.py --script run_mlm_flax.py
```

- Run pytorch
```
python3 run_test.py --script run_mlm_pytorch.py
```

- Run auto-parallel
```
python3 run_test.py --script run_mlm_auto_parallel.py
```
