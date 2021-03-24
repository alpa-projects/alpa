# BERT Language Model Pretraining
Modified from https://github.com/huggingface/transformers/tree/master/examples/language-modeling

- Run flax
```
python run_test.py --script run_mlm_flax.py
```

- Run pytorch
```
python run_test.py --script run_mlm_pytorch.py
```

- Run auto-parallel
```
python run_test.py --script run_mlm_auto_parallel.py
```
