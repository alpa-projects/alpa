## Requirements
```
pip3 install transformers
```

Put [weights](https://drive.google.com/file/d/19DyZve_46SkR-kUMNxxg_fOzusHI0KNU/view?usp=sharing) under `/home/ubuntu/opt_weights/125M_np`

## Run
```
# with pytorch backend
python3 test_text_gen.py --model facebook/opt-125m

# with alpa backend
python3 test_text_gen.py --model alpa/opt-125m
```

## Files
- `test_text_gen.py`: Use opt for text generation.
- `test_cache.py`: Test opt inference with cache and jax.jit.
- `test_parallelize.py`: Test opt inference with cache and alpa.parallelize.
- `opt_model.py`: The model implementation.
