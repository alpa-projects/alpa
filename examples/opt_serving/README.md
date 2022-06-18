## Requirements
```
pip3 install -r requirement.txt
```

## Get OPT weights

1. You can  download the original OPT weights released by [Metaseq](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT), then use the scripts
[convert_to_numpy_weight.py](scripts/convert_to_numpy_weights.py) to convert it into Alpa-compatible formats. 

2. You can can 
3. 
Put the [weights](https://drive.google.com/file/d/19DyZve_46SkR-kUMNxxg_fOzusHI0KNU/view?usp=sharing) under `/home/ubuntu/opt_weights/125M_np`

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
