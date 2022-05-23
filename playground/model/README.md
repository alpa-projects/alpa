# OPT Demo

## Requirements
- Compile jaxlib with [opt](https://github.com/alpa-projects/tensorflow-alpa/tree/pr-opt) branch.
- OPT weight folders: `/home/ubuntu/opt_weights/125M_np` and `/home/ubuntu/opt_weights/125M_ts`

## Files
- `opt_model.py`: The model definition
- `test_cache.py`: Test single GPU + jax.jit
- `test_parallelize.py`: Test multi GPU + alpa.parallelize
- `test_text_gen.py`: Test alpa.parallelize + huggingface generator
