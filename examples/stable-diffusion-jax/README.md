# Setup

## PyTorch Version

GitHub: https://github.com/CompVis/stable-diffusion

Paper: [High-Resolution Image Synthesis With Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Notes

1. Clone the repository and setup

   `git clone https://github.com/CompVis/stable-diffusion.git`

   `cd stable-diffusion`

   `conda env create -f environment.yaml`

   `conda activate ldm`

   **Your pytorch version and CUDA version should match here!**

   You may use 

   ```python
   import torch
   torch.ones((2,3)).cuda()
   ```
   to check.

2. (Alternative)

   You may update the environment of [latent diffusion](https://github.com/CompVis/latent-diffusion) by running

   ```bash
   conda install pytorch torchvision -c pytorch
   pip install transformers==4.19.2 diffusers invisible-watermark
   pip install -e .
   ```

3. Get checkpoint files.

   Check https://huggingface.co/CompVis/ and download checkpoint files. You may simply run

   ```bash
   wget -c -t 0 https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
   ```

   then you may link the checkpoint file:

   ```bash
   mkdir -p models/ldm/stable-diffusion-v1/
   ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 
   ```
   If you are encountering the RuntimeError below, something might have gone wrong in downloading the checkpoint file. I fixed this error by simply re-download checkpoints.
   ```
   RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
   ```

4. Sample

   You should now be able to run stable-diffusion now, with

   ```bash
   python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms 
   ```

   Unfortunately, users without a stable network connection may suffer from the issue where it claims that certain models cannot be loaded from a cache directory, e.g. 

   ```bash
   ~/.cache/huggingface/transformers/05f4975775cba20dc0cb27eb7c3a6e44996ed577f3a855c5ebdce8303719c784.86d7f7d9367f5c2c47fdcd93b38959858180f30bf53f98952f1fa6107d7c158b
   ```
   
   This error happens simply due to the failure of `http_get()` .To solve this, you may download the objects required with`wget -c -t 0` beforewards, or simply clear the corresponding cache directory each time it fails.

## JAX Version

GitHub: https://github.com/patil-suraj/stable-diffusion-jax

### Notes

1. Cloen and setup

   - Clone the repository

   `git clone https://github.com/patil-suraj/stable-diffusion-jax.git`

   `cd stable-diffusion-jax`

   - You may start from a conda environment identical to that of `stable-diffusion`:

   `conda create -n jsd --clone ldm`

   `conda activate jsd`

   - And you may run 

   ```bash
   python setup.py install
   ```

   to setup.

   **Make sure that your CUDA version is no older than 11.1**

2. Download / convert models

   Again, I presonally recommend that you download the models loaclly if connection failures happen too often.

   Check https://huggingface.co/fusing/sd-v1-4-flax and download models there. However, there might be something wrong with the `unet` and `vae` models.  When getting the error below:

   ```bash
   kernel = self.param('kernel',
   flax.errors.ScopeParamNotFoundError: Could not find parameter named "kernel" in scope "/down_blocks_0/attentions_0/transformer_blocks_0/self_attn/to_q". (https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.ScopeParamNotFoundError)
   ```

   you may use models converted from `pytorch` pre-trained models.

   For example, you can download `pytorch` pre-trained models from https://huggingface.co/CompVis/stable-diffusion-v1-4 and use `convert_diffusers_to_jax.py` to convert the `unet` and `vae` models, and use the two newly-converted models to **replace** those downloaded from https://huggingface.co/fusing/sd-v1-4-flax.

3. Modify the source code

   **In `run_safety_checker.py`**

   - Line 34:

   ```python
   model = StableDiffusionSafetyCheckerModel.from_pretrained("/home/patrick/sd-v1-4-flax/safety_checker")
   ============>
   model = StableDiffusionSafetyCheckerModel.from_pretrained("{fx_path}/safety_checker")
   ```
   
   - Line 35:

   ```python
   pt_model = StableDiffusionSafetyChecker.from_pretrained("/home/patrick/stable-diffusion-v1-1/safety_checker")
   ============>
   pt_model = StableDiffusionSafetyChecker.from_pretrained("<path/to/stable-diffusion-v1-1/on/your/machine>/safety_checker")
   ```
   
   - set `fx_path` to the directory where you saved your downloaded & converted models.
   - Line 91:

   ```python
   prng_seed = jax.random.split(prng_seed, 8)
   ============>
   prng_seed = jax.random.split(prng_seed, jax.local_device_count())
   ```
   
   **In `run.py`**
   
   - set `fx_path` to the directory where you saved your converted models.
   - Set the save directory to where you want to save your output images.
   - Line 78:

   ```python
   prng_seed = jax.random.split(prng_seed, 8)
   ============>
   prng_seed = jax.random.split(prng_seed, jax.local_device_count())
   ```

4. Run and enjoy!

### Codelist

`run.py`: original version using `pmap`

`nopmap.py`: no parallelism

`alpa_run.py`: using alpa