import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image
from transformers import CLIPTokenizer, FlaxCLIPTextModel, AutoFeatureExtractor
import warnings

from stable_diffusion_jax import (
    AutoencoderKL,
    InferenceState,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2D,
    StableDiffusionSafetyCheckerModel
)
from stable_diffusion_jax.convert_diffusers_to_jax import convert_diffusers_to_jax

from functools import partial

import alpa
from alpa.testing import assert_allclose
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax

alpa.init(cluster="ray")
#logger = logging.getLogger(__name__)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


# convert diffusers checkpoint to jax
#pt_path = "/home/patrick/stable-diffusion-v1-3"
#fx_path = pt_path + "_jax"
#convert_diffusers_to_jax(pt_path, fx_path)

fx_path = "/data/wly/stable-diffusion-jax/local_hf/sd-v1-4-flax"

# inference with jax
dtype = jnp.float16
clip_model, clip_params = FlaxCLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14", _do_init=False, dtype=dtype
)
unet, unet_params = UNet2D.from_pretrained(f"{fx_path}/unet", _do_init=False, dtype=dtype)
vae, vae_params = AutoencoderKL.from_pretrained(f"{fx_path}/vae", _do_init=False, dtype=dtype)
safety_checker, safety_params = StableDiffusionSafetyCheckerModel.from_pretrained(f"{fx_path}/safety_checker", _do_init=False)
scheduler = PNDMScheduler.from_config(f"{fx_path}/scheduler", use_auth_token=True)
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# create inference state and replicate it across all TPU devices
inference_state = InferenceState(text_encoder_params=clip_params, unet_params=unet_params, vae_params=vae_params)


# inference_state = replicate(inference_state)


# create pipeline
pipe = StableDiffusionPipeline(text_encoder=clip_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler, vae=vae)


# prepare inputs
num_samples = 8
#p = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
#p = "a photograph of an astronaut riding a horse"
p = "In a Cyberpunk city, cyberpunk knights ride on cyberpunk horses"

input_ids = tokenizer(
    [p] * num_samples, padding="max_length", truncation=True, max_length=77, return_tensors="jax"
).input_ids
uncond_input_ids = tokenizer(
    [""] * num_samples, padding="max_length", truncation=True, max_length=77, return_tensors="jax"
).input_ids
prng_seed = jax.random.PRNGKey(1)

# shard inputs and rng
#input_ids = shard(input_ids)
#uncond_input_ids = shard(uncond_input_ids)
#prng_seed = jax.random.split(prng_seed, 2)


# pmap the sample function
num_inference_steps = 1
guidance_scale = 7.5


#sample = jax.pmap(pipe.sample, static_broadcasted_argnums=(4, 5))
sample = pipe.sample


def inf_step(
    input_ids,
    uncond_input_ids,
    inference_state
):
    return pipe.sample(
    input_ids,
    uncond_input_ids,
    prng_seed,
    inference_state,
    num_inference_steps,
    guidance_scale,
)

inf_step = alpa.parallelize(inf_step, method=alpa.DataParallel())

images = inf_step(
    input_ids,
    uncond_input_ids,
    inference_state,
)
images = images._value
images = images / 2 + 0.5
images = jnp.clip(images, 0, 1)
images = (images * 255).round().astype("uint8")
images = np.asarray(images).reshape((num_samples, 512, 512, 3))

pil_images = [Image.fromarray(image) for image in images]

# run safety checker
safety_cheker_input = feature_extractor(pil_images, return_tensors="np")

images, has_nsfw_concept = safety_checker(safety_cheker_input.pixel_values, params=safety_params, images=images)

pil_images = [Image.fromarray(image) for image in images]

grid = image_grid(pil_images, rows=1, cols=num_samples)
grid.save(f"/data/wly/stable-diffusion-jax/output/image_{p}.png")

