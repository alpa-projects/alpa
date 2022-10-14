import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image
from transformers import CLIPTokenizer, FlaxCLIPTextModel, CLIPConfig

from stable_diffusion_jax import (
    AutoencoderKL,
    InferenceState,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2D
    StableDiffusionSafetyCheckerModel,
)
from stable_diffusion_jax.convert_diffusers_to_jax import convert_diffusers_to_jax


# convert diffusers checkpoint to jax
pt_path = "path_to_diffusers_pt_ckpt"
fx_path = "save_path"
convert_diffusers_to_jax(pt_path, fx_path)


# inference with jax
dtype = jnp.bfloat16
clip_model, clip_params = FlaxCLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14", _do_init=False, dtype=dtype
)
unet, unet_params = UNet2D.from_pretrained(f"{fx_path}/unet", _do_init=False, dtype=dtype)
vae, vae_params = AutoencoderKL.from_pretrained(f"{fx_path}/vae", _do_init=False, dtype=dtype)
safety_model, safety_model_params = StableDiffusionSafetyCheckerModel.from_pretrained(f"{fx_path}/safety_model", _do_init=False, dtype=dtype)

config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
scheduler = PNDMScheduler()

# create inference state and replicate it across all TPU devices
inference_state = InferenceState(text_encoder_params=clip_params, unet_params=unet_params, vae_params=vae_params)
inference_state = replicate(inference_state)


# create pipeline
pipe = StableDiffusionPipeline(text_encoder=clip_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler, vae=vae)



# prepare inputs
num_samples = 8
p = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"

input_ids = tokenizer(
    [p] * num_samples, padding="max_length", truncation=True, max_length=77, return_tensors="jax"
).input_ids
uncond_input_ids = tokenizer(
    [""] * num_samples, padding="max_length", truncation=True, max_length=77, return_tensors="jax"
).input_ids
prng_seed = jax.random.PRNGKey(42)

# shard inputs and rng
input_ids = shard(input_ids)
uncond_input_ids = shard(uncond_input_ids)
prng_seed = jax.random.split(prng_seed, 8)

# pmap the sample function
num_inference_steps = 50
guidance_scale = 1.0

sample = jax.pmap(pipe.sample, static_broadcasted_argnums=(4, 5))

# sample images
images = sample(
    input_ids,
    uncond_input_ids,
    prng_seed,
    inference_state,
    num_inference_steps,
    guidance_scale,
)


# convert images to PIL images
images = images / 2 + 0.5
images = jnp.clip(images, 0, 1)
images = (images * 255).round().astype("uint8")
images = np.asarray(images).reshape((num_samples, 512, 512, 3))

pil_images = [Image.fromarray(image) for image in images]
