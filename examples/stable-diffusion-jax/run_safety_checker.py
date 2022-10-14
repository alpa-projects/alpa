import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image
from flax.traverse_util import flatten_dict, unflatten_dict
import warnings
import torch
from transformers import CLIPTokenizer, FlaxCLIPTextModel, CLIPConfig, CLIPFeatureExtractor
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionSafetyChecker

from stable_diffusion_jax import (
    AutoencoderKL,
    InferenceState,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2D,
    StableDiffusionSafetyCheckerModel,
)
from stable_diffusion_jax.convert_diffusers_to_jax import convert_diffusers_to_jax

feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")

shape = (2, 224, 224, 3)
images = np.random.rand(*shape)
images = (images * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
safety_checker_input_np = feature_extractor(pil_images, return_tensors="np").pixel_values
safety_checker_input_pt = feature_extractor(pil_images, return_tensors="pt").pixel_values


model = StableDiffusionSafetyCheckerModel.from_pretrained("/home/patrick/sd-v1-4-flax/safety_checker")
pt_model = StableDiffusionSafetyChecker.from_pretrained("/home/patrick/stable-diffusion-v1-1/safety_checker")

result = model(safety_checker_input_np)
has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]
for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
    if has_nsfw_concept:
        images[idx] = np.zeros(images[idx].shape)  # black image

    if any(has_nsfw_concepts):
        warnings.warn(
            "Potential NSFW content was detected in one or more images. A black image will be returned instead."
            " Try again with a different prompt and/or seed."
        )


import ipdb; ipdb.set_trace()

# inference with jax
dtype = jnp.bfloat16
clip_model, clip_params = FlaxCLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14", _do_init=False, dtype=dtype
)
unet, unet_params = UNet2D.from_pretrained(f"{fx_path}/unet", _do_init=False, dtype=dtype)
vae, vae_params = AutoencoderKL.from_pretrained(f"{fx_path}/vae", _do_init=False, dtype=dtype)

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
