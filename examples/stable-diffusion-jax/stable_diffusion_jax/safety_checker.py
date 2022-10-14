import numpy as np
from flax import linen as nn
import jax.numpy as jnp
import jax
from typing import Optional, Tuple
import optax
import warnings
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from transformers import CLIPConfig, FlaxCLIPVisionModel, FlaxPreTrainedModel
from transformers.models.clip.modeling_flax_clip import FlaxCLIPVisionModule


def cosine_distance(ten_1, ten_2, eps=1e-12):
    norm_ten_1 = jnp.divide(ten_1.T, jnp.clip(jnp.linalg.norm(ten_1, axis=1), a_min=eps)).T
    norm_ten_2 = jnp.divide(ten_2.T, jnp.clip(jnp.linalg.norm(ten_2, axis=1), a_min=eps)).T

    return jnp.matmul(norm_ten_1, norm_ten_2.T)


class StableDiffusionSafetyCheckerModule(nn.Module):
    config: CLIPConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.vision_model = FlaxCLIPVisionModule(self.config.vision_config)
        self.visual_projection = nn.Dense(self.config.projection_dim, use_bias=False)

        self.concept_embeds = self.param("concept_embeds", jax.nn.initializers.ones, (17, self.config.projection_dim))
        self.special_care_embeds = self.param("special_care_embeds", jax.nn.initializers.ones, (3, self.config.projection_dim))

        self.concept_embeds_weights = self.param("concept_embeds_weights", jax.nn.initializers.ones, (17,))
        self.special_care_embeds_weights = self.param("special_care_embeds_weights", jax.nn.initializers.ones, (3,))

    def __call__(self, clip_input, images=None):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        if images is None:
            return special_cos_dist, cos_dist

        special_cos_dist = np.asarray(special_cos_dist)
        cos_dist = np.asarray(cos_dist)

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign image inputs
            adjustment = 0.0

            for concet_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concet_idx]
                concept_threshold = self.special_care_embeds_weights[concet_idx].item()
                result_img["special_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concet_idx] > 0:
                    result_img["special_care"].append({concet_idx, result_img["special_scores"][concet_idx]})
                    adjustment = 0.01

            for concet_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concet_idx]
                concept_threshold = self.concept_embeds_weights[concet_idx].item()
                result_img["concept_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concet_idx] > 0:
                    result_img["bad_concepts"].append(concet_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        images_was_copied = False
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if not images_was_copied:
                    images_was_copied = True
                    images = images.copy()

                images[idx] = np.zeros(images[idx].shape)  # black image

            if any(has_nsfw_concepts):
                warnings.warn(
                    "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                    " Try again with a different prompt and/or seed."
                )

        return images, has_nsfw_concepts


class StableDiffusionSafetyCheckerModel(FlaxPreTrainedModel):
    config_class = CLIPConfig
    main_input_name = "pixel_values"
    module_class = StableDiffusionSafetyCheckerModule

    def __init__(
        self,
        config: CLIPConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs
    ):
        if input_shape is None:
            input_shape = (1, 224, 224, 3)
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensor
        pixel_values = jax.random.normal(rng, input_shape)


        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, pixel_values)["params"]

        return random_params

    def __call__(
        self,
        pixel_values,
        params: dict = None,
        images=None,
    ):
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            images,
            rngs={},
        )
