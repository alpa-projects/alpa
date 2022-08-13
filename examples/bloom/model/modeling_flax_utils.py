# coding=utf-8
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gc
import json
import os
import re
from functools import partial
from pickle import UnpicklingError
from typing import Any, Dict, Set, Tuple, Union

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
import msgpack.exceptions
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from requests import HTTPError

from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import custom_object_save
from transformers.modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
from transformers.utils import (
    FLAX_WEIGHTS_INDEX_NAME,
    FLAX_WEIGHTS_NAME,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    WEIGHTS_NAME,
    EntryNotFoundError,
    PushToHubMixin,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
    cached_path,
    copy_func,
    has_file,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
    logging,
    replace_return_docstrings,
)
from transformers.utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
from .generation_flax_utils import FlaxGenerationMixin


logger = logging.get_logger(__name__)


def quick_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
    "quick_gelu": quick_gelu,
}


def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`. Example:
    ```py
    >>> dtype_byte_size(np.float32)
    4
    ```
    """
    if dtype == np.bool:
        return 1 / 8
    bit_search = re.search("[^\d](\d+)$", dtype.name)
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def flax_shard_checkpoint(params, max_shard_size="10GB"):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size. The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so
    there is no optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For
    example, if the limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as
    [6GB], [6+2GB], [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger that `max_shard_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        params (`Union[Dict, FrozenDict]`): A `PyTree` of model parameters.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0

    # flatten the weights to chunk
    weights = flatten_dict(params, sep="/")
    for item in weights:
        weight_size = weights[item].size * dtype_byte_size(weights[item].dtype)

        # If this weight is going to tip up over the maximal size, we split.
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = {}
            current_block_size = 0

        current_block[item] = weights[item]
        current_block_size += weight_size
        total_size += weight_size

    # Add the last block
    sharded_state_dicts.append(current_block)

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {FLAX_WEIGHTS_NAME: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = FLAX_WEIGHTS_NAME.replace(".msgpack", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.msgpack")
        shards[shard_file] = shard
        for weight_name in shard.keys():
            weight_map[weight_name] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index


class FlaxPreTrainedModel(PushToHubMixin, FlaxGenerationMixin):
    r"""
    Base class for all models.

    [`FlaxPreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    _auto_class = None
    _missing_keys = set()

    def __init__(
        self,
        config: PretrainedConfig,
        module: nn.Module,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
    ):
        if config is None:
            raise ValueError("config cannot be None")

        if module is None:
            raise ValueError("module cannot be None")

        # Those are private to be exposed as typed property on derived classes.
        self._config = config
        self._module = module

        # Those are public as their type is generic to every derived classes.
        self.key = PRNGKey(seed)
        self.dtype = dtype
        self.input_shape = input_shape

        # To check if the model was intialized automatically.
        self._is_initialized = _do_init

        if _do_init:
            # randomly initialized parameters
            random_params = self.init_weights(self.key, input_shape)
            params_shape_tree = jax.eval_shape(lambda params: params, random_params)
        else:
            init_fn = partial(self.init_weights, input_shape=input_shape)
            params_shape_tree = jax.eval_shape(init_fn, self.key)

            logger.info(
                "Model weights are not initialized as `_do_init` is set to `False`. "
                f"Make sure to call `{self.__class__.__name__}.init_weights` manually to initialize the weights."
            )

        # get the shape of the parameters
        self._params_shape_tree = params_shape_tree

        # save required_params as set
        self._required_params = set(flatten_dict(unfreeze(params_shape_tree)).keys())

        # initialize the parameters
        if _do_init:
            self.params = random_params

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> Dict:
        raise NotImplementedError(f"init method has to be implemented for {self}")

    def enable_gradient_checkpointing(self):
        raise NotImplementedError(f"gradient checkpointing method has to be implemented for {self}")

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.
        """
        return cls(config, **kwargs)

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a Flax model.
        """
        return "flax"

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def module(self) -> nn.Module:
        return self._module

    @property
    def params(self) -> Union[Dict, FrozenDict]:
        if not self._is_initialized:
            raise ValueError(
                "`params` cannot be accessed from model when the model is created with `_do_init=False`. "
                "You must call `init_weights` manually and store the params outside of the model and "
                "pass it explicitly where needed."
            )
        return self._params

    @property
    def required_params(self) -> Set:
        return self._required_params

    @property
    def params_shape_tree(self) -> Dict:
        return self._params_shape_tree

    @params.setter
    def params(self, params: Union[Dict, FrozenDict]):
        # don't set params if the model is not initialized
        if not self._is_initialized:
            raise ValueError(
                "`params` cannot be set from model when the model is created with `_do_init=False`. "
                "You store the params outside of the model."
            )

        if isinstance(params, FrozenDict):
            params = unfreeze(params)
        param_keys = set(flatten_dict(params).keys())
        if len(self.required_params - param_keys) > 0:
            raise ValueError(
                "Some parameters are missing. Make sure that `params` include the following "
                f"parameters {self.required_params - param_keys}"
            )
        self._params = params

    def _cast_floating_to(self, params: Union[Dict, FrozenDict], dtype: jnp.dtype, mask: Any = None) -> Any:
        """
        Helper method to cast floating-point values of given parameter `PyTree` to given `dtype`.
        """

        # taken from https://github.com/deepmind/jmp/blob/3a8318abc3292be38582794dbf7b094e6583b192/jmp/_src/policy.py#L27
        def conditional_cast(param):
            if isinstance(param, jnp.ndarray) and jnp.issubdtype(param.dtype, jnp.floating):
                param = param.astype(dtype)
            return param

        if mask is None:
            return jax.tree_map(conditional_cast, params)

        flat_params = flatten_dict(params)
        flat_mask, _ = jax.tree_flatten(mask)

        for masked, key in zip(flat_mask, flat_params.keys()):
            if masked:
                param = flat_params[key]
                flat_params[key] = conditional_cast(param)

        return unflatten_dict(flat_params)

    def to_bf16(self, params: Union[Dict, FrozenDict], mask: Any = None):
        r"""
        Cast the floating-point `params` to `jax.numpy.bfloat16`. This returns a new `params` tree and does not cast
        the `params` in place.

        This method can be used on TPU to explicitly convert the model parameters to bfloat16 precision to do full
        half-precision training or to save weights in bfloat16 for inference in order to save memory and improve speed.

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params
                you want to cast, and should be `False` for those you want to skip.

        Examples:

        ```python
        >>> from transformers import FlaxBertModel

        >>> # load model
        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")
        >>> # By default, the model parameters will be in fp32 precision, to cast these to bfloat16 precision
        >>> model.params = model.to_bf16(model.params)
        >>> # If you want don't want to cast certain parameters (for example layer norm bias and scale)
        >>> # then pass the mask as follows
        >>> from flax import traverse_util

        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")
        >>> flat_params = traverse_util.flatten_dict(model.params)
        >>> mask = {
        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
        ...     for path in flat_params
        ... }
        >>> mask = traverse_util.unflatten_dict(mask)
        >>> model.params = model.to_bf16(model.params, mask)
        ```"""
        return self._cast_floating_to(params, jnp.bfloat16, mask)

    def to_fp32(self, params: Union[Dict, FrozenDict], mask: Any = None):
        r"""
        Cast the floating-point `parmas` to `jax.numpy.float32`. This method can be used to explicitly convert the
        model parameters to fp32 precision. This returns a new `params` tree and does not cast the `params` in place.

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params
                you want to cast, and should be `False` for those you want to skip

        Examples:

        ```python
        >>> from transformers import FlaxBertModel

        >>> # Download model and configuration from huggingface.co
        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")
        >>> # By default, the model params will be in fp32, to illustrate the use of this method,
        >>> # we'll first cast to fp16 and back to fp32
        >>> model.params = model.to_f16(model.params)
        >>> # now cast back to fp32
        >>> model.params = model.to_fp32(model.params)
        ```"""
        return self._cast_floating_to(params, jnp.float32, mask)

    def to_fp16(self, params: Union[Dict, FrozenDict], mask: Any = None):
        r"""
        Cast the floating-point `parmas` to `jax.numpy.float16`. This returns a new `params` tree and does not cast the
        `params` in place.

        This method can be used on GPU to explicitly convert the model parameters to float16 precision to do full
        half-precision training or to save weights in float16 for inference in order to save memory and improve speed.

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params
                you want to cast, and should be `False` for those you want to skip

        Examples:

        ```python
        >>> from transformers import FlaxBertModel

        >>> # load model
        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")
        >>> # By default, the model params will be in fp32, to cast these to float16
        >>> model.params = model.to_fp16(model.params)
        >>> # If you want don't want to cast certain parameters (for example layer norm bias and scale)
        >>> # then pass the mask as follows
        >>> from flax import traverse_util

        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")
        >>> flat_params = traverse_util.flatten_dict(model.params)
        >>> mask = {
        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
        ...     for path in flat_params
        ... }
        >>> mask = traverse_util.unflatten_dict(mask)
        >>> model.params = model.to_fp16(model.params, mask)
        ```"""
        return self._cast_floating_to(params, jnp.float16, mask)

    @classmethod
    def load_flax_sharded_weights(cls, shard_files):
        """
        This is the same as [`flax.serialization.from_bytes`]
        (https:lax.readthedocs.io/en/latest/_modules/flax/serialization.html#from_bytes) but for a sharded checkpoint.

        This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
        loaded in the model.

        Args:
            shard_files (`List[str]`:
                The list of shard files to load.

        Returns:
            `Dict`: A nested dictionary of the model parameters, in the expected format for flax models : `{'model':
            {'params': {'...'}}}`.
        """

        # Load the index
        state_sharded_dict = dict()

        for shard_file in shard_files:
            # load using msgpack utils
            try:
                with open(shard_file, "rb") as state_f:
                    state = from_bytes(cls, state_f.read())
            except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
                with open(shard_file) as f:
                    if f.read().startswith("version"):
                        raise OSError(
                            "You seem to have cloned a repository without having git-lfs installed. Please"
                            " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                            " folder you cloned."
                        )
                    else:
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                raise EnvironmentError(f"Unable to convert {shard_file} to Flax deserializable object. ")

            state = flatten_dict(state, sep="/")
            state_sharded_dict.update(state)
            del state
            gc.collect()

        # the state dict is unflattened to the match the format of model.params
        return unflatten_dict(state_sharded_dict, sep="/")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        dtype: jnp.dtype = jnp.float32,
        *model_args,
        **kwargs
    ):

        r"""
        Instantiate a pretrained flax model from a pre-trained model configuration.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *pt index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In this case,
                      `from_pt` should be set to `True`.
            dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
                The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
                `jax.numpy.bfloat16` (on TPUs).

                This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
                specified all the computation will be performed with the given `dtype`.

                **Note that this only specifies the dtype of the computation and does not influence the dtype of model
                parameters.**

                If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
                [`~FlaxPreTrainedModel.to_bf16`].
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_pt (`bool`, *optional*, defaults to `False`):
                Load the model weights from a PyTorch checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        Examples:

        ```python
        >>> from transformers import BertConfig, FlaxBertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = FlaxBertModel.from_pretrained("./test/saved_model/")
        >>> # Loading from a PyTorch checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file("./pt_model/config.json")
        >>> model = FlaxBertModel.from_pretrained("./pt_model/pytorch_model.bin", from_pt=True, config=config)
        ```"""
        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_pt = kwargs.pop("from_pt", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _do_init = kwargs.pop("_do_init", True)

        user_agent = {"file_type": "model", "framework": "flax", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Add the dtype to model_kwargs
        model_kwargs["dtype"] = dtype

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)):
                    # Load from a Flax checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_INDEX_NAME)):
                    # Load from a sharded Flax checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_INDEX_NAME)
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME):
                    raise EnvironmentError(
                        f"Error no file named {FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} "
                        "but there is a file for PyTorch weights. Use `from_pt=True` to load this model from those "
                        "weights."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME} found in directory "
                        f"{pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            else:
                filename = WEIGHTS_NAME if from_pt else FLAX_WEIGHTS_NAME
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=filename,
                    revision=revision,
                )

            # redirect to the cache, if necessary

            try:
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                )

            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                    "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                    "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                    "login` and pass `use_auth_token=True`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                    "this model name. Check the model page at "
                    f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                if filename == FLAX_WEIGHTS_NAME:
                    try:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        archive_file = hf_bucket_url(
                            pretrained_model_name_or_path,
                            filename=FLAX_WEIGHTS_INDEX_NAME,
                            revision=revision,
                        )
                        resolved_archive_file = cached_path(
                            archive_file,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            proxies=proxies,
                            resume_download=resume_download,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            user_agent=user_agent,
                        )
                        is_sharded = True
                    except EntryNotFoundError:
                        has_file_kwargs = {"revision": revision, "proxies": proxies, "use_auth_token": use_auth_token}
                        if has_file(pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {FLAX_WEIGHTS_NAME} but there is a file for PyTorch weights. Use `from_pt=True` to"
                                " load this model from those weights."
                            )
                        else:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}."
                            )
                else:
                    raise EnvironmentError(
                        f"{pretrained_model_name_or_path} does not appear to have a file named {filename}."
                    )
            except HTTPError as err:
                raise EnvironmentError(
                    f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n"
                    f"{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}.\nCheckout your"
                    " internet connection or see how to run the library in offline mode at"
                    " 'https://huggingface.co/docs/transformers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}."
                )

            if resolved_archive_file == archive_file:
                logger.info(f"loading weights file {archive_file}")
            else:
                logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, _ = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                revision=revision,
            )

        # init random models
        model = cls(config, *model_args, _do_init=_do_init, **model_kwargs)

        if from_pt:
            state = load_pytorch_checkpoint_in_flax_state_dict(model, resolved_archive_file)
        else:

            if is_sharded:
                state = cls.load_flax_sharded_weights(resolved_archive_file)
            else:
                try:
                    with open(resolved_archive_file, "rb") as state_f:
                        state = from_bytes(cls, state_f.read())
                except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
                    try:
                        with open(resolved_archive_file) as f:
                            if f.read().startswith("version"):
                                raise OSError(
                                    "You seem to have cloned a repository without having git-lfs installed. Please"
                                    " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                                    " folder you cloned."
                                )
                            else:
                                raise ValueError from e
                    except (UnicodeDecodeError, ValueError):
                        raise EnvironmentError(f"Unable to convert {archive_file} to Flax deserializable object. ")
            # make sure all arrays are stored as jnp.arrays
            # NOTE: This is to prevent a bug this will be fixed in Flax >= v0.3.4:
            # https://github.com/google/flax/issues/1261
            if _do_init:
                state = jax.tree_util.tree_map(jnp.array, state)
            else:
                # keep the params on CPU if we don't want to initialize
                state = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("cpu")[0]), state)

        # if model is base model only use model_prefix key
        if cls.base_model_prefix not in dict(model.params_shape_tree) and cls.base_model_prefix in state:
            state = state[cls.base_model_prefix]

        # if model is head model and we are loading weights from base model
        # we initialize new params dict with base_model_prefix
        if cls.base_model_prefix in dict(model.params_shape_tree) and cls.base_model_prefix not in state:
            state = {cls.base_model_prefix: state}

        # flatten dicts
        state = flatten_dict(state)

        random_state = flatten_dict(unfreeze(model.params if _do_init else model.params_shape_tree))

        missing_keys = model.required_params - set(state.keys())
        unexpected_keys = set(state.keys()) - model.required_params

        if missing_keys and not _do_init:
            logger.warning(
                f"The checkpoint {pretrained_model_name_or_path} is missing required keys: {missing_keys}. "
                "Make sure to call model.init_weights to initialize the missing weights."
            )
            cls._missing_keys = missing_keys

        # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
        # matching the weights in the model.
        mismatched_keys = []
        for key in state.keys():
            if key in random_state and state[key].shape != random_state[key].shape:
                if ignore_mismatched_sizes:
                    mismatched_keys.append((key, state[key].shape, random_state[key].shape))
                    state[key] = random_state[key]
                else:
                    raise ValueError(
                        f"Trying to load the pretrained weight for {key} failed: checkpoint has shape "
                        f"{state[key].shape} which is incompatible with the model shape {random_state[key].shape}. "
                        "Using `ignore_mismatched_sizes=True` if you really want to load this checkpoint inside this "
                        "model."
                    )

        # add missing keys as random parameters if we are initializing
        if missing_keys and _do_init:
            for missing_key in missing_keys:
                state[missing_key] = random_state[missing_key]

        # remove unexpected keys to not be saved again
        for unexpected_key in unexpected_keys:
            del state[unexpected_key]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        # dictionary of key: dtypes for the model params
        param_dtypes = jax.tree_map(lambda x: x.dtype, state)
        # extract keys of parameters not in jnp.float32
        fp16_params = [k for k in param_dtypes if param_dtypes[k] == jnp.float16]
        bf16_params = [k for k in param_dtypes if param_dtypes[k] == jnp.bfloat16]

        # raise a warning if any of the parameters are not in jnp.float32
        if len(fp16_params) > 0:
            logger.warning(
                f"Some of the weights of {model.__class__.__name__} were initialized in float16 precision from "
                f"the model checkpoint at {pretrained_model_name_or_path}:\n{fp16_params}\n"
                "You should probably UPCAST the model weights to float32 if this was not intended. "
                "See [`~FlaxPreTrainedModel.to_fp32`] for further information on how to do this."
            )

        if len(bf16_params) > 0:
            logger.warning(
                f"Some of the weights of {model.__class__.__name__} were initialized in bfloat16 precision from "
                f"the model checkpoint at {pretrained_model_name_or_path}:\n{bf16_params}\n"
                "You should probably UPCAST the model weights to float32 if this was not intended. "
                "See [`~FlaxPreTrainedModel.to_fp32`] for further information on how to do this."
            )

        if _do_init:
            # set correct parameters
            model.params = unflatten_dict(state)
            return model
        else:
            return model, unflatten_dict(state)

    def save_pretrained(
        self, save_directory: Union[str, os.PathLike], params=None, push_to_hub=False, max_shard_size="10GB", **kwargs
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~FlaxPreTrainedModel.from_pretrained`]` class method

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                <Tip warning={true}>

                Using `push_to_hub=True` will synchronize the repository you are pushing to with `save_directory`,
                which requires `save_directory` to be a local clone of the repo you are pushing to if it's an existing
                folder. Pass along `temp_dir=True` to use a temporary directory instead.

                </Tip>

            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        # get abs dir
        save_directory = os.path.abspath(save_directory)
        # save config as well
        self.config.architectures = [self.__class__.__name__[4:]]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        self.config.save_pretrained(save_directory)

        # save model
        output_model_file = os.path.join(save_directory, FLAX_WEIGHTS_NAME)

        shards, index = flax_shard_checkpoint(params if params is not None else self.params, max_shard_size)
        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            if (
                filename.startswith(FLAX_WEIGHTS_NAME[:-4])
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
            ):
                os.remove(full_filename)

        if index is None:
            with open(output_model_file, "wb") as f:
                params = params if params is not None else self.params
                model_bytes = to_bytes(params)
                f.write(model_bytes)

        else:
            save_index_file = os.path.join(save_directory, FLAX_WEIGHTS_INDEX_NAME)
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )
            for shard_file, shard in shards.items():
                # the shard item are unflattened, to save them we need to flatten them again
                with open(os.path.join(save_directory, shard_file), mode="wb") as f:
                    params = unflatten_dict(shard, sep="/")
                    shard_bytes = to_bytes(params)
                    f.write(shard_bytes)

        logger.info(f"Model weights saved in {output_model_file}")

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Model pushed to the hub in this commit: {url}")

    @classmethod
    def register_for_auto_class(cls, auto_class="FlaxAutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"FlaxAutoModel"`):
                The auto class to register this new model with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class


# To update the docstring, we need to copy the method, otherwise we change the original docstring.
FlaxPreTrainedModel.push_to_hub = copy_func(FlaxPreTrainedModel.push_to_hub)
FlaxPreTrainedModel.push_to_hub.__doc__ = FlaxPreTrainedModel.push_to_hub.__doc__.format(
    object="model", object_class="FlaxAutoModel", object_files="model checkpoint"
)


def overwrite_call_docstring(model_class, docstring):
    # copy __call__ function to be sure docstring is changed only for this function
    model_class.__call__ = copy_func(model_class.__call__)
    # delete existing docstring
    model_class.__call__.__doc__ = None
    # set correct docstring
    model_class.__call__ = add_start_docstrings_to_model_forward(docstring)(model_class.__call__)


def append_call_sample_docstring(model_class, tokenizer_class, checkpoint, output_type, config_class, mask=None):
    model_class.__call__ = copy_func(model_class.__call__)
    model_class.__call__ = add_code_sample_docstrings(
        processor_class=tokenizer_class,
        checkpoint=checkpoint,
        output_type=output_type,
        config_class=config_class,
        model_cls=model_class.__name__,
    )(model_class.__call__)


def append_replace_return_docstrings(model_class, output_type, config_class):
    model_class.__call__ = copy_func(model_class.__call__)
    model_class.__call__ = replace_return_docstrings(
        output_type=output_type,
        config_class=config_class,
    )(model_class.__call__)
