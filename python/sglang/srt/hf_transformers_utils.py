# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Utilities for Huggingface Transformers."""

import contextlib
import enum
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from huggingface_hub import (file_exists, hf_hub_download,
                              snapshot_download,
                              try_to_load_from_cache)
from huggingface_hub.utils import (EntryNotFoundError, LocalEntryNotFoundError,
                                   RepositoryNotFoundError,
                                   RevisionNotFoundError)

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME

from sglang.srt.configs import ChatGLMConfig, DbrxConfig, ExaoneConfig, Qwen2VLConfig
from sglang.srt.tokenizers.mistral import MistralTokenizer

_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    ChatGLMConfig.model_type: ChatGLMConfig,
    DbrxConfig.model_type: DbrxConfig,
    ExaoneConfig.model_type: ExaoneConfig,
    Qwen2VLConfig.model_type: Qwen2VLConfig,
}


for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


def download_from_hf(model_path: str):
    if os.path.exists(model_path):
        return model_path

    return snapshot_download(model_path, allow_patterns=["*.json", "*.bin", "*.model"])

MISTRAL_CONFIG_NAME = "params.json"
HF_TOKEN = os.getenv('HF_TOKEN', None)

class ConfigFormat(str, enum.Enum):
    AUTO = "auto"
    HF = "hf"
    MISTRAL = "mistral"


def file_or_path_exists(model: Union[str, Path], config_name: str,
                        revision: Optional[str]) -> bool:
    if Path(model).exists():
        return (Path(model) / config_name).is_file()

    # Offline mode support: Check if config file is cached already
    cached_filepath = try_to_load_from_cache(repo_id=model,
                                             filename=config_name,
                                             revision=revision)
    if isinstance(cached_filepath, str):
        # The config file exists in cache- we can continue trying to load
        return True

    # NB: file_exists will only check for the existence of the config file on
    # hf_hub. This will fail in offline mode.
    try:
        return file_exists(model,
                           config_name,
                           revision=revision,
                           token=HF_TOKEN)
    except huggingface_hub.errors.OfflineModeIsEnabled:
        # Don't raise in offline mode, all we know is that we don't have this
        # file cached.
        return False

def get_hf_file_to_dict(file_name: str,
                        model: Union[str, Path],
                        revision: Optional[str] = 'main'):
    """
    Downloads a file from the Hugging Face Hub and returns 
    its contents as a dictionary.

    Parameters:
    - file_name (str): The name of the file to download.
    - model (str): The name of the model on the Hugging Face Hub.
    - revision (str): The specific version of the model. 

    Returns:
    - config_dict (dict): A dictionary containing 
    the contents of the downloaded file.
    """
    file_path = Path(model) / file_name

    if file_or_path_exists(model=model,
                           config_name=file_name,
                           revision=revision):

        if not file_path.is_file():
            try:
                hf_hub_file = hf_hub_download(model,
                                              file_name,
                                              revision=revision)
            except (RepositoryNotFoundError, RevisionNotFoundError,
                    EntryNotFoundError, LocalEntryNotFoundError) as e:
                logger.debug("File or repository not found in hf_hub_download",
                             e)
                return None
            file_path = Path(hf_hub_file)

        with open(file_path) as file:
            return json.load(file)
    return None

def load_params_config(model: Union[str, Path], revision: Optional[str],
                       **kwargs) -> PretrainedConfig:
    # This function loads a params.json config which
    # should be used when loading models in mistral format

    config_file_name = "params.json"

    config_dict = get_hf_file_to_dict(config_file_name, model, revision)
    assert isinstance(config_dict, dict)

    config_mapping = {
        "dim": "hidden_size",
        "norm_eps": "rms_norm_eps",
        "n_kv_heads": "num_key_value_heads",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "hidden_dim": "intermediate_size",
    }

    def recurse_elems(elem: Any):
        if isinstance(elem, dict):
            config_dict = {}
            for key, value in elem.items():
                key = config_mapping.get(key, key)
                config_dict[key] = recurse_elems(value)
            return PretrainedConfig(**config_dict)
        else:
            return elem

    config_dict["model_type"] = config_dict.get("model_type", "transformer")
    config_dict["hidden_act"] = config_dict.get("activation", "silu")
    config_dict["tie_word_embeddings"] = config_dict.get(
        "tie_embeddings", False)
    config_dict["max_seq_len"] = config_dict.get("max_seq_len", 128_000)
    config_dict["max_position_embeddings"] = config_dict.get(
        "max_position_embeddings", 128_000)

    if config_dict.get("moe") is not None:
        config_dict["architectures"] = ["MixtralForCausalLM"]
    else:
        config_dict["architectures"] = ["MistralForCausalLM"]

    if config_dict.get("vision_encoder") is not None:
        multimodal_config = config_dict.pop("vision_encoder")

        config_dict = {
            "text_config": config_dict,
            "vision_config": multimodal_config
        }
        config_dict["architectures"] = ["PixtralForConditionalGeneration"]
        config_dict["model_type"] = "pixtral"

    config_dict.update(kwargs)

    config = recurse_elems(config_dict)
    return config

def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    config_format: ConfigFormat = ConfigFormat.AUTO,
    **kwargs,
):

    is_gguf = check_gguf_file(model)
    if is_gguf:
        kwargs["gguf_file"] = model
        model = Path(model).parent

    if config_format == ConfigFormat.AUTO:
        if is_gguf or file_or_path_exists(model,
                                          HF_CONFIG_NAME,
                                          revision=revision):
            config_format = ConfigFormat.HF
        elif file_or_path_exists(model,
                                 MISTRAL_CONFIG_NAME,
                                 revision=revision):
            config_format = ConfigFormat.MISTRAL
        else:
            raise ValueError(f"No supported config format found in {model}")

    if config_format == ConfigFormat.HF:
        config = AutoConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
        )
        if config.model_type in _CONFIG_REGISTRY:
            config_class = _CONFIG_REGISTRY[config.model_type]
            config = config_class.from_pretrained(model, revision=revision)
            # NOTE(HandH1998): Qwen2VL requires `_name_or_path` attribute in `config`.
            setattr(config, "_name_or_path", model)
        if model_override_args:
            config.update(model_override_args)

        # Special architecture mapping check for GGUF models
        if is_gguf:
            if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
                raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
            model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
            config.update({"architectures": [model_type]})

    elif config_format == ConfigFormat.MISTRAL:
        config = load_params_config(model, revision)
    else:
        try:
            config = AutoConfig.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                code_revision=code_revision,
                **kwargs)
        except ValueError as e:
            if (not trust_remote_code
                    and "requires you to execute the configuration file"
                    in str(e)):
                err_msg = (
                    "Failed to load the model config. If the model is a custom "
                    "model not yet available in the HuggingFace transformers "
                    "library, consider setting `trust_remote_code=True` in LLM "
                    "or using the `--trust-remote-code` flag in the CLI.")
                raise RuntimeError(err_msg) from e
            else:
                raise e
        raise ValueError(f"Unsupported config format: {config_format}")

    return config


# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]


def get_context_length(config):
    """Get the context length of a model from a huggingface model configs."""
    text_config = config
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = rope_scaling.get("factor", 1)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(text_config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        kwargs["gguf_file"] = tokenizer_name
        tokenizer_name = Path(tokenizer_name).parent

    if (tokenizer_mode == "mistral"):
        tokenizer = MistralTokenizer.from_pretrained(
            tokenizer_name,
            revision=tokenizer_revision
        )
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                tokenizer_revision=tokenizer_revision,
                clean_up_tokenization_spaces=False,
                **kwargs,
            )
        except TypeError as e:
            # The LLaMA tokenizer causes a protobuf error in some environments.
            err_msg = (
                "Failed to load the tokenizer. If you are using a LLaMA V1 model "
                f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
                "original tokenizer."
            )
            raise RuntimeError(err_msg) from e
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported, suggest using the --trust-remote-code flag.
            if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer is a custom "
                    "tokenizer not yet available in the HuggingFace transformers "
                    "library, consider setting `trust_remote_code=True` in LLM "
                    "or using the `--trust-remote-code` flag in the CLI."
                )
                raise RuntimeError(err_msg) from e
            else:
                raise e

        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            warnings.warn(
                "Using a slow tokenizer. This might cause a significant "
                "slowdown. Consider using a fast tokenizer instead."
            )

    attach_additional_stop_token_ids(tokenizer)
    return tokenizer


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
):
    processor = AutoProcessor.from_pretrained(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        **kwargs,
    )

    attach_additional_stop_token_ids(processor.tokenizer)
    return processor


def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set(
            [tokenizer.get_added_vocab()["<|eom_id|>"]]
        )
    else:
        tokenizer.additional_stop_token_ids = None


def check_gguf_file(model: Union[str, os.PathLike]) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    with open(model, "rb") as f:
        header = f.read(4)
    return header == b"GGUF"
