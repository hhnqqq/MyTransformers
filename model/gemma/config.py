# Copyright 2024 Google LLC
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

"""Gemma model config."""

import immutabledict
import torch
from typing import Optional, List
from common.registry import registry
from dataclasses import dataclass, field
from common.utils.utils import STR_DTYPE_TO_TORCH_DTYPE


@dataclass
class GemmaConfig:
    # The number of tokens in the vocabulary.
    vocab_size: int = 256000
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 8192
    # The number of blocks in the model.
    n_layers: int = 28
    # The number of attention heads used in the attention layers of the model.
    n_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    dim: int = 3072
    # The dimension of the MLP representations.
    intermediate_size: int = 24576
    # The number of head dimensions.
    head_dim: int = 256
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6
    # The dtype of the weights.
    dtype: str = 'bfloat16'
    # Whether a quantized version of the model is used.
    quant: bool = False
    # The path to the model tokenizer.
    tokenizer: Optional[str] = ''
    rope_theta: float = 100000.0
    lora_layers: List[str] = field(default_factory=lambda: ["qkv_proj", "o_proj", "gate_proj", "down_proj", "up_proj"])

    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""
        return STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)


@registry.register_model_config("gemma_7b")
def get_config_for_7b() -> GemmaConfig:
    return GemmaConfig()

@registry.register_model_config("gemma_2b")
def get_config_for_2b() -> GemmaConfig:
    return GemmaConfig(
        n_layers=18,
        n_heads=8,
        num_key_value_heads=1,
        dim=2048,
        intermediate_size=16384
    )

@registry.register_model_config("gemma_test")
def get_config_for_test() -> GemmaConfig:
    return GemmaConfig(
        n_layers=6,
        n_heads=4,
        num_key_value_heads=1,
        dim=256,
        intermediate_size=2048,
        dtype='float16'
    )


def get_model_config(variant: str) -> GemmaConfig:
    if variant == '7b':
        return get_config_for_7b()
    elif variant == '2b':
        return get_config_for_2b()
    elif variant == 'test':
        return get_config_for_test()
    return ValueError(f'Invalid variant {variant}. Supported variants are "2b"'
                      'and "7b"')