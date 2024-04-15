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


# Keep a mapping from dtype strings to the supported torch dtypes.
_STR_DTYPE_TO_TORCH_DTYPE = immutabledict.immutabledict({
    'float16': torch.float16,
    'float': torch.float32,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
})


@dataclass
class GemmaConfig:
    # The number of tokens in the vocabulary.
    vocab_size: int = 256000
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 8192
    # The number of blocks in the model.
    num_hidden_layers: int = 28
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    hidden_size: int = 3072
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
    lora_layers: List[str] = field(default_factory=lambda: ["qkv_proj"])

    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""
        return _STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)


@registry.register_model_config("gemma_7b")
def get_config_for_7b() -> GemmaConfig:
    return GemmaConfig()

@registry.register_model_config("gemma_2b")
def get_config_for_2b() -> GemmaConfig:
    return GemmaConfig(
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        hidden_size=2048,
        intermediate_size=16384
    )

@registry.register_model_config("gemma_test")
def get_config_for_test() -> GemmaConfig:
    return GemmaConfig(
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        hidden_size=256,
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