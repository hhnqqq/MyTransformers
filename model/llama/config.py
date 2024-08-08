
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from common.registry import registry
from common.utils.utils import STR_DTYPE_TO_TORCH_DTYPE


@dataclass
class ModelArgs:
    dim: int = 4096
    head_dim: int = 128
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    max_batch_size: int = 32
    max_seq_len: int = 2048
    tokenizer: Optional[str] = ''
    lora_layers: List[str] = field(default_factory=lambda: ['wk', 'wv', 'wq', 'wo', 'w1', 'w2', 'w3'])
    dtype: str = 'float16'
    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""
        return STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)

@registry.register_model_config("llama3_8b")
def get_config_for_llama3_8b() -> ModelArgs:
    return ModelArgs(
        n_kv_heads=8,
        multiple_of=1024,
        ffn_dim_multiplier=1.3,
        vocab_size=128256,
        rope_theta=500000.0
    )

@registry.register_model_config(["llama_7b", "llama1_7b", "llama2_7b"])
def get_config_for_7b() -> ModelArgs:
    return ModelArgs()


@registry.register_model_config(["llama_13b", "llama1_13b", "llama2_13b"])
def get_config_for_13b() -> ModelArgs:
    return ModelArgs(
        dim=5120,
        n_layers=40,
        n_heads=40,
    )


@registry.register_model_config(["llama_test", "llama1_test", "llama2_test"])
def get_config_for_test() -> ModelArgs:
    return ModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=16,
    )

@registry.register_model_config("llama3_test")
def get_config_for_llama3_test() -> ModelArgs:
    return ModelArgs(
        dim=1024,
        n_layers=8,
        vocab_size=128256,
        n_heads=8,
        n_kv_heads=8,
    )