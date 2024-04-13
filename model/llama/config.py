from common.registry import registry
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    head_dim: int = 128
    hideen_size: int = 4096
    n_layers: int = 32
    num_hidden_layers: int = 32
    n_heads: int = 32
    num_attention_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


@registry.register_model_config("llama_7b")
def get_config_for_7b() -> ModelArgs:
    return ModelArgs()

@registry.register_model_config("llama_13b")
def get_config_for_2b() -> ModelArgs:
    return ModelArgs(
        dim=5120,
        hidden_size=5120,
        n_layers=40,
        num_hidden_layers=40,
        n_head=40,
        num_attention_heads=40,
    )

@registry.register_model_config("llama_test")
def get_config_for_test() -> ModelArgs:
    return ModelArgs(
        dim=2048,
        hidden_size=2048,
        n_layers=16,
        num_hidden_layers=16,
        n_head=16,
        num_attention_heads=16,
    )