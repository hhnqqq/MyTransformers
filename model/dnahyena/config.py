import torch

from dataclasses import dataclass
from common.registry import registry
from common.utils import STR_DTYPE_TO_TORCH_DTYPE
from typing import Optional, Any

@dataclass 
class Layerconfig:
    l_max: int
    dim: Optional[int] = None
    filter_order: int = 64
    dropout: float = 0.0
    filter_dropout: float = 0.0
    channels: int = 1
    activation_freq =1
    num_inner_mlps = 2
    train_freq = True
    use_bias = True

@dataclass
class HyenaConfig:
    dim: int
    n_layer: int
    vocab_size: int
    d_inner: Optional[int] = None
    layer_config: Layerconfig = None
    max_position_embeddings: int = 0
    layer_norm_epsilon: float = 1e-5
    initializer_cfg: Any = None
    pad_vocab_size_multiple: int = 1
    n_classes: int = 2
    mode: str = 'pool'
    l_output: Optional[int] = None
    initializer_range: float = 0.02
    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    resid_dropout1: float = 0.0
    resid_dropout2: float = 0.0
    drop_path1: float = 0.0
    drop_path2: float = 0.0
    filter_dropout: float = 0.0
    prenorm: bool = True
    use_head: bool = False
    return_residual: bool = False
    residual_in_fp32: bool = False
    device: Optional[str] = None
    dtype: str = 'float32'
    emb_dim: Optional[int] = None
    word_embed_proj_dim: Optional[int] = None
    short_filter_order: int = 3
    order: int = 2
    padding_idx: Optional[int] = None
    use_lengths: bool = False
    projector_type: str = 'linear'
    tokenizer = None

    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""
        return STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)

@registry.register_model_config("hyena_large_1m")
def get_config_for_large() -> HyenaConfig:
    layer_config = Layerconfig(
        l_max = 1000002
    )

    return HyenaConfig(
        dim = 256,
        d_inner = 1024,
        vocab_size = 12,
        emb_dim = 5,
        embed_dropout = 0.1,
        initializer_range = 0.02,
        layer_norm_epsilon = 1e-05,
        n_layer = 8,
        layer_config = layer_config,
        padding_idx=4,
        pad_vocab_size_multiple = 8,
        short_filter_order = 3,
        dtype = 'float16',
        l_output=32,
        mode='pool',
        device='cpu',
        projector_type='mlp2x_gelu'
    )

@registry.register_model_config("llama1_with_hyena_large")
def get_llama_hyena_config_large():
    model_config = registry.get_model_config_class("llama1_7b")()
    model_config.multimodal_model_config = registry.get_model_config_class("hyena_large_1m")()
    return model_config

@registry.register_model_config("llama2_with_hyena_large")
def get_llama_hyena_config_large():
    model_config = registry.get_model_config_class("llama2_7b")()
    model_config.multimodal_model_config = registry.get_model_config_class("hyena_large_1m")()
    return model_config

@registry.register_model_config("llama3_with_hyena_large")
def get_llama_hyena_config_large():
    model_config = registry.get_model_config_class("llama3_8b")()
    model_config.multimodal_model_config = registry.get_model_config_class("hyena_large_1m")()
    return model_config