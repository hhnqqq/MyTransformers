import re
import math
import torch
import torch.nn as nn

from typing import Any, Optional

from model.attention import attention_func
from model.llama.model import repeat_kv, FeedForward, RMSNorm

class LinearWithSample(nn.Linear):
    def __init__(self,
                 model_config,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device: Optional[Any] = None,
                 dtype: Optional[Any] = None):
        super().__init__(in_features,
                         out_features,
                         bias,
                         device,
                         dtype)
        self.sample_mode = model_config.multimodal_sample_mode
        self.k_tokens = model_config.multimodal_k_tokens

    def forward(self, x):
        if self.sample_mode == "last":
            restrict = lambda x: x[..., -self.k_tokens:, :]
        elif self.sample_mode == "first":
            restrict = lambda x: x[..., :self.k_tokens, :]
        elif self.sample_mode == "pool":
            restrict = lambda x: (
                torch.cumsum(x, dim=-2)
                / torch.arange(
                    1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                ).unsqueeze(-1)
            )[..., -self.k_tokens:, :]
        elif self.sample_mode == "sum":
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -self.k_tokens:, :]
            # TODO use same restrict function as pool case
        elif self.sample_mode == 'adaptive_average_pool':
            def restrict(x):
                x = x.transpose(1,2)
                pool = nn.AdaptiveAvgPool1d(self.k_tokens)
                x = pool(x)
                x = x.transpose(1,2)
                return x
        elif self.sample_mode == 'ragged':
            assert self.lengths is not None, "lengths must be provided for ragged mode"
            # remove any additional padding (beyond max length of any sequence in the batch)
            restrict = lambda x: x[..., : max(self.lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum']"
            )
        x = restrict(x)
    
        return super().forward(x)
    
class PerceiverResamplerAttention(nn.Module):
    def __init__(self, model_config, learned_query_dim: Optional[int] = None, kv_dim: Optional[int] = None):
        super().__init__()
        self.n_kv_heads = model_config.n_heads if model_config.n_kv_heads is None else model_config.n_kv_heads
        self.n_local_heads = model_config.n_heads 
        self.n_local_kv_heads = self.n_kv_heads 
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = model_config.dim // model_config.n_heads
        if learned_query_dim is None:
            learned_query_dim = model_config.dim
        if kv_dim is None:
            kv_dim = model_config.dim

        self.w_q = nn.Linear(
            learned_query_dim,
            model_config.n_heads * self.head_dim,
            bias=False,
        )
        self.w_k = nn.Linear(
            kv_dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.w_v = nn.Linear(
            kv_dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.w_o = nn.Linear(
            model_config.n_heads * self.head_dim,
            learned_query_dim,
            bias=False,
        )

        self.k_tokens = model_config.multimodal_k_tokens
        self.input_norm = RMSNorm(model_config.dim, model_config.norm_eps)
        self.learned_query_norm = RMSNorm(model_config.dim, model_config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        learned_query: torch.Tensor,
        atten_type: str = ''
    ):
        bsz, seq_len, _ = x.shape
        x = self.input_norm(x)
        learned_query = self.learned_query_norm(learned_query)

        kv_input = torch.cat([x, learned_query], dim=1)
        kv_input_len = seq_len + self.k_tokens
        xq, xk, xv = self.w_q(learned_query), self.w_k(kv_input), self.w_v(kv_input)

        xq = xq.view(bsz, self.k_tokens, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, kv_input_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, kv_input_len, self.n_local_kv_heads, self.head_dim)

        keys = repeat_kv(xk, self.n_rep)  # (bsz, learned_query_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bsz, learned_query_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bsz, n_local_heads, learned_query_len, head_dim)
        keys = keys.transpose(1, 2) # (bsz, n_local_heads, learned_query_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bsz, n_local_heads, learned_query_len + seqlen, head_dim)
        output = attention_func(q=xq, 
                                k=keys, 
                                v=values, 
                                atten_mask=None, 
                                dropout_p=0.0, 
                                scaling=1/math.sqrt(self.head_dim),
                                is_causal=False,
                                atten_type=atten_type) 
        # attention score (bsz, n_local_heads, learned_query_dim, kv_dim)
        # (bsz, n_local_heads, learned_query_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, self.k_tokens, -1)
        return self.w_o(output)
    
class PerceiverResamplerLayer(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.attention = PerceiverResamplerAttention(model_config)
        self.ffn = FeedForward(
                dim=model_config.dim,
                hidden_dim=4 * model_config.dim,
                multiple_of=model_config.multiple_of,
                ffn_dim_multiplier=model_config.ffn_dim_multiplier,
            )
        self.ffn_norm = RMSNorm(model_config.dim, model_config.norm_eps)

    def forward(self, x, learned_query):
        learned_query = learned_query +  self.attention(x, learned_query)
        learned_query = learned_query + self.ffn(self.ffn_norm(learned_query))
        return learned_query

class PerceiverResampler(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        multimodal_model_config = model_config.multimodal_model_config
        self.project_first = model_config.project_first
        self.project_final = not self.projec_first
        self.up_proj = nn.Linear(multimodal_model_config.dim, model_config.dim)

        if self.project_first:
            learned_query_dim = kv_dim = model_config.dim
            self.projector_norm = RMSNorm(model_config.dim, model_config.norm_eps)
            self.learned_query = nn.Parameter(
                torch.empty(
                    model_config.multimodal_k_tokens, 
                    model_config.dim
            ))
        else:
            learned_query_dim = kv_dim = multimodal_model_config.dim
            self.projector_norm = RMSNorm(multimodal_model_config.dim, model_config.norm_eps)
            self.learned_query = nn.Parameter(
                torch.empty(
                    model_config.multimodal_k_tokens, 
                    multimodal_model_config.dim
            ))

        self.layers = nn.ModuleList([])
        for _ in range(model_config.multimodal_projector_layers):
            self.layers.append(PerceiverResamplerLayer(model_config,
                                                       learned_query_dim = learned_query_dim,
                                                       kv_dim = kv_dim))


    def forward(self, x):
        # Project the encoder vector to word embedding space first.
        if self.project_first:
            x = self.up_proj(x)
        # Repeat the learned query in batch size.
        learned_query = self.learned_query.unsqueeze(0).expand(x.shape[0], -1, -1)
        # Compute the result through layers.
        for layer in self.layers:
            learned_query = layer(x, learned_query)
        learned_query = self.projector_norm(learned_query)
        if self.project_final:
            learned_query = self.up_proj(learned_query)
        # Return learned queries to large language model after a final norm of perceiver resampler.
        return self.projector_norm(learned_query)

class QFormer(nn.Module):
    pass

def get_multimodal_projector(model_config):
    multimodal_model_config = model_config.multimodal_model_config
    project_from = multimodal_model_config.dim
    project_to = model_config.dim

    projector_type = getattr(model_config, 'multimodal_projector_type', 'linear')
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if projector_type == 'linear':
        # If the projector type is linear, return LinearWithSample, which will reduce the sequence length first.
        return LinearWithSample(model_config,
                                project_from, 
                                project_to)
    elif projector_type == 'resampler':
        return PerceiverResampler(model_config)
    elif projector_type == 'qformer':
        return QFormer(model_config)
    elif projector_type == 'mlp':
        modules = [LinearWithSample(model_config,
                                    project_from, 
                                    project_to)]
        for _ in range(1, model_config.multimodal_projector_layers):
            modules.append(nn.GELU())
            modules.append(nn.Linear(project_to, project_to))
        return nn.Sequential(*modules)
