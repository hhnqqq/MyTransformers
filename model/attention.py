import torch
import torch.nn.functional as F
from typing import Optional
from packaging import version
from contextlib import nullcontext
from deepspeed.sequence.layer import DistributedAttention
from torch.nn.attention import SDPBackend, sdpa_kernel

import common.utils.parallel_states as parallel_states

def naive_attention_func(
   q: torch.Tensor,
   k: torch.Tensor,
   v: torch.Tensor,
   attn_mask: Optional[torch.Tensor],
   dropout_p: float,
   scale: float,
   is_causal: bool
):
    """
    The general attention implementation.

    Args:
        q (torch.Tensor): Query tensor of shape [batch_size, n_local_heads, input_len, head_dim].
        k (torch.Tensor): Key tensor of shape [batch_size, n_local_heads, input_len, head_dim].
        v (torch.Tensor): Value tensor of shape [batch_size, n_local_heads, input_len, head_dim].
        atten_mask (torch.Tensor): Attention mask tensor of shape [batch_size, 1, 1, input_len].
        dropout_p (float): Dropout probability.
        scale (float): Scaling factor for the attention scores.
        is_causal (bool): Whether the attention is causal (only attend to past tokens).

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, n_local_heads, input_len, head_dim].
    """
    q_len, k_len = q.size(-2), k.size(-2)
    scores = torch.matmul(q, k.transpose(2, 3)) * scale
    atten_bias = torch.zeros(q_len, k_len, dtype=q.dtype)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(q_len, k_len, dtype=torch.bool).tril(diagonal=0)
        atten_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        atten_bias = atten_bias.to(q.dtype).to(q.device)
        scores = scores + atten_bias
    elif attn_mask is not None:
        scores = scores + attn_mask

    scores = F.softmax(scores.float(), dim=-1).type_as(q)
    scores = torch.dropout(scores, dropout_p, train=True)
    output = torch.matmul(scores, v)
    return output


def attention_func(
   q: torch.Tensor,
   k: torch.Tensor,
   v: torch.Tensor,
   atten_mask: Optional[torch.Tensor],
   dropout_p: float,
   scaling: float,
   is_causal: bool,
   atten_type: str = ''
):
    """
    Attention function that supports different attention types.

    Args:
        q (torch.Tensor): Query tensor of shape [batch_size, n_local_heads, input_len, head_dim].
        k (torch.Tensor): Key tensor of shape [batch_size, n_local_heads, input_len, head_dim].
        v (torch.Tensor): Value tensor of shape [batch_size, n_local_heads, input_len, head_dim].
        atten_mask (torch.Tensor): Attention mask tensor of shape [batch_size, 1, input_len, input_len].
        dropout_p (float): Dropout probability.
        scaling (float): Scaling factor for the attention scores.
        is_causal (bool): Whether the attention is causal (only attend to past tokens).
        atten_type (str, optional): Type of attention to use. Can be 'flash_atten', 'ulysses_flash_atten', 'ulysses_atten', or leave empty for the default naive_attention_func.

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, n_local_heads, input_len, head_dim].
    """
    atten_func = F.scaled_dot_product_attention
    ctx_manager = nullcontext()

    if is_causal and atten_mask is not None:
        is_causal = False

    if version.parse(torch.__version__) > version.parse("2.0"):
        # This context manager is beta and subject to change.
        if 'flash' in atten_type:
            ctx_manager = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
            if atten_mask is not None:
            # Flash Attention does not support non-null attn_mask. 
            # (Triggered internally at ../aten/src/ATen/native/transformers/sdp_utils_cpp.h:271.) 
            # Triggered at torch 2.4 and higher.
                atten_mask = None
                is_causal = True

        elif 'memory_efficient' in atten_type:
            ctx_manager = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        elif 'math' in atten_type:
            ctx_manager = sdpa_kernel(SDPBackend.MATH)
        elif 'all' in atten_type:
            # Ensure that optimized attention is used
            ctx_manager = sdpa_kernel([SDPBackend.FLASH_ATTENTION, 
                                       SDPBackend.EFFICIENT_ATTENTION,
                                       SDPBackend.MATH])

    if 'ulysses' in atten_type:
        # Enables sequence parallel attention computation.
        atten_func = DistributedAttention(atten_func, parallel_states.get_sequence_parallel_group(), scatter_idx=1, gather_idx=2)
    
    with ctx_manager:
        output = atten_func(q, k, v, attn_mask=atten_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scaling)

    return output