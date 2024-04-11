import torch
import torch.nn.functional as F
from transformers.utils.versions import require_version
from deepspeed.sequence.layer import DistributedAttention
import common.utils.parallel_states as parallel_states

def naive_attention_func(
   q: torch.Tensor,
   k: torch.Tensor,
   v: torch.Tensor,
   atten_mask: torch.Tensor,
   dropout_p: float,
   scaling: float,
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
       scaling (float): Scaling factor for the attention scores.
       is_causal (bool): Whether the attention is causal (only attend to past tokens).

   Returns:
       torch.Tensor: Output tensor of shape [batch_size, n_local_heads, input_len, head_dim].
   """
   L, S = q.size(-2), k.size(-2)
   scores = torch.matmul(q, k.transpose(2, 3)) * scaling
   atten_bias = torch.zeros(L, S, dtype=q.dtype)

   if is_causal:
       assert atten_mask is None
       temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
       atten_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
       atten_bias = atten_bias.to(q.dtype)
       scores = scores + atten_bias

   scores = F.softmax(scores.float(), dim=-1).type_as(q)
   scores = torch.dropout(scores, dropout_p, train=True)
   output = torch.matmul(scores, v)
   return output

def attention_func(
   q: torch.Tensor,
   k: torch.Tensor,
   v: torch.Tensor,
   atten_mask: torch.Tensor,
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
       atten_mask (torch.Tensor): Attention mask tensor of shape [batch_size, 1, 1, input_len].
       dropout_p (float): Dropout probability.
       scaling (float): Scaling factor for the attention scores.
       is_causal (bool): Whether the attention is causal (only attend to past tokens).
       atten_type (str, optional): Type of attention to use. Can be 'flash_atten', 'ulysses_flash_atten', 'ulysses_atten', or leave empty for the default naive_attention_func.

   Returns:
       torch.Tensor: Output tensor of shape [batch_size, n_local_heads, input_len, head_dim].
   """
   if atten_type == 'flash_atten':
       require_version("torch>=2.0.0")
       with torch.backends.cuda.sdp_kernel(enable_flash=True):
           output = F.scaled_dot_product_attention(q, k, v, attn_mask=atten_mask, dropout_p=dropout_p, is_causal=is_causal)
   elif atten_type == 'ulysses_flash_atten':
       require_version("torch>=2.0.0")
       with torch.backends.cuda.sdp_kernel(enable_flash=True):
           flash_atten = F.scaled_dot_product_attention
           dist_atten = DistributedAttention(flash_atten, parallel_states.get_sequence_parallel_group(), scatter_idx=1, gather_idx=2)
           output = dist_atten(q, k, v, attn_mask=atten_mask, dropout_p=dropout_p, is_causal=is_causal)
   elif atten_type == 'ulysses_atten':
       dist_atten = DistributedAttention(naive_attention_func, parallel_states.get_sequence_parallel_group(), scatter_idx=1, gather_idx=2)
       output = dist_atten(q, k, v, atten_mask=atten_mask, dropout_p=dropout_p, scaling=scaling, is_causal=is_causal)
   else:
       output = naive_attention_func(q, k, v, atten_mask=atten_mask, dropout_p=dropout_p, scaling=scaling, is_causal=is_causal)
   return output