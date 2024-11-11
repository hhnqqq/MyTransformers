import math
import torch
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any

from model.llama.config import *
from model.attention import attention_func
from model.tokenizer import BaseTokenizer, Llama3Tokenizer


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0,
                         train_pi: Union[bool,None] = None,
                         train_pipeline: bool = False) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    if train_pi is not None:
        t = (torch.arange(end) / torch.tensor(train_pi)).to(freqs.device)
    # [end,dim]
    freqs = torch.outer(t, freqs).float() # Complex64
    # When utilizing pipeline parallelism, it is important to note that complex values should not be transposed between layers.
    if train_pipeline:
        return freqs
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)         
        return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_complex(freqs_cis):
        freqs_cis = torch.polar(torch.ones_like(freqs_cis), freqs_cis)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (Linear): Linear transformation for queries.
            wk (Linear): Linear transformation for keys.
            wv (Linear): Linear transformation for values.
            wo (Linear): Linear transformation for output.
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads 
        self.n_local_kv_heads = self.n_kv_heads 
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        atten_type: str = '',
        cache_kv = None
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.
            atten_type (str): type of attention function
            cache_kv (torch.Tensor): cached kv values

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache_kv is not None:
            cache_k, cache_v = cache_kv
            cache_k = cache_k.to(xq)
            cache_v = cache_v.to(xq)

            # cache_k.index_copy_(1, torch.tensor(start_pos, device=xk.device), xk)
            # cache_v.index_copy_(1, torch.tensor(start_pos, device=xv.device), xv)
            cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = cache_k[:bsz, : start_pos + seqlen]
            values = cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        is_causal = mask is None
        output = attention_func(q=xq, 
                                k=keys, 
                                v=values, 
                                atten_mask=mask, 
                                dropout_p=0.0, 
                                scaling=1/math.sqrt(self.head_dim),
                                is_causal=is_causal,
                                atten_type=atten_type) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (Linear): Linear transformation for the first layer.
            w2 (Linear): Linear transformation for the second layer.
            w3 (Linear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        atten_type: str = '',
        cache_kv = None
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(
                x=self.attention_norm(x), 
                start_pos=start_pos, 
                freqs_cis=freqs_cis, 
                mask=mask, 
                atten_type=atten_type, 
                cache_kv=cache_kv
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (lLinear): Linear layer for final output.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # can move this to llama attrs, but need to transfrom the pretrained ckpt
        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

    @torch.inference_mode()
    def forward(self, 
                tokens: torch.Tensor, 
                start_pos: int,
                freqs_cis: torch.Tensor, 
                atten_type: str = '',
                caches_kv=None,
                is_embed=False):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        if not is_embed:
            _bsz, seqlen = tokens.shape
            h = self.tok_embeddings(tokens)
        else:
            _bsz, seqlen, hidden_size = tokens.shape
            h = tokens
        freqs_cis = freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for i, layer in enumerate(self.layers):
            h = layer(x=h, 
                      start_pos=start_pos, 
                      freqs_cis=freqs_cis, 
                      mask=mask, 
                      atten_type=atten_type,
                      cache_kv=caches_kv[i] if caches_kv is not None else None)
        h = self.norm(h)
        output = self.output(h).float()
        return output
    
class LlamaGenerate(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model = Transformer(model_args)
        self.model_args = model_args
        # remove freqs_cis from transformer attrs
        self.freqs_cis = precompute_freqs_cis(
            model_args.dim // model_args.n_heads, 
            model_args.max_seq_len * 2,
            model_args.rope_theta
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        device,
        output_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        eos: bool = False
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        assert hasattr(self, "tokenizer"), "Can inference with out a provieded tokenizer"
        if isinstance(prompt_tokens, str):
            prompt_tokens = [prompt_tokens]
        prompt_tokens = [self.tokenizer.encode(x, eos=eos) for x in prompt_tokens]
        params = self.model.params
        bsz = len(prompt_tokens)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = output_len + max_prompt_len

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        eot_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id

        # remove kv cache from attention attrs, for unified API
        caches_kv = []
        for _ in range(params.n_layers):
            n_kv_heads = params.n_kv_heads if params.n_kv_heads else params.n_heads
            size = (bsz, params.max_seq_len, n_kv_heads,
                    params.head_dim)
            dtype = params.get_dtype()
            cache_k = torch.zeros(size=size, dtype=dtype, device=device)
            cache_v = torch.zeros(size=size, dtype=dtype, device=device)
            caches_kv.append((cache_k, cache_v))

        if min_prompt_len == total_len:
            logits = self.model.forward(tokens=tokens, 
                                        start_pos=prev_pos, 
                                        freqs_cis=self.freqs_cis,
                                        atten_type='',
                                        caches_kv=caches_kv)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens=tokens[:, prev_pos:cur_pos], 
                                        start_pos=prev_pos, 
                                        freqs_cis=self.freqs_cis, 
                                        atten_type='',
                                        caches_kv=caches_kv)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            eot_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eot_id
            )
            prev_pos = cur_pos
            if all(eos_reached) or all(eot_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_words, out_tokens, out_logprobs = [], [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + output_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + output_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            if self.tokenizer.eot_id in toks:
                eot_idx = toks.index(self.tokenizer.eot_id)
                toks = toks[:eot_idx]
                probs = probs[:eot_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
            out_words.append(self.tokenizer.decode(toks))
        return (out_words, out_tokens, out_logprobs if logprobs else None)


    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
    
    def load_weights(self, model_path: str):
        try:
            state_dict = torch.load(
                    model_path
                )['model_state_dict']
            self.model.load_state_dict(
                state_dict,
                strict=True,
            )
        except:
            state_dict = torch.load(
                    model_path
                )
            import re
            state_dict = {re.sub('module.','',k):v for k, v in state_dict.items()}
            state_dict = {re.sub('embedder.','tok_embeddings.',k):v for k, v in state_dict.items()}
            self.model.load_state_dict(
                state_dict,
                strict=True,
            )    

@registry.register_model(["llama", "llama1", "llama2"])
class Llama(LlamaGenerate):
    def __init__(self, model_args: ModelArgs):
        try:
            self.tokenizer = BaseTokenizer(model_args.tokenizer)
            model_args.vocab_size = self.tokenizer.n_words
        except:
            pass
        self.tokenizer.eos_id = 1000000
        super().__init__(model_args=model_args)

@registry.register_model("llama3")
class Llama3(LlamaGenerate):
    def __init__(self, model_args: ModelArgs):
        try:
            self.tokenizer = Llama3Tokenizer(model_args.tokenizer)
            model_args.vocab_size = self.tokenizer.n_words
        except:
            pass
        super().__init__(model_args=model_args)

