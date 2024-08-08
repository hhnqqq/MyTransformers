import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import asdict
from einops import rearrange
from functools import partial
from functools import partial
from model.dnahyena.config import HyenaConfig

def stochastic_depth(input: torch.Tensor, p: float, mode: str, training: bool = True) -> torch.Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise

def next_power_of_2(x):
    return 2 ** math.ceil(math.log2(x))

def fftconv(u, k, D):
    """
    We apply a convolution through the fourier domain (from the Convolution Theorem)
    """
    seqlen = u.shape[-1]
    fft_size = next_power_of_2(2 * seqlen)

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


    
class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s


class HyenaSin(nn.Module):
    """The Sin activation function for the Hyena Filter function."""
    def __init__(self, layer_config):
        super().__init__()
        self.freq = nn.Parameter((layer_config.activation_freq * torch.ones(1, layer_config.filter_order)) 
                                 if layer_config.train_freq 
                                 else layer_config.activation_freq * torch.ones(1, layer_config.filter_order))

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    def __init__(self, hyena_config: HyenaConfig):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = hyena_config.layer_config.l_max
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None] # 1, L, 1

        if hyena_config.emb_dim > 1:
            bands = (hyena_config.emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, self.seq_len - 1, self.seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / self.seq_len # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)

        self.register_buffer("z", z)
        self.register_buffer("t", t)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    """The window function applied to the output of the (MLP) filter function."""
    def __init__(
        self,
        dim,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulate: bool=True,
        shift: float = 0.05,
        **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, dim)[None, None]
        self.register_buffer("deltas", deltas)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    def __init__(
            self,
            hyena_config: HyenaConfig,
            **kwargs
        ):
        """
        Implicit long filter with modulation.

        Args:
            dim: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """
        super().__init__()

        layer_config = hyena_config.layer_config
        self.dim = hyena_config.dim * (hyena_config.order - 1)
        self.use_bias = layer_config.use_bias
        self.bias = nn.Parameter(torch.randn(self.dim))
        self.dropout = nn.Dropout(layer_config.filter_dropout)

        act = HyenaSin(layer_config)
        self.emb_dim = hyena_config.emb_dim
        assert hyena_config.emb_dim % 2 != 0 and hyena_config.emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = layer_config.l_max

        self.pos_emb = PositionalEmbedding(hyena_config)

        self.implicit_filter = nn.Sequential(
            nn.Linear(self.emb_dim, layer_config.filter_order),
            act,
        )
        for i in range(layer_config.num_inner_mlps):
            self.implicit_filter.append(nn.Linear(layer_config.filter_order, layer_config.filter_order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(layer_config.filter_order, layer_config.dim, bias=False))

        self.modulation = ExponentialModulation(self.dim, **kwargs)

        self.normalized = False

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k

        y = fftconv(x, k, bias)
        return y


class HyenaOperator(nn.Module):
    def __init__(
            self,
            hyena_config: HyenaConfig,
            **filter_args,
        ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            dim (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()

        layer_config = hyena_config.layer_config
        self.dim = hyena_config.dim
        self.order = hyena_config.order
        self.l_max = layer_config.l_max
        inner_width = self.dim * (self.order + 1)
        self.dropout = nn.Dropout(layer_config.dropout)
        self.in_proj = nn.Linear(hyena_config.dim, inner_width)
        self.out_proj = nn.Linear(hyena_config.dim, layer_config.dim)

        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            hyena_config.short_filter_order,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            hyena_config,
            **filter_args
        )

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')

        uc = self.short_filter(u)[...,:l_filter]
        *x, v = uc.split(self.dim, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, '(o d) -> o d', o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], 'b d l -> b l d')

        y = self.out_proj(y)
        return y


class Mlp(nn.Module):

    def __init__(self, 
                 hyena_config, 
                 activation_func,
                 device=None, 
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = hyena_config.dim
        hidden_features = hyena_config.d_inner if hyena_config.d_inner is not None else 4 * hyena_config.dim
        self.return_residual = hyena_config.return_residual
        self.fc1 = nn.Linear(hyena_config.dim, hidden_features, **factory_kwargs)
        self.activation = activation_func
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

#@title Block layer (Hyena + MLP layers)

class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input

class Block(nn.Module):

    def __init__(self, 
                 hyena_config, 
                 mixer_cls=None, 
                 mlp_cls=None, 
                 norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/block.py
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.
        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.prenorm = hyena_config.prenorm
        self.return_residual = hyena_config.return_residual
        self.residual_in_fp32 = hyena_config.residual_in_fp32
        if self.residual_in_fp32:
            assert self.prenorm, 'residual_in_fp32 is only compatible with prenorm=True'
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * hyena_config.dim)
        self.mixer = mixer_cls()
        self.dropout1 = dropout_cls(hyena_config.resid_dropout1)
        self.drop_path1 = StochasticDepth(hyena_config.drop_path1, mode='row')
        self.norm1 = norm_cls(hyena_config.dim)
        self.mlp = mlp_cls(hyena_config)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(hyena_config.resid_dropout2)
            self.drop_path2 = StochasticDepth(hyena_config.drop_path2, mode='row')
            self.norm2 = norm_cls(hyena_config.dim)

    def forward(self, hidden_states, residual = None,
                mixer_subset=None, mixer_kwargs=None):
        r"""Pass the input through the encoder layer.
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        if self.prenorm:
            dropped = self.drop_path1(self.dropout1(hidden_states))
            # dropped = self.dropout1(hidden_states)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                dropped = self.drop_path2(self.dropout2(hidden_states))
                # dropped = self.dropout2(hidden_states)
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {})
            )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out

            hidden_states = self.norm1((self.drop_path1(self.dropout1(mixer_out))
                                        + hidden_states).to(dtype=self.norm1.weight.dtype))

            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out

                hidden_states = self.norm2((self.drop_path2(self.dropout2(mlp_out))
                                            + hidden_states).to(dtype=self.norm2.weight.dtype))

            return hidden_states


def create_mlp_cls(hyena_config):
    inner_dim = hyena_config.d_inner if hyena_config.d_inner is not None else 4 * hyena_config.dim

    mlp_cls = partial(Mlp, 
                      activation_func=partial(F.gelu, approximate='tanh'))
    return mlp_cls


def create_block(hyena_config: HyenaConfig,
                 layer_idx=None):
    mixer_cls = partial(HyenaOperator, hyena_config)
    mlp_cls = create_mlp_cls(hyena_config)
    norm_cls = partial(nn.LayerNorm, eps=hyena_config.layer_norm_epsilon)
    block = Block(hyena_config, 
                  mixer_cls, 
                  mlp_cls, 
                  norm_cls)
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, 
                  hyena_config,
                  rescale_prenorm_residual=True,
                  glu_act=False):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=hyena_config.initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=hyena_config.initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=hyena_config.initializer_range / math.sqrt(2 * hyena_config.n_layer))
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    nn.init.normal_(p, mean=0.0, std=hyena_config.initializer_range / math.sqrt(2 * hyena_config.n_layer))
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    nn.init.normal_(p[:out_features // 2], mean=0.0, std=hyena_config.initializer_range / math.sqrt(2 * hyena_config.n_layer) * 2)

#@title Backbone model (stack of blocks)

"""
A backbone model consists of a stack of blocks. If you use attention, then
positional embeddings are included. When using Hyena, then the pos emb
revert to doing nothing.
"""

class GPT2Embeddings(nn.Module):

    def __init__(self, hyena_config: HyenaConfig):
        """
            If max_position_embeddings <= 0, there's no position embeddings
            If word_embe_proj_dim is not None (e.g., OPT-350m), we embed to that dimension
                the project up to embed_dim
        """
        factory_kwargs = {'device': torch.device(hyena_config.device), 'dtype': hyena_config.get_dtype()}
        super().__init__()
        if hyena_config.word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(hyena_config.vocab_size, 
                                                hyena_config.dim, 
                                                padding_idx=hyena_config.padding_idx,
                                                **factory_kwargs)
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(hyena_config.vocab_size, 
                                                hyena_config.word_embed_proj_dim,
                                                padding_idx=hyena_config.padding_idx, 
                                                **factory_kwargs)
            self.project_in = nn.Linear(hyena_config.word_embed_proj_dim, 
                                        hyena_config.dim, 
                                        bias=False,
                                        **factory_kwargs)
        self.max_position_embeddings = hyena_config.max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(hyena_config.max_position_embeddings, 
                                                    hyena_config.dim,
                                                    **factory_kwargs)

    def forward(self, input_ids, position_ids=None):
        """
            input_ids: (batch, seqlen)
            position_ids: (batch, seqlen)
        """
        if input_ids.dim() == 2:
            batch_size, seqlen = input_ids.shape
        else:
            seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings

class LMBackbone(nn.Module):

    def __init__(self, hyena_config:HyenaConfig, **kwargs) -> None:
        super().__init__()
        # note max_position_embeddings is 0 for Hyena, and therefore isn't used
        self.embeddings = GPT2Embeddings(hyena_config)

        self.layers = nn.ModuleList([create_block(
            hyena_config,
            layer_idx=i) 
            for i in range(hyena_config.n_layer)])

        self.drop_f = nn.Dropout(hyena_config.resid_dropout)
        self.ln_f = nn.LayerNorm(hyena_config.dim, 
                                 eps=hyena_config.layer_norm_epsilon, 
                                 device=torch.device(hyena_config.device))
        initialize_config = hyena_config.initializer_cfg if hyena_config.initializer_cfg is not None else {}

        self.apply(partial(_init_weights, hyena_config=hyena_config, **(initialize_config)))

    def forward(self, input_ids, position_ids=None, is_embed=False):
        if not is_embed:
            hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))

        return hidden_states

#@title Decoder head layer

class SequenceDecoder(nn.Module):
    def __init__(
        self, 
        hyena_config: HyenaConfig
    ):
        super().__init__()


        if hyena_config.l_output is None:
            self.l_output = None
            self.squeeze = False
        elif hyena_config.l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert hyena_config.l_output > 0
            self.l_output = hyena_config.l_output
            self.squeeze = False

        self.use_lengths = hyena_config.use_lengths
        self.mode = hyena_config.mode

        if hyena_config.mode == 'ragged':
            assert not hyena_config.use_lengths

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, dim)
        Returns: (n_batch, l_output, d_output)
        """

        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(-2)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            restrict = lambda x: x[..., -l_output:, :]
        elif self.mode == "first":
            restrict = lambda x: x[..., :l_output, :]
        elif self.mode == "pool":
            restrict = lambda x: (
                torch.cumsum(x, dim=-2)
                / torch.arange(
                    1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                ).unsqueeze(-1)
            )[..., -l_output:, :]

            # def restrict(x):
            #     L = x.size(-2)
            #     s = x.sum(dim=-2, keepdim=True)
            #     if l_output > 1:
            #         c = torch.cumsum(x[..., -(l_output - 1) :, :].flip(-2), dim=-2)
            #         c = F.pad(c, (0, 0, 1, 0))
            #         s = s - c  # (B, l_output, D)
            #         s = s.flip(-2)
            #     denom = torch.arange(
            #         L - l_output + 1, L + 1, dtype=x.dtype, device=x.device
            #     )
            #     s = s / denom
            #     return s

        elif self.mode == "sum":
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]
            # TODO use same restrict function as pool case
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"
            # remove any additional padding (beyond max length of any sequence in the batch)
            restrict = lambda x: x[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(-2) == 1
            x = x.squeeze(-2)

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        return self.output_transform(x)

#@title Model (backbone + head)
class HyenaDNAModel(nn.Module):
    def __init__(self, hyena_config: HyenaConfig, **kwargs) -> None:
        super().__init__()

        # def tie_weights(self):
        #     self.head.weight = self.backbone.embeddings.word_embeddings.weight
            
        if hyena_config.vocab_size % hyena_config.pad_vocab_size_multiple != 0:
            hyena_config.vocab_size += hyena_config.pad_vocab_size_multiple - (hyena_config.vocab_size % hyena_config.pad_vocab_size_multiple)

        # check if layer (config) has dim (HF code differs from main Safari code)
        if not hasattr(hyena_config.layer_config, 'dim') or hyena_config.layer_config.dim is None:
            hyena_config.layer_config.dim = hyena_config.dim

        self.backbone = LMBackbone(hyena_config, **kwargs)
        self.use_head = hyena_config.use_head
        initialize_config = hyena_config.initializer_cfg if hyena_config.initializer_cfg is not None else {}

        # we only need a head if doing classification, otherwise we'll use the
        # hidden states as embeddings
        if self.use_head:
            self.head = SequenceDecoder(hyena_config)
            # self.tie_weights()
        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, hyena_config=hyena_config, **(initialize_config)))

    def forward(self, input_ids, position_ids=None, state=None, is_embed=False): # state for the repo interface
        hidden_states = self.backbone(input_ids, position_ids=position_ids, is_embed=is_embed)

        if self.use_head:
            return self.head(hidden_states)
        else:
            return hidden_states

if __name__ == '__main__':
    from common.registry import registry

    hyena_config = registry.get_model_config_class('hyena_large_1m')()
    hyena_config.device='cpu'
    hyena_config.mode='pool'
    model = HyenaDNAModel(hyena_config=hyena_config)

    # 准备输入数据
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, hyena_config.vocab_size, (batch_size, seq_len), device=hyena_config.device)
    position_ids = torch.arange(seq_len, device=hyena_config.device).unsqueeze(0).expand(batch_size, -1)

    # 测试前向传播
    if model.use_head:
        outputs = model(input_ids, position_ids)
        print(outputs)
        print(f"Output shape: {outputs.shape}")  # 输出形状为 (batch_size, n_classes)
    else:
        hidden_states = model(input_ids, position_ids)
        print(f"Hidden states shape: {hidden_states.shape}")  # 输出形状为 (batch_size, seq_len, dim)

    # 测试模型参数
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # 测试梯度计算
    if model.use_head:
        outputs.sum().backward()
    else:
        hidden_states.mean().backward()

    print("Gradient check passed!")