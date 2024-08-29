""" Implements VeRA"""
import torch
import torch.nn as nn

from common.lora_modules.lora import *
    
class LinearWithVeRA(LinearWithLoRA):
    def __init__(self,
        in_features: int,
        out_features: int,
        lora_rank: int = 4,
        lora_scaler: float = 32.0,
        lora_dropout: Optional[float] = None,
        use_dora: bool = False,
        use_mos_lora: bool = False,
        quant: bool = False,
        plora_steps: Union[int, None] = None,
        weight_a_init_method: Optional[str] = None,
        weight_b_init_method: Optional[str] = None,
        weight_ab_mixer_init_method: Optional[str] = None,
        scaling_vector_b_init_method: str = 'zero',
        scaling_vector_d_init_method: str = 'ones'
    ):
        """
        Initialize the LinearWithVeRA layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            lora_rank (int, optional): Rank of LoRA decomposition. Default is 4.
            lora_scaler (float, optional): Scaler for LoRA weights. Default is 32.0.
            use_dora (bool, optional): Whether to use DoRA (Weight-Decomposed Low-Rank Adaptation). Default is False.
            use_mos_lora (bool, optional): Whether to use MosLoRA (Mixture-of-Subspaces in Low-Rank Adaptation). Default is False.
            quant (bool, optional): Whether to apply weight quantization. Default is False.
            plora_steps (Union(int, None), optional): Steps to merge and reset lora weight.  Deault is None.
            scaling_vector_b_init_method (str, optional): Initialization method for scaling vector b. ['zero', 'ones', 'small_constant', 'random']. Default is 'zeros'.
            scaling_vector_d_init_method (str, optional): Initialization method for scaling vector d. ['zero', 'ones', 'small_constant', 'random']. Default is 'ones'.
        """
        super().__init__(in_features,
                         out_features,
                         lora_rank,
                         lora_scaler,
                         lora_dropout,
                         use_dora,
                         use_mos_lora,
                         quant,
                         plora_steps,
                         weight_a_init_method,
                         weight_b_init_method,
                         weight_ab_mixer_init_method)
        self._init_scaling_vectors(scaling_vector_b_init_method, scaling_vector_d_init_method)
    
    def _init_lora_weights(self):
        dtype = torch.int8 if self.quant else None
        requires_grad = not self.quant

        # Initialize shared matrix a and b, and frozen them.
        self.weight_a = nn.Parameter(torch.randn((self.lora_rank, self.in_features), dtype=dtype), requires_grad=False)
        self.weight_b = nn.Parameter(torch.randn((self.out_features, self.lora_rank), dtype=dtype), requires_grad=False)

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

        if self.mos_lora:
            self.weight_ab_mixer = nn.Parameter(torch.empty((self.lora_rank, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
            if self.quant:
                self.weight_ab_mixer_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self._init_weight('weight_ab_mixer')

        self._init_weight('weight_a')
        self._init_weight('weight_b')
    
    def _init_scaling_vectors(self, b_init_method: str, d_init_method: str):
        """
        Initialize scaling vectors with different methods.
        Args:
            b_init_method: The method to initialize scaling vector b.
            d_init_method: The method to initialize scaling vector d.
        """
        # Initialize vector b
        if b_init_method == 'zero':
            self.scaling_vector_b = torch.zeros(self.out_features)
        elif b_init_method == 'ones':
            self.scaling_vector_b = torch.ones(self.out_features)
        elif b_init_method == 'small_constant':
            self.scaling_vector_b = 0.1 * torch.ones(self.out_features)
        elif b_init_method == 'random':
            self.scaling_vector_b = torch.rand(self.out_features)
        else:
            raise ValueError(f"Unknown b_init_method: {b_init_method}")

        # Initialize vector d
        if d_init_method == 'zero':
            self.scaling_vector_d = torch.zeros(self.lora_rank)
        elif d_init_method == 'ones':
            self.scaling_vector_d = torch.ones(self.lora_rank)
        elif d_init_method == 'small_constant':
            self.scaling_vector_d = 0.1 * torch.ones(self.lora_rank)
        elif d_init_method == 'random':
            self.scaling_vector_d = torch.rand(self.lora_rank)
        else:
            raise ValueError(f"Unknown d_init_method: {d_init_method}")

    def _compute_lora(self): 
        if self.has_lora_weights:
            # Compute adapted lora weights.
            adapted_weight_a = torch.matmul(self.weight_a, torch.diag(self.scaling_vector_b))
            adapted_weight_b = torch.matmul(self.weight_b, torch.diag(self.scaling_vector_d))
            # Compute lora weights.
            weight_a = self._quantize_weight(adapted_weight_a, self.weight_a_quantizer)
            weight_b = self._quantize_weight(adapted_weight_b, self.weight_b_quantizer)
            if self.mos_lora:
                # When using vanilla lora, the ab mixer is a identical matrix
                weight_ab_mixer = self._quantize_weight(self.weight_ab_mixer, self.weight_ab_quantizer)
                weight_a_forward = torch.matmul(weight_ab_mixer, weight_a)
                lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a_forward)
            else:
                lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)

            return lora_weight
