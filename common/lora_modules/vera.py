""" Implements VeRA"""
import torch
import torch.nn as nn

from common.lora_modules.lora import *
    
class LinearWithVeRA(LinearWithLoRA):
    def __init__(self,
        lora_config: LoRAConfig,
        scaling_vector_b_init_method: str = 'zero',
        scaling_vector_d_init_method: str = 'ones'
    ):
        """
        Initialize the LinearWithVeRA layer.

        Args:
            scaling_vector_b_init_method (str, optional): Initialization method for scaling vector b. ['zero', 'ones', 'small_constant', 'random']. Default is 'zeros'.
            scaling_vector_d_init_method (str, optional): Initialization method for scaling vector d. ['zero', 'ones', 'small_constant', 'random']. Default is 'ones'.
        """
        super().__init__(lora_config)
        self._init_scaling_vectors(scaling_vector_b_init_method, scaling_vector_d_init_method)
    
    def _init_lora_weights(self):
        dtype = torch.int8 if self.quant else None

        # Initialize shared matrix a and b, and frozen them.
        self.weight_a = nn.Parameter(torch.randn((self.lora_rank, self.in_features), dtype=dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(torch.randn((self.out_features, self.lora_rank), dtype=dtype), requires_grad=requires_grad)

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

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

            lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)

            return lora_weight
        