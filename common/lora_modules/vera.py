# @author: minglei li
# @date: 2025-6-19
"""Implementation of VeRA (Vector-based Random Matrix Adaptation) based on the paper:
https://arxiv.org/abs/2310.11454.

VeRA builds upon LoRA, where LoRA updates the weight matrix W by training two
low-rank matrices, A and B, with an intermediate rank r. In VeRA, these matrices
are frozen, shared across all layers, and adapted using trainable vectors d and b.
This significantly reduces the number of trainable parameters.

In both approaches, the low-rank matrices and vectors can be merged into the
original weight matrix W without introducing additional inference latency. This
means VeRA can maintain a much smaller number of trainable parameters even with
a larger rank.

This implementation provides true sharing of A and B matrices across layers,
similar to the HuggingFace PEFT implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LinearWithLoRA, LoRAConfig


class LinearWithVeRA(LinearWithLoRA):
    def __init__(self,
        lora_config: LoRAConfig,
        lambda_b_init_method: str = 'zero',
        lambda_d_init_method: str = 'small_constant',
    ):
        """
        Initialize the LinearWithVeRA layer.

        Args:
            lambda_b_init_method (str, optional): Initialization method for lambda b. ['zero', 'ones', 'small_constant', 'random']. Default is 'zero'.
            lambda_d_init_method (str, optional): Initialization method for lambda d. ['zero', 'ones', 'small_constant', 'random']. Default is 'small_constant'.
        """
        super().__init__(lora_config)
        self.share_lora_weights = True
        self.lambda_b_init_method = lambda_b_init_method
        self.lambda_d_init_method = lambda_d_init_method
        
        if lora_config.weight_b_init_method is None:
            raise ValueError('The init method for weight b in vera cannot be None.')
    
    def init_lora_weights(self):
        """
        For VeRA, we don't initialize A and B here as they are shared.
        Only initialize the lambda vectors.
        The shared A and B will be set externally via update_shared_weights.
        """
        self._init_lambdas(self.lambda_b_init_method, self.lambda_d_init_method)

    def update_shared_weights(
        self,
        weight_a,
        weight_b
    ):
        """
        Update this layer to use shared VeRA weights.
        
        Args:
            adapter_name: Name of the adapter
            weight_a: Shared A matrix buffer
            weight_b: Shared B matrix buffer
        """
        self.weight_a = weight_a
        self.weight_b = weight_b

    def _init_lambdas(self, b_init_method: str, d_init_method: str):
        """
        Initialize lambdas with different methods.
        Args:
            b_init_method: The method to initialize lambda b.
            d_init_method: The method to initialize lambda d.
        """
        # Initialize vector b
        dtype = self._get_lora_dtype()
        requires_grad = True  # Always make lambda parameters trainable
        
        if b_init_method == 'zero':
            lambda_b = torch.zeros(self.out_features, dtype=dtype)
        elif b_init_method == 'ones':
            lambda_b = torch.ones(self.out_features, dtype=dtype)
        elif b_init_method == 'small_constant':
            lambda_b = 0.1 * torch.ones(self.out_features, dtype=dtype)
        elif b_init_method == 'random':
            lambda_b = torch.rand(self.out_features, dtype=dtype)
        else:
            raise ValueError(f"Unknown b_init_method: {b_init_method}")
        
        # Initialize vector d
        if d_init_method == 'zero':
            lambda_d = torch.zeros(self.lora_rank, dtype=dtype)
        elif d_init_method == 'ones':
            lambda_d = torch.ones(self.lora_rank, dtype=dtype)
        elif d_init_method == 'small_constant':
            lambda_d = 0.1 * torch.ones(self.lora_rank, dtype=dtype)
        elif d_init_method == 'random':
            lambda_d = torch.rand(self.lora_rank, dtype=dtype)
        else:
            raise ValueError(f"Unknown d_init_method: {d_init_method}")
            
        self.lambda_b = nn.Parameter(lambda_b, requires_grad=requires_grad)
        self.lambda_d = nn.Parameter(lambda_d, requires_grad=requires_grad)

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        """
        Called by forward method in LinearWithLoRA
        Uses shared VeRA A and B matrices with layer-specific lambda vectors.
        """ 
        # Get shared matrices (sliced to current layer dimensions)
        weight_a = self.weight_a[:self.lora_rank, :self.in_features].to(self._get_lora_dtype())
        weight_b = self.weight_b[:self.out_features, :self.lora_rank].to(self._get_lora_dtype())
        
        lambda_b = self.lambda_b.to(self._get_lora_dtype())
        lambda_d = self.lambda_d.to(self._get_lora_dtype())
        
        # VeRA forward pass: lambda_b * linear(lambda_d * linear(x, A), B)
        lora_result = lambda_b * F.linear(
            lambda_d * F.linear(
                self.lora_dropout(x), 
                weight_a
            ), 
            weight_b
        )

        return result + self.lora_scaler * lora_result.to(result.dtype)
    
    def _compute_lora_weight(self):
        """
        Called by merge lora method in LinearWithLoRA
        Computes the effective weight using shared VeRA matrices and layer-specific lambdas.
        """
        if not self.has_vera_weights:
            return None
        
        # Get shared matrices (sliced to current layer dimensions)
        weight_a = self.weight_a[:self.lora_rank, :self.in_features]
        weight_b = self.weight_b[:self.out_features, :self.lora_rank]
        
        # Apply layer-specific scaling
        weight_a = weight_a.to(self._get_lora_dtype()) * self.lambda_d.to(self._get_lora_dtype()).unsqueeze(1)
        weight_b = weight_b.to(self._get_lora_dtype()) * self.lambda_b.to(self._get_lora_dtype()).unsqueeze(1)

        lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)
        return lora_weight.to(self.weight.dtype)
        
    @property
    def has_lora_weights(self):
        """
        Check if this layer has VeRA weights.
        """
        has_lambda_b = hasattr(self, 'lambda_b') and self.lambda_b is not None
        has_lambda_d = hasattr(self, 'lambda_d') and self.lambda_d is not None
        return has_lambda_b and has_lambda_d and super().has_lora_weights

    def _del_lora(self):
        """
        Delete VeRA weights. Only delete layer-specific parameters (lambdas).
        """
        if hasattr(self, 'lambda_b'):
            delattr(self, "lambda_b")
        if hasattr(self, 'lambda_d'):
            delattr(self, "lambda_d")
        
        # Don't delete shared matrices, they are managed globally