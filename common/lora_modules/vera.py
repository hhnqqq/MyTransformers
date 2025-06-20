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
from typing import Optional

from .lora import LinearWithLoRA, LoRAConfig
from ._buffer_dict import BufferDict


class LinearWithVeRA(LinearWithLoRA):
    def __init__(self,
        lora_config: LoRAConfig,
        lambda_b_init_method: str = 'zero',
        lambda_d_init_method: str = 'small_constant'
    ):
        """
        Initialize the LinearWithVeRA layer.

        Args:
            lambda_b_init_method (str, optional): Initialization method for lambda b. ['zero', 'ones', 'small_constant', 'random']. Default is 'zero'.
            lambda_d_init_method (str, optional): Initialization method for lambda d. ['zero', 'ones', 'small_constant', 'random']. Default is 'small_constant'.
        """
        super().__init__(lora_config)
        self.lambda_b_init_method = lambda_b_init_method
        self.lambda_d_init_method = lambda_d_init_method
        
        # References to shared VeRA A and B matrices
        self.vera_A: Optional[BufferDict] = None
        self.vera_B: Optional[BufferDict] = None
        
        if lora_config.weight_b_init_method is None:
            raise ValueError('The init method for weight b in vera cannot be None.')
    
    def init_lora_weights(self):
        """
        For VeRA, we don't initialize A and B here as they are shared.
        Only initialize the lambda vectors.
        The shared A and B will be set externally via update_vera_layer.
        """
        self._init_lambdas(self.lambda_b_init_method, self.lambda_d_init_method)

    def update_vera_layer(
        self,
        adapter_name: str,
        vera_A: BufferDict,
        vera_B: BufferDict
    ):
        """
        Update this layer to use shared VeRA weights.
        
        Args:
            adapter_name: Name of the adapter
            vera_A: Shared A matrix buffer
            vera_B: Shared B matrix buffer
        """
        self.vera_A = vera_A
        self.vera_B = vera_B

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
        if self.vera_A is None or self.vera_B is None:
            return result
            
        adapter_name = "default"  # Use default adapter name
        if adapter_name not in self.vera_A or adapter_name not in self.vera_B:
            return result
            
        # Get shared matrices (sliced to current layer dimensions)
        vera_A = self.vera_A[adapter_name][:self.lora_rank, :self.in_features].to(self._get_lora_dtype())
        vera_B = self.vera_B[adapter_name][:self.out_features, :self.lora_rank].to(self._get_lora_dtype())
        
        lambda_b = self.lambda_b.to(self._get_lora_dtype())
        lambda_d = self.lambda_d.to(self._get_lora_dtype())
        
        # VeRA forward pass: lambda_b * linear(lambda_d * linear(x, A), B)
        lora_result = lambda_b * F.linear(
            lambda_d * F.linear(
                self.lora_dropout(x), 
                vera_A
            ), 
            vera_B
        )

        return result + self.lora_scaler * lora_result.to(result.dtype)
    
    def _compute_lora_weight(self):
        """
        Called by merge lora method in LinearWithLoRA
        Computes the effective weight using shared VeRA matrices and layer-specific lambdas.
        """
        if not self.has_vera_weights:
            return None
            
        adapter_name = "default"
        
        # Get shared matrices (sliced to current layer dimensions)
        vera_A = self.vera_A[adapter_name][:self.lora_rank, :self.in_features]
        vera_B = self.vera_B[adapter_name][:self.out_features, :self.lora_rank]
        
        # Apply layer-specific scaling
        weight_a = vera_A.to(self._get_lora_dtype()) * self.lambda_d.to(self._get_lora_dtype()).unsqueeze(1)
        weight_b = vera_B.to(self._get_lora_dtype()) * self.lambda_b.to(self._get_lora_dtype()).unsqueeze(1)

        lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)
        return lora_weight.to(self.weight.dtype)
        
    @property
    def has_lora_weights(self):
        """
        Check if this layer has VeRA weights (shared matrices + lambdas).
        """
        return self.has_vera_weights

    @property 
    def has_vera_weights(self):
        """
        Check if this layer has VeRA weights.
        """
        has_lambda_b = hasattr(self, 'lambda_b') and self.lambda_b is not None
        has_lambda_d = hasattr(self, 'lambda_d') and self.lambda_d is not None
        has_vera_matrices = (self.vera_A is not None and self.vera_B is not None and
                           "default" in self.vera_A and "default" in self.vera_B)
        return has_lambda_b and has_lambda_d and has_vera_matrices
    
    def _del_lora(self):
        """
        Delete VeRA weights. Only delete layer-specific parameters (lambdas).
        """
        if hasattr(self, 'lambda_b'):
            delattr(self, "lambda_b")
        if hasattr(self, 'lambda_d'):
            delattr(self, "lambda_d")
        
        # Don't delete shared matrices, they are managed globally


def prepare_vera_shared_weights(model: nn.Module, args) -> tuple[BufferDict, BufferDict]:
    """
    Prepare shared VeRA weights that will be used across all layers.
    
    Args:
        model: The model containing VeRA layers
        args: Arguments containing VeRA configuration
        
    Returns:
        Tuple of (vera_A, vera_B) BufferDicts
    """
    # Find the maximum dimensions needed across all layers
    max_in_features = 0
    max_out_features = 0
    
    for module in model.modules():
        if isinstance(module, LinearWithVeRA):
            max_in_features = max(max_in_features, module.in_features)
            max_out_features = max(max_out_features, module.out_features)
    
    if max_in_features == 0 or max_out_features == 0:
        raise ValueError("No LinearWithVeRA layers found in the model")
    
    # Create shared buffers
    vera_A = BufferDict({}, persistent=True)
    vera_B = BufferDict({}, persistent=True)
    
    # Initialize shared weights with deterministic random initialization
    adapter_name = "default"
    dtype = torch.float32 if args.run_lora_in_fp32 else torch.float16
    device = next(model.parameters()).device
    
    # Use a fixed seed for reproducible initialization
    generator = torch.Generator().manual_seed(42)
    
    # Initialize A matrix (Kaiming uniform)
    weight_a = torch.empty((args.lora_rank, max_in_features), dtype=dtype, device=device)
    with torch.no_grad():
        if args.weight_a_init_method == 'kaiming':
            nn.init.kaiming_uniform_(weight_a, a=5**0.5, mode='fan_in')
        else:
            bound = (6.0 / max_in_features) ** 0.5
            nn.init.uniform_(weight_a, -bound, bound)
    
    # Initialize B matrix based on method
    weight_b = torch.empty((max_out_features, args.lora_rank), dtype=dtype, device=device)
    with torch.no_grad():
        if args.weight_b_init_method == 'kaiming':
            nn.init.kaiming_uniform_(weight_b, a=5**0.5, mode='fan_in')
        elif args.weight_b_init_method == 'normal':
            nn.init.normal_(weight_b, mean=0.0, std=0.02)
        elif args.weight_b_init_method == 'zeros':
            nn.init.zeros_(weight_b)
        else:
            nn.init.uniform_(weight_b, -0.01, 0.01)
    
    vera_A[adapter_name] = weight_a
    vera_B[adapter_name] = weight_b
    
    return vera_A, vera_B


def apply_vera_to_model(model: nn.Module, vera_A: BufferDict, vera_B: BufferDict):
    """
    Apply shared VeRA weights to all LinearWithVeRA layers in the model.
    
    Args:
        model: The model containing VeRA layers
        vera_A: Shared A matrix buffer
        vera_B: Shared B matrix buffer
    """
    adapter_name = "default"
    
    for module in model.modules():
        if isinstance(module, LinearWithVeRA):
            module.update_vera_layer(adapter_name, vera_A, vera_B)
    
    # Attach the shared buffers to the model for proper state dict handling
    if not hasattr(model, 'vera_A'):
        model.vera_A = vera_A
        model.vera_B = vera_B