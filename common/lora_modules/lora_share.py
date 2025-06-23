# @author: minglei li
# @date: 2025-6-19
""" Implementation of Shared LoRA where A and B matrices are shared across all layers.
All layers share the same A and B parameters, which are trainable. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lora import LinearWithLoRA, LoRAConfig

class LinearWithSharedLoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig):
        """
        Initialize the LinearWithSharedLoRA layer.
        """
        self.share_lora_weights = True
        super().__init__(lora_config)

    def init_lora_weights(self):
        """
        For shared LoRA, we don't initialize A and B here.
        The shared A and B will be set externally via update_shared_weights.
        """
        pass

    def update_shared_weights(
        self,
        weight_a: nn.Parameter,
        weight_b: nn.Parameter
    ):
        """
        Update this layer to use shared LoRA weights.
        
        Args:
            weight_a: Shared A matrix parameter
            weight_b: Shared B matrix parameter
        """
        self.weight_a = weight_a
        self.weight_b = weight_b

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using shared LoRA weights.
        """
        # Get the required slices from shared matrices
        weight_a = self.weight_a[:self.lora_rank, :self.in_features].to(self._get_lora_dtype())
        weight_b = self.weight_b[:self.out_features, :self.lora_rank].to(self._get_lora_dtype())

        # Apply shared transformation
        lora_result = F.linear(F.linear(self.lora_dropout(x), weight_a), weight_b)
            
        return result + self.lora_scaler * lora_result.to(result.dtype)

    def _compute_lora_weight(self):
        """
        Compute the effective LoRA weight for this layer.
        """
        if self.has_lora_weights:
            weight_a = self.weight_a[:self.lora_rank, :self.in_features].to(self._get_lora_dtype())
            weight_b = self.weight_b[:self.out_features, :self.lora_rank].to(self._get_lora_dtype())

            lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)
            return lora_weight.to(self.weight.dtype)
        
    def _del_lora(self):
        """
        Delete LoRA weights. For shared LoRA, we only delete references.
        """
        # Don't delete shared weights, they are managed globally
        self.weight_a = None
        self.weight_b = None
        
        # Call parent to delete any individual weights that might exist
        try:
            super()._del_lora()
        except AttributeError:
            pass

    def print_details(self) -> None:
        """
        Print details about this shared LoRA layer.
        """
        print(f"{self.__class__.__name__} Layer: in_features={self.in_features}, out_features={self.out_features}")
        print(f"Shared LoRA Enabled: {self.has_shared_lora_weights}, LoRA Rank: {self.lora_rank}, Quantized: {self.quant}")


def prepare_shared_lora_weights(model: nn.Module, args) -> tuple[nn.Parameter, nn.Parameter]:
    """
    Prepare shared LoRA weights that will be used across all layers.
    
    Args:
        model: The model containing LoRA layers
        args: Arguments containing LoRA configuration
        
    Returns:
        Tuple of (weight_a, weight_b) Parameters
    """
    # Find the maximum dimensions needed across all layers
    max_in_features = 0
    max_out_features = 0
    
    for module in model.modules():
        if getattr(module, 'share_lora_weights', False):
            max_in_features = max(max_in_features, module.in_features)
            max_out_features = max(max_out_features, module.out_features)
    
    if max_in_features == 0 or max_out_features == 0:
        raise ValueError("No LinearWithLoRA layers with shared weights found in the model")
    
    # Create shared parameters
    dtype = torch.float32 if args.run_lora_in_fp32 else torch.float16
    device = next(model.parameters()).device
    
    # Initialize A matrix
    weight_a = nn.Parameter(
        torch.empty((args.lora_rank, max_in_features), dtype=dtype, device=device))
    
    # Initialize B matrix  
    weight_b = nn.Parameter(
        torch.zeros((max_out_features, args.lora_rank), dtype=dtype, device=device))
    
    # Initialize weights
    with torch.no_grad():
        if args.weight_a_init_method == 'kaiming':
            nn.init.kaiming_uniform_(weight_a, a=5**0.5, mode='fan_in')
        else:
            nn.init.normal_(weight_a, mean=0.0, std=1 / (max_in_features ** 0.5))
        
        if args.weight_b_init_method == 'kaiming':
            nn.init.kaiming_uniform_(weight_b, a=5**0.5, mode='fan_in')
        elif args.weight_b_init_method == 'normal':
            nn.init.normal_(weight_b, mean=0.0, std=0.02)

    model.weight_a = weight_a
    model.weight_b = weight_b


def update_shared_weights_to_layer(model: nn.Module):
    """
    Apply shared LoRA weights to all LinearWithSharedLoRA layers in the model.
    
    Args:
        model: The model containing LoRA layers
        weight_a: Shared A matrix parameter
        weight_b: Shared B matrix parameter
    """
    for module in model.modules():
        if getattr(module, 'share_lora_weights', False):
            module.update_shared_weights(model.weight_a, model.weight_b)