# @author: minglei li
# @date: 2025-6-19
""" Implementation of Shared LoRA where A and B matrices are shared across all layers.
All layers share the same A and B parameters, which are trainable. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .lora import LinearWithLoRA, LoRAConfig


class LinearWithSharedLoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig):
        """
        Initialize the LinearWithSharedLoRA layer.
        """
        super().__init__(lora_config)
        # References to shared A and B parameters
        self.shared_lora_A: Optional[nn.Parameter] = None
        self.shared_lora_B: Optional[nn.Parameter] = None

    def init_lora_weights(self):
        """
        For shared LoRA, we don't initialize A and B here.
        The shared A and B will be set externally via update_shared_layer.
        """
        pass

    def update_shared_layer(
        self,
        shared_lora_A: nn.Parameter,
        shared_lora_B: nn.Parameter
    ):
        """
        Update this layer to use shared LoRA weights.
        
        Args:
            shared_lora_A: Shared A matrix parameter
            shared_lora_B: Shared B matrix parameter
        """
        self.shared_lora_A = shared_lora_A
        self.shared_lora_B = shared_lora_B

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using shared LoRA weights.
        """
        if self.shared_lora_A is None or self.shared_lora_B is None:
            return result
            
        # Get the required slices from shared matrices
        weight_a = self.shared_lora_A[:self.lora_rank, :self.in_features].to(self._get_lora_dtype())
        weight_b = self.shared_lora_B[:self.out_features, :self.lora_rank].to(self._get_lora_dtype())

        # Apply shared transformation
        lora_result = F.linear(F.linear(self.lora_dropout(x), weight_a), weight_b)
            
        return result + self.lora_scaler * lora_result.to(result.dtype)

    def _compute_lora_weight(self):
        """
        Compute the effective LoRA weight for this layer.
        """
        if self.shared_lora_A is None or self.shared_lora_B is None:
            return None
            
        weight_a = self.shared_lora_A[:self.lora_rank, :self.in_features].to(self._get_lora_dtype())
        weight_b = self.shared_lora_B[:self.out_features, :self.lora_rank].to(self._get_lora_dtype())

        lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)
        return lora_weight.to(self.weight.dtype)

    def _merge_lora(self) -> bool:
        """
        Merge the shared LoRA weight into full rank weight if possible.
        """
        if self.has_shared_lora_weights:
            lora_weight = self._compute_lora_weight()
            if lora_weight is not None:
                self.weight.data += lora_weight
                return True
        return False

    def _unmerge_lora(self) -> bool:
        """
        Unmerge the shared LoRA weight from full rank weight.
        """
        if self.has_shared_lora_weights:
            lora_weight = self._compute_lora_weight()
            if lora_weight is not None:
                self.weight.data -= lora_weight
                return True
        return False

    @property
    def has_lora_weights(self):
        """
        Check if this layer has LoRA weights (either individual or shared).
        For shared LoRA, we check for shared weights.
        """
        return self.has_shared_lora_weights or super().has_lora_weights

    @property
    def has_shared_lora_weights(self):
        """
        Check if this layer has shared LoRA weights.
        """
        return (self.shared_lora_A is not None and 
                self.shared_lora_B is not None)

    def _del_lora(self):
        """
        Delete LoRA weights. For shared LoRA, we only delete references.
        """
        # Don't delete shared weights, they are managed globally
        self.shared_lora_A = None
        self.shared_lora_B = None
        
        # Call parent to delete any individual weights that might exist
        try:
            super()._del_lora()
        except AttributeError:
            pass

    def merge_and_del(self):
        """
        Merge and delete LoRA weights.
        """
        if self._merge_lora():
            self._del_lora()
            if self.quant:
                # Reset quantization scalers if they exist
                if hasattr(self, 'weight_a_scaler'):
                    self.weight_a_scaler = None
                if hasattr(self, 'weight_b_scaler'):
                    self.weight_b_scaler = None

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
        Tuple of (shared_lora_A, shared_lora_B) Parameters
    """
    # Find the maximum dimensions needed across all layers
    max_in_features = 0
    max_out_features = 0
    
    for module in model.modules():
        if isinstance(module, LinearWithSharedLoRA):
            max_in_features = max(max_in_features, module.in_features)
            max_out_features = max(max_out_features, module.out_features)
    
    if max_in_features == 0 or max_out_features == 0:
        raise ValueError("No LinearWithSharedLoRA layers found in the model")
    
    # Create shared parameters
    dtype = torch.float32 if args.run_lora_in_fp32 else torch.float16
    device = next(model.parameters()).device
    
    # Initialize A matrix
    shared_lora_A = nn.Parameter(
        torch.empty((args.lora_rank, max_in_features), dtype=dtype, device=device),
        requires_grad=True
    )
    
    # Initialize B matrix  
    shared_lora_B = nn.Parameter(
        torch.zeros((max_out_features, args.lora_rank), dtype=dtype, device=device),
        requires_grad=True
    )
    
    # Initialize weights
    with torch.no_grad():
        if args.weight_a_init_method == 'kaiming':
            nn.init.kaiming_uniform_(shared_lora_A, a=5**0.5, mode='fan_in')
        else:
            nn.init.normal_(shared_lora_A, mean=0.0, std=1 / (max_in_features ** 0.5))
        
        if args.weight_b_init_method == 'kaiming':
            nn.init.kaiming_uniform_(shared_lora_B, a=5**0.5, mode='fan_in')
        elif args.weight_b_init_method == 'normal':
            nn.init.normal_(shared_lora_B, mean=0.0, std=0.02)
        # else: keep as zeros (default)
    
    return shared_lora_A, shared_lora_B


def apply_shared_lora_to_model(model: nn.Module, shared_lora_A: nn.Parameter, shared_lora_B: nn.Parameter):
    """
    Apply shared LoRA weights to all LinearWithSharedLoRA layers in the model.
    
    Args:
        model: The model containing LoRA layers
        shared_lora_A: Shared A matrix parameter
        shared_lora_B: Shared B matrix parameter
    """
    for module in model.modules():
        if isinstance(module, LinearWithSharedLoRA):
            module.update_shared_layer(shared_lora_A, shared_lora_B)
    
    # Attach the shared parameters to the model for proper parameter registration
    if not hasattr(model, 'shared_lora_A'):
        model.shared_lora_A = shared_lora_A
        model.shared_lora_B = shared_lora_B