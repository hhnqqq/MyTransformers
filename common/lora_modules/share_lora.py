# @author: minglei li (modified by haonan he)
# @date: 2025-6-19
""" Implementation of ShareLoRA where A or B matrices are shared across all layers."""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import defaultdict
from common.lora_modules.lora import LinearWithLoRA, LoRAConfig

class LinearWithShareLoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, share_part: Optional[str] = None):
        """
        Initialize the LinearWithSharedLoRA layer.
        """
        self.share_lora_weights = True
        self.share_part = share_part
        super().__init__(lora_config)

    def init_lora_weights(self):
        """
        For share-lora, we initialize the low-rank weights according to share_part.
        """
        # Defualt is fp32 when LinearWithLora init.
        dtype = self._get_lora_dtype()
        requires_grad = not self.quant

        if self.share_part == 'A':
            self.weight_b = nn.Parameter(torch.zeros((self.out_features, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
            self._init_weight('weight_b')
        elif self.share_part == 'B':
            self.weight_a = nn.Parameter(torch.empty((self.lora_rank, self.in_features), dtype=dtype), requires_grad=requires_grad)
            self._init_weight('weight_a')

    def update_shared_weights(
        self,
        shared_weight_a: Optional[nn.Parameter] = None,
        shared_weight_b: Optional[nn.Parameter] = None
    ):
        """
        Update this layer to use shared LoRA weights.
        The shared parameters will be registered in the first layer that cite them.
        
        Args:
            shared_weight_a: Shared A matrix parameter
            shared_weight_b: Shared B matrix parameter
        """
        if shared_weight_a is not None:
            self.shared_weight_a = shared_weight_a
            self.weight_a = shared_weight_a
        if shared_weight_b is not None:
            self.shared_weight_b = shared_weight_b
            self.weight_b = shared_weight_b

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
        else:
            raise RuntimeError('Can not compute lora weight without lora_weights!')
        
    def _del_lora(self):
        """
        Delete LoRA weights. For shared LoRA, we only delete references.
        """
        # Don't delete shared weights, they are managed globally
        self.shared_weight_a = None
        self.shared_weight_b = None
        
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
        print(f"Shared LoRA Enabled: {self.has_lora_weights}, LoRA Rank: {self.lora_rank}, Quantized: {self.quant}")

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
    if 'A' in args.sharelora_share_part:
        shared_weight_a = nn.Parameter(
            torch.empty((args.lora_rank, max_in_features), dtype=dtype, device=device))
    else:
        shared_weight_a = None
    
    # Initialize B matrix  
    if 'B' in args.sharelora_share_part:
        shared_weight_b = nn.Parameter(
            torch.zeros((max_out_features, args.lora_rank), dtype=dtype, device=device))
    else:
        shared_weight_b = None
    
    # Initialize weights
    with torch.no_grad():
        if shared_weight_a is not None:
            if args.weight_a_init_method == 'kaiming':
                nn.init.kaiming_uniform_(shared_weight_a, a=5**0.5, mode='fan_in')
            else:
                nn.init.normal_(shared_weight_a, mean=0.0, std=1 / (max_in_features ** 0.5))
        
        if shared_weight_b is not None:
            if args.weight_b_init_method == 'kaiming':
                nn.init.kaiming_uniform_(shared_weight_b, a=5**0.5, mode='fan_in')
            elif args.weight_b_init_method == 'normal':
                nn.init.normal_(shared_weight_b, mean=0.0, std=0.02)

    return shared_weight_a, shared_weight_b


def update_shared_weights_to_layer(model: nn.Module, shared_weight_a, shared_weight_b):
    """
    Apply shared LoRA weights to all LinearWithSharedLoRA layers in the model.
    
    Args:
        model: The model containing LoRA layers
        weight_a: Shared A matrix parameter
        weight_b: Shared B matrix parameter
    """
    for module in model.modules():
        if getattr(module, 'share_lora_weights', False):
            module.update_shared_weights(shared_weight_a, shared_weight_b)

# def get_module_groups(model):
#     """
#     Group modules by their type across layers and verify shape consistency.
    
#     Args:
#         model: The model to analyze
        
#     Returns:
#         Dictionary mapping module names to their shape and layer information
#     """
#     module2shape = {}
#     for key, module in model.named_modules():
#         if isinstance(module, LinearWithLoRA):
#             module2shape[key] = tuple(module.weight.shape)

#     if not module2shape:
#         raise ValueError("No LinearWithLoRA layer was found.")

#     # Group modules across layers
#     module_groups = defaultdict(list)
#     pattern = re.compile(r'layers\.(\d+)\.(.+)')

#     for key, value in module2shape.items():
#         match = pattern.search(key)
#         if match:
#             layer_id = match.group(1)
#             module_name = match.group(2).replace('.', '__')
#             module_groups[module_name].append((layer_id, value))

#     module_groups = dict(module_groups)

#     # Assert each type of module has the same shape across layers
#     for key, value in module_groups.items():
#         assert all([v[1] == value[0][1] for v in value]), f"Shape mismatch for {key} layers: {value}"

#     # Add the number of layers for each module type
#     for key in module_groups.keys():
#         module_groups[key] = {
#             "shape": module_groups[key][0][1],
#             "layer_ids": [int(v[0]) for v in module_groups[key]],
#             "num_layers": len(module_groups[key]),
#         }

#     return module_groups

def get_module_groups(model):
    """
    Group modules by their type across layers and verify shape consistency.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary mapping module names to their shape and layer information
    """
    module2shape = {}
    for key, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            module2shape[key] = tuple(module.weight.shape)

    if not module2shape:
        raise ValueError("No LinearWithLoRA layer was found.")

    # Group modules across layers
    module_groups = defaultdict(list)
    pattern = re.compile(r'roberta\.encoder\.layer\.(\d+)\.(.+)')

    for key, value in module2shape.items():
        match = pattern.search(key)
        if match:
            layer_id = match.group(1)
            module_name = match.group(2).replace('.', '_')
            module_groups[module_name].append((layer_id, value))

    module_groups = dict(module_groups)

    # Assert each type of module has the same shape across layers
    for key, value in module_groups.items():
        assert all([v[1] == value[0][1] for v in value]), f"Shape mismatch for {key} layers: {value}"

    # Add the number of layers for each module type
    for key in module_groups.keys():
        module_groups[key] = {
            "shape": module_groups[key][0][1],
            "layer_ids": [int(v[0]) for v in module_groups[key]],
            "num_layers": len(module_groups[key]),
        }

    return module_groups

# def update_grouped_shared_weights_to_layer(model: nn.Module, shared_weight_a, shared_weight_b):
#     """
#     Apply shared LoRA weights to all LinearWithRASA layers in the model.
    
#     Args:
#         model: The model containing LoRA layers
#     """
#     for name, module in model.named_modules():
#         if getattr(module, 'share_lora_weights', False):
#             pattern = re.compile(r'layers\.(\d+)\.(.+)')
#             match = pattern.search(name)
#             module_name = match.group(2).replace('.', '__')
#             module.update_shared_weights(shared_weight_a, shared_weight_b, module_name)

# For Roberta
def update_grouped_shared_weights_to_layer(model: nn.Module, shared_weight_a, shared_weight_b):
    """
    Apply shared LoRA weights to all LinearWithRASA layers in the model.
    
    Args:
        model: The model containing LoRA layers
        shared_weight_a: The shared weight 'A' matrix
        shared_weight_b: The shared weight 'B' matrix
    """
    for name, module in model.named_modules():
        if getattr(module, 'share_lora_weights', False):
            # The pattern now captures the part after the fourth '.'
            # (e.g., query, key, value, output.dense, intermediate.dense, output.dense)
            pattern = re.compile(r'roberta\.encoder\.layer\.(\d+)\.(.+)')
            match = pattern.search(name)
            
            if match:
                # Get the captured group and replace '.' with '_'
                module_name = match.group(2).replace('.', '_')
                print(f"Updating module: {name} with module_name: {module_name}")
                # Update the module's weights
                module.update_shared_weights(shared_weight_a, shared_weight_b, module_name)
