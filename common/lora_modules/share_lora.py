# @author: minglei li (modified by haonan he)
# @date: 2025-6-19
""" Implementation of ShareLoRA where A matrices are shared per module type across layers."""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import defaultdict
from common.lora_modules.lora import LinearWithLoRA, LoRAConfig

def extract_module_type_for_sharing(module_name: str) -> str:
    """Extract module type from module name for A matrix sharing."""
    module_type_patterns = [
        r'\.([qkvo]_proj)$',  # q_proj, k_proj, v_proj, o_proj
        r'\.([qkvo])$',       # q, k, v, o
        r'\.(query|key|value|output)$',
        r'\.(q_linear|k_linear|v_linear|o_linear)$',
        r'\.(out_proj)$',
        r'\.(dense_h_to_4h|dense_4h_to_h)$',
        r'\.(dense|fc1|fc2)$',
        r'\.(gate_proj|up_proj|down_proj)$',
        r'\.(wi_0|wi_1|wo)$',
        r'\.(wq|wk|wv|wo)$',
        r'\.([^.]+)$'
    ]
    
    for pattern in module_type_patterns:
        match = re.search(pattern, module_name)
        if match:
            module_type = match.group(1)
            return module_type
    
    return module_name

class LinearWithShareLoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, share_part: Optional[str] = None):
        """
        Initialize the LinearWithSharedLoRA layer.
        """
        self.share_lora_weights = True
        self.share_part = share_part
        self.module_type = None  # Will be set during weight sharing
        super().__init__(lora_config)

    def init_lora_weights(self):
        """
        For share-lora, we initialize the low-rank weights according to share_part.
        """
        # Default is fp32 when LinearWithLora init.
        dtype = self._get_lora_dtype()
        requires_grad = not self.quant

        if self.share_part == 'A':
            # A is shared per module type, so we don't initialize it here
            self.weight_b = nn.Parameter(torch.zeros((self.out_features, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
            self._init_weight('weight_b')
        elif self.share_part == 'B':
            # B is shared, so we don't initialize it here
            self.weight_a = nn.Parameter(torch.empty((self.lora_rank, self.in_features), dtype=dtype), requires_grad=requires_grad)
            self._init_weight('weight_a')

    def update_shared_weights(
        self,
        shared_weight_a: Optional[nn.Parameter] = None,
        shared_weight_b: Optional[nn.Parameter] = None,
        module_type: Optional[str] = None
    ):
        """
        Update this layer to use shared LoRA weights.
        
        Args:
            shared_weight_a: Module-type-specific shared A matrix parameter
            shared_weight_b: Shared B matrix parameter (if sharing B)
            module_type: The module type for this layer
        """
        self.module_type = module_type
        
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
        print(f"Shared LoRA Enabled: {self.has_lora_weights}, LoRA Rank: {self.lora_rank}, Module Type: {self.module_type}")

def prepare_shared_lora_weights(model: nn.Module, args) -> tuple:
    """
    Prepare shared LoRA weights based on the sharing strategy.
    - If sharing A: Create per-module-type A matrices and optionally one global B matrix
    - If sharing B: Create one global B matrix and optionally per-module-type A matrices
    - If sharing AB: Create per-module-type A matrices and one global B matrix
    
    Args:
        model: The model containing LoRA layers
        args: Arguments containing LoRA configuration
        
    Returns:
        Tuple of (weight_a_dict_or_single, weight_b_dict_or_single)
    """
    # Collect module information
    module_info = defaultdict(lambda: {'max_in': 0, 'max_out': 0, 'layers': []})
    
    for name, module in model.named_modules():
        if getattr(module, 'share_lora_weights', False):
            module_type = extract_module_type_for_sharing(name)
            module_info[module_type]['max_in'] = max(module_info[module_type]['max_in'], module.in_features)
            module_info[module_type]['max_out'] = max(module_info[module_type]['max_out'], module.out_features)
            module_info[module_type]['layers'].append(name)
    
    if not module_info:
        raise ValueError("No LinearWithLoRA layers with shared weights found in the model")
    
    # Create shared parameters
    dtype = torch.float32 if args.run_lora_in_fp32 else torch.float16
    device = next(model.parameters()).device
    
    shared_weight_a_dict = {}
    shared_weight_b = None
    
    # Create per-module-type A matrices if sharing A
    if 'A' in args.sharelora_share_part:
        for module_type, info in module_info.items():
            weight_a = nn.Parameter(
                torch.empty((args.lora_rank, info['max_in']), dtype=dtype, device=device))
            
            # Initialize A matrix
            if args.weight_a_init_method == 'kaiming':
                nn.init.kaiming_uniform_(weight_a, a=5**0.5, mode='fan_in')
            else:
                nn.init.normal_(weight_a, mean=0.0, std=1 / (info['max_in'] ** 0.5))
            
            shared_weight_a_dict[module_type] = weight_a
            print(f"Created shared weight_a for module type '{module_type}': {weight_a.shape}")
    
    # Create global B matrix if sharing B
    if 'B' in args.sharelora_share_part:
        max_out_features = max(info['max_out'] for info in module_info.values())
        shared_weight_b = nn.Parameter(
            torch.zeros((max_out_features, args.lora_rank), dtype=dtype, device=device))
        
        # Initialize B matrix
        if args.weight_b_init_method == 'kaiming':
            nn.init.kaiming_uniform_(shared_weight_b, a=5**0.5, mode='fan_in')
        elif args.weight_b_init_method == 'normal':
            nn.init.normal_(shared_weight_b, mean=0.0, std=0.02)
        
        print(f"Created shared weight_b: {shared_weight_b.shape}")
    
    return shared_weight_a_dict, shared_weight_b

def update_shared_weights_to_layer(model: nn.Module, shared_weight_a, shared_weight_b):
    """
    Apply shared LoRA weights to all LinearWithShareLoRA layers in the model.
    
    Args:
        model: The model containing LoRA layers
        shared_weight_a: Dict of module-type-specific A matrices or single A matrix
        shared_weight_b: Shared B matrix parameter or None
    """
    for name, module in model.named_modules():
        if getattr(module, 'share_lora_weights', False):
            module_type = extract_module_type_for_sharing(name)
            
            # Get the appropriate A matrix for this module type
            weight_a_to_use = None
            if isinstance(shared_weight_a, dict):
                weight_a_to_use = shared_weight_a.get(module_type)
            else:
                weight_a_to_use = shared_weight_a
            
            module.update_shared_weights(weight_a_to_use, shared_weight_b, module_type)
            print(f"Updated module {name} (type: {module_type}) with shared weights")

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

    # Group modules across layers by module type
    module_groups = defaultdict(list)
    pattern = re.compile(r'layers\.(\d+)\.(.+)')

    for key, value in module2shape.items():
        match = pattern.search(key)
        if match:
            layer_id = match.group(1)
            module_name = match.group(2).replace('.', '__')
            module_type = extract_module_type_for_sharing(key)
            module_groups[module_type].append((layer_id, value, key))

    module_groups = dict(module_groups)

    # Verify each module type has consistent shapes across layers
    for module_type, module_list in module_groups.items():
        shapes = [item[1] for item in module_list]
        if not all(shape == shapes[0] for shape in shapes):
            print(f"Warning: Shape mismatch for module type {module_type}: {shapes}")

    # Add metadata for each module type
    for module_type in module_groups.keys():
        module_info = module_groups[module_type]
        module_groups[module_type] = {
            "shape": module_info[0][1] if module_info else None,
            "layer_ids": [int(item[0]) for item in module_info],
            "num_layers": len(module_info),
            "module_names": [item[2] for item in module_info]
        }

    return module_groups

def update_grouped_shared_weights_to_layer(model: nn.Module, shared_weight_a, shared_weight_b):
    """
    Apply shared LoRA weights to all LinearWithShareLoRA layers in the model with grouping.
    
    Args:
        model: The model containing LoRA layers
        shared_weight_a: Dict of module-type-specific shared A matrices
        shared_weight_b: Shared B matrix or dict of shared B matrices
    """
    for name, module in model.named_modules():
        if getattr(module, 'share_lora_weights', False):
            module_type = extract_module_type_for_sharing(name)
            
            # Get appropriate weights
            weight_a_to_use = None
            weight_b_to_use = None
            
            if isinstance(shared_weight_a, dict):
                weight_a_to_use = shared_weight_a.get(module_type)
            else:
                weight_a_to_use = shared_weight_a
                
            if isinstance(shared_weight_b, dict):
                weight_b_to_use = shared_weight_b.get(module_type)
            else:
                weight_b_to_use = shared_weight_b
            
            module.update_shared_weights(weight_a_to_use, weight_b_to_use, module_type)
