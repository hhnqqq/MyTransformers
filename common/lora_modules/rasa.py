import re
from collections import defaultdict
from common.lora_modules.lora import *
from common.lora_modules.lora import LoRAConfig
from torch import Tensor

class LinearWithRASA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, shared_lora_rank):
        super().__init__(lora_config)
        if shared_lora_rank > self.lora_rank:
            raise ValueError("RASA's shared_lora_rank can not be larger than lora_rank!"
                             "Please check your rank configuration.")
        self.lora_rank -= shared_lora_rank
        self.lora_scaler = lora_config.lora_scaler
        self.share_lora_weights = True

    def update_shared_weights(
        self,
        weight_a,
        weight_b,
        module_name
    ):
        self.shared_weight_a = weight_a
        self.shared_weight_b = weight_b
        self.module_name = module_name
        shared_rank = weight_a[module_name].shape[0]
        self.effect_rank = shared_rank + self.lora_rank
        self.weight_e = nn.Parameter(torch.ones(self.effect_rank, 1))
        with torch.no_grad():
            # More stable initialization
            shared_part = (0.5 * self.lora_scaler) / shared_rank
            lora_part = (0.5 * self.lora_scaler) / self.lora_rank
            self.weight_e.normal_(mean=0, std=0.02)
            self.weight_e[self.lora_rank:].fill_(shared_part)
            self.weight_e[:self.lora_rank].fill_(lora_part)

    def _concat_lora_weights(self):
        """Concatenate shared and private LoRA weights."""
        if not self._shared_weights_initialized:
            raise RuntimeError("Shared weights not initialized. Call update_shared_weights first.")
            
        dtype = self._get_lora_dtype()
        weight_a = torch.cat([
            self.weight_a.to(dtype), 
            self.shared_weight_a[self.module_name].to(dtype)
        ], dim=0)
        
        weight_b = torch.cat([
            self.weight_b.to(dtype), 
            self.shared_weight_b[self.module_name].to(dtype)
        ], dim=1)
        
        return weight_a, weight_b, self.weight_e.to(dtype)
    
    
    def _lora_forward(self, x: Tensor, result: Tensor) -> Tensor:
        weight_a, weight_b, weight_e = self._concat_lora_weights()

        lora_result = F.linear(
            F.linear(self.lora_dropout(x), weight_a * weight_e),
            weight_b,
            ).to(result.dtype)
        return result + lora_result

    def _compute_lora(self):
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a, weight_b, weight_e = self._concat_lora_weights()
            # When using vanilla lora, the ab mixer is a identical matrix
        return F.linear(weight_a * weight_e, weight_b)
    
    @property
    def has_lora_weights(self):
        """
        Check if this layer has VeRA weights.
        """
        has_shared_weight_a = hasattr(self, 'shared_weight_a')
        has_shared_weight_b = hasattr(self, 'shared_weight_a')
        return has_shared_weight_a and has_shared_weight_b and super().has_lora_weights

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
        if isinstance(module, LinearWithRASA):
            module2shape[key] = tuple(module.weight.shape)

    if not module2shape:
        raise ValueError("No LinearWithRASA layer was found.")

    # Group modules across layers
    module_groups = defaultdict(list)
    pattern = re.compile(r'layers\.(\d+)\.(.+)')

    for key, value in module2shape.items():
        match = pattern.search(key)
        if match:
            layer_id = match.group(1)
            module_name = match.group(2).replace('.', '__')
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

def prepare_shared_lora_weights_rasa(model: nn.Module, args) -> tuple[nn.Parameter, nn.Parameter]:
    """
    Prepare shared LoRA weights for RASA based on model architecture.
    
    Args:
        model: The model to prepare weights for
        args: Configuration arguments
        
    Returns:
        Tuple of (shared_weight_a, shared_weight_b) ParameterDicts
    """
    # Find the maximum dimensions needed across all layers
    module_groups = get_module_groups(model)
    
    # Create shared parameters
    dtype = torch.float32 if args.run_lora_in_fp32 else torch.float16
    device = next(model.parameters()).device
    
    shared_weight_a = nn.ParameterDict()
    shared_weight_b = nn.ParameterDict()
    for name, info in module_groups.items():
        out_features, in_features = info["shape"]
        rasa_shared_rank = args.rasa_shared_lora_rank * info["num_layers"]
        # Initialize A matrix
        weight_a = nn.Parameter(
            torch.empty((rasa_shared_rank, in_features), dtype=dtype, device=device))
        
        # Initialize B matrix  
        weight_b = nn.Parameter(
            torch.zeros((out_features, rasa_shared_rank), dtype=dtype, device=device))
        # Initialize weights
        with torch.no_grad():
            if args.weight_a_init_method == 'kaiming':
                nn.init.kaiming_uniform_(weight_a, a=5**0.5, mode='fan_in')
            else:
                nn.init.normal_(weight_a, mean=0.0, std=1 / (in_features ** 0.5))
            
            if args.weight_b_init_method == 'kaiming':
                nn.init.kaiming_uniform_(weight_b, a=5**0.5, mode='fan_in')
            elif args.weight_b_init_method == 'normal':
                nn.init.normal_(weight_b, mean=0.0, std=0.02)

        shared_weight_a[name] = weight_a
        shared_weight_b[name] = weight_b

    model.weight_a = shared_weight_a
    model.weight_b = shared_weight_b


def update_shared_weights_to_layer_rasa(model: nn.Module):
    """
    Apply shared LoRA weights to all LinearWithRASA layers in the model.
    
    Args:
        model: The model containing LoRA layers
    """
    for name, module in model.named_modules():
        if getattr(module, 'share_lora_weights', False):
            pattern = re.compile(r'layers\.(\d+)\.(.+)')
            match = pattern.search(name)
            module_name = match.group(2).replace('.', '__')
            module.update_shared_weights(model.weight_a, model.weight_b, module_name)