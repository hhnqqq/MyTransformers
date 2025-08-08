from common.lora_modules.qlora import *
from common.utils.utils import safe_cat
from typing import Tuple
from collections import defaultdict

from torch import Tensor

class LinearWithBSLoRA(LinearWithQLoRA):
    def __init__(self, lora_config: LoRAConfig, forward_method: str = 'slice'):
        self.share_lora_weights = False
        self.has_inter_weights = False
        self.has_intra_weights = False
        self.has_unique_weights = False
        self.forward_method = forward_method
        super().__init__(lora_config)

    def init_lora_weights(self):
        if self.lora_rank > 0:
            super().init_lora_weights()
            self.has_unique_weights = True
    
    def update_shared_weights(
        self,
        shared_weight_a: Optional[Dict[str, nn.Parameter]] = None,
        shared_weight_b: Optional[Dict[str, nn.Parameter]] = None,
    ):
        # Reset shared weights and flags
        self.inter_shared_weight_a = self.inter_shared_weight_b = None
        self.intra_shared_weight_a = self.intra_shared_weight_b = None
        self.has_inter_weights = self.has_intra_weights = False
        
        # Update shared weights if provided
        if shared_weight_a is not None:
            self.inter_shared_weight_a = shared_weight_a.get('inter_layer')
            self.intra_shared_weight_a = shared_weight_a.get('intra_layer')
        
        if shared_weight_b is not None:
            self.inter_shared_weight_b = shared_weight_b.get('inter_layer')
            self.intra_shared_weight_b = shared_weight_b.get('intra_layer')
        
        # Update flags
        self.has_inter_weights = (self.inter_shared_weight_a is not None and 
                                self.inter_shared_weight_b is not None)
        self.has_intra_weights = (self.intra_shared_weight_a is not None and 
                                self.intra_shared_weight_b is not None)
        self.share_lora_weights = self.has_inter_weights or self.has_intra_weights
        
        # Setup forward method if sharing weights
        if self.share_lora_weights:
            self._setup_forward_method()

    def _setup_forward_method(self):
        """Setup the appropriate forward method based on configuration."""
        if self.forward_method == 'kron':
            self._setup_kron_weights()
            self._cat_weights = self._cat_weights_kron
        elif self.forward_method == 'gate':
            self._setup_gate_weights()
            self._cat_weights = self._cat_weights_gate
        else:  # default to slice method
            self._cat_weights = self._cat_weights_slice

    def _setup_kron_weights(self):
        """Initialize weights for Kronecker product method."""
        def init_kron_weights(prefix, shared_weight_a, shared_weight_b):
            # Calculate the required dimensions for the sampler matrices
            a_blocks = self.in_features // shared_weight_a.shape[1]
            b_blocks = self.out_features // shared_weight_b.shape[0]
            
            # Initialize sampler matrices
            setattr(self, f"{prefix}_sampler_a", 
                    nn.Parameter(torch.empty(1, a_blocks)))
            setattr(self, f"{prefix}_sampler_b", 
                    nn.Parameter(torch.empty(b_blocks, 1)))
            
            # Initialize the sampler weights
            nn.init.kaiming_uniform_(getattr(self, f"{prefix}_sampler_a"), a=5**0.5)
            nn.init.kaiming_uniform_(getattr(self, f"{prefix}_sampler_b"), a=5**0.5)
        
        if self.has_inter_weights:
            init_kron_weights("inter", self.inter_shared_weight_a, self.inter_shared_weight_b)
        
        if self.has_intra_weights:
            init_kron_weights("intra", self.intra_shared_weight_a, self.intra_shared_weight_b)

    def _setup_gate_weights(self):
        """Initialize weights for gating method."""
        def init_gate_weights(prefix, shared_weight_a, shared_weight_b):
            setattr(self, f"{prefix}_gate_a_down", 
                    nn.Parameter(torch.empty(shared_weight_a.shape[1], 1)))
            setattr(self, f"{prefix}_gate_a_up", 
                    nn.Parameter(torch.empty(1, self.in_features)))
            setattr(self, f"{prefix}_gate_b_down", 
                    nn.Parameter(torch.empty(1, shared_weight_b.shape[0])))
            setattr(self, f"{prefix}_gate_b_up", 
                    nn.Parameter(torch.empty(self.out_features, 1)))
            
            for param in [f"{prefix}_gate_a_down", f"{prefix}_gate_a_up",
                        f"{prefix}_gate_b_down", f"{prefix}_gate_b_up"]:
                nn.init.kaiming_uniform_(getattr(self, param), a=5**0.5)
        
        if self.has_inter_weights:
            init_gate_weights("inter", self.inter_shared_weight_a, self.inter_shared_weight_b)
        
        if self.has_intra_weights:
            init_gate_weights("intra", self.intra_shared_weight_a, self.intra_shared_weight_b)

    def _get_weights(self, weight_a, weight_b, condition):
        """Helper method to get weights if condition is met."""
        if condition:
            return weight_a.to(self._get_lora_dtype()), weight_b.to(self._get_lora_dtype())
        return None, None

    def _get_unique_weights(self):
        return self._get_weights(self.weight_a, self.weight_b, self.lora_rank > 0)
    
    def _get_inter_shared_weights(self):
        return self._get_weights(self.inter_shared_weight_a, self.inter_shared_weight_b, 
                               self.has_inter_weights)
    
    def _get_intra_shared_weights(self):
        return self._get_weights(self.intra_shared_weight_a, self.intra_shared_weight_b,
                               self.has_intra_weights)
    
    def _cat_weights_slice(self) -> Tuple[torch.Tensor, torch.Tensor]:
        unique_a, unique_b = self._get_unique_weights()
        inter_a, inter_b = self._get_inter_shared_weights()
        intra_a, intra_b = self._get_intra_shared_weights()

        weight_a = safe_cat([
            unique_a,
            inter_a[:, :self.in_features] if inter_a is not None else None,
            intra_a[:, :self.in_features] if intra_a is not None else None
        ], dim=0)
        
        weight_b = safe_cat([
            unique_b,
            inter_b[:self.out_features, :] if inter_b is not None else None,
            intra_b[:self.out_features, :] if intra_b is not None else None
        ], dim=1)

        return weight_a, weight_b
    
    def _cat_weights_kron(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self._get_lora_dtype()
        unique_a, unique_b = self._get_unique_weights()
        inter_a, inter_b = self._get_inter_shared_weights()
        intra_a, intra_b = self._get_intra_shared_weights()

        # Process inter weights
        inter_kron_a = inter_kron_b = None
        if self.has_inter_weights:
            inter_sampler_a = getattr(self, "inter_sampler_a", None)
            inter_sampler_b = getattr(self, "inter_sampler_b", None)
            
            if inter_sampler_a is not None and inter_sampler_b is not None:
                inter_sampler_a = inter_sampler_a.to(dtype)
                inter_sampler_b = inter_sampler_b.to(dtype)
                inter_kron_a = torch.kron(inter_sampler_a, inter_a)
                
                inter_kron_b = torch.kron(inter_sampler_b, inter_b)

        # Process intra weights
        intra_kron_a = intra_kron_b = None
        if self.has_intra_weights:
            intra_sampler_a = getattr(self, "intra_sampler_a", None)
            intra_sampler_b = getattr(self, "intra_sampler_b", None)
            
            if intra_sampler_a is not None and intra_sampler_b is not None:
                intra_sampler_a = intra_sampler_a.to(dtype)
                intra_sampler_b = intra_sampler_b.to(dtype)
                intra_kron_a = torch.kron(intra_sampler_a, intra_a)
                
                intra_kron_b = torch.kron(intra_sampler_b, intra_b)

        weight_a = safe_cat([unique_a, inter_kron_a, intra_kron_a], dim=0)
        weight_b = safe_cat([unique_b, inter_kron_b, intra_kron_b], dim=1)
        
        return weight_a, weight_b

    def _cat_weights_gate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self._get_lora_dtype()
        unique_a, unique_b = self._get_unique_weights()
        inter_a, inter_b = self._get_inter_shared_weights()
        intra_a, intra_b = self._get_intra_shared_weights()

        # Process inter weights
        if self.has_inter_weights:
            inter_gate_a_down = self.inter_gate_a_down.to(dtype)
            inter_gate_a_up = self.inter_gate_a_up.to(dtype)
            inter_gate_b_down = self.inter_gate_b_down.to(dtype)
            inter_gate_b_up = self.inter_gate_b_up.to(dtype)
            inter_a = torch.matmul(torch.matmul(inter_a, inter_gate_a_down), inter_gate_a_up)
            inter_b = torch.matmul(inter_gate_b_up, torch.matmul(inter_gate_b_down, inter_b))

        # Process intra weights
        if self.has_intra_weights:
            intra_gate_a_down = self.intra_gate_a_down.to(dtype)
            intra_gate_a_up = self.intra_gate_a_up.to(dtype)
            intra_gate_b_down = self.intra_gate_b_down.to(dtype)
            intra_gate_b_up = self.intra_gate_b_up.to(dtype)
            intra_a = torch.matmul(torch.matmul(intra_a, intra_gate_a_down), intra_gate_a_up)
            intra_b = torch.matmul(intra_gate_b_up, torch.matmul(intra_gate_b_down, intra_b))

        weight_a = safe_cat([unique_a, inter_a, intra_a], dim=0)
        weight_b = safe_cat([unique_b, inter_b, intra_b], dim=1)

        return weight_a, weight_b
    
    def _lora_forward(self, x: Tensor, result: Tensor) -> Tensor:
        weight_a, weight_b = self._cat_weights()
        lora_result = F.linear(F.linear(self.lora_dropout(x), weight_a), weight_b)
        return result + self.lora_scaler * lora_result.to(result.dtype)

    def _compute_lora_weight(self) -> torch.Tensor:
        if not self.has_lora_weights:
            raise RuntimeError('Cannot compute LoRA weight without LoRA weights!')
            
        weight_a, weight_b = self._cat_weights()
        return (self.lora_scaler * torch.matmul(weight_b, weight_a)).to(self.weight.dtype)

    @property
    def has_lora_weights(self):
        return self.has_unique_weights or self.has_inter_weights or self.has_intra_weights
    

def prepare_shared_lora_weights_bslora(model: nn.Module, args) -> Tuple[Dict[str, nn.Parameter], Dict[str, Dict[int, nn.Parameter]]]:
    """
    Prepare shared LoRA weights for BSLoRA that includes both intra-layer and inter-layer sharing.
    
    Args:
        model: The model containing BSLoRA layers
        args: Arguments containing BSLoRA configuration
        
    Returns:
        Tuple of (shared_weight_a_dict, shared_weight_b_dict) containing:
            - inter_layer: Parameters shared across all layers
            - intra_layer: Parameters shared within each layer
    """
    # Find the maximum dimensions needed across all layers for inter and intra sharing
    max_inter_in = max_inter_out = 0
    max_intra_in_dict = max_intra_out_dict = defaultdict(int)
    intra_shared_weights_a_dict, intra_shared_weights_b_dict = {}, {}
    
    for name, module in model.named_modules():
        if 'layer' in name and isinstance(module, nn.ModuleList):
            for layer_idx, layer in enumerate(module):
                for sub_module in layer.modules():
                    if isinstance(sub_module, LinearWithBSLoRA):
                        max_inter_in = args.bslora_share_size if args.bslora_forward_method != 'slice' else max(max_inter_in, sub_module.in_features)
                        max_inter_out = args.bslora_share_size if args.bslora_forward_method != 'slice' else max(max_inter_out, sub_module.out_features)

                        max_intra_in_dict[layer_idx] = args.bslora_share_size if args.bslora_forward_method != 'slice' else max(max_intra_in_dict[layer_idx], sub_module.in_features)
                        max_intra_out_dict[layer_idx] = args.bslora_share_size if args.bslora_forward_method != 'slice' else max(max_intra_out_dict[layer_idx], sub_module.out_features)
            break
    
    # Create shared parameters for inter and intra layer sharing
    dtype = torch.float32 if args.run_lora_in_fp32 else torch.float16
    device = next(model.parameters()).device
    
    shared_weight_a_dict = {}
    shared_weight_b_dict = {}
    
    # Initialize inter-layer shared weights (global sharing)
    if args.bslora_inter_shared_rank > 0:
        inter_a = nn.Parameter(
            torch.empty((args.bslora_inter_shared_rank, max_inter_in), dtype=dtype, device=device))
        inter_b = nn.Parameter(
            torch.zeros((max_inter_out, args.bslora_inter_shared_rank), dtype=dtype, device=device))
        
        with torch.no_grad():
            if args.weight_a_init_method == 'kaiming':
                nn.init.kaiming_uniform_(inter_a, a=5**0.5, mode='fan_in')
            else:
                nn.init.normal_(inter_a, mean=0.0, std=1 / (max_inter_in ** 0.5))
            
            if args.weight_b_init_method == 'kaiming':
                nn.init.kaiming_uniform_(inter_b, a=5**0.5, mode='fan_in')
            elif args.weight_b_init_method == 'normal':
                nn.init.normal_(inter_b, mean=0.0, std=0.02)
        
        shared_weight_a_dict['inter_layer'] = inter_a
        shared_weight_b_dict['inter_layer'] = inter_b
    
    # Initialize intra-layer shared weights (within layer sharing)
    if args.bslora_intra_shared_rank > 0:
        for layer_idx in max_intra_in_dict.keys():
            intra_a = nn.Parameter(
                torch.empty((args.bslora_intra_shared_rank, max_intra_in_dict[layer_idx]), dtype=dtype, device=device))
            intra_b = nn.Parameter(
                torch.zeros((max_intra_out_dict[layer_idx], args.bslora_intra_shared_rank), dtype=dtype, device=device))
            
            with torch.no_grad():
                if args.weight_a_init_method == 'kaiming':
                    nn.init.kaiming_uniform_(intra_a, a=5**0.5, mode='fan_in')
                else:
                    nn.init.normal_(intra_a, mean=0.0, std=1 / (max_intra_in_dict[layer_idx] ** 0.5))
                
                if args.weight_b_init_method == 'kaiming':
                    nn.init.kaiming_uniform_(intra_b, a=5**0.5, mode='fan_in')
                elif args.weight_b_init_method == 'normal':
                    nn.init.normal_(intra_b, mean=0.0, std=0.02)
            
            intra_shared_weights_a_dict[layer_idx] = intra_a
            intra_shared_weights_b_dict[layer_idx] = intra_b
        shared_weight_a_dict['intra_layer'] = intra_shared_weights_a_dict
        shared_weight_b_dict['intra_layer'] = intra_shared_weights_b_dict
        
    
    return shared_weight_a_dict, shared_weight_b_dict


def update_shared_weights_to_layer_bslora(model: nn.Module, 
                                        shared_weight_a_dict, 
                                        shared_weight_b_dict):
    """
    Apply shared LoRA weights to all LinearWithBSLoRA layers in the model.
    
    Args:
        model: The model containing BSLoRA layers
        shared_weight_a_dict: Dictionary containing inter_layer and/or intra_layer A matrices
        shared_weight_b_dict: Dictionary containing inter_layer and/or intra_layer B matrices
    """
    inter_layer_shared_weight_a = shared_weight_a_dict.get('inter_layer')
    inter_layer_shared_weight_b = shared_weight_b_dict.get('inter_layer')
    intra_layer_shared_weight_a_dict = shared_weight_a_dict.get('intra_layer')
    intra_layer_shared_weight_b_dict = shared_weight_b_dict.get('intra_layer')

    for name, module in model.named_modules():
        if 'layer' in name and isinstance(module, nn.ModuleList):
            for layer_idx, layer in enumerate(module):
                for sub_module in layer.modules():
                    if isinstance(sub_module, LinearWithBSLoRA):
                        sub_module.update_shared_weights({'inter_layer': inter_layer_shared_weight_a, 
                                                          'intra_layer': intra_layer_shared_weight_a_dict[layer_idx] if intra_layer_shared_weight_a_dict else None}, 
                                                         {'inter_layer': inter_layer_shared_weight_b, 
                                                          'intra_layer': intra_layer_shared_weight_b_dict[layer_idx] if intra_layer_shared_weight_b_dict else None})