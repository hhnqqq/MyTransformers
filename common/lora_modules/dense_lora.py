# @author: haonan he
"""
Implementation of DenseLoRA: Dense Low-Rank Adaptation of Large Language Models [ACL 2025]
Paper link: https://arxiv.org/abs/2505.23808
Code reference: https://github.com/mulin-ahu/DenseLoRA/blob/main/commonsense_reasoning/peft/src/peft/tuners/lora.py

DenseLoRA enhances parameter efficiency while achieving superior performance
compared to LoRA. DenseLoRA builds upon the concept of representation fine-tuning, 
incorporating a single Encoder-Decoder to refine and compress hidden representations 
across all adaptation layers before applying adaptation. Instead of relying on two redundant
low-rank matrices as in LoRA, DenseLoRA adapts LLMs through a dense low-rank matrix, 
improving parameter utilization and adaptation efficiency. 
"""
from common.lora_modules.lora import *
from common.lora_modules.share_lora import get_module_groups

class LinearWithDenseLoRA(LinearWithLoRA):
    def __init__(self,
        lora_config: LoRAConfig,
        weight_ab_mixer_init_method: Optional[str] = None):
        self.weight_ab_mixer_init_method = weight_ab_mixer_init_method
        self.share_lora_weights = True
        self.first_update = True
        super().__init__(lora_config)

    def update_shared_weights(
        self,
        shared_weight_a: nn.Parameter,
        shared_weight_b: nn.Parameter,
        module_name: str = None
    ):
        """
        Update this layer to use shared LoRA weights.
        
        Args:
            weight_a: Shared A matrix parameter
            weight_b: Shared B matrix parameter
        """
        dtype = self._get_lora_dtype()
        self.shared_weight_a = shared_weight_a
        self.shared_weight_b = shared_weight_b
        self.weight_a = shared_weight_a
        self.weight_b = shared_weight_b
        if self.first_update:
            if module_name is None:
                raise ValueError("Module name can not be None")
            self.module_name = module_name
            self.weight_ab_mixer = nn.Parameter(torch.empty((self.lora_rank, self.lora_rank), dtype=dtype))
            nn.init.kaiming_uniform_(self.weight_ab_mixer, a=5**0.5, mode='fan_in')
            self.first_update = False
        
    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        weight_a = self._quantize_weight(self.shared_weight_a[self.module_name], self.weight_a_quantizer).to(self._get_lora_dtype())
        weight_b = self._quantize_weight(self.shared_weight_b[self.module_name], self.weight_b_quantizer).to(self._get_lora_dtype())
        weight_ab_mixer = self._quantize_weight(self.weight_ab_mixer, self.weight_ab_quantizer).to(self._get_lora_dtype())
        lora_result = F.gelu(F.linear(F.linear(F.gelu(F.linear(self.lora_dropout(x), weight_a)), weight_ab_mixer), weight_b)).to(result.dtype)
        return result + self.lora_scaler * lora_result
        
    def init_lora_weights(self):
        pass

    def _merge_lora(self):
        raise NotImplementedError('Weights of DenseLoRA can not be merged!')
    
    @property
    def weight_ab_quantizer(self) -> Optional[torch.Tensor]:
        return getattr(self, "weight_ab_scaler", None)
    
    def _del_lora(self):
        delattr(self, "weight_ab_mixer")

    @property
    def has_lora_weights(self):
        has_ab_mixer = hasattr(self, 'weight_ab_mixer') and self.weight_ab_mixer is not None
        return has_ab_mixer and super().has_lora_weights

def prepare_shared_lora_weights_denselora(model: nn.Module, args) -> tuple[nn.Parameter, nn.Parameter]:
    """
    Prepare shared LoRA weights for DenseLoRA based on model architecture.
    
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
    
    shared_weight_a_dict = nn.ParameterDict()
    shared_weight_b_dict = nn.ParameterDict()
    for name, info in module_groups.items():
        out_features, in_features = info["shape"]
        # Initialize A matrix
        shared_weight_a = nn.Parameter(
            torch.empty((args.lora_rank, in_features), dtype=dtype, device=device))
        
        # Initialize B matrix  
        shared_weight_b = nn.Parameter(
            torch.zeros((out_features, args.lora_rank), dtype=dtype, device=device))
        # Initialize weights
        with torch.no_grad():
            if args.weight_a_init_method == 'kaiming':
                nn.init.kaiming_uniform_(shared_weight_a, a=5**0.5, mode='fan_in')
            else:
                nn.init.normal_(shared_weight_a, mean=0.0, std=1 / (in_features ** 0.5))
            
            if args.weight_b_init_method == 'kaiming':
                nn.init.kaiming_uniform_(shared_weight_b, a=5**0.5, mode='fan_in')
            elif args.weight_b_init_method == 'normal':
                nn.init.normal_(shared_weight_b, mean=0.0, std=0.02)

        shared_weight_a_dict[name] = shared_weight_a
        shared_weight_b_dict[name] = shared_weight_b

    return shared_weight_a_dict, shared_weight_b_dict
