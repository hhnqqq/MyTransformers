import torch
import torch.nn as nn
import bitsandbytes as bnb

from common.lora_modules.lora import *

class LinearWithQLoRA(LinearWithLoRA):
    def __init__(
        self,
        lora_config: LoRAConfig
    ):
        super().__init__(lora_config)

        self.quant = lora_config.quant
        self.qlora_config = lora_config
        self.quant_type = lora_config.quant_type

        self.is_quantized = False
        self.quantized_linear = None

    def quantize_base_layer(self):
        if self.is_quantized:
            return

        if self.weight is None or self.weight.data.numel() == 0:
            raise RuntimeError("Cannot quantize layer because self.weight has not been loaded.")

        device = self.weight.device
        self.weight_dtype = self.weight.dtype
        
        self.quantized_linear = bnb.nn.Linear4bit(
            self.in_features,
            self.out_features,
            bias=(self.bias is not None),
            compute_dtype=self.weight_dtype,
            quant_type=self.quant_type,
            device=device,
        )

        self.quantized_linear.weight = bnb.nn.Params4bit(
            data=self.weight.data.detach().clone(),
            requires_grad=False
        )
        
        if self.bias is not None:
            self.quantized_linear.bias = self.bias

        self.is_quantized = True

        del self.weight
        self.weight = None
        if hasattr(self, 'bias') and self.bias is not None:
            del self.bias
            self.bias = None

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant:
            return super().forward(x)
        
        self.quantize_base_layer()
        self.quantized_linear.to(x.device)
        result = self.quantized_linear(x)

        if self.disable_lora or not self.has_lora_weights:
            return result
        else:
            return self._lora_forward(x.to(self._get_lora_dtype()), result)

    def _merge_lora(self) -> bool:
        if not self.has_lora_weights:
            return False

        if self.quant and self.is_quantized:
            if self.quantized_linear is None: return False
            
            base_weight = self.quantized_linear.weight.dequantize()
            lora_weight = self._compute_lora_weight()
            
            self.weight = nn.Parameter(base_weight + lora_weight)
            if self.quantized_linear.bias is not None:
                self.bias = nn.Parameter(self.quantized_linear.bias.data)
            
            delattr(self, 'quantized_linear')
            self.is_quantized = False
            return True
        else:
            return super()._merge_lora()
    
    def _get_lora_dtype(self):
        weight_dtype = self.weight.dtype if getattr(self, "weight", None) is not None else self.weight_dtype
        return torch.float32 if self.run_lora_in_fp32 else weight_dtype