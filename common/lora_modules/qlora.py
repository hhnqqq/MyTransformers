import torch
import torch.nn as nn
import torch.nn.functional as F
from common.lora_modules.lora import *

class DoubleQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_level1 = lambda x: self.quantize(x, bits=4)
        self.quant_level2 = lambda x: self.quantize(x, bits=8)

    def quantize(self, x, bits=8):
        qmax = 2**(bits-1) - 1
        abs_max = x.abs().max()
        abs_max = abs_max.clamp(min=1e-8)  # Prevent division by zero
        scale = abs_max / qmax
        quantized = torch.clamp(torch.round(x / scale), -qmax, qmax)
        return quantized.to(torch.int8), scale

    def dequantize(self, quantized, scale):
        return quantized.to(torch.bfloat16) * scale

    def forward(self, x):
        quant_weight, scale1 = self.quant_level1(x)
        quant_scale, scale2 = self.quant_level2(scale1)
        return quant_weight, quant_scale, scale2

    def reconstruct(self, quant_weight, quant_scale, scale2):
        scale1 = self.dequantize(quant_scale, scale2)
        return self.dequantize(quant_weight, scale1)

class LinearWithQLoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig):
        super().__init__(lora_config)
        self.quantizer = DoubleQuantizer()
        self.register_buffer("quant_weight", None)
        self.register_buffer("quant_scale", None)
        self.register_buffer("scale2", None)
        self._orig_weight = None
        self._register_load_state_dict_pre_hook(self._pre_load_hook)

    def _quantize_weights(self):
        if self.quant_weight is None and self.weight is not None:
            self.quant_weight, self.quant_scale, self.scale2 = self.quantizer.forward(
                self.weight.data
            )
            self._orig_weight = self.weight.data
            # Keep weight tensor but set to zero to reduce memory usage
            self.weight.data = torch.zeros_like(self.weight.data)

    def _pre_load_hook(self, state_dict, prefix, *args, **kwargs):
        # Check if quantized weights are present in state dict
        if f"{prefix}quant_weight" in state_dict:
            # Load quantized weights and scales
            self.quant_weight = state_dict.pop(f"{prefix}quant_weight")
            self.quant_scale = state_dict.pop(f"{prefix}quant_scale")
            self.scale2 = state_dict.pop(f"{prefix}scale2")
            # Ensure weight tensor is zeroed
            if f"{prefix}weight" in state_dict:
                state_dict[f"{prefix}weight"] = torch.zeros_like(
                    state_dict[f"{prefix}weight"]
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._quantize_weights()
        
        # Reconstruct weights only when needed
        if self.quant_weight is not None:
            weight = self.quantizer.reconstruct(
                self.quant_weight,
                self.quant_scale,
                self.scale2
            ).to(x.dtype)
        else:
            weight = self._orig_weight.to(x.dtype)

        # Perform linear transformation
        result = F.linear(x, weight, self.bias)
        
        # Apply LoRA if enabled and weights exist
        if not self.disable_lora and self.has_lora_weights:
            x_lora = x.to(self._get_lora_dtype())
            result = self._lora_forward(x_lora, result)
            
        return result