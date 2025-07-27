import torch
import torch.nn as nn
import bitsandbytes as bnb

from common.lora_modules.lora import *

class LinearWithQLoRA(LinearWithLoRA):
    def __init__(
        self,
        lora_config: LoRAConfig,
        compute_dtype = torch.bfloat16,
        quant_type: str = "nf4"
    ):
        super().__init__(lora_config)

        self.qlora_config = lora_config
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type

        self.is_quantized = False
        self.quantized_linear = None

    def quantize_base_layer(self):
        if self.is_quantized:
            return

        if self.weight is None or self.weight.data.numel() == 0:
            raise RuntimeError("Cannot quantize layer because self.weight has not been loaded.")

        device = self.weight.device

        self.quantized_linear = bnb.nn.Linear4bit(
            self.in_features,
            self.out_features,
            bias=(self.bias is not None),
            compute_dtype=self.compute_dtype,
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
        self.quantize_base_layer()
        self.quantized_linear = self.quantized_linear.to(x.device)
        result = self.quantized_linear(x)

        if self.disable_lora or not self.has_lora_weights:
            return result
        else:
            return self._lora_forward(x, result)

    def _get_lora_dtype(self):
        if self.is_quantized:
            return self.compute_dtype
        else:
            return super()._get_lora_dtype()

    def _merge_lora(self) -> bool:
        if not self.has_lora_weights or self.merged:
            return False

        if self.is_quantized:
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
            
    def _compute_lora_weight(self):
        # 确保在任何状态下都能正确计算lora增量
        if not self.has_lora_weights:
            return None
        
        dtype = self._get_lora_dtype()
        device = self.weight_a.device
        
        lora_weight = self.lora_scaler * torch.matmul(
            self.weight_b.to(device=device, dtype=dtype),
            self.weight_a.to(device=device, dtype=dtype)
        )
        return lora_weight