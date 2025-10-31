from common.lora_modules.qlora import *

class LinearWithLoDA(LinearWithQLoRA):
    def __init__(self, lora_config: LoRAConfig, weight_ab_mixer_init_method):
        super().__init__(lora_config)
        self.lora_scaler = 1 / self.out_features**0.5
        self.weight_ab_mixer_1_init_method = weight_ab_mixer_init_method
        self.weight_ab_mixer_2_init_method = weight_ab_mixer_init_method

    def init_lora_weights(self):
        super().init_lora_weights()
        dtype = self._get_lora_dtype()

        self.weight_ab_mixer_1 = nn.Parameter(torch.empty((self.lora_rank, self.lora_rank), dtype=dtype), requires_grad=True)
        self.weight_ab_mixer_2 = nn.Parameter(torch.empty((self.lora_rank, self.lora_rank), dtype=dtype), requires_grad=True)
        self._init_weight('weight_ab_mixer_1')
        self._init_weight('weight_ab_mixer_2')

    def get_weight_init_kwargs(self, weight_name: str, method: Optional[str] = None) -> Dict[str, Any]:
        if 'weight_ab_mixer' in weight_name:
            weight_name = 'weight_ab_mixer'

        return super().get_weight_init_kwargs(weight_name, method)
    
    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        # If self.run_lora_in_fp32, then the dtype of lora_result will be fp32.
        weight_a = self.weight_a.to(self._get_lora_dtype())
        weight_b = self.weight_b.to(self._get_lora_dtype())
        A_out = F.leaky_relu(F.linear(self.lora_dropout(x), weight_a), negative_slope=0.8)
        mixer_1_out = F.leaky_relu(F.linear(A_out, self.weight_ab_mixer_1), negative_slope=0.8)
        f1_out = F.leaky_relu(F.linear(mixer_1_out, self.weight_ab_mixer_2), negative_slope=0.8)
        lora_result = F.linear(F.linear(self.lora_dropout(x), weight_a), weight_b) + F.leaky_relu(F.linear(f1_out, weight_b), negative_slope=0.8)
        return result + self.lora_scaler * lora_result.to(result.dtype)