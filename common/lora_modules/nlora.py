from common.lora_modules.mos_lora import *

class LinearWithNLoRA(LinearWithMosLoRA):
    def init_lora_weights(self):
        dtype = self._get_lora_dtype()
        requires_grad = not self.quant

        self.weight_a = nn.Parameter(torch.empty((self.lora_rank, self.in_features), dtype=dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(torch.zeros((self.out_features, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
        self.weight_ab_mixer = nn.Parameter(torch.empty((self.lora_rank, self.lora_rank), dtype=dtype), requires_grad=requires_grad)

        weight_ab_mixer_data = self.weight[:self.lora_rank, :self.lora_rank]
        weight_a_data = self.weight[:self.lora_rank, :]         
        weight_b_data = self.weight[:, :self.lora_rank]
        
        self.weight_a = nn.Parameter(weight_a_data.clone().to(dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(weight_b_data.clone().to(dtype), requires_grad=requires_grad)
        self.weight_ab_mixer = nn.Parameter(weight_ab_mixer_data.clone().to(dtype), requires_grad=requires_grad)

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))
            self.weight_ab_mixer_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
