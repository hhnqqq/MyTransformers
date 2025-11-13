from common.lora_modules.qlora import *

class LinearWithLoRADash(LinearWithQLoRA):
    def __init__(self, lora_config, init_t=100, index=8):
        super().__init__(lora_config)
        self.step = 0
        self.init_t = init_t
        self.index = index

    def calculate_change_rate(self, a, bb, r):
        self.lora_change_a = nn.Parameter(a)
        self.lora_change_bb = nn.Parameter(bb)

        change_rate = abs(bb) / abs(a)
        _, top_r_indices = torch.topk(change_rate, r)
        return top_r_indices

    def init_lora_weights(self):
        super().init_lora_weights()
        dtype = self._get_lora_dtype()
        self.weight_v_top = nn.Parameter(torch.zeros((self.out_features, self.index), dtype=dtype))
        self.weight_uh_top = nn.Parameter(torch.zeros((self.index, self.in_features), dtype=dtype))
        self.weight_index = nn.Parameter(torch.zeros(self.index, dtype=dtype))

    
    def _lora_forward(self, x, result):
        self.step += 1
        if self.step < self.init_t:
            return super()._lora_forward(x, result)
        elif self.step == self.init_t:
            V, S, Uh = torch.linalg.svd(self.weight.data.float())
            V, S, Uh = V.to(self.weight.dtype), S.to(self.weight.dtype), Uh.to(self.weight.dtype)
            delta_S = torch.diag(torch.matmul(torch.matmul(V.T, self._compute_lora_weight()), Uh.T))
            top_index = self.calculate_change_rate(S, delta_S, self.index)
            self.weight_v_top.data = V.to(self._get_lora_dtype())[:, top_index]
            self.weight_uh_top.data = Uh.to(self._get_lora_dtype())[top_index, :]
            
        result += self.lora_dropout(x) @ (self.weight_v_top.to(self._get_lora_dtype()) @ torch.diag(self.weight_index) @ self.weight_uh_top.to(self._get_lora_dtype())).T
        return result