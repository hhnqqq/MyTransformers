from common.lora_modules.lora import LoRAConfig
from common.lora_modules.qlora import *
from torch import Tensor

class LinearWithAurora(LinearWithQLoRA):
    def __init__(self, 
        lora_config: LoRAConfig,
        grid_size=5,
        spline_order=3,  
        scale_noise=0.1,
        base_activation=nn.Tanh,
        grid_range=[-1, 1],
    ):
        super().__init__(lora_config)
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", grid)  

        self.weight_ab_mixer = nn.Parameter(torch.Tensor(self.lora_rank, self.lora_rank), requires_grad=True)
        self.weight_a_spline = nn.Parameter(torch.Tensor(self.lora_rank, grid_size + spline_order))
        self.scale_noise = scale_noise
        self.base_activation = base_activation()

    def init_lora_weights(self):
        super().init_lora_weights()
        nn.init.xavier_normal_(self.weight_ab_mixer)
        nn.init.uniform_(self.weight_a_spline, -self.scale_noise, self.scale_noise)
             
    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.lora_rank
        x = x.unsqueeze(-1)  
        grid = self.grid.to(self._get_lora_dtype())
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()  

        for k in range(1, self.spline_order + 1):
            denom1 = grid[k:-1] - grid[:-k - 1]
            denom2 = grid[k + 1:] - grid[1:-k]
            denom1[denom1 == 0] = 1 
            denom2[denom2 == 0] = 1  

            term1 = ((x - grid[:-k - 1]) / denom1) * bases[:, :, :-1]
            term2 = ((grid[k + 1:] - x) / denom2) * bases[:, :, 1:]

            bases = term1 + term2  

        return bases.to(self._get_lora_dtype())

    def ANL_forward(self, x):
        original_shape = x.shape
        weight_ab_mixer = self.weight_ab_mixer.to(self._get_lora_dtype())
        weight_a_spline = self.weight_a_spline.to(self._get_lora_dtype())
        x = x.view(-1, self.lora_rank) 

        fixed_act_output = self.base_activation(F.linear(self.base_activation(x), weight_ab_mixer))  

        spline_act_output = F.linear(self.b_splines(x).sum(dim=1), weight_a_spline)
        
        act_output = fixed_act_output + spline_act_output
        act_output = act_output.view(*original_shape[:-1], self.lora_rank)
        return act_output

    def _lora_forward(self, x: Tensor, result: Tensor) -> Tensor:
        weight_a = self.weight_a.to(self._get_lora_dtype())
        weight_b = self.weight_b.to(self._get_lora_dtype())
        
        lora_result = F.linear(self.ANL_forward(F.linear(self.lora_dropout(x), weight_a)), weight_b).to(result.dtype)
        return result + self.lora_scaler * lora_result
    
    def _compute_lora_weight(self): 
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self.ANL_forward(self.weight_a.to(self._get_lora_dtype()))
            weight_b = self.weight_b.to(self._get_lora_dtype())
            lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)
            return lora_weight.to(self.weight.dtype)
        