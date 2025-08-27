from common.lora_modules.lora import LoRAConfig
from common.lora_modules.qlora import *

class LinearWithPROLoRA(LinearWithQLoRA):
    def __init__(self, lora_config: LoRAConfig, shared_lora_rank: int = 1, repeat_times: int = 2):
        super().__init__(lora_config)
        self.shared_lora_rank = shared_lora_rank * repeat_times
        self.lora_rank -= shared_lora_rank
        self.repeat_times = repeat_times

        if shared_lora_rank >= self.lora_rank:
            raise ValueError("shared rank must be smaller than total rank")
        if self.in_features % self.repeat_times != 0:
            raise ValueError(f"In features {self.in_features} must be divisible by repeat times {self.repeat_times}")
        if self.out_features % self.repeat_times != 0:
            raise ValueError(f"Out features {self.out_features} must be divisible by repeat times {self.repeat_times}")
        
        self.mini_in_features = self.in_features // repeat_times
        self.mini_out_features = self.out_features // repeat_times


    def init_lora_weights(self):
        dtype = self._get_lora_dtype()
        # initialize unshared part
        super().init_lora_weights()
        
        # Initialize shared base chunks (reduced feature dimension)
        # For A matrix shared base: [shared_lora_rank * repeat_times, in_features // repeat_times]
        self.weight_a_s = nn.Parameter(
            torch.empty((self.shared_lora_rank, self.mini_in_features), dtype=dtype),
            requires_grad=True
        )
        
        # For B matrix shared base: [out_features // repeat_times, shared_lora_rank * repeat_times]
        self.weight_b_s = nn.Parameter(
            torch.zeros((self.mini_out_features, self.shared_lora_rank), dtype=dtype),
            requires_grad=True
        )
        
        self._init_weight('weight_a_s')
        self._init_weight('weight_b_s')
        
    def _init_weight(self, weight_name: str):
        weight = getattr(self, weight_name)
        init_method = getattr(self, f"{weight_name.strip('_s')}_init_method")
        init_kwargs = self.get_weight_init_kwargs(weight_name.strip('_s'), init_method)
        if init_method and 'method' not in init_kwargs.keys():
            init_kwargs['method'] = init_method
        self.get_weight_init_method(**init_kwargs)(weight)

    def _construct_full_weight(self, unshared: torch.Tensor, shared: torch.Tensor, is_A: bool) -> torch.Tensor:
        """
        Construct the full weight matrix by combining unshared and rolled shared parts.
        
        Args:
            unshared: the unshared weight part
            shared: the base shared chunk
            is_A: whether this is for matrix A (True) or B (False)
            
        Returns:
            The full weight matrix with rolled shared chunks
        """
        # For shared part, we need to repeat it with roll
        shared_chunk_size = shared.size(1) if is_A else shared.size(0)
        rolled_shared_chunks = []
        
        for i in range(self.repeat_times):
            # Calculate stride for this chunk
            stride = max(1, shared_chunk_size // self.repeat_times) * i
            
            # Roll the shared chunk
            if is_A:
                rolled_chunk = torch.roll(shared, shifts=stride, dims=1)
            else:
                rolled_chunk = torch.roll(shared, shifts=stride, dims=0)
            
            rolled_shared_chunks.append(rolled_chunk)
        
        # Combine all rolled chunks
        if is_A:
            rolled_shared = torch.cat(rolled_shared_chunks, dim=1)
            return torch.cat([unshared, rolled_shared], dim=0)
        else:
            rolled_shared = torch.cat(rolled_shared_chunks, dim=0)
            return torch.cat([unshared, rolled_shared], dim=1)
        
    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        # Get the base weights in correct dtype
        dtype = self._get_lora_dtype()
        
        # Construct full weight_a with unshared and rolled shared parts
        weight_a = self._construct_full_weight(self.weight_a, self.weight_a_s, is_A=True).to(dtype)
        
        # Construct full weight_b with unshared and rolled shared parts
        weight_b = self._construct_full_weight(self.weight_b, self.weight_b_s, is_A=False).to(dtype)
        
        # Compute PROLoRA forward pass
        lora_result = F.linear(F.linear(self.lora_dropout(x), weight_a), weight_b).to(result.dtype)
        return result + self.lora_scaler * lora_result