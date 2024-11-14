from common.lora_modules.lora import *

class LinearWithMoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, mora_type: str):
        if mora_type not in {'rope', 'sharing'}:
            raise ValueError(f'Not supported mora type: {mora_type}!')
        self.mora_type = mora_type
        super().__init__(lora_config)

    def _init_lora_weights(self):
        dtype = self._get_lora_dtype()
        requires_grad = not self.quant

        self.weight_matrix_a = nn.Parameter(torch.empty((self.lora_rank, self.lora_rank), dtype=dtype), requires_grad=requires_grad)

        if self.quant:
            self.weight_matrix_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))

        self._init_weight('weight_a')

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        x = self.lora_dropout(x)
        quantized_weight_a = self._quantize_weight(self.weight_matrix_a, self.weight_a_quantizer).to(self._get_lora_dtype())

        compressed_input = self.compress_input(x)
        mora_result = torch.matmul(compressed_input, quantized_weight_a)
        mora_result = self.decompress_result(mora_result, x)

        return result + mora_result.to(result.dtype)

    def compress_input(self, x):
        num_full_blocks = self.in_features // self.lora_rank
        if self.in_features % self.lora_rank:
            padding_size = self.lora_rank - self.in_features % self.lora_rank
            x = torch.cat([x, x[..., :padding_size]], dim=-1)
            num_full_blocks+=1
        compressed_input = x.view(*x.shape[:-1], num_full_blocks, self.lora_rank)

        if self.mora_type == 'rope':
            if not hasattr(self, 'cos') or not hasattr(self, 'sin'):
                inverse_frequency = 1.0 / (10000 ** (torch.arange(0, self.lora_rank, 2).float() / self.lora_rank))
                time_indices = torch.arange(num_full_blocks)
                freqs = torch.outer(time_indices, inverse_frequency)
                embedding = torch.cat((freqs, freqs), dim=-1)
                self.cos = embedding.cos().unsqueeze(0).to(x.device, x.dtype)
                self.sin = embedding.sin().unsqueeze(0).to(x.device, x.dtype)

            rotated_half_input = torch.cat((-compressed_input[..., self.lora_rank//2:], compressed_input[..., :self.lora_rank//2]), dim=-1)
            compressed_input = compressed_input * self.cos + rotated_half_input * self.sin

        elif self.mora_type == 'sharing':
            compressed_input = compressed_input.sum(dim=-2)

        return compressed_input

    def decompress_result(self, mora_result, x):
        if self.mora_type == 'rope':
            mora_result = mora_result.view(*x.shape[:-1], -1)[..., :self.out_features]
            if mora_result.shape[-1] < self.out_features:
                repeat_time = -(-self.out_features // mora_result.shape[-1])  # Ceiling division
                mora_result = torch.cat([mora_result] * repeat_time, dim=-1)[..., :self.out_features]
        elif self.mora_type == 'sharing':
            repeat_count = -(-self.out_features // self.lora_rank)  # Ceiling division
            mora_result = torch.cat([mora_result] * repeat_count, dim=-1)[..., :self.out_features]
        return mora_result

    def _merge_lora(self):
        raise NotImplementedError('Weights of MoRA can not be merged!')

    def _del_lora(self):
        raise NotImplementedError('Weights of MoRA can not be deleted!')
    
    @property
    def has_lora_weights(self):
        has_attr = hasattr(self, 'weight_a')
        if has_attr:
            is_not_None = self.weight_a is not None
        return has_attr and is_not_None