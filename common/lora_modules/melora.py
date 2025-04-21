# @author: haonan he
# @date: 2024-08-21
""" Implements MELORA"""

from common.lora_modules.lora import *

class LinearWithMELoRA(LinearWithLoRA):
    def __init__(self,
        lora_config: LoRAConfig,
        me_lora_n_split: int = 2,
        forward_method: str = 'for'):
        """
        Initialize the LinearWithMELoRA layer.

        Args:
            me_lora_n_split int: Number of groups of LoRA weight.

        Note:
            For detailed explanations of in_features, out_features, lora_rank, lora_scaler, 
            lora_dropout, quant, weight_a_init_method, and weight_b_init_method, 
            please refer to the parent class LinearWithLoRA.
        """
        self._prepare_melora_attrs(me_lora_n_split, 
                                   lora_config.lora_rank, 
                                   lora_config.in_features, 
                                   lora_config.out_features)
        
        super().__init__(lora_config)
        if forward_method == "for":
            self.init_lora_weights = self.init_lora_weights_for
            self._lora_forward = self._lora_forward_for
        elif forward_method == "einsum":
            self.init_lora_weights = self.init_lora_weights_einsum
            self._lora_forward = self._lora_forward_einsum
        if lora_config.quant:
            print(f'Currently MELoRA is incompatible with quant, skipped quant')

    def _prepare_melora_attrs(self, me_lora_n_split, lora_rank, in_features, out_features):
        self.melora_n_split = me_lora_n_split
        self.lora_rank = lora_rank
        self.in_features = in_features
        self.out_features = out_features

        self._check_exact_division()
        self.mini_lora_rank = int(self.lora_rank / self.melora_n_split)
        self.mini_in_features = int(self.in_features / self.melora_n_split)
        self.mini_out_features = int(self.out_features / self.melora_n_split)

    def _check_exact_division(self):
        if self.lora_rank % self.melora_n_split != 0:
            raise ValueError(f"lora_rank ({self.lora_rank}) must be divisible by melora_n_split ({self.melora_n_split})")
        if self.in_features % self.melora_n_split != 0:
            raise ValueError(f"in_features ({self.in_features}) must be divisible by melora_n_split ({self.melora_n_split})")
        if self.out_features % self.melora_n_split != 0:
            raise ValueError(f"out_features ({self.out_features}) must be divisible by melora_n_split ({self.melora_n_split})")

    def init_lora_weights_for(self):
        dtype = torch.int8 if self.quant else None
        requires_grad = not self.quant

        self.weight_a, self.weight_b =nn.ParameterList(), nn.ParameterList()  
        for _ in range(self.melora_n_split):
            mini_weight_a = nn.Parameter(torch.empty((self.mini_lora_rank, self.mini_in_features), dtype=dtype), requires_grad=requires_grad)
            mini_weight_b = nn.Parameter(torch.zeros((self.mini_out_features, self.mini_lora_rank), dtype=dtype), requires_grad=requires_grad)
            self.weight_a.append(mini_weight_a)
            self.weight_b.append(mini_weight_b)
        self._init_weight('weight_a')
        self._init_weight('weight_b')

    def init_lora_weights_einsum(self):
        dtype = torch.int8 if self.quant else None
        requires_grad = not self.quant

        self.weight_a = nn.Parameter(torch.empty((self.melora_n_split, self.mini_lora_rank, self.mini_in_features), dtype=dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(torch.zeros((self.melora_n_split, self.mini_out_features, self.mini_lora_rank), dtype=dtype), requires_grad=requires_grad)
        super()._init_weight('weight_a')
        super()._init_weight('weight_b')

    def _init_weight(self, weight_name: str):
        weight_list = getattr(self, weight_name)
        init_method = getattr(self, f"{weight_name}_init_method")
        init_kwargs = self.get_weight_init_kwargs(weight_name, init_method)
        for weight in weight_list:
            self.get_weight_init_method(**init_kwargs)(weight)

    def _lora_forward_for(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        lora_result = []
        for i in range(self.melora_n_split):
            mini_x = x[..., i*self.mini_in_features:(i+1)*self.mini_in_features]
            mini_weight_a = self.weight_a[i].to(self._get_lora_dtype())
            mini_weight_b = self.weight_b[i].to(self._get_lora_dtype())
            mini_lora_result = F.linear(F.linear(self.lora_dropout(mini_x), mini_weight_a), mini_weight_b)
            lora_result.append(mini_lora_result)
        lora_result = torch.cat(lora_result, dim=-1).to(result.dtype)

        return result + self.lora_scaler * lora_result
    
    def _lora_forward_einsum(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        bsz_seq_len = x.shape[:-1]
        x = x.view(*bsz_seq_len, self.melora_n_split, self.mini_in_features)
        xa = torch.einsum("...si,sri->...sr", self.lora_dropout(x), self.weight_a.to(self._get_lora_dtype()))
        lora_result = torch.einsum("...sr,sor->...so", xa, self.weight_b.to(self._get_lora_dtype())).reshape(*bsz_seq_len, self.out_features)
        return result + self.lora_scaler * lora_result

    def _compute_lora_weight(self):
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self._diagonal_concat_weight_a().to(self._get_lora_dtype())
            weight_b = self._diagonal_concat_weight_b().to(self._get_lora_dtype())
            lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)
            return lora_weight.to(self.weight.dtype)
        
    def _diagonal_concat_weight_a(self):
        weight_a = torch.zeros(self.lora_rank, self.in_features)
        
        for i in range(self.melora_n_split):
            start_row = i * self.mini_lora_rank
            start_col = i * self.mini_in_features
            weight_a[start_row:start_row+self.mini_lora_rank, start_col:start_col+self.mini_in_features] = self.weight_a[i]

        return weight_a
    
    def _diagonal_concat_weight_b(self):
        weight_b = torch.zeros(self.out_features, self.lora_rank)
        
        for i in range(self.melora_n_split):
            start_row = i * self.mini_out_features
            start_col = i * self.mini_lora_rank
            weight_b[start_row:start_row+self.mini_out_features, start_col:start_col+self.mini_lora_rank] = self.weight_b[i]
        
        return weight_b


if __name__ == "__main__":
    # --- Configuration ---
    import time
    in_features = 128
    out_features = 256
    lora_rank = 32
    me_lora_n_split = 4 # Must divide features and rank
    lora_alpha = 32
    lora_dropout = 0.1 # Set dropout > 0, but we'll use .eval() for comparison
    batch_size = 4
    seq_len = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Using device: {device}")
    print(f"Input: ({batch_size}, {seq_len}, {in_features})")
    print(f"Output: ({batch_size}, {seq_len}, {out_features})")
    print(f"Rank: {lora_rank}, Splits: {me_lora_n_split}")
    print(f"Mini Rank: {lora_rank // me_lora_n_split}, Mini In: {in_features // me_lora_n_split}, Mini Out: {out_features // me_lora_n_split}")

    # --- Create Config ---
    lora_config = LoRAConfig(in_features=in_features, 
                        out_features=out_features, 
                        lora_rank=lora_rank, 
                        lora_scaler=lora_alpha, 
                        quant=False)

    # --- Instantiate Layers ---
    # Use torch.manual_seed for reproducible initializations if needed *before* layer creation
    torch.manual_seed(42)
    layer_for = LinearWithMELoRA(lora_config, me_lora_n_split, forward_method='for').to(device).to(dtype)
    layer_for.init_lora_weights()
    torch.manual_seed(42) # Reset seed to ensure einsum layer *would* get same init if method was identical
    layer_einsum = LinearWithMELoRA(lora_config, me_lora_n_split, forward_method='einsum').to(device).to(dtype)
    layer_einsum.init_lora_weights()

    # --- CRITICAL: Synchronize Weights ---
    # Copy weights from 'for' version (ParameterList) to 'einsum' version (batched tensor)
    # This ensures the *logic* is tested with identical parameters.
    with torch.no_grad():
        # Weight A
        weight_a_list = [p.data.clone() for p in layer_for.weight_a]
        layer_einsum.weight_a.data = torch.stack(weight_a_list, dim=0)

        # Weight B
        weight_b_list = [p.data.clone() for p in layer_for.weight_b]
        layer_einsum.weight_b.data = torch.stack(weight_b_list, dim=0)

    # --- Prepare Input Data ---
    x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=dtype)
    # Base result (output of the main linear layer, which LoRA adds to)
    # We use zeros to isolate the LoRA calculation for this test.
    base_result = torch.zeros(batch_size, seq_len, out_features, device=device, dtype=dtype)

    # --- Set to Evaluation Mode (Disable Dropout) ---
    layer_for.eval()
    layer_einsum.eval()

    # --- Perform Forward Pass ---
    with torch.no_grad(): # Disable gradient calculation for inference
        output_for = layer_for._lora_forward(x, base_result.clone()) # Use clone if base_result shouldn't be modified
        output_einsum = layer_einsum._lora_forward(x, base_result.clone())

    # --- Compare Outputs ---
    # Use allclose for floating point comparisons
    are_close = torch.allclose(output_for, output_einsum, atol=1e-6) # Adjust tolerance if needed

    print(f"\nOutputs are close: {are_close}")

    if not are_close:
        print("Outputs differ!")
        print("Max absolute difference:", torch.max(torch.abs(output_for - output_einsum)))
        print("Output shape (for):", output_for.shape)
        print("Output shape (einsum):", output_einsum.shape)
        # Optionally print parts of the tensors to debug
        # print("Output for (part):", output_for[0, 0, :10])
        # print("Output einsum (part):", output_einsum[0, 0, :10])

    # --- Optional: Test Merged Weight Calculation ---
    # Also check if the merged weight computation yields the same result
    # Note: This depends on _diagonal_concat methods correctly handling both weight formats
    print("\nTesting merged weight computation equivalence:")
    with torch.no_grad():
        time_1 = time.time()
        merged_weight_for = layer_for._compute_lora_weight()
        time_2 = time.time()
        merged_weight_einsum = layer_einsum._compute_lora_weight()
        time_3 = time.time()
        print(time_2-time_1, time_3-time_2)

        if merged_weight_for is None or merged_weight_einsum is None:
             print("Skipping merged weight comparison (LoRA not active or error).")
        else:
            are_merged_weights_close = torch.allclose(merged_weight_for, merged_weight_einsum, atol=1e-6)
            print(f"Merged weights are close: {are_merged_weights_close}")
            if not are_merged_weights_close:
                print("Merged weights differ!")
                print("Max absolute difference:", torch.max(torch.abs(merged_weight_for - merged_weight_einsum)))
