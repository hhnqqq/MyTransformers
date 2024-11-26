# @author: mengqi li
# @date: 2024-11-03
"""Un-official implements AdaLoRA(https://arxiv.org/pdf/2303.10512)."""

from common.lora_modules.lora import *


class LinearWithAdaLoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, init_r):
        """
        Initialize the LinearWithAdaLoRA layer.

        Args:

        Note:
        
        """
        super().__init__(lora_config)
        self.lora_scaler = lora_config.lora_scaler
        self.lora_rank = init_r

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer).to(
            self._get_lora_dtype()
        )
        weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer).to(
            self._get_lora_dtype()
        )
        weight_e = self.weight_e.to(self._get_lora_dtype())
        ranknum = self.ranknum + 1e-5

        lora_result = F.linear(
            F.linear(self.lora_dropout(x), weight_a * weight_e),
            weight_b,
            ).to(result.dtype)

        return result + lora_result * self.lora_scaler / ranknum

    def _compute_lora(self):
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer)
            weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer)
            weight_e = self.weight_quantizer
            # When using vanilla lora, the ab mixer is a identical matrix

        ranknum = self.ranknum + 1e-5
        lora_result = F.linear(weight_a * weight_e,weight_b,)
        lora_weight =  lora_result * self.lora_scaler / ranknum
        return lora_weight

    def init_lora_weights(self):
        # called by __init__ in LinearWithLoRA
        dtype = self._get_lora_dtype()
        requires_grad = not self.quant

        self.weight_a = nn.Parameter(torch.randn((self.lora_rank, self.in_features), dtype=dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(torch.randn((self.out_features, self.lora_rank), dtype=dtype), requires_grad=requires_grad)

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

        nn.init.normal_(self.weight_a, mean=0.0, std=0.02)
        nn.init.normal_(self.weight_b, mean=0.0, std=0.02)

        requires_grad = not self.quant

        # In adalora, we have a diagonal matrix, but stored in singular values vector
        # Here we have no need to specify a weight init method, just using zeros is fine
        self.weight_e = nn.Parameter(
            torch.zeros((self.lora_rank, 1), dtype=dtype), requires_grad=requires_grad
        )
        # The current rank
        self.ranknum = nn.Parameter(torch.randn(1), requires_grad=False)
        self.ranknum.data.fill_(float(self.lora_rank))
        self.ranknum.requires_grad = False


class RankAllocator:
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config):
        self.peft_config = peft_config
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1

        self.reset_ipt()
        self._set_budget_scheduler(model)

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        self.init_bgt = 0
        self.name_set = set()
        for n, p in model.named_parameters():
            if "weight_a" in n:
                self.init_bgt += p.size(0)
                self.name_set.add(n.replace("weight_a", "%s"))
        self.name_set = sorted(self.name_set)
        # The total final rank budget
        self.target_bgt = self.peft_config.target_r * len(self.name_set)

    def budget_schedule(self, step: int):
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.num_global_update_steps
        # Initial warmup
        if step <= tinit:
            budget = self.init_bgt
            mask_ind = False
        # Final fine-tuning
        elif step > total_step - tfinal:
            budget = self.target_bgt
            mask_ind = True
        else:
            # Budget decreasing with a cubic scheduler
            mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
            budget = int(
                (self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt
            )
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return budget, mask_ind

    def update_ipt(self, model, saved_gradients):
        """
        Compute the sensitivity I(t) in (8) for every parameter in {P,E,Q};
        Update I (t) as (9) and U(t) as (10) for every parameter in {P,E,Q};
        Compute S(t)
        k,i by (7), for k = 1,...,n and i = 1,...,r ;
        """
        for n, p in model.named_parameters():
            if "weight_" in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    # grad = deepspeed.utils.safe_get_full_grad(p)
                    self.ipt[n] = (p * saved_gradients[n]).abs().detach()
                    # TODO: check if this is correct
                    # self.ipt[n] = ((p+grad) * grad).abs().detach()
                    # Sensitivity smoothing
                    self.exp_avg_ipt[n] = (
                        self.beta1 * self.exp_avg_ipt[n]
                        + (1 - self.beta1) * self.ipt[n]
                    )
                    # Uncertainty quantification
                    self.exp_avg_unc[n] = (
                        self.beta2 * self.exp_avg_unc[n]
                        + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    )

    def _element_score(self, n):
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_budget(self, model, budget):
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        # Get the importance score for A, E, B
        for n, p in model.named_parameters():
            if "weight_a" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace("weight_a", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if "weight_b" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace("weight_b", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if "weight_e" in n:
                entry_ipt = self._element_score(n)
                name_m = n.replace("weight_e", "%s")
                value_ipt[name_m] = entry_ipt

        all_score = []
        # Calculate the score for each triplet
        for name_m in vector_ipt:
            ipt_E = value_ipt[name_m]
            ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_m % "weight_e"
            triplet_ipt[name_E] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))

        mask_threshold = torch.kthvalue(
            torch.cat(all_score),
            k=self.init_bgt - budget,
        )[0].item()

        # print(f"torch.cat(all_score): f{torch.cat(all_score)}")
        # print(f"self.init_bgt, budget: {self.init_bgt}, {budget}")
        # print("mask_threshold: ", mask_threshold)

        rank_pattern = {}
        # Mask the unimportant triplets
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "weight_e" in n:
                    p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
                    rank_pattern[n] = (
                        (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
                    )
        return rank_pattern

    def update_and_allocate(
        self, model, global_step, saved_gradients, force_mask=False
    ):
        # # Update the importance score and allocate the budget
        if (
            global_step
            < self.peft_config.num_global_update_steps - self.peft_config.tfinal
        ):
            self.update_ipt(model, saved_gradients)
        budget, mask_ind = self.budget_schedule(global_step)
        # Allocate the budget according to importance scores
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
        else:
            rank_pattern = None
        return budget, rank_pattern

    def mask_using_rank_pattern(self, model, rank_pattern):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "weight_e" in n:
                    key = n
                    mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1).to(p.device)
                    p.masked_fill_(~mask.bool(), 0.0)

def update_and_allocate(model, global_step, saved_gradients):
    """
    This method updates Adalora budget and mask.

    This should be called in every training step after `loss.backward()` and before `zero_grad()`.

    `tinit`, `tfinal` and `deltaT` are handled with in the method.

    Args:
        global_step (`int`): The current training step, it is used to calculate adalora budget.

    Example:

    ```python
    >>> loss = model(**input).loss
    >>> loss.backward()
    >>> optimizer.step()
    >>> model.base_model.update_and_allocate(i_step)
    >>> optimizer.zero_grad()
    ```
    """
    lora_config = model.rankallocator.peft_config
    # Update the importance score and allocate the budget
    if global_step < lora_config.num_global_update_steps - lora_config.tfinal:
        _, rank_pattern = model.rankallocator.update_and_allocate(model, global_step, saved_gradients)
        if rank_pattern:
            lora_config.rank_pattern = rank_pattern
    # Finalize the budget allocation
    elif global_step == lora_config.num_global_update_steps - lora_config.tfinal:
        _, rank_pattern = model.rankallocator.update_and_allocate(model, global_step, saved_gradients, force_mask=True)
        # for some reason, this freezes the trainable parameters and nothing gets updates
        # self.resize_modules_by_rank_pattern(rank_pattern, self.trainable_adapter_name)
        lora_config.rank_pattern = rank_pattern
        model.rankallocator.reset_ipt()
    # Currently using inefficient way to mask the unimportant weights using the rank pattern
    #  due to problem mentioned above
    elif global_step > lora_config.num_global_update_steps - lora_config.tfinal:
        model.rankallocator.mask_using_rank_pattern(model, lora_config.rank_pattern)
    # Pass the function and do forward propagation
    else:
        return None