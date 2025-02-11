# @author: mengqi li
# @date: 2024-11-21
"""Un-official implements IncreLoRA(https://arxiv.org/abs/2308.12043)."""

from common.lora_modules.lora import *
import pdb
import math


class LinearWithIncreLoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, init_r):
        super().__init__(lora_config)
        self.lora_scaler = lora_config.lora_scaler
        self.lora_rank = init_r
        self.W = loraW()
        self.score = 0
        self.gradMatrix_trace = 0
        self.hook_handle = self.W.register_full_backward_hook(self.backward_hook)

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        weight_a = self.weight_a.to(self._get_lora_dtype())
        weight_b = self.weight_b.to(self._get_lora_dtype())
        weight_e = self.weight_e.to(self._get_lora_dtype())

        try:
            result += self.lora_dropout(x) @ self.W(
                weight_a, weight_e, weight_b, self.lora_scaler, self.ranknum
            ).T.to(self._get_lora_dtype())
        except:
            pdb.set_trace()
            print(self.W)

        return result

    def _compute_lora(self):
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer)
            weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer)
            weight_e = self.weight_quantizer
            # When using vanilla lora, the ab mixer is a identical matrix

        return self.W(weight_a, weight_e, weight_b, self.lora_scaler, self.ranknum).T

    def init_lora_weights(self):
        # called by __init__ in LinearWithLoRA
        dtype = self._get_lora_dtype()
        requires_grad = not self.quant

        self.weight_a = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn((self.lora_rank, self.in_features), dtype=dtype),
                    requires_grad=requires_grad,
                )
            ]
        )
        self.weight_e = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros((self.lora_rank, 1), dtype=dtype),
                    requires_grad=requires_grad,
                )
            ]
        )
        self.weight_b = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn((self.out_features, self.lora_rank), dtype=dtype),
                    requires_grad=requires_grad,
                )
            ]
        )

        self.weight_a.to(self._get_lora_dtype())
        self.weight_b.to(self._get_lora_dtype())
        self.weight_e.to(self._get_lora_dtype())

        self.ranknum = nn.Parameter(torch.randn(1), requires_grad=False)
        self.ranknum.data.fill_(float(self.lora_rank))
        self.ranknum.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "weight_a"):
            # initialize A,B the same way as the default for nn.Linear
            # and E (singular values) for zero
            nn.init.zeros_(self.weight_e[0])
            nn.init.normal_(self.weight_a[0], mean=0.0, std=0.02)
            nn.init.normal_(self.weight_b[0], mean=0.0, std=0.02)

    def backward_hook(self, module, grad_input, grad_output):
        # print("Output_Grad:", grad_output)
        grad_Matrix = grad_output[0]
        try:
            W = (
                self.W(
                    self.weight_a,
                    self.weight_e,
                    self.weight_b,
                    self.lora_scaler,
                    self.ranknum,
                )
            ).abs()
            scale_W = 1
            self.score = torch.sum(
                ((W / scale_W) * grad_Matrix).abs().detach()
            ) / math.sqrt(W.numel())
        except:
            pdb.set_trace()

    def add_reserve_param(self, add_r, advance_learn=True):
        for _ in range(add_r):
            e = nn.Parameter(self.weight.new_zeros(1, 1), requires_grad=False)
            a = nn.Parameter(
                self.weight.new_zeros((1, self.in_features)),
                requires_grad=advance_learn,
            )
            b = nn.Parameter(
                self.weight.new_zeros((self.out_features, 1)),
                requires_grad=advance_learn,
            )
            e[0][0] = 1e-5 if advance_learn else 0.0
            nn.init.normal_(a, mean=0.0, std=0.02)
            nn.init.normal_(b, mean=0.0, std=0.02)
            self.weight_e.append(e)
            self.weight_a.append(a)
            self.weight_b.append(b)


class loraW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, E, B, scaling, ranknum):
        return (
            torch.cat([b for b in B], 1)
            @ (torch.cat([a for a in A], 0) * torch.cat([e for e in E], 0))
            * scaling
            / (ranknum + 1e-5)
        )


class IncreRankAllocator:
    """
    The RankAllocator for IncreLoRA Model that will be called every training step.

    Args:
        model: the model that we apply IncreLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        incre_interval (`int`): The time internval between two budget allocations.
        top_h (`int`): The number of modules selected.
        advance_learn (`bool`): With or without advance learning.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank.
    """

    def __init__(self, model, peft_config):
        self.peft_config = peft_config

        self.ave_target_rank = peft_config.target_r
        self.target_rank = None
        self.lora_init_rank = peft_config.init_r

        self.init_warmup = peft_config.tinit
        self.incre_interval = peft_config.deltaT
        self.advance_learn = True
        self.top_h = peft_config.top_h
        incre_rank_num = None
        if incre_rank_num:
            self.incre_rank_num = incre_rank_num
        else:
            rank_dic = {2: 1, 4: 2, 6: 3, 8: 4}
            self.incre_rank_num = rank_dic[self.ave_target_rank]

        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        self.total_step = peft_config.num_global_update_steps

        self.model = model
        self.weight_decay = None
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {}
        self.get_lora_param_name()
        self.total_rank = self.initial_total_rank

        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 < 1 and self.beta1 > 0
        assert self.beta2 < 1 and self.beta2 > 0

        self.print_total_step()

    def get_lora_param_name(self):
        # Prepare the budget scheduler
        self.name_set = set()
        self.initial_total_rank = 0
        self.shape_dict = {}
        for n, layer in self.model.named_modules():
            if isinstance(layer, LinearWithIncreLoRA):
                self.name_set.add(n)
                self.initial_total_rank += layer.weight_a[0].size(0)
                self.shape_dict[n + ".lora_A"] = layer.weight_a[0].shape
                self.shape_dict[n + ".lora_B"] = layer.weight_b[0].shape

        self.name_set = list(sorted(self.name_set))
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set)

    def print_total_step(self):
        # Set total step number
        rank_per_round = self.top_h * self.incre_rank_num
        total_round = math.ceil(
            (self.target_rank - self.initial_total_rank) / rank_per_round
        )
        total_incre_step = self.incre_interval * total_round

        print(
            "Total incremental step: total_incre_step: {}, of total steps: {:.0%}".format(
                total_incre_step, total_incre_step / self.total_step
            )
        )

    def get_rank_pattern(self):
        # Return rank pattern
        return self.rank_pattern

    def update_ipt(self, model):
        for n, layer in model.named_modules():
            if isinstance(layer, LinearWithIncreLoRA):
                if n not in self.ipt:
                    self.ipt[n] = 0
                    self.exp_avg_ipt[n] = 0
                    self.exp_avg_unc[n] = 0

                # self.tb_writter.add_scalar("GradMatrix_Rank/%s"%(n[:-7],), layer.gradMatrix_rank, global_step)
                try:
                    self.ipt[n] = layer.score

                    # Update sensitivity
                    self.exp_avg_ipt[n] = (
                        self.beta1 * self.exp_avg_ipt[n]
                        + (1 - self.beta1) * self.ipt[n]
                    )
                    # Update uncertainty
                    self.exp_avg_unc[n] = (
                        self.beta2 * self.exp_avg_unc[n]
                        + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    )
                except:
                    pdb.set_trace()
                    print(layer)

    def calculate_score(self, n, layer, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = 0.0
            for n, p in layer.named_parameters():
                ipt_score += p.abs().detach().clone()
        else:
            raise ValueError("Unexcptected Metric: %s" % metric)
        return ipt_score

    def increase_to_target_rank(
        self, model, global_step, optimizer, lr_scheduler, args
    ):
        is_dict = {}
        all_is = []
        # Calculate the importance score for each sub matrix
        for n, layer in model.named_modules():
            if isinstance(layer, LinearWithIncreLoRA):
                ipt_score = self.calculate_score(n, layer, metric="ipt")
                is_dict[n] = ipt_score
                all_is.append(ipt_score)

        # Calculate the increasing threshold
        k = min(self.top_h, self.target_rank - self.total_rank)
        increase_threshold = torch.topk(torch.tensor(all_is), k)[0][-1].item()
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            new_param_list = []
            add_r = self.incre_rank_num
            for n, layer in model.named_modules():
                if isinstance(layer, LinearWithIncreLoRA):
                    if is_dict[n] >= increase_threshold:
                        # rank increase 1
                        layer.ranknum += add_r
                        self.total_rank += add_r

                        # add weight_e
                        for param in layer.weight_e[-add_r:]:
                            param.requires_grad = True
                            new_param_list.append(param)

                        if self.advance_learn:
                            layer.add_reserve_param(add_r, True)
                            new_param_list.extend(layer.weight_a[-add_r:])
                            new_param_list.extend(layer.weight_b[-add_r:])
                        else:
                            for param in layer.weight_a[-add_r:]:
                                param.requires_grad = True
                                new_param_list.append(param)
                            for param in layer.weight_b[-add_r:]:
                                param.requires_grad = True
                                new_param_list.append(param)
                            layer.add_reserve_param(add_r, False)

                        print(
                            "The lora parameters rank of {} increased by {}".format(
                                n, add_r
                            )
                        )

            # update_deepspeed_optimizer(model, optimizer, lr_scheduler, args)
            optimizer.add_param_group(
                {
                    "params": new_param_list,
                    "weight_decay": args.weight_decay,
                }
            )

            if self.total_rank == self.target_rank:
                for name, module in model.named_modules():
                    if isinstance(module, LinearWithIncreLoRA):
                        module.hook_handle.remove()
                        for param in module.weight_e[-add_r:]:
                            param.fill_(0.0)

        return increase_threshold

    def update_and_increase_increlora(
        self, model, global_step, optimizer, lr_scheduler, args
    ):
        self.global_step = global_step
        increase_threshold = None
        add_r = self.incre_rank_num

        if global_step == 0:
            new_param_list = []
            for name, module in model.named_modules():
                if isinstance(module, LinearWithIncreLoRA):
                    module.add_reserve_param(add_r, self.advance_learn)
                    new_param_list.extend(module.weight_a[-add_r:])
                    new_param_list.extend(module.weight_b[-add_r:])
            if self.advance_learn:
                # update_deepspeed_optimizer(model, optimizer, lr_scheduler, args)
                optimizer.add_param_group(
                    {
                        "params": new_param_list,
                        "weight_decay": args.weight_decay,
                    }
                )
                pass

        if self.total_rank < self.target_rank:
            self.update_ipt(model)
            if (
                global_step > self.init_warmup
                and global_step % self.incre_interval == 0
            ):
                increase_threshold = self.increase_to_target_rank(
                    model, global_step, optimizer, lr_scheduler, args
                )

        return self.top_h, increase_threshold
