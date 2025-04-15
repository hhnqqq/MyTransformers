from common.lora_modules.lora import *
import torch.nn.functional as F
import math

class LinearWithSALoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, init_r):
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
        # Use structure gate instead of weight_e
        structure_gate = self.structure_gate.to(self._get_lora_dtype())
        ranknum = self.ranknum + 1
        
        gated_weight_a = weight_a * structure_gate
        
        lora_result = F.linear(
            F.linear(self.lora_dropout(x), gated_weight_a),
            weight_b,
        ).to(result.dtype)
        
        return result + lora_result * self.lora_scaler / ranknum
    
    def _compute_lora(self):
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer)
            weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer)
            structure_gate = self.structure_gate
            
            # Apply structure-aware gating
            gated_weight_a = weight_a * structure_gate
            
        ranknum = self.ranknum + 1
        lora_result = F.linear(gated_weight_a, weight_b,)
        lora_weight = lora_result * self.lora_scaler / ranknum
        return lora_weight
    
    def init_lora_weights(self):
        # called by __init__ in LinearWithLoRA
        dtype = self._get_lora_dtype()
        requires_grad = not self.quant
        
        # Initialize A and B matrices with structural awareness
        self.weight_a = nn.Parameter(torch.randn((self.lora_rank, self.in_features), dtype=dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(torch.randn((self.out_features, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
        
        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))
        
        nn.init.orthogonal_(self.weight_a)
        nn.init.orthogonal_(self.weight_b)
        
        requires_grad = not self.quant
        
        self.structure_gate = nn.Parameter(
            torch.ones((self.lora_rank, 1), dtype=dtype), requires_grad=requires_grad
        )
        
        nn.init.normal_(self.structure_gate, mean=1.0, std=0.1)
        
        self.ranknum = nn.Parameter(torch.randn(1), requires_grad=False)
        self.ranknum.data.fill_(float(self.lora_rank))
        self.ranknum.requires_grad = False


class StructureAwareRankAllocator:
    """
    The StructureAwareRankAllocator for SALoRA Model.
    
    Args:
        config: The configuration of the SALoRA model.
        model: the model that we apply SALoRA to.
    """
    
    def __init__(self, model, peft_config):
        self.peft_config = peft_config
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        self.rank_pattern = {}
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1
        
        # Structure correlation tracking
        self.structure_correlation = {}
        self.layer_importance = {}
        
        self.reset_ipt()
        self._set_budget_scheduler(model)
    
    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.structure_correlation = {}
        
    def _set_budget_scheduler(self, model):
        self.init_bgt = 0
        self.name_set = set()
        
        # Find model structure and correlations between layers
        layer_dims = {}
        for n, p in model.named_parameters():
            if "weight_a" in n:
                self.init_bgt += p.size(0)
                self.name_set.add(n.replace("weight_a", "%s"))
                
                # Store layer dimensions for structural analysis
                module_name = n.rsplit(".", 1)[0]
                layer_dims[module_name] = (p.size(1), None)  # input_dim
                
            elif "weight_b" in n:
                module_name = n.rsplit(".", 1)[0]
                if module_name in layer_dims:
                    layer_dims[module_name] = (layer_dims[module_name][0], p.size(0))  # add output_dim
                    
        # Calculate structural importance based on layer position and connectivity
        self._analyze_layer_structure(model, layer_dims)
            
        self.name_set = sorted(self.name_set)
        # The total final rank budget
        self.target_bgt = self.peft_config.target_r * len(self.name_set)
    
    def _analyze_layer_structure(self, model, layer_dims):
        """Analyze model structure and assign importance scores to layers based on their position"""
        # Simple heuristic: deeper attention layers are more important for reasoning
        layer_depths = {}
        max_depth = 0
        
        # Approximate depth by counting dots in parameter name (more dots = deeper nesting)
        for name in layer_dims:
            depth = name.count('.')
            layer_depths[name] = depth
            max_depth = max(max_depth, depth)
        
        # Assign structural importance scores normalized to [0.5, 1.5]
        for name, depth in layer_depths.items():
            # Structural bias: deeper layers get more capacity
            # Layers in the middle of the network receive higher importance
            normalized_depth = depth / max_depth
            
            # Bell curve importance - layers in middle get highest importance
            # This reflects that middle layers often handle more complex representations
            importance = 0.5 + math.exp(-(normalized_depth - 0.5)**2 / 0.15)
            
            # # Additional importance to attention layers (generally more critical)
            # if 'attention' in name.lower():
            #     importance *= 1.2
                
            # # Feed-forward layers also get a boost but less than attention
            # if 'feed_forward' in name.lower() or 'mlp' in name.lower():
            #     importance *= 1.1
                
            self.layer_importance[name] = importance
    
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
            # Budget decreasing with a smoothed cosine scheduler for better transitions
            progress = (step - tinit) / (total_step - tfinal - tinit)
            # Cosine schedule gives smoother transition than cubic
            cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
            budget = int((self.init_bgt - self.target_bgt) * cos_factor + self.target_bgt)
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return budget, mask_ind
    
    def update_ipt(self, model, saved_gradients):
        """
        Compute the sensitivity incorporating structural information
        """
        # First update importance scores for individual parameters
        for n, p in model.named_parameters():
            if "weight_" in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    # Calculate importance as product of parameter and gradient
                    self.ipt[n] = (p * saved_gradients[n]).abs().detach()
                    
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
        
        # Now incorporate structural information
        self._update_structure_correlation(model)
    
    def _update_structure_correlation(self, model):
        """Update the correlation between different adaptations in the model"""
        with torch.no_grad():
            # Group parameters by layer/module
            layer_params = {}
            for n, p in model.named_parameters():
                if "weight_" in n:
                    module_name = n.rsplit(".", 1)[0]
                    if module_name not in layer_params:
                        layer_params[module_name] = []
                    layer_params[module_name].append((n, p))
            
            # Calculate correlation within each layer's adaptations
            for module_name, params in layer_params.items():
                if module_name not in self.structure_correlation:
                    self.structure_correlation[module_name] = 1.0
                
                # Skip if we don't have at least A and B weights
                if len(params) < 2:
                    continue
                
                # Calculate structural alignment within the layer
                # This is a measure of how coherently the different weights are adapting
                grad_norms = []
                for n, _ in params:
                    if n in self.exp_avg_ipt:
                        grad_norms.append(self.exp_avg_ipt[n].norm().item())
                
                if grad_norms:
                    # Higher variance in gradients indicates less structural coherence
                    # We want to allocate more parameters to highly coherent layers
                    mean_norm = sum(grad_norms) / len(grad_norms)
                    variance = sum((x - mean_norm)**2 for x in grad_norms) / len(grad_norms)
                    # Convert to a correlation-like metric (higher is better)
                    coherence = 1.0 / (1.0 + math.sqrt(variance) / (mean_norm + 1e-6))
                    # Update with EMA
                    self.structure_correlation[module_name] = (
                        0.9 * self.structure_correlation[module_name] + 0.1 * coherence
                    )
    
    def _element_score(self, n):
        """Get importance score incorporating uncertainty"""
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]
    
    def _structure_aware_score(self, n):
        """Get structure-aware importance score"""
        module_name = n.rsplit(".", 1)[0]
        base_score = self._element_score(n)
        
        # Apply structure-aware scaling factor based on:
        # 1. Layer importance (from static analysis)
        # 2. Dynamic correlation within the layer
        structure_factor = 1.0
        if module_name in self.layer_importance:
            structure_factor *= self.layer_importance[module_name]
        if module_name in self.structure_correlation:
            structure_factor *= self.structure_correlation[module_name]
            
        return base_score * structure_factor
    
    def _combine_ipt(self, ipt_gate, ipt_AB):
        """Combine importance scores, giving higher weight to gate importance"""
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        # Give 2x weight to gate importance in SALORA - more focus on structural elements
        sum_ipt = 2.0 * ipt_gate.view(-1) + ipt_AB.view(-1)
        return sum_ipt
    
    def mask_to_budget(self, model, budget):
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        
        # Get the importance score for A, gate, B
        for n, p in model.named_parameters():
            if "weight_a" in n:
                # Use structure-aware scores
                entry_ipt = self._structure_aware_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace("weight_a", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if "weight_b" in n:
                entry_ipt = self._structure_aware_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace("weight_b", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if "structure_gate" in n:  # Changed from weight_e to structure_gate
                entry_ipt = self._structure_aware_score(n)
                name_m = n.replace("structure_gate", "%s")
                value_ipt[name_m] = entry_ipt
        
        all_score = []
        # Calculate the score for each triplet
        for name_m in vector_ipt:
            if name_m not in value_ipt:
                continue
            ipt_gate = value_ipt[name_m]
            ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
            sum_ipt = self._combine_ipt(ipt_gate, ipt_AB)
            name_gate = name_m % "structure_gate"  # Changed from weight_e to structure_gate
            triplet_ipt[name_gate] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))
        
        # Determine mask threshold
        if not all_score:
            return {}
            
        mask_threshold = torch.kthvalue(
            torch.cat(all_score),
            k=min(self.init_bgt - budget, len(torch.cat(all_score))),
        )[0].item()
        
        rank_pattern = {}
        # Mask the unimportant triplets
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "structure_gate" in n:  # Changed from weight_e to structure_gate
                    if n in triplet_ipt:
                        p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
                        rank_pattern[n] = (
                            (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
                        )
        return rank_pattern
    
    def update_and_allocate(
        self, model, global_step, saved_gradients, force_mask=False
    ):
        # Update the importance score and allocate the budget
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
                if "structure_gate" in n and n in rank_pattern:  # Changed from weight_e to structure_gate
                    key = n
                    mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1).to(p.device)
                    p.masked_fill_(~mask.bool(), 0.0)


def update_and_allocate(model, global_step, saved_gradients):
    """
    This method updates SALORA budget and mask.
    
    This should be called in every training step after `loss.backward()` and before `zero_grad()`.
    
    `tinit`, `tfinal` and `deltaT` are handled with in the method.
    
    Args:
        global_step `int`): The current training step, it is used to calculate SALORA budget.
    
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
            model.rankallocator.rank_pattern = rank_pattern
    # Finalize the budget allocation
    elif global_step == lora_config.num_global_update_steps - lora_config.tfinal:
        _, rank_pattern = model.rankallocator.update_and_allocate(model, global_step, saved_gradients, force_mask=True)
        model.rankallocator.rank_pattern = rank_pattern
        model.rankallocator.reset_ipt()
    # Mask unimportant weights using the rank pattern
    elif global_step > lora_config.num_global_update_steps - lora_config.tfinal and model.rankallocator.rank_pattern:
        model.rankallocator.mask_using_rank_pattern(model, model.rankallocator.rank_pattern)
    # Pass the function and do forward propagation
    else:
        return None
