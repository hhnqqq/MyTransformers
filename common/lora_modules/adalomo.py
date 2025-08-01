import math

import torch
from torch.optim import Optimizer

class AdaLomo(Optimizer):

    def __init__(
        self,
        model,
        lr=1e-3,
        loss_scale=2**10,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        clip_grad_norm=None,
        clip_grad_value=None,
        weight_decay=0.0,
    ):
        self.model = model
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.weight_decay = weight_decay
        self.loss_scale = loss_scale
        if self.weight_decay > 0.0:
            self.do_weight_decay = True
        else:
            self.do_weight_decay = False
        self.eps = eps
        self.step_num = 0
        self.decay_rate = decay_rate
        self.clip_threshold = clip_threshold

        # for grad norm
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError(
                f"clip_grad_norm should be positive, got {self.clip_grad_norm}."
            )
        self.gather_norm = False
        self.grad_norms = []
        self.hooks = []
        self.clip_coef = None

        self.exp_avg_sq = {}
        self.exp_avg_sq_row = {}
        self.exp_avg_sq_col = {}

        for n, p in self.model.named_parameters():
            if len(p.data.shape) == 1:
                self.exp_avg_sq[n] = torch.zeros(p.data.shape[0], dtype=torch.float32).cuda()
            else:
                self.exp_avg_sq_row[n] = torch.zeros(p.data.shape[0], dtype=torch.float32).cuda()
                self.exp_avg_sq_col[n] = torch.zeros(p.data.shape[1], dtype=torch.float32).cuda()
            if p.requires_grad:
                self.hooks.append(p.register_hook(self.fuse_update()))
        defaults = dict(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm,
            clip_grad_value=clip_grad_value,
        )
        super(AdaLomo, self).__init__(self.model.parameters(), defaults)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def fuse_update(self):

        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        torch.distributed.all_reduce(
                            p.grad, op=torch.distributed.ReduceOp.AVG, async_op=False
                        )
                        grad_fp32 = p.grad.to(torch.float32)
                        p.grad = None
                        if self.loss_scale:
                            grad_fp32.div_(self.loss_scale)
                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm computation and parameter update, respectively.
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:
                            # grad clip or norm
                            if (
                                self.clip_grad_value is not None
                                and self.clip_grad_value > 0
                            ):
                                # Clipping gradients by their value
                                grad_fp32.clamp_(
                                    min=-self.clip_grad_value, max=self.clip_grad_value
                                )
                            if (
                                self.clip_grad_norm is not None
                                and self.clip_grad_norm > 0
                                and self.clip_coef is not None
                            ):
                                # Normalize the gradient according to its norm (computed in another pass)
                                grad_fp32.mul_(self.clip_coef)

                            # To avoid math errors for edge cases
                            if self.step_num == 0 and self.decay_rate < 0:
                                decay_rate = - self.decay_rate
                            else:
                                decay_rate = self.decay_rate

                            beta2t = 1.0 - math.pow(self.step_num, decay_rate)
                            update = (grad_fp32**2) + self.eps[0]

                            if len(p.data.shape) > 1:
                                self.exp_avg_sq_row[n].mul_(beta2t).add_(
                                    update.mean(dim=-1), alpha=1.0 - beta2t
                                )
                                self.exp_avg_sq_col[n].mul_(beta2t).add_(
                                    update.mean(dim=-2), alpha=1.0 - beta2t
                                )
                                update = self._approx_sq_grad(
                                    self.exp_avg_sq_row[n], self.exp_avg_sq_col[n]
                                )
                                update.mul_(grad_fp32)
                            else:
                                self.exp_avg_sq[n].mul_(beta2t).add_(
                                    update, alpha=1.0 - beta2t
                                )
                                update = self.exp_avg_sq[n].rsqrt().mul_(grad_fp32)

                            update.div_(
                                (self._rms(update) / self.clip_threshold).clamp_(
                                    min=1.0
                                )
                            )

                            p_fp32 = p.data.to(torch.float32)
                            p_rms = torch.norm(p_fp32, 2.0) / math.sqrt(p.numel())
                            torch.distributed.all_reduce(p_rms, op=torch.distributed.ReduceOp.SUM)
                            lr = self.lr
                            param_scale = max(self.eps[1], p_rms)
                            lr = lr * param_scale

                            if self.do_weight_decay:
                                p_fp32.mul_(1.0 - lr * self.weight_decay)
                            p_fp32.add_(update, alpha=-lr)
                            p.data.copy_(p_fp32)

            return x

        return func

    def fused_backward(self, loss, lr):
        self.lr = lr
        if self.loss_scale:
            loss = loss * self.loss_scale
        self.step_num += 1
        loss.backward()
        # update the last parameter since the last parameter in the computaiton graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything.
        self.fuse_update()

    def grad_norm(self, loss):
        self.gather_norm = True
        self.grad_norms = []
        if self.loss_scale:
            loss = loss * self.loss_scale
        loss.backward(retain_graph=True)
        # update the last parameter since the last parameter in the computaiton graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything.
        self.fuse_update()

        with torch.no_grad():
            # The norm is computed over all gradients together, as if they were
            # concatenated into a single vector. Gradients are modified in-place.
            self.grad_norms = torch.stack(self.grad_norms)

            total_norm = torch.norm(self.grad_norms, 2.0)
            self.clip_coef = float(self.clip_grad_norm) / (total_norm + 1e-6)
            self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
        self.gather_norm = False