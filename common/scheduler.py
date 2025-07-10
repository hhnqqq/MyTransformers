# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

from common.utils import print_rank_0
from copy import deepcopy


class AnnealingLR(_LRScheduler):
    """Anneals the learning rate from start to zero along a cosine curve."""

    DECAY_STYLES = ['linear', 'cosine', 'exponential', 'cosine_restarts', 'constant', 'None']

    def __init__(self, 
                 optimizer, 
                 start_lr, 
                 warmup_iter, 
                 num_iters, 
                 decay_style=None, 
                 last_iter=-1, 
                 decay_ratio=0.5, 
                 auto_warmup_steps=50, 
                 auto_warmup_rate=0.05,
                 restart_warmup_steps=None,
                 restart_every=None,
                 global_rank=0
                 ):
        """
        Initializes the AnnealingLR scheduler.

        Args:
            optimizer: The optimizer for which to schedule the learning rate.
            start_lr: Base learning rate for the scheduler.
            warmup_iter: Number of iterations for the warmup phase.
            num_iters: Total number of iterations.
            decay_style: Learning rate decay style after warmup. Defaults to None.
            last_iter: Number of iterations already trained. Defaults to -1.
            decay_ratio: The decay ratio for learning rate. Defaults to 0.5.
            auto_warmup_steps: The fixed number of warmup steps. Defaults to 50.
            auto_warmup_rate: The learning rate ratio for fixed warmup steps. Defaults to 0.05.

        Raises:
            AssertionError: If warmup_iter is greater than num_iters.
        """
        assert warmup_iter <= num_iters
        self.optimizer = optimizer
        self.lr_scale = deepcopy([x['lr'] if 'lr' in x else 1. for x in optimizer.param_groups])
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.init_step = last_iter
        self.num_iters = last_iter + 1
        self.end_iter = num_iters
        self.decay_style = decay_style.lower() if isinstance(decay_style, str) else None
        self.decay_ratio = 1 / decay_ratio
        self.auto_warmup_steps = auto_warmup_steps
        self.auto_warmup_rate = auto_warmup_rate
        self.global_rank = global_rank

        if warmup_iter < 0 or num_iters <= warmup_iter:
            raise ValueError(f"warmup_iter ({warmup_iter}) must be in range [0, num_iters ({num_iters})]")
        if self.decay_style == self.DECAY_STYLES[3]:
            if restart_every is None:
                raise ValueError("restart_every must be specified for cosine_restarts scheduler")
            if restart_warmup_steps is None:
                raise ValueError("restart_warmup_steps must be specified for cosine_restarts scheduler")
        self.restart_warmup_steps = restart_warmup_steps
        self.restart_every = restart_every

        self.step(self.num_iters)
        print_rank_0(f'--->learning rate decaying style {self.decay_style}, ratio {self.decay_ratio}%', global_rank)

    def get_lr(self):
        # auto_warmup_steps并不取决于warmup的设置，而是固定的进行warmup
        if self.num_iters <= self.init_step + self.auto_warmup_steps:
            auto_lr = float(self.start_lr) * self.auto_warmup_rate
            scheduled_lr = float(self.start_lr) * self.num_iters / self.warmup_iter
            return min(auto_lr, scheduled_lr)

        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            return float(self.start_lr) * self.num_iters / self.warmup_iter
        else:
            if self.decay_style == self.DECAY_STYLES[0]:
                return self.start_lr*((self.end_iter-(self.num_iters-self.warmup_iter))/self.end_iter)
            elif self.decay_style == self.DECAY_STYLES[1]:
                decay_step_ratio = min(1.0, (self.num_iters - self.warmup_iter) / self.end_iter)
                return self.start_lr / self.decay_ratio * (
                        (math.cos(math.pi * decay_step_ratio) + 1) * (self.decay_ratio - 1) / 2 + 1)
            elif self.decay_style == self.DECAY_STYLES[2]:
                return self.start_lr
            elif self.decay_style == self.DECAY_STYLES[3]:
                return self._get_cosine_schedule_with_multiple_warmups()
            else:
                return self.start_lr

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group, scale in zip(self.optimizer.param_groups, self.lr_scale):
            group['lr'] = new_lr * scale

    def state_dict(self):
        sd = {
                'start_lr': self.start_lr,
                'warmup_iter': self.warmup_iter,
                'num_iters': self.num_iters,
                'decay_style': self.decay_style,
                'end_iter': self.end_iter,
                'decay_ratio': self.decay_ratio
        }
        return sd

    def _get_cosine_schedule_with_multiple_warmups(self):

        restart_step = self.num_iters % self.restart_every
        restart_number = self.num_iters // self.restart_every

        if restart_step < self.restart_warmup_steps and self.num_iters >= self.restart_every:
            # get expected lr multipler at the end of the warmup
            warmup_lr_multiplier = self.start_lr * 0.5 * (1 + math.cos(math.pi * (
                    float(restart_number * self.restart_every + self.restart_warmup_steps - self.warmup_iter) /
                    float(max(1, self.end_iter - self.warmup_iter))
            )))

            return float(restart_step) / float(max(1, self.restart_warmup_steps)) * warmup_lr_multiplier

        progress = float(self.num_iters - self.warmup_iter) / float(max(1, self.end_iter - self.warmup_iter))

        return self.start_lr * 0.5 * (1 + math.cos(math.pi * progress))
    

if __name__ == '__main__':
    import torchvision
    import torch.optim as optim
    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define a simple model for demonstration
    model = torchvision.models.resnet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Create an instance of AnnealingLR scheduler
    # 实验验证
    decay_style = 'cosine_restarts'
    lr_scheduler = AnnealingLR(optimizer=optimizer, start_lr=0.1, warmup_iter=500, num_iters=5000,
                            decay_style=decay_style, restart_every=1000, restart_warmup_steps=100)

    # Train the model with lr_scheduler
    iters = []
    lrs = []
    for epoch in range(10):
        print(f"Epoch: {epoch}")
        for i in range(500):
            optimizer.zero_grad()
            output = model(torch.randn(4, 3, 224, 224).to(device))
            loss = output.sum()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            iters.append(epoch*500 + i)
            lrs.append(optimizer.param_groups[0]['lr'])
            if i % 100 == 0:
                print(f"Iteration: {i}, LR: {optimizer.param_groups[0]['lr']}")
                
    plt.plot(iters, lrs)
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Decay')
    plt.legend()
    plt.grid(True)
    plt.show()
