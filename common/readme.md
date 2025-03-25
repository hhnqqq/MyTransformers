## 该文件夹中包含一些训练常用的内容，包括：
- 超参数管理
- 并行状态管理（目前仅在序列并行时有用）
- 常用函数如print_rank_0
- 数据集
- 优化器
- parser
- 注册器等
- 各种LoRA算法，本仓库实现了大部分的lora算法，比peft多，比peft清晰！！

## 常见用法如下：
### 常用函数的导入：
```python
from common.utils import print_rank_0, read_config, set_random_seed, init_dist
```
其中，print_rank_0负责打印和日志记录，read_config负责读取ds config文件，init_dist负责初始化并行环境
### optimizer的使用：
```python
from common.utils.optimizer import get_optimizer
optimizer, lr_scheduler = get_optimizer(ds_config, args, model=model)
# 将优化器和学习率规划器传入deepspeed, 否则无效
model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, 
                                                         optimizer=optimizer,
                                                         lr_scheduler=lr_scheduler,
                                                         config=ds_config,
                                                         model_parameters=[p for p in model.parameters() if p.requires_grad],
                                                         mpu=parallel_states)
```
### 超参数管理的使用：
```python
from common.utils.params_manager import (
    refresh_config, 
    print_trainable_module_names, 
    enable_trainable_params, 
    disable_untrainable_params
)
# refresh_config会根据argument parser中的参数来更新ds配置文件（默认argument parser中参数优先级最高）
# print_trainable_module_names简单的打印可训练参数
# enable_trainable_params传入一个列表，参数名字包含列表中的字符串时可训练，否则不可
# disable_untrainable_params与上面的函数相反
```
### parser的使用
```python
# 必须这样调用，否则参数空间会不全，导致报错
from common.parser import get_args
args = get_args()
```

### LoRA的使用
```python
from train.load_model import load_model
from common.lora_modules import * # 或者导入setup函数也可以

model, tokenizer, model_config, return_dataset_kwargs = load_model(args)

setup_lora(model, args, model_config)

# 注意，如果需要使用gora或者lora-ga或者adalora等，那么以下代码是必须的
if args.use_lora_ga:
    lora_ga_reinit(model=model,
                   dataloader=train_dataloader,
                   args=args,
                   iters=args.lora_ga_n_steps)
if args.use_gora:
    gora_reinit(model=model,
                  dataloader=train_dataloader,
                  args=args,
                  iters=args.gora_n_steps)
if args.use_adalora:
    rank_allocator = RankAllocator(model, args)
    model.rankallocator = rank_allocator
if args.use_increlora:
    rank_allocator = IncreRankAllocator(model, args)
    model.rankallocator = rank_allocator
```
