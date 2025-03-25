## 使用方法：
- 训练的入口统一是u_train，在使用时只需要指定训练相关的参数，用deepspeed或者torchrun启动程序即可

## 代码可读性
- 加载模型相关的代码在load_model.py中，支持transformers模型和torch实现的模型，但是transformers库模型未经过完整测试
- 加载数据集相关的代码在lora_dataset.py中。
- forward_step和backward_step则在dp_train.py和pp_train.py中
- 本仓库基于注册器机制构建，具体请参考common.registry
