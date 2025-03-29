## 使用方法：

- python setup.py install 配置环境
- scripts/有一些训练脚本的示例
- 调整脚本中的参数，具体参数的意思可以查看common/parser.py中的注释
- 需要注意的是参数中的地址配置。本项目配置的是既可以通过指明地址来配置，也可以通过指明名字来配置
    - 比如配置ckpt path可以在脚本中直接写明地址
    - 也可以事先在MyTransfomers/paths.json文件中写入地址, 如下代码中配置了llama的tokenizer地址，那么通过llama的名字就可以取出该地址

- 设置好参数之后运行该脚本即可启动训练

## 支持的功能：
1. 使用liger_kernel减小训练显存使用，加速训练
2. 使用deepspeed分布式训练，支持流水线并行，序列并行
3. 支持使用transformers模型和tokenizer
4. 支持20+ LoRA算法，架构清晰，易学习
5. 支持灵活的optimizer设置
6. 支持使用多种不同的attention implementation，支持对sdpa后端进行设置
7. 支持多种不同的数据集
8. 支持多节点训练
9. 灵活的注册器机制

   
## 详细的文档：
[MyTransformer使用文档](https://github.com/hhnqqq/MyTransformers/blob/main/MyTransformers_document.pdf)
## 多机多卡训练
[sacc](https://github.com/hhnqqq/sacc_beijingcloud)
