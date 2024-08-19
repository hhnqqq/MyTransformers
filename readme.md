## 使用方法：

- python setup.py install 配置环境
- scripts/有一些训练脚本的示例
- 调整脚本中的参数，具体参数的意思可以查看common/parser.py中的注释
- 需要注意的是参数中的地址配置。本项目配置的是既可以通过指明地址来配置，也可以通过指明名字来配置
    - 比如配置ckpt path可以在脚本中直接写明地址
    - 也可以事先在MyTransfomers/paths.json文件中写入地址, 如下代码中配置了llama的tokenizer地址，那么通过llama的名字就可以取出该地址

- 设置好参数之后运行该脚本即可启动训练


## 详细的文档：
[MyTransformer使用文档](https://github.com/hhnqqq/MyTransformers/blob/main/MyTransformers%E9%A1%B9%E7%9B%AE%E4%BB%A3%E7%A0%81%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3.pdf)
