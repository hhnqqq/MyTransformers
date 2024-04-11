使用方法：

- 打开scripts/pp_scripts文件夹下的流水线训练运行脚本
- 调整脚本中的参数，具体参数的意思可以查看common/parser.py中的注释
- 需要注意的是参数中的地址配置。本项目配置的是既可以通过指明地址来配置，也可以通过指明名字来配置
    - 比如配置ckpt path可以在脚本中直接写明地址
    - 也可以事先在common/path.py文件中写入地址, 如下代码中配置了llama的tokenizer地址，那么通过llama的名字就可以取出该地址
```python
paths = {"model":{
    "llama":"path_of_model",
    "gemma":"path_of_model"
}, 
"tokenizer":{
    "llama":"/workspace/tokenizer.model",
    "gemma":"/workspace/tokenizer.model"
},
"dataset":{
    "dna_pretrain":"/workspace/longtext-2k-clean.jsonl"
}}
```
- 设置好参数之后运行该脚本即可启动训练