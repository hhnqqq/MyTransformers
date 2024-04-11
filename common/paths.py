"""please insert all paths into the paths dict"""
from common.registry import registry

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

for key, value in paths.items():
    for sub_k, sub_v in value.items():
        name = '_'.join([key, sub_k])
        registry.register_path(name=name, path=sub_v)