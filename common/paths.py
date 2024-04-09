"""please insert all paths into the paths dict"""
from common.registry import registry

paths = {"model":{
    "llama":"path_of_llama",
    "gemma":"path_of_gemma"
}, 
"tokenizer":{
    "llama":"path_of_llama_tokenizer",
    "gemma":"path_of_gemma_tokenizer"
},
"dataset":{
    "dna_pretrain":"path_of_dna_pretrain_dataset"
}}

for key, value in paths.items():
    for sub_k, sub_v in value.items():
        name = '_'.join(key, sub_k)
        registry.register_path(name=name, path=value)