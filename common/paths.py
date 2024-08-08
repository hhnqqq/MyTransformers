"""please insert all paths into the paths dict"""
from common.registry import registry

paths = {"model":{
    "llama3_8b":"/home/bingxing2/ailab/scx6mh7/workspace/llama/llama3.pth",
    "llama2_7b":"/home/bingxing2/ailab/scx6mh7/workspace/llama/llama2.pth",
    "llama1_7b":"/home/bingxing2/ailab/scx6mh7/workspace/llama/llama.pth",
    "gemma_2b":"/home/bingxing2/ailab/scx6mh7/workspace/gemma/gemma-2b.ckpt",
    "gemma_7b":"path_of_model"
}, 
"tokenizer":{
    "llama3":"/home/bingxing2/ailab/scx6mh7/workspace/llama/llama3_tokenizer.model",
    "llama2":"/home/bingxing2/ailab/scx6mh7/workspace/llama/llama2_tokenizer.model",
    "llama1":"/home/bingxing2/ailab/scx6mh7/workspace/llama/llama1_tokenizer.model",
    "gemma":"/home/bingxing2/ailab/scx6mh7/workspace/gemma/tokenizer.model",
    "dnabert2":"/home/bingxing2/ailab/scx6mh7/workspace/dnabert2/tokenizer.json"
},
"train_dataset":{
    "dna_pretrain_tokenized":"/home/bingxing2/ailab/scx6mh7/workspace/data/dna_pretrain.jsonl",
    "dna_pretrain":"/home/bingxing2/ailab/group/ai4bio/renyuchen/pretraining_data/human8k/GRCh38.p13.genome.txt",
    "dna_pretrain_2k":"/home/bingxing2/ailab/group/ai4bio/renyuchen/pretraining_data/human8k/GRCh38_2k.txt",
    "dna_pretrain_1k":"/home/bingxing2/ailab/group/ai4bio/renyuchen/pretraining_data/human8k/GRCh38_1k.txt",
    "gue":"/home/bingxing2/ailab/group/ai4bio/public/multi-omics/DNA/downstream/GUE/human.csv",
    "ruozhi":"/home/bingxing2/ailab/scx6mh7/workspace/data/ruozhiba_qa.jsonl",
    "splice":"/home/bingxing2/ailab/scx6mh7/workspace/data/train_splice.jsonl",
    "gue_except_covid":"/home/bingxing2/ailab/scx6mh7/workspace/data/train_gue_except_covid.jsonl",
    "gue_except_covid_resample":"/home/bingxing2/ailab/scx6mh7/workspace/data/train_gue_except_covid_resample.jsonl",
    "emp_H3K36me3": "/home/bingxing2/ailab/scx6mh7/workspace/data/train_emp_H3K36me3.jsonl",
    "emp_H3": "/home/bingxing2/ailab/scx6mh7/workspace/data/train_emp_H3.jsonl",
    "emp":"/home/bingxing2/ailab/scx6mh7/workspace/data/train_emp.jsonl",
    "pd":"/home/bingxing2/ailab/scx6mh7/workspace/data/train_pd.jsonl",
    "cpd":"/home/bingxing2/ailab/scx6mh7/workspace/data/train_cpd.jsonl"
},
"eval_dataset":{"gue_human":"/home/bingxing2/ailab/scx6mh7/workspace/data/gue_human.txt",
                "splice":"/home/bingxing2/ailab/scx6mh7/workspace/data/dev_splice.jsonl",
                "gue_except_covid":"/home/bingxing2/ailab/scx6mh7/workspace/data/dev_gue_except_covid.jsonl",
                "emp_H3K36me3": "/home/bingxing2/ailab/scx6mh7/workspace/data/dev_emp_H3K36me3.jsonl",
                "emp_H3": "/home/bingxing2/ailab/scx6mh7/workspace/data/dev_emp_H3.jsonl",
                "emp":"/home/bingxing2/ailab/scx6mh7/workspace/data/dev_emp.jsonl",
                "pd":"/home/bingxing2/ailab/scx6mh7/workspace/data/dev_pd.jsonl",
                "cpd":"/home/bingxing2/ailab/scx6mh7/workspace/data/dev_cpd.jsonl"}}

for key, value in paths.items():
    for sub_k, sub_v in value.items():
        name = '_'.join([key, sub_k])
        registry.register_path(name=name, path=sub_v)