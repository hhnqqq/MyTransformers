#! /bin/bash

options="--batch_size 48 \
    --dataset_path your_dataset_path \
    --pretrained_ckpt your_pretrained_ckpt_path \
    --ckpt your_ckpt_path(for lora) \
    --use_lora \
    --output_len 1024 \
    --tokenizer your_tokenizer_path \
    --model_name llama2 \
    --variant 7b \
    --result_path your_output_path \
"
run_cmd="deepspeed --include localhost:6 --master_port 16666 run.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x