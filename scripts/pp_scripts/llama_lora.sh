#! /bin/bash

base_options="--train-dataset-name ruozhi \
--model-name gemma \
--tokenizer-name base \
--output-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output \
--ckpt-path /home/bingxing2/ailab/scx6mh7/workspace/gemma/gemma-2b.ckpt \
--tb-log-dir /home/bingxing2/ailab/scx6mh7/workspace/dnallama/tb_files/new_runs \
"

enable_list=("weight_a" "weight_b" "norm")

options="$base_options \
    --experiment-name lora_test \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode sft \
    --from-pretrained \
    --epochs 3 \
    --batch-size-per-gpu 16 \
    --eval-batch-size-per-gpu 96 \
    --eval-interval 100000 \
    --save-interval 500000 \
    --bf16 \
    --warmup 0.03 \
    --variant 2b \
    --device cuda \
    --max-len 600 \
    --max-src-len 650 \
    --eval-max-len 410 \
    --eval-max-src-len 400 \
    --seed 42 \
    --ds-config-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/ds_config/pp_config.json \
    --lr 1e-5 \
    --lr-decay-ratio 0.1 \
    --auto-warmup-steps 50 \
    --auto-warmup-rate 0.05 \
    --use-lora \
    --lora-rank 16 \
    --activation-checkpoint \
    --atten-type flash_atten \
    --tensorboard \
    --diy-optimizer \
    --save-trainable \
    --test-code \
    --enable-list \
    "


for item in "${enable_list[@]}"; do
    options+=" \"$item\""
done
    
run_cmd="deepspeed --include localhost:0 --master_port 16666 /home/bingxing2/ailab/scx6mh7/workspace/dnallama/train/u_train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x