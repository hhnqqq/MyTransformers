#! /bin/bash

#SBATCH --gpus=4

source /home/bingxing2/ailab/scx6mh7/workspace/env.sh

base_options="--train-dataset-name gue_except_covid_resample \
--eval-dataset-name gue_except_covid \
--model-name llama3 \
--tokenizer-name llama3 \
--output-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output \
--ckpt-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output/dnallama_bpe_1em4/step_15000.ckpt \
--tb-log-dir /home/bingxing2/ailab/scx6mh7/workspace/dnallama/tb_files/new_runs \
"

enable_list=("weight_a" "weight_b" "norm")

options="$base_options \
    --experiment-name gue_llama3_10epochs \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode sft \
    --from-pretrained \
    --epochs 10 \
    --batch-size-per-gpu 48 \
    --eval-batch-size-per-gpu 48 \
    --eval-interval 10 \
    --save-interval 10000 \
    --bf16 \
    --warmup 0.03 \
    --variant 8b \
    --device cuda \
    --max-len 410 \
    --max-src-len 400 \
    --eval-max-len 410 \
    --eval-max-src-len 400 \
    --seed 42 \
    --ds-config-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/ds_config/pp_config.json \
    --lr 1e-5 \
    --lr-decay-ratio 0.1 \
    --auto-warmup-steps 50 \
    --auto-warmup-rate 0.05 \
    --use-lora \
    --use-lora-plus \
    --lora-rank 16 \
    --activation-checkpoint \
    --atten-type flash_atten \
    --tensorboard \
    --diy-optimizer \
    --save-trainable \
    --enable-list \
    "


for item in "${enable_list[@]}"; do
    options+=" \"$item\""
done
    
run_cmd="deepspeed --include localhost:0,1,2,3 --master_port 16666 /home/bingxing2/ailab/scx6mh7/workspace/dnallama/train/u_train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x