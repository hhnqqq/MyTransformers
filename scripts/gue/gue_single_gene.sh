#! /bin/bash

#SBATCH --gpus=4

source /home/bingxing2/ailab/scx6mh7/workspace/env.sh

base_options="--train-dataset-name dna_pretrain_2k \
--eval-dataset-name gue_human \
--model-name gemma \
--tokenizer-name base \
--output-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output \
--ckpt-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output/gemma_single_encode/step_180000_full.ckpt \
--tb-log-dir /home/bingxing2/ailab/scx6mh7/workspace/dnallama/tb_files/new_runs \
"

enable_list=("weight_a" "weight_b" "norm")

options="$base_options \
    --experiment-name gue_gemma_single_encode \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode pretrain \
    --from-pretrained \
    --epochs 3 \
    --batch-size-per-gpu 8 \
    --eval-batch-size-per-gpu 12 \
    --eval-interval 10 \
    --save-interval 10000 \
    --bf16 \
    --warmup 0.03 \
    --variant 2b \
    --device cuda \
    --max-len 1800  \
    --max-src-len 1600 \
    --eval-max-len 1800 \
    --eval-max-src-len 1600 \
    --seed 42 \
    --ds-config-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/ds_config/pp_config.json \
    --lr 1e-5 \
    --lr-decay-ratio 0.1 \
    --auto-warmup-steps 10 \
    --auto-warmup-rate 0.05 \
    --use-lora \
    --use-lora-plus \
    --lora-rank 16 \
    --activation-checkpoint \
    --atten-type flash_atten \
    --tensorboard \
    --diy-optimizer \
    --save-trainable \
    --encode-single-gene \
    --enable-list \
    "


for item in "${enable_list[@]}"; do
    options+=" \"$item\""
done
    
run_cmd="deepspeed --include localhost:0,1,2,3 --master_port 16666 /home/bingxing2/ailab/scx6mh7/workspace/dnallama/train/u_train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x