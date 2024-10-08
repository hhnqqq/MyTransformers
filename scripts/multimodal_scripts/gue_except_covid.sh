#! /bin/bash

#SBATCH --gpus=4

source /home/bingxing2/ailab/scx6mh7/workspace/env.sh

base_options="--train-dataset-name gue_except_covid_resample \
--eval-dataset-name gue_except_covid \
--model-name llama2_with_bert \
--tokenizer-name base \
--tokenizer-path /home/bingxing2/ailab/scx6mh7/workspace/llama/llama2_tokenizer.model \
--output-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output \
--ckpt-path /home/bingxing2/ailab/scx6mh7/workspace/llama/llama2.pth \
--tb-log-dir /home/bingxing2/ailab/scx6mh7/workspace/dnallama/tb_files/new_runs_1 \
--multimodal-tokenizer-name dnabert2 \
--multimodal \
--multimodal-model-ckpt-path /home/bingxing2/ailab/scx6mh7/workspace/dnabert2/pytorch_model.bin \
--multimodal-projector-type mlp \
--multimodal-sample-mode pool \
--multimodal-projector-layers 2 \
--multimodal-k-tokens 128 \
--dataset-class-name iterable_multimodal_dna_dataset \
"

enable_list=("norm" "multimodal")

options="$base_options \
    --experiment-name gue_multimodal_mlp_128_tokens \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode sft \
    --from-pretrained \
    --epochs 3 \
    --batch-size-per-gpu 96 \
    --eval-batch-size-per-gpu 48 \
    --eval-interval 10 \
    --save-interval 10000 \
    --bf16 \
    --warmup 0.05 \
    --variant large \
    --device cuda \
    --max-len 220 \
    --max-src-len 210 \
    --eval-max-len 220 \
    --eval-max-src-len 210 \
    --seed 42 \
    --ds-config-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/ds_config/pp_config.json \
    --lr 1e-5 \
    --lr-decay-ratio 0.1 \
    --auto-warmup-steps 50 \
    --auto-warmup-rate 0.05 \
    --activation-checkpoint \
    --atten-type flash_atten \
    --tensorboard \
    --use-lora \
    --use-lora-plus \
    --lora-rank 128 \
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