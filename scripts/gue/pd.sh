#! /bin/bash

module load compilers/cuda/12.1
module load nccl/2.18.3-1_cuda12.1
module load compilers/gcc/12.2.0
module load cudnn/8.9.5.29_cuda12.x 
module load tensorboard/2.11.2   
module load anaconda/2021.11
source activate hhn

base_options="--train-dataset-name pd \
--eval-dataset-name pd \
--model-name llama3 \
--tokenizer-name llama3 \
--output-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output \
--ckpt-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output/dnallama_bpe_1em4/step_15000.ckpt \
--tb-log-dir /home/bingxing2/ailab/scx6mh7/workspace/dnallama/tb_files/new_runs \
"

enable_list=("weight_a" "weight_b" "norm")

options="$base_options \
    --experiment-name pd_small_bsz \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode sft \
    --from-pretrained \
    --epochs 3 \
    --batch-size-per-gpu 64 \
    --eval-batch-size-per-gpu 96 \
    --eval-interval 10 \
    --save-interval 5000 \
    --bf16 \
    --warmup 0.10 \
    --variant 8b \
    --device cuda \
    --max-len 240 \
    --max-src-len 230 \
    --eval-max-len 240 \
    --eval-max-src-len 230 \
    --seed 42 \
    --ds-config-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/ds_config/pp_config.json \
    --lr 1e-5 \
    --lr-decay-ratio 0.5 \
    --auto-warmup-steps 10 \
    --auto-warmup-rate 0.01 \
    --use-lora-plus \
    --lora-rank 8 \
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