#! /bin/bash

#SBATCH --gpus=4

module load compilers/cuda/12.1
module load nccl/2.18.3-1_cuda12.1
module load compilers/gcc/12.2.0
module load cudnn/8.9.5.29_cuda12.x 
module load tensorboard/2.11.2   
module load anaconda/2021.11
source activate hhn

base_options="--train-dataset-name dna_pretrain_2k \
--eval-dataset-name gue_human \
--model-name gemma \
--tokenizer-name base \
--output-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output \
--ckpt-path /home/bingxing2/ailab/scx6mh7/workspace/gemma/gemma-2b.ckpt \
--tb-log-dir /home/bingxing2/ailab/scx6mh7/workspace/dnallama/tb_files/new_runs \
"

enable_list=("weight_a" "weight_b" "norm")

options="$base_options \
    --experiment-name gemma_single_encode \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode pretrain \
    --from-pretrained \
    --epochs 3 \
    --batch-size-per-gpu 6 \
    --eval-batch-size-per-gpu 6 \
    --eval-interval 10 \
    --save-interval 10000 \
    --bf16 \
    --warmup 0.03 \
    --variant 2b \
    --device cuda \
    --max-len 2400 \
    --max-src-len 2400 \
    --eval-max-len 2400 \
    --eval-max-src-len 2400 \
    --seed 42 \
    --ds-config-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/ds_config/pp_config.json \
    --lr 1e-4 \
    --lr-decay-ratio 0.1 \
    --auto-warmup-steps 100 \
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
    --test-code \
    --enable-list \
    "


for item in "${enable_list[@]}"; do
    options+=" \"$item\""
done
    
run_cmd="deepspeed --include localhost:0,1,2,3 --master_port 16666 /home/bingxing2/ailab/scx6mh7/workspace/dnallama/train/u_train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x