#! /bin/bash

#SBATCH --gpus=4

module load compilers/cuda/12.1
module load nccl/2.18.3-1_cuda12.1
module load compilers/gcc/12.2.0
module load cudnn/8.9.5.29_cuda12.x 
module load tensorboard/2.11.2   
module load anaconda/2021.11
source activate hhn

base_options="--train-dataset-name dna_pretrain \
--eval-dataset-name gue_human \
--model-name llama3 \
--tokenizer-name llama3 \
--output-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output \
"

meta_prompt="Below is a nucleotide sequence of human dna, surrounded by special word:"
meta_prompt_escaped="${meta_prompt// /\\ }"
enable_list=("weight_a" "weight_b" "embedder" "norm" "bias")

options="$base_options \
    --experiment-name dnallama_bpe_1em4 \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode pretrain \
    --from-pretrained \
    --epochs 2 \
    --batch-size-per-gpu 6 \
    --eval-batch-size-per-gpu 64 \
    --eval-interval 10 \
    --save-interval 1000 \
    --bf16 \
    --gradient-accumulation-steps 8 \
    --warmup 0.05 \
    --variant 8b \
    --device cuda \
    --num-pp-stages 4 \
    --max-len 4400 \
    --max-src-len 4400 \
    --eval-max-len 250 \
    --eval-max-src-len 250 \
    --seed 42 \
    --ds-config-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/ds_config/pp_config.json \
    --lr 1e-4 \
    --warmup-min-lr 1e-5 \
    --warmup-max-lr 1e-4 \
    --auto-warmup-steps 100 \
    --auto-warmup-rate 0.05 \
    --use-lora-plus \
    --lora-rank 128 \
    --activation-checkpoint \
    --atten-type flash_atten \
    --tensorboard \
    --diy-optimizer \
    --meta-prompt $meta_prompt_escaped \
    --prefix %dna% \
    --postfix %dna% \
    --enable-list \
    "


for item in "${enable_list[@]}"; do
    options+=" \"$item\""
done
    
run_cmd="deepspeed --include localhost:0,1,2,3 --master_port 16666 /home/bingxing2/ailab/scx6mh7/workspace/dnallama/train/pp_train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x