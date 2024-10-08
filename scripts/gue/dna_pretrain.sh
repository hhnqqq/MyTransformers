#! /bin/bash

#SBATCH --gpus=4

module load compilers/cuda/12.1
module load nccl/2.18.3-1_cuda12.1
module load compilers/gcc/12.2.0
module load cudnn/8.9.5.29_cuda12.x 
module load tensorboard/2.11.2   
module load anaconda/2021.11
source activate hhn
export LD_PRELOAD=/home/bingxing2/ailab/scx6mh7/peng_tmp_test/miniconda3/envs/hhn/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0


base_options="--train-dataset-name dna_pretrain_2k \
--eval-dataset-name gue_human \
--model-name llama2 \
--tokenizer-name base \
--output-path /home/bingxing2/ailab/scx6mh7/workspace/MyTransformers/output \
--tokenizer-path /home/bingxing2/ailab/scx6mh7/workspace/dnabert2/merged_tokenizer.model \
--ckpt-path /home/bingxing2/ailab/scx6mh7/workspace/llama/llama2.pth \
--tb-log-dir /home/bingxing2/ailab/scx6mh7/workspace/MyTransformers/tb_files/new_runs_1 \
--partial-ckpt-path /home/bingxing2/ailab/scx6mh7/workspace/dnabert2/merged_embedding.ckpt \
"

enable_list=("weight_a" "weight_b" "norm" "embedding" "output")

options="$base_options \
    --experiment-name llama2_dna_pretrain_merged_tokenizer \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode pretrain \
    --from-pretrained \
    --epochs 5 \
    --batch-size-per-gpu 48 \
    --eval-batch-size-per-gpu 48 \
    --eval-interval 10 \
    --save-interval 10000 \
    --bf16 \
    --warmup 0.03 \
    --variant 7b \
    --device cuda \
    --max-len 700 \
    --max-src-len 700 \
    --eval-max-len 700 \
    --eval-max-src-len 700 \
    --seed 42 \
    --ds-config-path /home/bingxing2/ailab/scx6mh7/workspace/MyTransformers/ds_config/pp_config.json \
    --lr 1e-4 \
    --lr-decay-ratio 0.1 \
    --auto-warmup-steps 100 \
    --auto-warmup-rate 0.05 \
    --use-lora \
    --use-lora-plus \
    --lora-rank 128 \
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
    
run_cmd="deepspeed --include localhost:0,1,2,3 --master_port 16666 /home/bingxing2/ailab/scx6mh7/workspace/MyTransformers/train/u_train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x