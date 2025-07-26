#! /bin/bash
base_options="--train-dataset-name wizardlm_52k \
--eval-dataset-name wizardlm \
--model-name llama3 \
--tokenizer-name llama3 \
--output-path your_output_ckpt_path \
--tokenizer-path your_tokenizer_path \
--ckpt-path your_pretrained_ckpt_path \
--prompt-path your_prompt_path \
--wandb-dir your_wandb_dir \
--wandb-cache-dir your_cache_dir \
--dataset-class-name iterable \
"

replace_modules="wq wk wv wo"
lora_options="--use-lora \
    --use-lora-plus \
    --lora-plus-scaler 16 \
    --use-gora \
    --gora-n-steps 32 \
    --gora-init-method compress \
    --gora-max-rank 32 \
    --gora-min-rank 4 \
    --gora-scale-by-lr \
    --gora-lr 5e-2 \
    --gora-rank-stablize \
    --gora-dynamic-scaling \
    --gora-importance-type union_mean \
    --lora-rank 8 \
    --lora-scaler 16 \
    --replace-modules $replace_modules \
    --run-lora-in-fp32 \
    --weight-a-init-method kaiming "


train_options="--experiment-name chat-llama3.1-gora-seed${SEED:-42}
    --wandb \
    --all-reduce-loss \
    --fuse-linear-loss \
    --show-loss-step 1 \
    --epochs 1 \
    --mode sft \
    --batch-size-per-gpu 4 \
    --eval-batch-size-per-gpu 4 \
    --eval-interval 50 \
    --bf16 \
    --from-pretrained \
    --show-avg-loss-step 1 \
    --variant 8b \
    --save-interval 10000 \
    --gradient-accumulation-steps 2 \
    --device cuda \
    --max-len 1024 \
    --max-src-len 1024 \
    --eval-max-len 1024 \
    --eval-max-src-len 1024 \
    --seed ${SEED:-42} \
    --zero-stage 2 \
    --lr 5e-5 \
    --warmup 0.03 \
    --auto-warmup-steps 10 \
    --auto-warmup-rate 0.05 \
    --weight-decay 5e-4 \
    --lr-decay-style cosine \
    --lr-decay-ratio 0.1 \
    --atten-type flash \
    --save-trainable \
    --diy-optimizer \
    "

options="$base_options \
    $lora_options \
    $train_options \
    "
run_cmd="deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 16667 ../../train/u_train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}