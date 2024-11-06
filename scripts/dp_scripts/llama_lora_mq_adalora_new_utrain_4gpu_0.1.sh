#! /bin/bash
base_options="--train-dataset-name gsm8k \
--eval-dataset-name gsm8k \
--model-name llama3 \
--tokenizer-name llama3 \
--output-path /dssg/home/acct-aemzl/aemzl-user1/qbadam/loraeden/checkpoint/adalora_gsm8k_4gpu_orth01 \
--tokenizer-path /dssg/home/acct-aemzl/aemzl-user1/modelscope_models/llama/llama3.1/original/tokenizer.model \
--ckpt-path /dssg/home/acct-aemzl/aemzl-user1/modelscope_models/llama/llama3.1/original/consolidated.00.pth \
--tb-log-dir /dssg/home/acct-aemzl/aemzl-user1/qbadam/loraeden/tensorboard_logs/adalora_gsm8k_4gpu_orth01 \
"

common_lora_options="--use-lora \
--run-lora-in-fp32 \
--replace-modules wq_wv \
--lora-rank 8 \
--lora-scaler 16 \
--weight-a-init-method kaiming \
"

common_train_options="--from-pretrained \
--show-loss-step 1 \
--show-avg-loss-step 1 \
--gradient-accumulation-steps 8 \
--mode sft \
--prompt-path /dssg/home/acct-aemzl/aemzl-user1/qbadam/loraeden/MyTransformers/prompt_configs/llama3_general.json \
--epochs 1 \
--batch-size-per-gpu 8 \
--eval-batch-size-per-gpu 8 \
--eval-interval 10 \
--save-interval 100000 \
--bf16 \
--variant 8b \
--device cuda \
--max-len 1024 \
--max-src-len 1024 \
--eval-max-len 1024 \
--eval-max-src-len 1024 \
--seed 42 \
--zero-stage 2 \
--lr 5e-5 \
--lr-decay-ratio 0.1 \
--warmup 0.03 \
--auto-warmup-steps 10 \
--auto-warmup-rate 0.05 \
--tensorboard \
--diy-optimizer \
--save-trainable \
--all-reduce-loss \
--activation-checkpoint \
--enable-list \
"

enable_list=("weight_a" "weight_b", "weight_e")

lora_options="$common_lora_options \
    --use-adalora \
    "

options="--experiment-name gsm8k_adalora_fp32_4gpu_orth01 \
    $base_options \
    $lora_options \
    $common_train_options \
    "

for item in "${enable_list[@]}"; do
    options+=" \"$item\""
done

run_cmd="deepspeed --include localhost:0,1,2,3 --master_port 16666 /dssg/home/acct-aemzl/aemzl-user1/qbadam/loraeden/MyTransformers/train/u_train_adalora.py ${options}"
echo ${run_cmd}
eval ${run_cmd}
