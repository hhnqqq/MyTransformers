#! /bin/bash
base_options="--dataset-name dna_pretrain \
--model-name llama \
--tokenizer-name base \
--output-path /workspace/dnallama \
"

options="$base_options \
    --experiment-name llama_test \
    --show-loss-step 1 \
    --epochs 3 \
    --batch-size-per-gpu 1 \
    --fp16 \
    --gradient-accumulation-steps 1 \
    --warmup 0.02 \
    --variant 7b \
    --device cuda \
    --num-pp-stages 1 \
    --max-len 1024 \
    --max-src-len 512 \
    --seed 42 \
    --read-nums 100 \
    --ds-config-path /workspace/dnallama/ds_config/pp_config.json \
    --lr 1e-5 \
    --warmup-min-lr 1e-6 \
    --warmup-max-lr 2e-5 \
    --use-lora-plus \
    --lora-rank 128 \
    --activation-checkpoint \
    --atten-type flash_atten \
    --tensorboard \
    "
    
run_cmd="deepspeed --include localhost:0 --master_port 16666 /workspace/dnallama/train/pp_train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x