#! /bin/bash
base_options="--train-dataset-name ruozhi \
--model-name llama3 \
--tokenizer-name llama3 \
--output-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output \
"

options="$base_options \
    --experiment-name ruozhiba \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode sft \
    --epochs 2 \
    --batch-size-per-gpu 8 \
    --fp16 \
    --gradient-accumulation-steps 1 \
    --warmup 0.02 \
    --variant 8b \
    --device cuda \
    --num-pp-stages 1 \
    --max-len 1024 \
    --max-src-len 512 \
    --seed 42 \
    --ds-config-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/ds_config/pp_config.json \
    --lr 1e-5 \
    --warmup-min-lr 1e-6 \
    --warmup-max-lr 2e-5 \
    --use-lora-plus \
    --lora-rank 128 \
    --activation-checkpoint \
    --atten-type flash_atten \
    --tensorboard \
    --diy-optimizer \
    "
    
run_cmd="deepspeed --include localhost:0 --master_port 16666 /home/bingxing2/ailab/scx6mh7/workspace/dnallama/train/pp_train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x