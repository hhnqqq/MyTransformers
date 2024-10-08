#! /bin/bash

base_options="--train-dataset-name dna_pretrain \
--eval-dataset-name gue_human \
--model-name gemma \
--tokenizer-name base \
--output-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output \
"

meta_prompt="here is a string of human dna"
meta_prompt_escaped="${meta_prompt// /\\ }"

options="$base_options \
    --experiment-name dnallama_gemma_bpe_test \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode pretrain \
    --from-pretrained \
    --epochs 2 \
    --batch-size-per-gpu 4 \
    --eval-batch-size-per-gpu 64 \
    --eval-interval 10 \
    --save-interval 5000 \
    --fp16 \
    --gradient-accumulation-steps 1 \
    --warmup 0.05 \
    --variant 2b \
    --device cuda \
    --num-pp-stages 4 \
    --max-len 3500 \
    --max-src-len 3500 \
    --eval-max-len 250 \
    --eval-max-src-len 250 \
    --seed 42 \
    --ds-config-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/ds_config/pp_config.json \
    --lr 5e-4 \
    --warmup-min-lr 5e-5 \
    --warmup-max-lr 5e-3 \
    --use-lora-plus \
    --lora-rank 128 \
    --activation-checkpoint \
    --atten-type flash_atten \
    --tensorboard \
    --diy-optimizer \
    --meta-prompt $meta_prompt_escaped \
    --prefix %dna% \
    --postfix %dna% \
    "
    
run_cmd="deepspeed --include localhost:0,1,2,3 --master_port 16666 /home/bingxing2/ailab/scx6mh7/workspace/dnallama/train/pp_train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x