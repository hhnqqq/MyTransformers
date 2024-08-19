#! /bin/bash

base_options="--train-dataset-name dna_pretrain \
--model-name llama3 \
--tokenizer-name llama3 \
--output-path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output \
"

meta_prompt="here is a string of human dna"
meta_prompt_escaped="${meta_prompt// /\\ }"

options="$base_options \
    --experiment-name dnallama_bpe_test \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode pretrain \
    --from-pretrained \
    --epochs 2 \
    --batch-size-per-gpu 1 \
    --save-interval 5000 \
    --bf16 \
    --gradient-accumulation-steps 1 \
    --warmup 0.05 \
    --variant 8b \
    --device cuda \
    --num-pp-stages 2 \
    --max-len 4400 \
    --max-src-len 4400 \
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
    --meta-prompt $meta_prompt_escaped \
    --prefix %dna% \
    --postfix %dna% \
    "
    
run_cmd="deepspeed --include localhost:0,1 --master_port 16666 /home/bingxing2/ailab/scx6mh7/workspace/dnallama/train/pp_train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x