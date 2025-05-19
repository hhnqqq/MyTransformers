base_options="--train-dataset-name chat_multi_omics \
--eval-dataset-name biology_instrucitons \
--model-name llama3 \
--tokenizer-name llama3 \
--output-path your_output_checkpoint_path \
--tokenizer-path your_tokenizer_path \
--ckpt-path your_pretrained_checkpoint_path \
--tb-log-dir your_tensorboard_log_path \
--dataset-class-name iterable \
"

enable_list="wq wv weight_a weight_b norm"
replace_modules="w1 w2 w3 wk wo"

lora_options="--use-lora \
    --use-lora-plus \
    --lora-plus-scaler 16 \
    --replace-modules $replace_modules \
    --lora-rank 64 \
    --lora-scaler 32 \
    --weight-a-init-method kaiming \
     "

options="$base_options \
    $lora_options \
    --experiment-name instruct_tuning_exp_1 \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode sft \
    --prompt-path your_prompt_path \
    --from-pretrained \
    --epochs 1 \
    --batch-size-per-gpu 2 \
    --eval-batch-size-per-gpu 2 \
    --eval-interval 10 \
    --save-interval 1000000 \
    --fp16 \
    --variant 8b \
    --device cuda \
    --max-len 1024 \
    --max-src-len 1024 \
    --eval-max-len 1024 \
    --eval-max-src-len 1024 \
    --seed 42 \
    --zero-stage 2 \
    --lr 1e-4 \
    --lr-decay-ratio 0.1 \
    --warmup 0.03 \
    --auto-warmup-steps 50 \
    --auto-warmup-rate 0.05 \
    --atten-type flash_atten \
    --tensorboard \
    --diy-optimizer \
    --save-trainable \
    --enable-list $enable_list \
    "
    
sacc --num_nodes 5 --gpu_per_nodes 4 --group your_slurm_group ../../train/u_train.py ${options}
