base_options="--train-dataset-name multi_omics_pretrain \
--eval-dataset-name gue_human \ 
--model-name llama3 \
--tokenizer-name llama3 \
--output-path your_output_checkpoint_path \
--tokenizer-path your_tokenizer_path \
--ckpt-path your_pretrained_checkpoint_path \
--tb-log-dir your_tensorboard_log_path \
--dataset-class-name concat_iterable \
"
     
replace_modules="wq wk wv wo"
enable_list="weight_a weight_b norm"

lora_options="--use-lora \
    --use-lora-plus \
    --lora-plus-scaler 4 \
    --replace-modules $replace_modules \
    --lora-rank 128 \
    --lora-scaler 64 \
    --weight-a-init-method kaiming \
     "
     
options="$base_options \
    --experiment-name llama3.1_lora_pretrain_exp_1 \
    --dataset-weights 2_1_1 \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode pretrain \
    --prompt-path your_prompt_path \
    --from-pretrained \
    --epochs 2 \
    --batch-size-per-gpu 2 \
    --eval-batch-size-per-gpu 2 \
    --batching-stretegy packing \
    --eval-interval 200 \
    --save-interval 20000 \
    --bf16 \
    --eps 1e-9 \
    --variant 8b \
    --device cuda \
    --max-len 1200 \
    --max-src-len 1200 \
    --eval-max-len 1200 \
    --eval-max-src-len 1200 \
    --seed 42 \
    --zero-stage 2 \ 
    --lr 1e-4 \
    --lr-decay-ratio 0.1 \
    --warmup 0.1 \
    --auto-warmup-steps 100 \
    --auto-warmup-rate 0.05 \
    --atten-type flash_atten \
    --tensorboard \ 
    --diy-optimizer \
    --save-trainable \
    --enable-list $enable_list \ 
    "
    
sacc --num_nodes 8 --gpu_per_nodes 4 --group your_slurm_group ../../train/u_train.py ${options}