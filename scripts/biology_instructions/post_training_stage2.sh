base_options="--train-dataset-name biology-instructions-stage3 \
--eval-dataset-name chat_multi_omics \
--model-name llama3 \
--tokenizer-name llama3 \
--output-path your_output_checkpoint_path \
--tokenizer-path your_tokenizer_path \
--ckpt-path your_pretrained_checkpoint_path \
--tb-log-dir your_tensorboard_log_path \
--dataset-class-name concat_iterable \
"

enable_list="wq wk wv wo w1 w2 w3"
options="$base_options \
    --experiment-name stage3_exp1 \
    --dataset-weights 1_1 \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode sft \
    --prompt-path your_prompt_path \
    --epochs 2 \
    --batch-size-per-gpu 1 \
    --eval-batch-size-per-gpu 1 \
    --gradient-accumulation-steps 4 \
    --eval-interval 10 \
    --save-interval 20000 \
    --bf16 \
    --variant test \
    --device cuda \
    --max-len 1024 \
    --max-src-len 1024 \
    --eval-max-len 1024 \
    --eval-max-src-len 1024 \
    --seed 42 \
    --zero-stage 2 \
    --lr 1e-5 \
    --lr-decay-ratio 0.5 \
    --warmup 0.03 \
    --auto-warmup-steps 1 \
    --auto-warmup-rate 0.05 \
    --atten-type flash_atten \
    --tensorboard \
    --diy-optimizer \
    --save-trainable \
    --enable-list $enable_list \
    "

sacc --num_nodes 3 --gpu_per_nodes 4 --group your_slurm_group ../../train/u_train.py ${options}