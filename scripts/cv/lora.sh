#! /bin/bash
export TRANSFORMERS_CACHE=/fs-computility/mabasic/yepeng/liminglei/MyTransformers/.hf_cache
export HF_HOME=/fs-computility/mabasic/yepeng/liminglei/MyTransformers/.hf_home
export WANDB_DIR=/fs-computility/mabasic/yepeng/liminglei/MyTransformers/.wandb_logs
export TMPDIR=/fs-computility/mabasic/yepeng/liminglei/MyTransformers/.tmp
export LOG_FLODER=/fs-computility/mabasic/yepeng/liminglei/MyTransformers/cv_logs

# name=train_clip_lora_1gpu
name=tmp_lora
# seeds=("42" "0" "18")
seeds=("42")
gpu_id=0
# datasets=("stanford_cars" "RESISC45" "dtd" "svhn" "EuroSAT" "GTSRB" "sun397")
datasets=("EuroSAT")
# datasets=("svhn")
for dataset in "${datasets[@]}"
do
    for seed in "${seeds[@]}"
    do
        lora_options="--use-lora \
            --run-lora-in-fp32 \
            --lora-scaler 16 \
            --lora-rank 8 \
            --replace-modules vision_model \
            --weight-a-init-method kaiming \
            --use-lora-plus \
            --lora-plus-scaler 1 \
            "

        options="--experiment-name ${name}_${dataset}_seed${seed} \
            --lr 1e-4 \
            --batch-size-per-gpu 64 \
            --eval-batch-size-per-gpu 128 \
            --output-path /fs-computility/mabasic/yepeng/liminglei/MyTransformers/checkpoint/clip_lora_finetuned \
            --tensorboard \
            --tb-log-dir /fs-computility/mabasic/yepeng/liminglei/MyTransformers/tensorboard_logs/train_clip/${name}_${dataset}_seed${seed} \
            --model-name clip \
            --device cuda \
            --seed $seed \
            --cv-dataset-name $dataset \
            $lora_options \
            "

        CUDA_VISIBLE_DEVICES=$gpu_id python /fs-computility/mabasic/yepeng/liminglei/MyTransformers/train/train_clip.py $options
    done
done