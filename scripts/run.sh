#! /bin/bash

#SBATCH --gpus=1

module load compilers/cuda/12.1
module load nccl/2.18.3-1_cuda12.1
module load compilers/gcc/12.2.0
module load cudnn/8.9.5.29_cuda12.x 
module load tensorboard/2.11.2   
module load anaconda/2021.11
source activate hhn
export LD_PRELOAD=/home/bingxing2/ailab/scx6mh7/peng_tmp_test/miniconda3/envs/hhn/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

options="--batch_size 48 \
    --dataset_path /home/bingxing2/ailab/scx6mh7/workspace/data/test_gue_except_covid.jsonl \
    --pretrained_ckpt /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output/dnallama_bpe_1em4/step_15000.ckpt \
    --ckpt /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output/gue_llama3_10epochs/step_10000.ckpt \
    --use_lora \
    --tokenizer /home/bingxing2/ailab/scx6mh7/workspace/llama/llama3_tokenizer.model \
    --model-name llama3 \
    --variant 8b \
    --result_path /home/bingxing2/ailab/scx6mh7/workspace/dnallama/output \
"
run_cmd="deepspeed --include localhost:0 --master_port 16666 /home/bingxing2/ailab/scx6mh7/workspace/dnallama/scripts/run.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x