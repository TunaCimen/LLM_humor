#!/bin/bash -l
#SBATCH --nodes=1                    # account name
#SBATCH --partition=ai                      # partition / queue
#SBATCH --gres=gpu:ampere_a40:2          # request 1 A100 GPU
#SBATCH -c 4                        # request 4 CPU cores
#SBATCH --mem=400G                  # request 128 GB RAM
#SBATCH --time=06:00:00           # max runtime (1 day)
#SBATCH -o results/results_%j.txt   # STDOUT log
#SBATCH -e errors/errors_%j.txt     # STDERR log


export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_7b_humor_tmp_more_epochs1.txt"

conda activate r1_env_c
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=2

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    grpo.py \
    --output_dir ./outputs_hum_tmp_more_epochs1 \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_name mervehmv/my-subset-grpo1 \
    --dataset_config ranking \
    --deepspeed ./src/r1-v/local_scripts/zero3_offload.json \
    --learning_rate 5e-6 \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation sdpa \
    --max_pixels 401408 \
    --num_train_epochs 4 \
    --run_name Qwen2-VL-7B-GRPO-NYCC-200 \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 4   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
