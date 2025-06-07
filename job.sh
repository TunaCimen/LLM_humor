#!/bin/bash
#!/bin/bash
#SBATCH --nodes=1                    # account name
#SBATCH --partition=ai                      # partition / queue
#SBATCH --gres=gpu:ampere_a40          # request 1 A100 GPU
#SBATCH -c 4                        # request 4 CPU cores
#SBATCH --mem=128G                  # request 128 GB RAM
#SBATCH --time=1-00:00:00           # max runtime (1 day)
#SBATCH -o results/results_%j.txt   # STDOUT log
#SBATCH -e errors/errors_%j.txt     # STDERR log

module load git
srun python SFT.py \
    --model_name "unsloth/Qwen2.5-VL-7B-Instruct" \
    --dataset_path sft_data4.json \
    --max_steps 250
