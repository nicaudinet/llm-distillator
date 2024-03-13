#!/usr/bin/env bash
#SBATCH -A NAISS2024-22-58
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 0-24:00:00

time apptainer exec container.sif python main.py alvis \
    --prompt_mode "cot" \
	--data_path "/cephyr/users/audinet/Alvis/llm-distillator/data/amazon_reviews/original.txt" \
	--out_dir "/cephyr/users/audinet/Alvis/llm-distillator/data/amazon_reviews/cot" \
	--num_samples 10 \
    --start_sample 0
	--batch_size 4
