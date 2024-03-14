#!/usr/bin/env bash
#SBATCH -A NAISS2024-22-58
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1  # We're launching 1 node with 1 Nvidia T4 GPUs each
#SBATCH -t 0-24:00:00

time apptainer exec container.sif python classify.py \
    --original_reviews="/cephyr/users/audinet/Alvis/llm-distillator/data/amazon_reviews/original_reviews.txt" \
    --distilled_reviews="/cephyr/users/audinet/Alvis/llm-distillator/data/amazon_reviews/few-shot/distilled_reviews.txt" \
    --out_file="/cephyr/users/audinet/Alvis/llm-distillator/data/amazon_reviews/few-shot/results_alvis.txt" \
    --num_reviews=20 \
    --batch_size=4 \
    --test_size=0.2 \
    --num_bootstrap=20 \
    --confidence_level=0.95 \
