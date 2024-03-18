#!/usr/bin/env bash

python main.py gpt4 \
    --prompt_mode identity \
	--data_path "/Users/audinet/Projects/llm-distillator/data/amazon_reviews/original_reviews.txt" \
	--out_dir "/Users/audinet/Projects/llm-distillator/data/amazon_reviews/identity/gpt4/" \
	--num_samples 450 \
    --start_sample 1550 \
