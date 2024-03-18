#!/usr/bin/env bash

python main.py gpt4 \
    --prompt_mode cot \
	--data_path "/Users/audinet/Projects/llm-distillator/data/amazon_reviews/original_reviews.txt" \
	--out_dir "/Users/audinet/Projects/llm-distillator/data/amazon_reviews/cot/gpt4/" \
	--num_samples 300 \
    --start_sample 1700 \
