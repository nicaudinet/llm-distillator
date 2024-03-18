#!/usr/bin/env bash

python main.py gpt4 \
	--data_path "/Users/audinet/Projects/llm-distillator/data/amazon_reviews/original_reviews.txt" \
	--out_dir "/Users/audinet/Projects/llm-distillator/data/amazon_reviews/few-shot/gpt4/reviews" \
	--num_samples 800 \
