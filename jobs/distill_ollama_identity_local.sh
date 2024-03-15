#!/usr/bin/env bash

python main.py ollama \
    --prompt_mode="identity" \
	--data_path="/Users/audinet/Projects/llm-distillator/data/amazon_reviews/original_reviews.txt" \
    --num_samples=1 \
