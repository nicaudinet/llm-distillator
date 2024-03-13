#!/usr/bin/env bash

python main.py ollama \
    --prompt_mode="cot" \
	--data_path="/Users/audinet/Projects/llm-distillator/data/amazon_reviews/original_reviews.txt" \
    --num_samples=10 \
