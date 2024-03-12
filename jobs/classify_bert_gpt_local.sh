#!/usr/bin/env bash

# Use --embedding_method="bert" instead of --embedding_path to generate
# embeddings as well
#
python classify.py \
    --embedding_path="/Users/audinet/Datasets/amazon_reviews/few-shot/gpt4/" \
    --original_reviews="data/amazon_reviews/original_reviews.txt" \
    --distilled_reviews="data/amazon_reviews/few-shot/gpt4/distilled.txt" \
    --out_file="data/amazon_reviews/few-shot/gpt4/results.txt" \
    --num_reviews=1600 \
    --batch_size=4 \
    --test_size=0.2 \
    --num_bootstrap=500 \
    --confidence_level=0.95 \
