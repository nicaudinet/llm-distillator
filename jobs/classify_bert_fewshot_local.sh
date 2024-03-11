#!/usr/bin/env bash

# Use --embedding_method="bert" instead of --embedding_path to generate
# embeddings as well
#
python classify.py \
    --embedding_path="embeddings/bert" \
    --original_reviews="data/amazon_reviews/original_reviews.txt" \
    --distilled_reviews="data/amazon_reviews/few-shot/distilled_reviews.txt" \
    --out_file="data/amazon_reviews/few-shot/results_bert_local.txt" \
    --num_reviews=5000 \
    --batch_size=4 \
    --test_size=0.2 \
    --num_bootstrap=500 \
    --confidence_level=0.95 \
