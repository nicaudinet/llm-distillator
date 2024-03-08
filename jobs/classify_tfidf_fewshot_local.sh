#!/usr/bin/env bash

python classify.py \
    --embedding_method="tfidf" \
    --original_reviews="data/amazon_reviews/original_reviews.txt" \
    --distilled_reviews="data/amazon_reviews/few-shot/distilled_reviews.txt" \
    --out_file="data/amazon_reviews/few-shot/results_tfidf_local.txt" \
    --num_reviews=20 \
    --max_features=256 \
    --test_size=0.2 \
    --num_bootstrap=20 \
    --confidence_level=0.95 \
