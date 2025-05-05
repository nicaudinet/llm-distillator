#!/usr/bin/env bash

# Use --embedding_method="bert" instead of --embedding_path to generate
# embeddings as well
#
python classify.py \
    --mean_projection \
    --embedding_in="embeddings/mistral/cot/" \
    --original_reviews="reviews/original_reviews.txt" \
    --distilled_reviews="reviews/mistral/cot/mistral_cot_reviews.txt" \
    --out_file="reviews/projection/projection_results.sh" \
    --num_reviews=2000 \
    --batch_size=4 \
    --test_size=0.2 \
    --num_bootstrap=500 \
    --confidence_level=0.95 \
