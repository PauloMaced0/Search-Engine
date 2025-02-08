#!/bin/bash

# This script indexes the corpus using the specified parameters.
python main.py --index \
    --corpus data/MEDLINE_2024_Baseline.jsonl \
    --output_dir output \
    --min_token_length 3 \
    --lowercase True \
    --stemming False \
    --batch_size 10000

# This script performs the search using a questions file and generates the ranked results.
python main.py --search \
    --questions_file data/questions.jsonl \
    --load_index output/merged_index.msgpack \
    --num_results 10

# This script computes the average nDCG@10 score using the questions file and the ranked results.
python compute_ndcg.py \
    --questions_file data/questions.jsonl \
    --results_file output/ranked_questions.jsonl \
    --k 10
