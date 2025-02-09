#!/bin/bash

# This script indexes the corpus using the specified parameters.
python cli.py --index \
    --corpus data/MEDLINE_2024_Baseline.jsonl \
    --output_dir output \
    --min_token_length 3 \
    --lowercase True \
    --stemming False \
    --batch_size 10000

# This script performs the search using a questions file and generates the ranked results.
python cli.py --search \
    --questions_file data/questions.jsonl \
    --load_index output/merged_index.msgpack \
    --num_results 10

# To search using a query

python main.py --search --query "your search query" --load_index path/to/index.msgpack

# Or, to evaluate the system using provided questions:

python cli.py --search --questions_file data/questions.jsonl --load_index output/merged_index.msgpack
