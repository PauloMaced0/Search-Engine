#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define directories
PROJECT_DIR=$(pwd)
DATA_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$PROJECT_DIR/output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 2: Train the CNN Reranker Model and Perform Reranking
echo "Training the CNN reranker model and performing reranking..."

python train_and_rerank.py \
    --pretrained_embeddings "$DATA_DIR/glove.42B.300d.txt" \
    --corpus "$DATA_DIR/MEDLINE_2024_Baseline.jsonl" \
    --questions_file "$DATA_DIR/questions.jsonl" \
    --bm25_ranked_file "$DATA_DIR/questions_bm25_ranked.jsonl" \
    --training_data "$DATA_DIR/training_data.jsonl" \
    --training_data_bm25_ranked "$DATA_DIR/training_data_bm25_ranked.jsonl" \
    --output_file "$OUTPUT_DIR/final_ranked_questions.jsonl" \
    --model_checkpoint "$OUTPUT_DIR/trained_cnn_model.pt" \
    --batch_size 64 \
    --number_documents_ranked 10 \
    --epochs 5 \
    --learning_rate 0.001

echo "Training and reranking completed."

# Step 3: Compute nDCG@10
echo "Computing nDCG@10..."

python compute_ndcg.py \
    --questions_file "$DATA_DIR/questions.jsonl" \
    --results_file "$OUTPUT_DIR/final_ranked_questions.jsonl" \
    --k 10

echo "nDCG computation completed."

# Deactivate the virtual environment
deactivate

echo "All steps completed successfully."
