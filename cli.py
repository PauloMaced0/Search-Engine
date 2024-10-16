import time
import argparse
import os
from src.corpus_reader import CorpusReader
from src.spimi_indexer import SPIMIIndexer
from src.searcher import BM25Searcher
from src.tokenizer import Tokenizer

RESET = "\033[0m"  # Reset all styles
UNDERLINE = "\033[4m"

def main():
    parser = argparse.ArgumentParser(description="Information Retrieval System")
    
    # Add program input args for the tokenizer and ranking
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--index', action='store_true', help="Run the indexer")
    group.add_argument('--search', action='store_true', help="Run the searcher")
    
    # Query and corpus arguments
    parser.add_argument('--query', type=str, help="Query for searching documents")
    parser.add_argument('--corpus', type=str, help="Path to the corpus file", default='data/MEDLINE_2024_Baseline.jsonl')
    parser.add_argument('--output_dir', type=str, help="Directory to store partial and merged indexes", default='output')

    # Tokenizer configuration
    parser.add_argument('--min_token_length', type=int, default=3, help="Minimum token length for tokenization")
    parser.add_argument('--lowercase', type=bool, default=True, help="Normalize tokens to lowercase")
    parser.add_argument('--stopwords', type=str, help="Path to stopwords file (optional)")
    parser.add_argument('--stemming', type=bool, default=False, help="Whether to apply stemming to tokens")

    # Ranking model configuration
    parser.add_argument('--k1', type=float, default=1.2, help="BM25 parameter k1 (controls term frequency saturation)")
    parser.add_argument('--b', type=float, default=0.75, help="BM25 parameter b (controls document length normalization)")

    # Batch size argument for indexer
    parser.add_argument('--batch_size', type=int, default=10000, help="Number of documents per batch for indexing")

    # Load specific index for search
    parser.add_argument('--load_index', type=str, help="Path to pre-built index file to load for searching")

    args = parser.parse_args()

    if args.search and not args.query:
        parser.error("--query is required when --search is selected.")

    merged_index_dir = os.path.join(args.output_dir, "merged_index.msgpack")
    tokenizer_config_dir = os.path.join(args.output_dir, "tokenizer_config.msgpack")

    if args.index:
        print(f"Indexing the corpus at {UNDERLINE}{args.corpus}{RESET}...")
        print(f"Using a batch size of {UNDERLINE}{args.batch_size}{RESET}...")
        print(f"Output directory at {UNDERLINE}{args.output_dir}{RESET}...")
        print(f"Minimum token length: {UNDERLINE}{args.min_token_length}{RESET}")
        print(f"Lowercase normalization: {UNDERLINE}{args.lowercase}{RESET}")
        print(f"Stemming: {UNDERLINE}{args.stemming}{RESET}")

        start = time.time()

        tokenizer = Tokenizer(min_token_length=args.min_token_length, lowercase=args.lowercase, stem=args.stemming, remove_stopwords=args.stopwords)
        corpus_reader = CorpusReader(args.corpus)
        indexer = SPIMIIndexer(tokenizer, args.output_dir, merged_index_dir, args.batch_size)
        indexer.process_corpus(corpus_reader)
        
        end = time.time()
        print(f'Doc tokenization and indexing: {end-start:.6f} seconds')

    elif args.search:
        print(f"Searching with query: {UNDERLINE}{args.query}{RESET}")

        if args.load_index:
            print(f"Loading index from: {UNDERLINE}{args.load_index}{RESET}")

        print(f"BM25 parameters: k1 = {UNDERLINE}{args.k1}{RESET}, b = {UNDERLINE}{args.b}{RESET}")

        searcher = BM25Searcher(index_path=merged_index_dir, metadata_path=tokenizer_config_dir)
        results = searcher.search(args.query)
        # Later to be stored in ranked_questions.jsonl
        # {
        # “id”:“<question_id>”,
        # “question”: “<question_text>”, 
        # “retrieved_documents”: 
        #   [ “<doc_id_pos_1>”, “<doc_id_pos_2>”, ..., “<doc_id_pos_100>”,]
        # },
        # ...
        for doc_id, score in results:
            print(f"Document {doc_id} - Score: {score}")

if __name__ == '__main__':
    main()

