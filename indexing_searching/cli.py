import time
import argparse
import os
import ujson
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
    
    # Corpus arguments
    parser.add_argument('--corpus', type=str, help="Path to the corpus file", default='../data/MEDLINE_2024_Baseline.jsonl')
    parser.add_argument('--output_dir', type=str, help="Directory to store partial and merged indexes", default='../output')

    # Tokenizer configuration
    parser.add_argument('--min_token_length', type=int, default=3, help="Minimum token length for tokenization")
    parser.add_argument('--lowercase', type=bool, default=False, help="Normalize tokens to lowercase")
    parser.add_argument('--stopwords', type=str, help="String with stopwords (space split)")
    parser.add_argument('--stemming', type=bool, default=False, help="Whether to apply stemming to tokens")

    # Ranking model configuration
    parser.add_argument('--k1', type=float, default=1.2, help="BM25 parameter k1 (controls term frequency saturation)")
    parser.add_argument('--b', type=float, default=0.75, help="BM25 parameter b (controls document length normalization)")

    # Batch size argument for indexer
    parser.add_argument('--batch_size', type=int, default=10000, help="Number of documents per batch for indexing")

    # Load specific index for search and number of results
    parser.add_argument('--load_index', type=str, help="Path to pre-built index file to load for searching")
    parser.add_argument('--num_results', type=int, default=100, help="Number of results to return for the query")

    search_group = parser.add_mutually_exclusive_group()
    search_group.add_argument('--query', type=str, help="Query for searching documents")
    search_group.add_argument('--questions_file', type=str, help="Path to the questions file (JSONL) for batch processing")
    search_group.add_argument('--interactive', action='store_true', help="Run in interactive mode for manual query input")

    args = parser.parse_args()

    if args.search and not args.query and not args.questions_file and not args.interactive:
        parser.error("--query | --questions_file | --interactive is required when --search is selected.")

    index_dir = os.path.join(args.output_dir, "merged_index.msgpack")
    tokenizer_config_dir = os.path.join(args.output_dir, "tokenizer_config.msgpack")

    if args.index:
        print(f"Indexing the corpus at {UNDERLINE}{args.corpus}{RESET}...")
        print(f"Using a batch size of {UNDERLINE}{args.batch_size}{RESET}...")
        print(f"Output directory at {UNDERLINE}{args.output_dir}{RESET}...")
        print(f"Minimum token length: {UNDERLINE}{args.min_token_length}{RESET}")
        print(f"Lowercase normalization: {UNDERLINE}{args.lowercase}{RESET}")
        print(f"Stemming: {UNDERLINE}{args.stemming}{RESET}")

        stopwords = None
        if args.stopwords:
            stopwords = args.stopwords.split(' ')

        start = time.time()
        tokenizer = Tokenizer(min_token_length=args.min_token_length, lowercase=args.lowercase, stem=args.stemming, stopwords=stopwords)
        corpus_reader = CorpusReader(args.corpus)
        indexer = SPIMIIndexer(tokenizer, args.output_dir, index_dir, args.batch_size)
        indexer.process_corpus(corpus_reader)
        
        end = time.time()
        print(f'Doc tokenization and indexing: {end-start:.6f} seconds')

    elif args.search:
        if args.query:
            print(f"Searching with query: {UNDERLINE}{args.query}{RESET}")
        elif args.questions_file:
            print(f"Processing questions from file: {UNDERLINE}{args.questions_file}{RESET}")
        elif args.interactive:
            print("Entering interactive mode for manual query input.")

        if args.load_index:
            print(f"Loading index from: {UNDERLINE}{args.load_index}{RESET}")
            index_dir = args.load_index

        print(f"BM25 parameters: k1 = {UNDERLINE}{args.k1}{RESET}, b = {UNDERLINE}{args.b}{RESET}")

        start = time.time()

        searcher = BM25Searcher(index_path=index_dir, tokenizer_config_path=tokenizer_config_dir)

        output_data = []
        if args.interactive:
            query_count = 1
            while True:
                query = input("Enter your query (or type 'exit' to quit): ")
                if query.lower() == 'exit':
                    break
                results = searcher.search(query, n_scores=args.num_results)
                retrieved_docs = [doc_id for doc_id, _ in results]

                output_entry = {
                    "id": f"interactive_{query_count}",
                    "question": query,
                    "retrieved_documents": retrieved_docs
                }
                output_data.append(output_entry)
                query_count += 1

        elif args.questions_file:
            output_data = []
            with open(args.questions_file, 'r') as f:
                for line in f:
                    question_data = ujson.loads(line)
                    question = question_data['question']
                    query_id = question_data['query_id']

                    results = searcher.search(question, n_scores=args.num_results)

                    retrieved_docs = [doc_id for doc_id, _ in results]
                    
                    output_entry = {
                        "id": query_id,
                        "question": question,
                        "retrieved_documents": retrieved_docs
                    }
                    output_data.append(output_entry)
        else:
            results = searcher.search(args.query, args.num_results)
            retrieved_docs = [doc_id for doc_id, _ in results]

            output_entry = {
                "id": "Command_line_query",
                "question": args.query,
                "retrieved_documents": retrieved_docs
            }
            output_data.append(output_entry)
        
        output_file_path = os.path.join(args.output_dir, 'ranked_questions.jsonl')
        with open(output_file_path, 'w') as outfile:
            for entry in output_data:
                json_line = ujson.dumps(entry)
                outfile.write(json_line + '\n')

        end = time.time()
        print(f'Searching document collection: {end-start:.6f} seconds')

if __name__ == '__main__':
    main()

