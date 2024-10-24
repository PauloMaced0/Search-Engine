import math
import msgpack
from src.tokenizer import Tokenizer
from typing import List, Dict, Tuple

class BM25Searcher:
    def __init__(self, index_path: str, tokenizer_config_path: str, k1: float = 1.2, b: float = 0.75):
        self.index, self.doc_lengths = self.load_index(index_path)
        
        self.tokenizer = self.load_tokenizer(tokenizer_config_path)
        self.k1 = k1
        self.b = b
        self.average_doc_length = self.calculate_average_doc_length()

        self.idf_cache = self.compute_idf()

    def load_index(self, index_path: str) -> Tuple[dict, dict]:
        """Load the index from a msgpack file."""
        try:
            with open(index_path, 'rb') as f:
                unpacker = msgpack.Unpacker(f, raw=False)
                data = next(unpacker)
                return data['index'], data['doc_lengths'] 
        except Exception as e:
            raise IOError(f"Error loading index from {index_path}: {e}")

    def load_tokenizer(self, tokenizer_config_path: str) -> Tokenizer:
        """Load the tokenizer configuration from a msgpack file."""
        try:
            with open(tokenizer_config_path, 'rb') as f:
                unpacker = msgpack.Unpacker(f, raw=False)
                tokenizer_config = next(unpacker)['tokenizer_config']
            return Tokenizer(
                min_token_length=tokenizer_config['min_token_length'],
                lowercase=tokenizer_config['lowercase'],
                remove_stopwords=tokenizer_config['remove_stopwords'],
                stopwords=set(tokenizer_config['stopwords']) if tokenizer_config['stopwords'] else None,
                stem=tokenizer_config['stem']
            )
        except Exception as e:
            raise IOError(f"Error loading tokenizer configuration from {tokenizer_config_path}: {e}")

    def calculate_average_doc_length(self) -> float:
        """Calculate the average document length."""
        return sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0.0

    def compute_idf(self) -> Dict[str, float]:
        """Compute the IDF for each term in the index and store it."""
        idf = {}
        total_docs = len(self.doc_lengths)
        for term, postings in self.index.items():
            doc_freq = len(postings)
            idf[term] = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return idf

    def calculate_bm25(self, query_tokens: List[str], doc_id: str) -> float:
        """Computes BM25 score for a single document given a query."""
        score = 0.0
        doc_length = self.doc_lengths[doc_id]
        b_times_doc_length = self.b * (doc_length / self.average_doc_length)

        for term in query_tokens:
            if term in self.idf_cache:
                term_freq = len(self.index[term][doc_id]) if doc_id in self.index[term] else 0
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + b_times_doc_length)
                score += self.idf_cache[term] * (numerator / denominator)
        return score

    def score_document(self, query_tokens: List[str], doc_id: str) -> Tuple[str, float]:
        """Scores a single document for a given query."""
        return doc_id, self.calculate_bm25(query_tokens, doc_id)

    def search(self, query: str, n_scores: int = 100) -> List[Tuple[str, float]]:
        """Searches for the query and returns the top ranked documents."""
        query_tokens = self.tokenizer.tokenize(query)
        
        candidate_docs = set()
        for term in query_tokens:
            if term in self.index:
                candidate_docs.update(self.index[term].keys())

        print(f"Searching through {len(candidate_docs)} docs")

        # Score only the candidate documents
        results = []
        for doc_id in candidate_docs:
            score = self.calculate_bm25(query_tokens, doc_id)
            results.append((doc_id, score))

        # Sort and return the top n_scores documents
        return sorted(results, key=lambda x: x[1], reverse=True)[:n_scores]
