import math
import msgpack
from src.tokenizer import Tokenizer

class BM25Searcher:
    def __init__(self, index_path, metadata_path, k1=1.2, b=0.75):
        with open(index_path, 'rb') as f:
            data = msgpack.unpack(f)
            self.index = data['index']

        with open(metadata_path, 'rb') as f:
            data = msgpack.unpack(f)

        tokenizer_config = data['tokenizer_config']
        self.tokenizer = Tokenizer(
            min_token_length=tokenizer_config['min_token_length'],
            lowercase=tokenizer_config['lowercase'],
            remove_stopwords=tokenizer_config['remove_stopwords'],
            stopwords=set(tokenizer_config['stopwords']) if tokenizer_config['stopwords'] else None,
            stem=tokenizer_config['stem']
        )
        self.k1 = k1
        self.b = b
        self.doc_lengths = {}  # Store document lengths for BM25 calculation
        self.average_doc_length = 1445

    def calculate_bm25(self, query_tokens, doc_id):
        """Computes BM25 score for a single document given a query."""
        score = 0.0
        for term in query_tokens:
            if term in self.index:
                doc_freq = len(self.index[term])
                term_freq = sum(1 for d, _ in self.index[term] if d == doc_id)

                idf = math.log((len(self.doc_lengths) - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
                tf = term_freq * (self.k1 + 1) / (term_freq + self.k1 * (1 - self.b + self.b * (len(self.doc_lengths[doc_id]) / self.average_doc_length)))
                
                score += idf * tf
        return score

    def search(self, query):
        """Searches for the query and returns the top 100 ranked documents."""
        query_tokens = self.tokenizer.tokenize(query)

        doc_scores = {doc_id: self.calculate_bm25(query_tokens, doc_id) for doc_id in self.doc_lengths}

        return sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:100]
