"""
BM25 search functions for document retrieval.
"""

import math
import msgpack
from typing import List, Dict, Tuple, Set


def load_index(index_path: str) -> Tuple[Dict, Dict]:
    """
    Load the inverted index and document lengths from a msgpack file.
    
    Args:
        index_path: Path to the index file
        
    Returns:
        Tuple of (index, document_lengths)
    """
    with open(index_path, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        data = next(unpacker)
    return data['index'], data['doc_lengths']


def compute_idf(index: Dict, num_docs: int) -> Dict[str, float]:
    """
    Compute IDF scores for all terms in the index.
    
    Args:
        index: Inverted index mapping terms to documents
        num_docs: Total number of documents
        
    Returns:
        Dictionary mapping terms to IDF scores
    """
    idf = {}
    for term, postings in index.items():
        doc_freq = len(postings)
        # BM25 IDF formula
        idf[term] = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    return idf


def calculate_bm25_score(
    query_terms: List[str],
    doc_id: str,
    index: Dict,
    doc_lengths: Dict,
    idf_scores: Dict,
    avg_doc_length: float,
    k1: float = 1.2,
    b: float = 0.75
) -> float:
    """
    Calculate BM25 score for a document given a query.
    
    Args:
        query_terms: List of query terms
        doc_id: Document ID to score
        index: Inverted index
        doc_lengths: Dictionary of document lengths
        idf_scores: Precomputed IDF scores
        avg_doc_length: Average document length in the corpus
        k1: BM25 parameter controlling term frequency saturation
        b: BM25 parameter controlling document length normalization
        
    Returns:
        BM25 score for the document
    """
    score = 0.0
    doc_length = doc_lengths.get(doc_id, 0)
    
    if doc_length == 0:
        return 0.0
    
    # Precompute normalization factor
    norm_factor = 1 - b + b * (doc_length / avg_doc_length)
    
    for term in query_terms:
        if term not in idf_scores:
            continue
        
        # Get term frequency in document
        term_freq = len(index[term].get(doc_id, []))
        
        if term_freq == 0:
            continue
        
        # BM25 formula
        numerator = term_freq * (k1 + 1)
        denominator = term_freq + k1 * norm_factor
        score += idf_scores[term] * (numerator / denominator)
    
    return score


def get_candidate_documents(query_terms: List[str], index: Dict) -> Set[str]:
    """
    Get candidate documents that contain at least one query term.
    
    Args:
        query_terms: List of query terms
        index: Inverted index
        
    Returns:
        Set of candidate document IDs
    """
    candidates = set()
    for term in query_terms:
        if term in index:
            candidates.update(index[term].keys())
    return candidates


def search(
    query: str,
    index: Dict,
    doc_lengths: Dict,
    tokenizer_config: Dict,
    n_results: int = 100,
    k1: float = 1.2,
    b: float = 0.75
) -> List[Tuple[str, float]]:
    """
    Search for documents matching a query using BM25 ranking.
    
    Args:
        query: Search query
        index_path: Path to the index file
        tokenizer_config: Configuration for tokenizing the query
        n_results: Number of results to return
        k1: BM25 parameter
        b: BM25 parameter
        
    Returns:
        List of (document_id, score) tuples sorted by relevance
    """
    # Import tokenization function
    from .data_processing import tokenize_text
    
    # Tokenize query using the same configuration as indexing
    query_tokens = tokenize_text(
        query,
        min_token_length=tokenizer_config.get('min_token_length', 3),
        lowercase=tokenizer_config.get('lowercase', True),
        stem=tokenizer_config.get('stem', False),
        stopwords=set(tokenizer_config.get('stopwords', [])) if tokenizer_config.get('stopwords') else None
    )
    
    # Get candidate documents
    candidates = get_candidate_documents(query_tokens, index)
    
    if not candidates:
        return []
    
    # Compute corpus statistics
    num_docs = len(doc_lengths)
    avg_doc_length = sum(doc_lengths.values()) / num_docs if num_docs > 0 else 0
    
    # Compute IDF scores
    idf_scores = compute_idf(index, num_docs)
    
    # Score candidate documents
    results = []
    for doc_id in candidates:
        score = calculate_bm25_score(
            query_tokens, doc_id, index, doc_lengths, 
            idf_scores, avg_doc_length, k1, b
        )
        if score > 0:
            results.append((f"PMID:{doc_id}", score))
    
    # Sort by score and return top N
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n_results]

def batch_search(
    queries: List[Dict[str, str]],
    index_path: str,
    tokenizer_config: Dict,
    n_results: int = 100,
    k1: float = 1.2,
    b: float = 0.75
) -> List[Dict]:
    """
    Process multiple queries and return results.
    
    Args:
        queries: List of query dictionaries with 'query_id' and 'question' keys
        index_path: Path to the index file
        tokenizer_config: Configuration for tokenizing queries
        n_results: Number of results per query
        k1: BM25 parameter
        b: BM25 parameter
        
    Returns:
        List of result dictionaries
    """
    results = []

    index, doc_lengths = load_index(index_path)
    
    for query_data in queries:
        query_id = query_data.get('query_id', query_data.get('id', 'unknown'))
        question = query_data.get('question', query_data.get('query', ''))
        
        print("Search: " + question) 
        # Search for documents
        search_results = search(
            question, index, doc_lengths, tokenizer_config, 
            n_results, k1, b
        )

        # Extract document IDs
        retrieved_docs = [doc_id for doc_id, _ in search_results]
        
        results.append({
            'id': query_id,
            'question': question,
            'retrieved_documents': retrieved_docs
        })
    
    return results
