"""
Data processing functions for document indexing and tokenization.
"""

import re
import os
import ujson
import msgpack
from collections import defaultdict
from typing import Iterator, Dict, List, Set, Optional
from nltk.stem import SnowballStemmer

def read_corpus(file_path: str) -> Iterator[Dict]:
    """
    Read documents from a JSONL corpus file.
    
    Args:
        file_path: Path to the corpus file
        
    Yields:
        Dictionary containing document data
    """
    with open(file_path, 'r', buffering=1024 * 1024) as f:
        for line in f:
            yield ujson.loads(line)

def tokenize_text(
    text: str,
    min_token_length: int = 3,
    lowercase: bool = True,
    stem: bool = False,
    stopwords: Optional[Set[str]] = None,
    stemmer: Optional[SnowballStemmer] = None,
    stem_cache: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Tokenize text into a list of processed tokens.
    
    Args:
        text: Input text to tokenize
        min_token_length: Minimum length for valid tokens
        lowercase: Whether to convert tokens to lowercase
        stem: Whether to apply stemming
        stopwords: Set of stopwords to filter out
        stemmer: Stemmer instance (created if not provided and stem=True)
        stem_cache: Cache for stemmed tokens
        
    Returns:
        List of processed tokens
    """
    if stem_cache is None:
        stem_cache = {}
    
    if stem and stemmer is None:
        stemmer = SnowballStemmer('english')
    
    if stopwords is None:
        stopwords = set()
    elif lowercase:
        stopwords = {word.lower() for word in stopwords}
    
    # Convert to lowercase if needed
    if lowercase:
        text = text.lower()
    
    # Tokenize using regex
    word_pattern = re.compile(r'\b\w+\b')
    tokens = word_pattern.findall(text)
    
    result_tokens = []
    for token in tokens:
        # Skip short tokens
        if len(token) < min_token_length:
            continue
        
        # Skip stopwords
        if token in stopwords:
            continue
        
        # Apply stemming if needed
        if stem:
            if token not in stem_cache:
                stem_cache[token] = stemmer.stem(token)
            result_tokens.append(stem_cache[token])
        
        result_tokens.append(token)
    
    return result_tokens

def index_documents(
    corpus_path: str,
    output_dir: str,
    batch_size: int = 10000,
    tokenizer_config: Optional[Dict] = None
) -> str:
    """
    Index documents from a corpus using SPIMI algorithm.
    
    Args:
        corpus_path: Path to the corpus file
        output_dir: Directory to store index files
        batch_size: Number of documents per batch
        tokenizer_config: Configuration for tokenization
        
    Returns:
        Path to the merged index file
    """
    if tokenizer_config is None:
        tokenizer_config = {
            'min_token_length': 3,
            'lowercase': True,
            'stem': False,
            'stopwords': None
        }
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize variables
    index = defaultdict(lambda: defaultdict(list))
    doc_lengths = {}
    current_batch = 0
    current_doc_count = 0
    
    # Create stemmer and cache if needed
    stemmer = SnowballStemmer('english') if tokenizer_config.get('stem', False) else None
    stem_cache = {}
    
    # Process corpus
    for doc in read_corpus(corpus_path):
        doc_id = doc['doc_id'].split(':')[1]
        
        # Tokenize document
        tokens = tokenize_text(
            doc['text'],
            min_token_length=tokenizer_config.get('min_token_length', 3),
            lowercase=tokenizer_config.get('lowercase', True),
            stem=tokenizer_config.get('stem', False),
            stopwords=tokenizer_config.get('stopwords'),
            stemmer=stemmer,
            stem_cache=stem_cache
        )
        
        # Store document length
        doc_lengths[doc_id] = len(tokens)
        
        # Create positional index
        for pos, term in enumerate(tokens):
            index[term][doc_id].append(pos)
        
        current_doc_count += 1
        
        # Write batch to disk when batch size is reached
        if current_doc_count % batch_size == 0:
            _write_partial_index(index, doc_lengths, output_dir, current_batch)
            index.clear()
            doc_lengths.clear()
            current_batch += 1
    
    # Write remaining documents
    if index:
        _write_partial_index(index, doc_lengths, output_dir, current_batch)
        current_batch += 1
    
    # Save tokenizer configuration
    config_path = os.path.join(output_dir, 'tokenizer_config.msgpack')
    with open(config_path, 'wb') as f:
        config = {
            'tokenizer_config': {
                'min_token_length': tokenizer_config.get('min_token_length', 3),
                'lowercase': tokenizer_config.get('lowercase', True),
                'stopwords': list(tokenizer_config.get('stopwords', [])) if tokenizer_config.get('stopwords') else None,
                'stem': tokenizer_config.get('stem', False)
            }
        }
        msgpack.pack(config, f)
    
    # Merge partial indexes
    merged_path = os.path.join(output_dir, 'merged_index.msgpack')
    _merge_partial_indexes(output_dir, current_batch, merged_path)
    
    return merged_path

def _write_partial_index(
    index: Dict,
    doc_lengths: Dict,
    output_dir: str,
    batch_num: int
) -> None:
    """Write a partial index to disk."""
    file_path = os.path.join(output_dir, f'partial_index_{batch_num}.msgpack')
    with open(file_path, 'wb') as f:
        msgpack.pack({'index': index, 'doc_lengths': doc_lengths}, f)

def _merge_partial_indexes(
    output_dir: str,
    num_batches: int,
    merged_path: str
) -> None:
    """Merge all partial indexes into a single file."""
    merged_data = defaultdict(lambda: defaultdict(list))
    doc_lengths = {}
    
    for batch_num in range(num_batches):
        partial_file = os.path.join(output_dir, f'partial_index_{batch_num}.msgpack')
        print(f"Merging partial index: {partial_file}")
        
        with open(partial_file, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            partial_index = next(unpacker)
        
        # Merge index data
        for word, doc_data in partial_index['index'].items():
            for doc_id, positions in doc_data.items():
                merged_data[word][doc_id].extend(positions)
        
        # Merge document lengths
        doc_lengths.update(partial_index['doc_lengths'])
    
    # Write merged index
    with open(merged_path, 'wb') as f:
        msgpack.pack({'index': dict(merged_data), 'doc_lengths': doc_lengths}, f)
    
    print(f"Merged index saved to: {merged_path}")

def load_tokenizer_config(config_path: str) -> Dict:
    """
    Load tokenizer configuration from a msgpack file.
    
    Args:
        config_path: Path to the tokenizer configuration file
        
    Returns:
        Dictionary containing tokenizer configuration
    """
    with open(config_path, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        config = next(unpacker)
    return config['tokenizer_config']
