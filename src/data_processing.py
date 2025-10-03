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
from multiprocessing import Pool, cpu_count

def _index_batch(docs, batch_id, output_dir, tokenizer_config):
    """Index one batch of documents and write a partial index."""
    index = defaultdict(lambda: defaultdict(list))
    doc_lengths = {}

    stemmer = SnowballStemmer('english') if tokenizer_config.get('stem', False) else None
    stem_cache = {}

    for doc in docs:
        doc_id = doc['doc_id'].split(':')[1]
        tokens = tokenize_text(
            doc['text'],
            min_token_length=tokenizer_config.get('min_token_length', 3),
            lowercase=tokenizer_config.get('lowercase', True),
            stem=tokenizer_config.get('stem', False),
            stopwords=tokenizer_config.get('stopwords'),
            stemmer=stemmer,
            stem_cache=stem_cache
        )

        doc_lengths[doc_id] = len(tokens)
        for pos, term in enumerate(tokens):
            index[term][doc_id].append(pos)

    # Write partial index
    _write_partial_index(index, doc_lengths, output_dir, batch_id)
    file_path = os.path.join(output_dir, f'partial_index_{batch_id}.msgpack')
    with open(file_path, 'wb') as f:
        msgpack.pack({'index': dict(index), 'doc_lengths': doc_lengths}, f)

    print(f"Created partial index: {file_path}")
    return file_path

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
    tokenizer_config: Optional[Dict] = None,
    num_workers: int = 0
) -> str:
    """
    Index documents from a corpus in parallel using SPIMI + multiprocessing.
    Each worker creates a partial index, then they are merged.
    """
    if tokenizer_config is None:
        tokenizer_config = {
            'min_token_length': 3,
            'lowercase': True,
            'stem': False,
            'stopwords': None
        }

    if num_workers < 1:
        num_workers = max(1, cpu_count() - 1)

    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    docs = list(read_corpus(corpus_path))
    batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]

    print(f"Indexing {len(docs)} documents in {len(batches)} batches with {num_workers} workers...")

    tasks = [(batches[i], i, output_dir, tokenizer_config) for i in range(len(batches))]

    with Pool(processes=num_workers) as pool:
        partial_files = pool.starmap(_index_batch, tasks)

    # Merge all partials
    merged_path = os.path.join(output_dir, 'merged_index.msgpack')
    _merge_partial_indexes(output_dir, len(partial_files), merged_path, num_workers=num_workers)
    # """
    # Index documents from a corpus using SPIMI algorithm.
    # 
    # Args:
    #     corpus_path: Path to the corpus file
    #     output_dir: Directory to store index files
    #     batch_size: Number of documents per batch
    #     tokenizer_config: Configuration for tokenization
    #     
    # Returns:
    #     Path to the merged index file
    # """
    # if tokenizer_config is None:
    #     tokenizer_config = {
    #         'min_token_length': 3,
    #         'lowercase': True,
    #         'stem': False,
    #         'stopwords': None
    #     }
    # 
    # # Create output directory if it doesn't exist
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # 
    # # Initialize variables
    # index = defaultdict(lambda: defaultdict(list))
    # doc_lengths = {}
    # current_batch = 0
    # current_doc_count = 0
    # 
    # # Create stemmer and cache if needed
    # stemmer = SnowballStemmer('english') if tokenizer_config.get('stem', False) else None
    # stem_cache = {}
    # 
    # # Process corpus
    # for doc in read_corpus(corpus_path):
    #     doc_id = doc['doc_id'].split(':')[1]
    #     
    #     # Tokenize document
    #     tokens = tokenize_text(
    #         doc['text'],
    #         min_token_length=tokenizer_config.get('min_token_length', 3),
    #         lowercase=tokenizer_config.get('lowercase', True),
    #         stem=tokenizer_config.get('stem', False),
    #         stopwords=tokenizer_config.get('stopwords'),
    #         stemmer=stemmer,
    #         stem_cache=stem_cache
    #     )
    #     
    #     # Store document length
    #     doc_lengths[doc_id] = len(tokens)
    #     
    #     # Create positional index
    #     for pos, term in enumerate(tokens):
    #         index[term][doc_id].append(pos)
    #     
    #     current_doc_count += 1
    #     
    #     # Write batch to disk when batch size is reached
    #     if current_doc_count % batch_size == 0:
    #         _write_partial_index(index, doc_lengths, output_dir, current_batch)
    #         index.clear()
    #         doc_lengths.clear()
    #         current_batch += 1
    # 
    # # Write remaining documents
    # if index:
    #     _write_partial_index(index, doc_lengths, output_dir, current_batch)
    #     current_batch += 1
    # 
    # # Save tokenizer configuration
    # config_path = os.path.join(output_dir, 'tokenizer_config.msgpack')
    # with open(config_path, 'wb') as f:
    #     config = {
    #         'tokenizer_config': {
    #             'min_token_length': tokenizer_config.get('min_token_length', 3),
    #             'lowercase': tokenizer_config.get('lowercase', True),
    #             'stopwords': list(tokenizer_config.get('stopwords', [])) if tokenizer_config.get('stopwords') else None,
    #             'stem': tokenizer_config.get('stem', False)
    #         }
    #     }
    #     msgpack.pack(config, f)
    # 
    # # Merge partial indexes
    # merged_path = os.path.join(output_dir, 'merged_index.msgpack')
    # _merge_partial_indexes(output_dir, current_batch, merged_path)
    # 
    return merged_path

def _write_partial_index(
    index: Dict,
    doc_lengths: Dict,
    output_dir: str,
    batch_num: int,
) -> None:
    """Write a partial index to disk."""
    file_path = os.path.join(output_dir, f'partial_index_{batch_num}.msgpack')
    with open(file_path, 'wb') as f:
        msgpack.pack({'index': index, 'doc_lengths': doc_lengths}, f)

def _load_partial(partial_file: str):
    """Load one partial index file from disk."""
    with open(partial_file, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        return next(unpacker)

def _merge_partial_files(file_list: List[str], merged_file: str):
    """Merge a small group of partial index files into one intermediate file."""
    merged_data = defaultdict(lambda: defaultdict(list))
    doc_lengths = {}

    for pf in file_list:
        partial_index = _load_partial(pf)
        for word, doc_data in partial_index['index'].items():
            for doc_id, positions in doc_data.items():
                merged_data[word][doc_id].extend(positions)
        doc_lengths.update(partial_index['doc_lengths'])

    # Write intermediate file
    with open(merged_file, 'wb') as f:
        msgpack.pack({'index': dict(merged_data), 'doc_lengths': doc_lengths}, f)

    print(f"Created intermediate file: {merged_file}")

    return merged_file

def _merge_partial_indexes(
    output_dir: str,
    num_batches: int,
    merged_path: str,
    chunk_size: int = 5,
    # num_workers: int = 0
) -> None:
    """
    Hierarchical merge of many partial index files.
    - Groups files into chunks of size `chunk_size`
    - Merges each chunk into an intermediate file
    - Repeats until only one file remains
    """
    round_num = 0
    files = [os.path.join(output_dir, f'partial_index_{i}.msgpack') for i in range(num_batches)]

    while len(files) > 1:
        round_num += 1
        print(f"=== Merge Round {round_num}, {len(files)} files ===")

        new_files = []
        for i in range(0, len(files), chunk_size):
            chunk = files[i:i+chunk_size]
            merged_file = os.path.join(output_dir, f"intermediate_round{round_num}_{i//chunk_size}.msgpack")
            _merge_partial_files(chunk, merged_file)
            new_files.append(merged_file)

        # Replace old list with intermediates for next round
        files = new_files

    # Final result
    os.rename(files[0], merged_path)
    print(f"Final merged index saved to: {merged_path}")
    # """
    # Hierarchical merge of many partial index files.
    # - Groups files into chunks of size `chunk_size`
    # - Merges each chunk into an intermediate file
    # - Repeats until only one file remains
    # """
    # if num_workers < 1:
    #     num_workers = max(1, cpu_count() - 1)
    #
    # round_num = 0
    # files = [os.path.join(output_dir, f'partial_index_{i}.msgpack') for i in range(num_batches)]
    #
    # while len(files) > 1:
    #     round_num += 1
    #     print(f"=== Merge Round {round_num}, {len(files)} files ===")
    #
    #     tasks = []
    #     for i in range(0, len(files), chunk_size):
    #         chunk = files[i:i+chunk_size]
    #         merged_file = os.path.join(output_dir, f"intermediate_round{round_num}_{i//chunk_size}.msgpack")
    #         tasks.append((chunk, merged_file))
    #
    #     # Run merges in parallel
    #     with Pool(processes=num_workers) as pool:
    #         new_files = pool.starmap(_merge_partial_files, tasks)
    #
    #     files = new_files  # intermediates become inputs for next round
    #
    # os.rename(files[0], merged_path)
    # print(f"Final merged index saved to: {merged_path}")

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
