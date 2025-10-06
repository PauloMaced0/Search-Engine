"""
Data processing functions for document indexing and tokenization.
"""

import os
import ujson
import msgpack
from collections import defaultdict
from typing import Iterator, Dict, List, Optional
from multiprocessing import Pool, cpu_count
from .tokenizer import Tokenizer

def _index_batch(docs, batch_id, output_dir, tokenizer_config):
    """Index one batch of documents and write a partial index."""
    index = defaultdict(lambda: defaultdict(list))
    doc_lengths = {}

    tokenizer = Tokenizer(
        tokenizer_config.get('min_token_length', 3),
        tokenizer_config.get('lowercase', True),
        tokenizer_config.get('stem', False),
        tokenizer_config.get('stopwords')
    )

    for doc in docs:
        doc_id = doc['doc_id'].split(':')[1]
        tokens = tokenizer.tokenize(doc['text'])

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
    _merge_partial_indexes(output_dir, len(partial_files), merged_path)
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
