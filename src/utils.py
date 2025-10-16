# Utility functions

import torch
import ujson
import numpy as np
import os
import msgpack
from typing import Dict

def load_pretrained_embeddings(embedding_path, tokenizer, embedding_dim=300, lowercase=True):
    """
    Load pretrained GloVe embeddings aligned with tokenizer's vocabulary.
    Unknown words get random vectors, PAD gets zeros, and <SEP> gets a random vector.
    """
    vocab = tokenizer.token_to_id
    vocab_size = len(vocab)

    rng = np.random.default_rng(seed=42)
    embedding_matrix = rng.uniform(-0.05, 0.05, (vocab_size, embedding_dim)).astype(np.float32)

    # Special tokens
    if "<PAD>" in vocab:
        embedding_matrix[vocab["<PAD>"]] = np.zeros(embedding_dim, dtype=np.float32)

    found = 0

    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) <= embedding_dim:
                continue  # skip malformed lines
            word = parts[0].lower() if lowercase else parts[0]
            if word in vocab:
                embedding_matrix[vocab[word]] = np.asarray(parts[1:], dtype=np.float32)
                found += 1
                if found == vocab_size:
                    break

    print(f"Loaded {found}/{vocab_size} words from GloVe ({found/vocab_size:.2%} coverage)")

    return torch.tensor(embedding_matrix, dtype=torch.float32)

def build_collate_fn(tokenizer, max_number_of_question_tokens, max_number_of_document_tokens):
    # Retrieve the pad token ID from the tokenizer
    pad_token_id = tokenizer.token_to_id["<PAD>"]

    def pad_to_fixed_length(seq, max_len):
        seq = seq[:max_len]
        if len(seq) < max_len:
            seq += [pad_token_id] * (max_len - len(seq))
        return seq

    def collate_fn(batch):
        """
        Args:
            batch (List[Dict]): A list of samples from the dataset.
        
        Returns:
            A dictionary containing:
                - "question_token_ids": Tensor of shape (batch_size, max_number_of_question_tokens)
                - "document_token_ids": Tensor of shape (batch_size, max_number_of_document_tokens)
                - "question_id": List of question IDs
                - "document_id": List of document IDs
        """

        question_ids = [s["query_id"] for s in batch]
        document_ids = [s["document_id"] for s in batch]

        query_seqs = [pad_to_fixed_length(s["question_token_ids"], max_number_of_question_tokens) for s in batch]
        doc_seqs   = [pad_to_fixed_length(s["document_token_ids"], max_number_of_document_tokens) for s in batch]
        labels     = [s["label"] for s in batch]

        query_tensor = torch.tensor(query_seqs, dtype=torch.long)
        doc_tensor   = torch.tensor(doc_seqs, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.float)

        return {
            "question_token_ids": query_tensor,
            "document_token_ids": doc_tensor,
            "query_ids": question_ids,
            "document_ids": document_ids,
            "label": label_tensor 
        }

    return collate_fn

def get_questions(questions_file):
    """
    Reads the questions file and stores the data in self.questions.
    Each question is a dictionary with keys 'question', 'goldstandard_documents', and 'query_id'.
    """
    questions = []
    with open(questions_file, 'r', buffering=1024 * 1024) as f:
        for line in f:
            data = ujson.loads(line)
            questions.append(data['question'])

    return questions

def get_all_doc_texts(corpus_file):
    """
    Return all document texts from the corpus file.
    """
    corpus_map = _load_corpus(corpus_file)
    return list(corpus_map.values())

def _load_corpus(corpus_file):
    """
    Load the corpus from MEDLINE_2024_Baseline.jsonl with structure:
    {"doc_id": "PMID:2451706", "text": "Organization of the genes ..."}

    Returns:
        corpus_map: dict mapping doc_id -> doc_text
    """
    corpus_map = {}
    with open(corpus_file, 'r') as cf:
        for line in cf:
            doc = ujson.loads(line.strip())
            doc_id = doc["doc_id"]
            doc_text = doc["text"]
            corpus_map[doc_id] = doc_text
    return corpus_map

def load_questions(questions_file) -> Dict[str, str]:
    """
    Load questions from a JSONL file and return a dictionary mapping query_id to question text.

    Args:
        questions_file (str): Path to the questions file (JSONL format).

    Returns:
        dict: A dictionary where keys are query_ids and values are question texts.
    """
    questions_dict = {}
    with open(questions_file, 'r') as f:
        for line in f:
            data = ujson.loads(line)
            qid = data.get('query_id') or data.get('id')  # Adjust based on your JSON structure
            question = data.get('question')
            if qid and question:
                questions_dict[qid] = question
    return questions_dict

def print_short_index_entries(merged_index_file, min_length=200, max_items=15):
    """Print index entries with fewer than min_length items."""

    # Check if the file exists
    if not os.path.exists(merged_index_file):
        print(f"Error: File '{merged_index_file}' does not exist.")
        return

    # Open and read the merged index file using MessagePack
    with open(merged_index_file, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        merged_data = next(unpacker)

    short_entries = [
        (word, entries) for word, entries in merged_data['index'].items() 
        if len(entries) < min_length
    ]
    
    for word, entries in short_entries[:max_items]:
        print(f"{word}: {entries}")

