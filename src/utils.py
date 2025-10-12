# Utility functions

import random
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
    if "<SEP>" in vocab:
        embedding_matrix[vocab["<SEP>"]] = rng.uniform(-0.05, 0.05, embedding_dim)

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
    sep_token_id = tokenizer.token_to_id["<SEP>"]

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
        sequences = []
        labels = []
        qids = []
        dids = []

        for sample in batch:
            q_tokens = sample["question_token_ids"]
            d_tokens = sample["document_token_ids"]

            # Combine [query] + [SEP] + [doc]
            combined = pad_to_fixed_length(q_tokens + [sep_token_id] + d_tokens, max_number_of_question_tokens + max_number_of_question_tokens) 

            sequences.append(combined)
            labels.append(sample["label"])
            qids.append(sample["query_id"])
            dids.append(sample["document_id"])

        seq_tensor = torch.tensor(sequences, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.float)

        return {
            "joint_input": seq_tensor,
            "label": label_tensor,
            "query_ids": qids,
            "document_ids": dids,
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

def get_all_doc_texts(questions_file, ranked_file, corpus_file, negative_sample_multiplier=2):
    """
    Return a list of document texts (both positive and negative) for all queries combined.
    """
    gold_data = _load_gold_standard(questions_file)
    ranked_data = _load_ranked_results(ranked_file)
    corpus_map = _load_corpus(corpus_file)

    all_doc_texts = []
    for qid, qdata in gold_data.items():
        gold_docs = qdata["goldstandard_documents"]
        retrieved_docs = ranked_data.get(qid, [])

        positives = [d for d in retrieved_docs if d in gold_docs]
        if not positives:
            continue

        negatives = [d for d in retrieved_docs if d not in gold_docs]

        for doc_id in positives + negatives:
            if doc_id in corpus_map:
                all_doc_texts.append(corpus_map[doc_id])

    return all_doc_texts

def _load_gold_standard(questions_file):
    """
    Load gold-standard documents from a file with structure:
    {
      "question": "Is erenumab effective for trigeminal neuralgia?", 
      "goldstandard_documents": ["PMID:36113495"], 
      "query_id": "63f73f1b33942b094c000008"
    }

    Returns:
        gold_data: dict mapping query_id -> {
            "question": question_text,
            "goldstandard_documents": set_of_doc_ids
        }
    """
    gold_data = {}
    with open(questions_file, 'r') as f:
        for line in f:
            record = ujson.loads(line.strip())
            qid = record["query_id"]
            gold_data[qid] = {
                "question": record["question"],
                "goldstandard_documents": set(record.get("goldstandard_documents", []))
            }
    return gold_data

def _load_ranked_results(ranked_file):
    """
    Load BM25 or other ranked results from a file with structure:
    {
      "query_id": "55031181e9bde69634000014", 
      "retrieved_documents": [
         {"id": "PMID:15617541", "score": 36.2235}, 
         {"id": "PMID:15829955", "score": 32.1627}, ...
      ]
    }

    Returns:
        ranked_data: dict mapping query_id -> list_of_doc_ids_sorted_by_score
    """
    ranked_data = {}
    with open(ranked_file, 'r') as f:
        for line in f:
            record = ujson.loads(line.strip())
            qid = record["query_id"]
            # Extract doc_ids and maintain order by score
            doc_ids = [doc["id"] for doc in record["retrieved_documents"]]
            ranked_data[qid] = doc_ids
    return ranked_data

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

