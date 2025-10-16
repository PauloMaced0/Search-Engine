# Utility functions

import torch
import ujson
import os
import msgpack
from typing import Dict
from tqdm import tqdm

def compute_dynamic_max_len(questions_file, bm25_ranked_file, corpus_file, tokenizer):
    """
    Computes the max tokenized length of question+document pairs in your corpus.

    Args:
        questions_file: path to questions JSONL
        bm25_ranked_file: path to BM25 ranked results JSONL
        corpus_file: path to corpus JSONL
        tokenizer: HuggingFace tokenizer (e.g. BertTokenizer)
        sample_size: optional number of pairs to sample (for large datasets)
        percentile: upper percentile to clip max length (default: 95)
    """
    from .utils import load_questions, load_corpus, load_ranked_results

    questions = load_questions(questions_file)
    corpus = load_corpus(corpus_file)
    ranked = load_ranked_results(bm25_ranked_file)

    pairs = []

    max_len = 0

    # Collect all (question, doc) pairs
    for qid, docs in ranked.items():
        for did in docs:
            if qid in questions and did in corpus:
                pairs.append((questions[qid], corpus[did]))

    for qtext, dtext in tqdm(pairs, desc="Computing token lengths"):
        tokens = tokenizer(qtext + dtext)
        if max_len < len(tokens):
            max_len = len(tokens)

    return max_len + 3

def build_collate_fn():
    def collate_fn(batch):
        question_ids = [s["query_id"] for s in batch]
        document_ids = [s["document_id"] for s in batch]

        input_seqs   = [s["input_ids"] for s in batch]
        labels       = [s["label"] for s in batch]

        input_tensor  = torch.tensor(input_seqs, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.float)

        return {
            "input_token_ids": input_tensor,
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
    corpus_map = load_corpus(corpus_file)
    return list(corpus_map.values())

def load_gold_standard(questions_file):
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

def load_ranked_results(ranked_file):
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

def load_corpus(corpus_file):
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

