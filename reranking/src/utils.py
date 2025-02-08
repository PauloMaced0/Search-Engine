import torch
import random
import ujson
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def load_pretrained_embeddings(embedding_path, tokenizer, embedding_dim=300):

    # Initialize embedding matrix
    embedding_matrix = np.random.uniform(-0.05, 0.05, (tokenizer.vocab_size, embedding_dim))
    embedding_matrix[tokenizer.token_to_id["<PAD>"]] = np.zeros(embedding_dim)

    # Create a set of tokens in your vocabulary for quick lookup
    vocab_tokens = set(tokenizer.token_to_id.keys())

    # Load embeddings
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            if word in vocab_tokens:
                vector = np.asarray(values[1:], dtype='float32')
                idx = tokenizer.token_to_id[word]
                embedding_matrix[idx] = vector

    return torch.tensor(embedding_matrix, dtype=torch.float)

def build_collate_fn(tokenizer, max_number_of_question_tokens, max_number_of_document_tokens):
    # Retrieve the pad token ID from the tokenizer
    pad_token_id = tokenizer.token_to_id["<PAD>"]

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

        # Extract each field
        question_ids = [sample["query_id"] for sample in batch]
        document_ids = [sample["document_id"] for sample in batch]

        query_ids = [sample["question_token_ids"] for sample in batch]
        doc_ids = [sample["document_token_ids"] for sample in batch]
        
        # Truncate sequences to max lengths
        query_ids = [q[:max_number_of_question_tokens] for q in query_ids]
        doc_ids = [d[:max_number_of_document_tokens] for d in doc_ids]
        
        # Convert to tensors
        query_tensors = [torch.tensor(q, dtype=torch.long) for q in query_ids]
        doc_tensors = [torch.tensor(d, dtype=torch.long) for d in doc_ids]

        # Pad sequences to the specified max length
        padded_questions = pad_sequence(query_tensors, batch_first=True, padding_value=pad_token_id)
        padded_documents = pad_sequence(doc_tensors, batch_first=True, padding_value=pad_token_id)

        # If sequences are shorter than max length, they are automatically padded by pad_sequence.
        labels = None
        if "label" in batch[0]:
            labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.float)
        
        return {
            "question_token_ids": padded_questions,
            "document_token_ids": padded_documents,
            "query_ids": question_ids,
            "document_ids": document_ids,
            "label": labels
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

        # Positive docs
        positive_doc_ids = [d for d in retrieved_docs if d in gold_docs]
        num_positives = len(positive_doc_ids)
        # Negative docs
        negative_doc_ids = [d for d in retrieved_docs if d not in gold_docs]

        desired_negatives = negative_sample_multiplier * num_positives
        actual_negatives = min(desired_negatives, len(negative_doc_ids))

        # Negative sampling
        negative_doc_ids = negative_doc_ids[:actual_negatives]

        # Collect texts
        for doc_id in positive_doc_ids + negative_doc_ids:
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

def load_questions(questions_file):
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
