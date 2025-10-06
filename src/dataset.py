import ujson
from typing import Dict
from torch.utils.data import Dataset
from .utils import _load_gold_standard, _load_ranked_results, load_questions

class PointWiseDataset(Dataset):
    """
    A dataset for (query, document) pairs, with optional relevance labels.

    Each item is a dict:
    {
        "query_id": str,
        "document_id": str,
        "question_token_ids": List[str],
        "document_token_ids": List[str],
        "label": float (if return_label=True)
    }
    """
    def __init__(self, questions_file, bm25_ranked_file, corpus_file, tokenizer, return_label=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.return_label = return_label

        self.questions = load_questions(questions_file)

        # Load gold standard if labels are needed
        self.gold_data = _load_gold_standard(questions_file) if return_label else {}

        # Create a flat list of (question_id, document_id) pairs
        needed_doc_ids = set()
        self.data = []
        for query_id, docs in _load_ranked_results(bm25_ranked_file).items():
            for doc_id in docs:
                needed_doc_ids.add(doc_id)
                self.data.append((query_id, doc_id))

        self.corpus = {}
        with open(corpus_file, 'r') as cf:
            for line in cf:
                doc = ujson.loads(line.strip())
                doc_id, doc_text = doc["doc_id"], doc["text"]
                if doc_id in needed_doc_ids:
                    self.corpus[doc_id] = doc_text

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, str]:
        # Get the query_id and document_id from the stored pairs
        qid, docid = self.data[idx]

        # Retrieve the question text
        question_text = self.questions[qid]
        document_text = self.corpus[docid]

        # Tokenize question and document
        question_token_ids = self.tokenizer(question_text)
        document_token_ids = self.tokenizer(document_text)

        sample = {
            "query_id": qid,
            "document_id": docid,
            "question_token_ids": question_token_ids,
            "document_token_ids": document_token_ids
        }

        if self.return_label:
            gold_docs = self.gold_data.get(qid, {}).get("goldstandard_documents", set())
            label = 1.0 if docid in gold_docs else 0.0
            sample["label"] = label

        return sample

