import random
import torch
from torch.utils.data import Dataset
from .utils import load_corpus, load_questions, load_gold_standard, load_ranked_results 

class PointWiseDataset(Dataset):
    def __init__(
        self,
        questions_file,
        bm25_ranked_file,
        corpus_file,
        tokenizer,
        mode: str = "train",  # "train" or "valid"
        negative_ratio: int = 2,
        include_random_negatives: bool = True,
        random_negatives_ratio: float = 0.25,  # fraction of negatives that are random
    ):
        """
        Flexible dataset for (query, document, label) training and validation.

        Args:
            questions_file: JSONL with queries and goldstandard_documents
            bm25_ranked_file: BM25-ranked docs for each query
            corpus_file: JSONL with {"doc_id": ..., "text": ...}
            tokenizer: tokenizer instance
            split: "train" or "valid"
            negative_ratio: number of negatives per positive (for training)
            include_random_negatives: add random negatives (for training)
            random_negatives_ratio: fraction of negatives that are random
        """
        super().__init__()
        assert mode in {"train", "valid"}, "mode must be 'train' or 'valid'"
        self.tokenizer = tokenizer
        self.split = mode
        self.negative_ratio = negative_ratio
        self.include_random_negatives = include_random_negatives
        self.random_negatives_ratio = random_negatives_ratio

        random.seed(42)

        # Load data
        self.questions = load_questions(questions_file)
        self.gold_data = load_gold_standard(questions_file)
        ranked_results = load_ranked_results(bm25_ranked_file)
        corpus_map = load_corpus(corpus_file)
        all_doc_ids = list(corpus_map.keys())

        self.data = []

        for qid, ranked_docs in ranked_results.items():
            gold_docs = self.gold_data.get(qid, {}).get("goldstandard_documents", set())
            if not gold_docs:
                continue  # skip queries without labels

            positives = [d for d in ranked_docs if d in gold_docs]
            negatives = [d for d in ranked_docs if d not in gold_docs]

            # Always include all positives
            for doc_id in positives:
                self.data.append((qid, doc_id, 1.0))

            if mode == "train":
                # --- TRAINING MODE: use sampled negatives
                if negatives and positives:
                    num_hard_needed = int(len(positives) * negative_ratio * (1 - random_negatives_ratio))
                    hard_samples = random.sample(negatives, min(len(negatives), num_hard_needed))

                    random_negatives = []
                    if include_random_negatives:
                        excluded_docs = set(ranked_docs)
                        available_docs = [d for d in all_doc_ids if d not in excluded_docs]
                        num_easy_needed = int(len(positives) * negative_ratio * random_negatives_ratio)
                        random_negatives = random.sample(available_docs, min(len(available_docs), num_easy_needed))

                    for doc_id in hard_samples + random_negatives:
                        self.data.append((qid, doc_id, 0.0))

            elif mode == "valid":
                # --- VALIDATION MODE: include all negatives from BM25 (no sampling)
                for doc_id in negatives:
                    self.data.append((qid, doc_id, 0.0))

        # Load only needed docs
        needed_doc_ids = {doc_id for _, doc_id, _ in self.data}
        self.corpus = {doc_id: corpus_map[doc_id] for doc_id in needed_doc_ids}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qid, docid, label = self.data[idx]
        question_text = self.questions[qid]
        document_text = self.corpus[docid]

        # Concatenate question and document
        encoding = self.tokenizer(
            question_text,
            document_text,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        )

        return {
            "query_id": qid,
            "document_id": docid,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": label,
        }
