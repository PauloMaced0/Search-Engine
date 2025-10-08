import random
from torch.utils.data import Dataset
from .utils import _load_gold_standard, _load_ranked_results, _load_corpus, load_questions

class PointWiseDataset(Dataset):
    def __init__(self, 
                 questions_file, 
                 bm25_ranked_file, 
                 corpus_file, 
                 tokenizer, 
                 negative_ratio=2,
                 use_negative_sampling: bool = False
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.negative_ratio = negative_ratio

        self.questions = load_questions(questions_file)
        self.gold_data = _load_gold_standard(questions_file)
        ranked_results = _load_ranked_results(bm25_ranked_file)
        corpus_map = _load_corpus(corpus_file)

        self.data = []

        random.seed(42)

        for qid, ranked_docs in ranked_results.items():
            gold_docs = self.gold_data.get(qid, {}).get("goldstandard_documents", set())

            positives = [doc_id for doc_id in ranked_docs if doc_id in gold_docs]
            negatives = [doc_id for doc_id in ranked_docs if doc_id not in gold_docs]

            # Keep all positives
            for doc_id in positives:
                self.data.append((qid, doc_id, 1.0))

            # Randomly sample negatives at 1:2 ratio
            if positives:
                if use_negative_sampling:
                    num_neg_needed = min(len(negatives), len(positives) * self.negative_ratio)
                    neg_samples = random.sample(negatives, num_neg_needed)
                else:
                    neg_samples = negatives

                for doc_id in neg_samples:
                    self.data.append((qid, doc_id, 0.0))

        # Load corpus only for the needed documents
        self.corpus = {doc_id: corpus_map[doc_id] for _, doc_id, _ in self.data}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qid, docid, label = self.data[idx]
        question_text = self.questions[qid]
        document_text = self.corpus[docid]
        return {
            "query_id": qid,
            "document_id": docid,
            "question_token_ids": self.tokenizer(question_text),
            "document_token_ids": self.tokenizer(document_text),
            "label": label
        }
