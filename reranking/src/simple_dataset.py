import ujson
from torch.utils.data import Dataset
from src.utils import _load_gold_standard

class SimpleDataset(Dataset):
    def __init__(self, questions_file, bm25_ranked_file, corpus_file, tokenizer, return_label=False):
        super().__init__()
        self.k = 0
        self.tokenizer = tokenizer
        self.return_label = return_label
        self.gold_data = {}

        # Load questions
        with open(questions_file, 'r') as qf:
            questions = [ujson.loads(line) for line in qf]

        # Convert questions into a dictionary for easy lookup by query_id
        self.questions_map = {q["query_id"]: q["question"] for q in questions}

        # Load gold standard if labels are needed
        if self.return_label:
            self.gold_data = _load_gold_standard(questions_file)

        # Load BM25 rankings
        with open(bm25_ranked_file, 'r') as bf:
            bm25_rankings = [ujson.loads(line) for line in bf]

        # Create a flat list of (question_id, document_id) pairs
        needed_doc_ids = set()
        self.data = []
        for ranking in bm25_rankings:
            query_id = ranking["query_id"]
            for doc_info in ranking["retrieved_documents"]:
                doc_id = doc_info["id"]
                needed_doc_ids.add(doc_id)
                self.data.append((query_id, doc_id))

        self.corpus_map = {}  
        with open(corpus_file, 'r') as cf:
            for line in cf:
                doc = ujson.loads(line.strip())
                doc_id = doc["doc_id"]
                doc_text = doc["text"]
                if doc_id in needed_doc_ids:
                    self.corpus_map[doc_id] = doc_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the query_id and document_id from the stored pairs
        query_id, document_id = self.data[idx]

        # Retrieve the question text
        question_text = self.questions_map[query_id]
        document_text = self.corpus_map[document_id]

        # Tokenize question and document
        question_token_ids = self.tokenizer(question_text)
        document_token_ids = self.tokenizer(document_text)

        sample = {
            "query_id": query_id,
            "document_id": document_id,
            "question_token_ids": question_token_ids,
            "document_token_ids": document_token_ids
        }

        if self.return_label:
            gold_docs = self.gold_data.get(query_id, {}).get("goldstandard_documents", set())
            label = 1.0 if document_id in gold_docs else 0.0
            sample["label"] = label

        # Return the dictionary as specified
        return sample
