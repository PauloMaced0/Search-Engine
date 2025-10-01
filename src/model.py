import re
import torch
import torch.nn as nn
from typing import Dict
import ujson
from torch.utils.data import Dataset
from reranking.src.utils import _load_gold_standard, _load_ranked_results, load_questions

class CNNInteractionBasedModel(nn.Module):
    def __init__(self, vocab_size, pretrained_embeddings=None, embedding_dim=300, num_filters=32, kernel_size=3, dropout=0.3):
        super().__init__()
        # Embedding layer to map token IDs to dense vectors
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Convolutional layer to process interaction matrix
        self.conv = nn.Conv2d(1, out_channels=num_filters, kernel_size=(kernel_size, kernel_size))

        # Activation function
        self.activation = nn.ReLU()

        # Pooling layer
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        self.dropout = nn.Dropout(dropout)

        # Fully connected layer for scoring
        self.fc = nn.Linear(num_filters, 1)

        # Sigmoid for probabilities
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, document):
        """
        Args:
        - query (torch.Tensor): Tensor of query token IDs (batch_size, query_len)
        - document (torch.Tensor): Tensor of document token IDs (batch_size, doc_len)
        
        Returns:
        - prob (torch.Tensor): Relevance score (batch_size,)
        """
        # Embed 
        query_embed = self.embedding(query)  # (B, Q, E)
        document_embed = self.embedding(document)  # (B, D, E)

        # Compute interaction matrix
        # Interaction matrix shape: (B, Q, D)
        interaction_matrix = torch.matmul(query_embed, document_embed.transpose(1, 2))

        # Convolution
        conv_out = self.activation(self.conv(interaction_matrix.unsqueeze(1)))  # (B, F, h, w)
        pooled = self.pool(conv_out).squeeze(-1).squeeze(-1)  # (B, F)

        # Dropout + FC
        logits = self.fc(self.dropout(pooled)).squeeze(-1)  # (B,)

        return logits

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

class Tokenizer:
    def __init__(self):
        self.token_to_id = {"<PAD>": 0}
        self.vocab_size = 1

    def __call__(self, text):
        """
        tokenizes the input text(s) into a sequence of token ids based on the vocabulary.
        if 'text' is a list of strings, it will combine them and tokenize the entire sequence.
        """
        if isinstance(text, list):
            text = " ".join(t.lower() for t in text)
        else:
            text = text.lower()
        tokens = self._preprocess_text(text).split()
        return [self.token_to_id.get(tok, self.token_to_id["<PAD>"]) for tok in tokens]

    def fit(self, texts) -> None:
        """
        fits the tokenizer on a list of texts, building a vocabulary of the most frequent words.
        """
        all_tokens = []
        for text in texts:
            text = text.lower()
            text = self._preprocess_text(text)
            text = text.split()
            all_tokens.extend(text)
        
        tokens_set = set(all_tokens)

        for token in tokens_set:
            if token not in self.token_to_id:
                self.token_to_id[token] = self.vocab_size
                self.vocab_size += 1

    def _preprocess_text(self, text) -> str:
        """
        removes punctuation and normalizes whitespace in the text.
        """
        # remove punctuation using regex
        text = re.sub(r'[^\w\s]', '', text)
        # replace newline and tab characters with a space
        text = re.sub(r'[\n\t]', ' ', text)
        # replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
