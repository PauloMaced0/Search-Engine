import re
from typing import List, Optional, Set, Dict
from nltk.stem import SnowballStemmer

class Tokenizer:
    def __init__(
        self,
        min_token_length: int = 3,
        lowercase: bool = True,
        stem: bool = False,
        stopwords: Optional[Set[str]] = None
    ):
        self.token_to_id = {"<PAD>": 0}
        self.vocab_size = 1

        # Config
        self.min_token_length = min_token_length
        self.lowercase = lowercase
        self.stem = stem
        self.stopwords = stopwords if stopwords else set()

        if lowercase:
            self.stopwords = {w.lower() for w in self.stopwords}

        # Stemming
        self.stemmer = SnowballStemmer("english") if stem else None
        self.stem_cache: Dict[str, str] = {}

    def __call__(self, text: str) -> List[int]:
        """Convert text into token IDs"""
        tokens = self.tokenize(text)
        return [self.token_to_id.get(tok, self.token_to_id["<PAD>"]) for tok in tokens]

    def tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()

        word_pattern = re.compile(r"\b\w+\b")
        tokens = word_pattern.findall(text)

        result_tokens = []
        for token in tokens:
            if len(token) < self.min_token_length:
                continue
            if token in self.stopwords:
                continue

            if self.stem:
                if token not in self.stem_cache:
                    self.stem_cache[token] = self.stemmer.stem(token)
                result_tokens.append(self.stem_cache[token])
            else:
                result_tokens.append(token)

        return result_tokens

    def fit(self, texts: List[str]) -> None:
        """Build vocabulary from a list of texts"""
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.vocab_size
                    self.vocab_size += 1
