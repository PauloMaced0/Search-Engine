import re
from nltk.stem import PorterStemmer

class Tokenizer:
    def __init__(self, min_token_length=3, lowercase=True, remove_stopwords=False, stem=False, stopwords=None):
        self.min_token_length = min_token_length
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.stopwords = stopwords or set()
        self.stemmer = PorterStemmer()

    def tokenize(self, text):
        """Tokenize the text into a list of tokens."""
        if self.lowercase:
            text = text.lower()
        
        # Tokenize by non-word characters (e.g., whitespace, punctuation)
        tokens = re.findall(r'\w+', text)

        # Filter by token length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        # Apply stemming
        if self.stem:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return tokens
