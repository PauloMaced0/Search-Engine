import re
from nltk.stem import SnowballStemmer

class Tokenizer:
    def __init__(self, min_token_length=3, lowercase=True, stem=False, stopwords=None):
        self.min_token_length = min_token_length
        self.lowercase = lowercase
        self.stem = stem
        self.stopwords = stopwords or set()
        if self.lowercase:
            self.stopwords = {word.lower() for word in self.stopwords}
        self.stemmer = SnowballStemmer('english')
        self.stem_cache = {}  # Cache for stemmed words
        self.word_pattern = re.compile(r'\b\w+\b')  # Precompile regex

    def stem_token(self, token):
        """Return the stemmed version of the token using caching."""
        if token in self.stem_cache:
            return self.stem_cache[token]
        stemmed_token = self.stemmer.stem(token)
        self.stem_cache[token] = stemmed_token
        return stemmed_token

    def tokenize(self, text):
        """Tokenize the text into a list of tokens."""
        if self.lowercase:
            text = text.lower()

        # Tokenize by non-word characters (e.g., whitespace, punctuation)
        tokens = self.word_pattern.findall(text)

        result_tokens = []

        # Process tokens in a single loop
        for token in tokens:
            if len(token) < self.min_token_length:
                continue  # Skip short tokens
            if self.stopwords and token in self.stopwords:
                continue  # Skip stopwords
            if self.stem:
                token = self.stem_token(token)  # Stem the token using cache
            result_tokens.append(token)

        return result_tokens

