import re

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
        text = self._preprocess_text(text)
        tokens = text.split()
        token_ids = [self.token_to_id.get(token, 0) for token in tokens]

        return token_ids

    def fit(self, texts):
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

    def _preprocess_text(self, text):
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
