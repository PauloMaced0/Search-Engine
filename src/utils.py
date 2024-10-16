# Utility functions
def index_document(tokenizer, index, doc_lengths, doc_id, text):
    """Indexes a single document by tokenizing and creating positional postings."""
    tokens = tokenizer.tokenize(text)
    doc_lengths[doc_id] = len(tokens)
    print("pls")

    for pos, term in enumerate(tokens):
        index[term][doc_id].append(pos)
