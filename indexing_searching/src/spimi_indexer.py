import os
import msgpack
from collections import defaultdict

class SPIMIIndexer:
    def __init__(self, tokenizer, output_dir, merge_index_dir, batch_size=10000):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.merged_index_file = merge_index_dir
        self.index = defaultdict(lambda: defaultdict(list))
        self.doc_lengths = {}
        self.current_batch = 0
        self.current_doc_id = 0

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def index_document(self, doc_id, text):
        """Indexes a single document by tokenizing and creating positional postings."""
        tokens = self.tokenizer.tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)

        for pos, term in enumerate(tokens):
            self.index[term][doc_id].append(pos)

    def write_partial_index(self):
        """Writes the current in-memory index and document lengths to disk, including tokenizer configuration."""
        file_path = os.path.join(self.output_dir, f'partial_index_{self.current_batch}.msgpack')

        with open(file_path, 'wb') as f:
            msgpack.pack({'index': self.index, 'doc_lengths': self.doc_lengths}, f)

        self.index.clear()  # Clear memory after writing
        self.doc_lengths.clear() # Clear memory after writing
        self.current_batch += 1

    def merge_partial_indexes(self, num_batches):
        """Merge all partial indexes into a single coherent inverted index."""

        # Create a buffer to write the final merged index in chunks
        merged_data = defaultdict(lambda: defaultdict(list))
        doc_lengths = {}

        # Iterate through partial index files in batches of 10
        for batch_num in range(num_batches):
            partial_index_file = os.path.join(self.output_dir, f'partial_index_{batch_num}.msgpack')

            print(f"Merging partial index: {partial_index_file}")

            # Load the partial index using MessagePack (binary mode)
            with open(partial_index_file, 'rb') as f:
                unpacker = msgpack.Unpacker(f, raw=False)
                partial_index = next(unpacker)

            for word, doc_data in partial_index['index'].items():
                for doc_id, positions in doc_data.items():
                    merged_data[word][doc_id].extend(positions)

            doc_lengths.update(partial_index['doc_lengths'])

        with open(self.merged_index_file, 'wb') as f:
            msgpack.pack({'index': dict(merged_data), 'doc_lengths': doc_lengths}, f)

    def process_corpus(self, corpus_reader):
        """Processes the entire corpus in batches of documents."""
        for doc in corpus_reader.read_documents():
            doc_id = doc['doc_id'].split(':')[1]
            self.index_document(doc_id, doc['text'])
            self.current_doc_id += 1

            if self.current_doc_id % self.batch_size == 0:
                self.write_partial_index()

        # Write any remaining index to disk
        if self.index:
            self.write_partial_index()

        config = {
            'tokenizer_config': {
                'min_token_length': self.tokenizer.min_token_length,
                'lowercase': self.tokenizer.lowercase,
                'stopwords': list(self.tokenizer.stopwords) if self.tokenizer.stopwords else None,
                'stem': self.tokenizer.stem
            },
        }

        file_path = os.path.join(self.output_dir, f'tokenizer_config.msgpack')

        with open(file_path, 'wb') as f:
            msgpack.pack(config, f)

        self.merge_partial_indexes(self.current_batch)
