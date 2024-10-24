# Information Retrieval System Documentation

## Overview

This Information Retrieval System allows users to index a corpus of documents and perform searches using a BM25 ranking model. The system is designed for processing large datasets efficiently by utilizing a SPIMI (Single Pass In-Memory Indexing) indexing technique.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Indexing](#indexing)
  - [Searching](#searching)
- [Classes](#classes)
  - [CorpusReader](#corpusreader)
  - [SPIMIIndexer](#spimiindexer)
  - [Tokenizer](#tokenizer)
  - [BM25Searcher](#bm25searcher)
- [Dependencies](#dependencies)
- [License](#license)

## Features

- **Indexing**: Efficiently indexes documents in batches, allowing for scalability.
- **Searching**: Implements the BM25 ranking model for retrieving relevant documents based on user queries.
- **Tokenizer**: Supports tokenization with options for case normalization, stopword removal, and stemming.

## Installation

To set up the Information Retrieval System, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-folder>
./setup.sh
```

Make sure to have Python 3.x installed on your machine.

## Usage

The system can be run from the command line. Use the following command structure:

```bash
python main.py --index|--search [options]
```

### Command Line Arguments

| Argument               | Type    | Default                        | Description                                                                                  |
|------------------------|---------|--------------------------------|----------------------------------------------------------------------------------------------|
| `--index`              | flag    | -                              | Run the indexer to index the documents in the specified corpus.                            |
| `--search`             | flag    | -                              | Run the searcher to query the indexed documents.                                            |
| `--query`              | string  | -                              | Query for searching documents (required if `--search` is selected).                        |
| `--corpus`             | string  | `data/MEDLINE_2024_Baseline.jsonl` | Path to the corpus file containing documents to index.                                      |
| `--output_dir`         | string  | `output`                       | Directory to store partial and merged indexes.                                             |
| `--min_token_length`   | int     | 3                              | Minimum token length for tokenization.                                                     |
| `--lowercase`          | bool    | True                           | Normalize tokens to lowercase if set to True.                                              |
| `--stopwords`          | string  | -                              | Path to a stopwords file (optional).                                                       |
| `--stemming`           | bool    | False                          | Apply stemming to tokens if set to True.                                                   |
| `--k1`                 | float   | 1.2                            | BM25 parameter k1 (controls term frequency saturation).                                     |
| `--b`                  | float   | 0.75                           | BM25 parameter b (controls document length normalization).                                  |
| `--batch_size`         | int     | 10000                          | Number of documents per batch for indexing.                                                |
| `--load_index`         | string  | -                              | Path to a pre-built index file to load for searching.                                      |

### Indexing 

To index a corpus, run the following command:

```bash
python main.py --index --corpus path/to/corpus.jsonl --output_dir path/to/output
```

### Searching 

To search using a query, run the following command:

```bash
python main.py --search --query "your search query" --load_index path/to/index.msgpack
```

## Classes 

### Corpus Reader

The `CorpusReader` class is responsible for reading documents from the specified corpus file.

#### Methods

- **`__init__(self, file_path)`**: Initializes the CorpusReader with the given file path.
- **`read_documents(self)`**: Reads documents from the corpus and yields them one by one.

### SPIMIIndexer

The `SPIMIIndexer` class handles the indexing of documents using the SPIMI technique.

#### Methods

- **`__init__(self, tokenizer, output_dir, merge_index_dir, batch_size=10000)`**: Initializes the indexer with tokenizer and output directory settings.
- **`index_document(self, doc_id, text)`**: Indexes a single document by tokenizing and creating positional postings.
- **`write_partial_index(self)`**: Writes the current in-memory index and document lengths to disk.
- **`merge_partial_indexes(self, num_batches)`**: Merges all partial indexes into a single coherent inverted index.
- **`process_corpus(self, corpus_reader)`**: Processes the entire corpus in batches of documents.

### Tokenizer

The `Tokenizer` class is responsible for tokenizing text and normalizing tokens based on user-defined settings.

#### Methods

- **`__init__(self, min_token_length=3, lowercase=True, remove_stopwords=False, stem=False, stopwords=None)`**: Initializes the tokenizer with specified configurations.
- **`tokenize(self, text)`**: Tokenizes the input text into a list of tokens, applying normalization and filtering.

### BM25Searcher

The `BM25Searcher` class is responsible for searching documents based on the BM25 ranking model.

The `BM25` retrieval function calculates a relevance score for each document based on a specific search query.

The algorithm looks at three things:

1. How often do the query terms appear in the document.
2. The length of the document.
3. The average length of all documents in the collection.

> **Note**: The `BM25Searcher` class is mentioned in the code but doesn't work as expected. Yet to be completed and tested for the searching functionality.
"""

## Dependencies 

- Python 3.x
- **ujson**: For fast JSON reading.
- **msgpack**: For binary serialization of data.
- **nltk**: For natural language processing tasks like stemming.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
