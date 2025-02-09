# Information Retrieval System Documentation

## Overview

This Information Retrieval System allows users to index a corpus of documents and perform searches using a BM25 ranking model. The system is designed for processing large datasets efficiently by utilizing a SPIMI (Single Pass In-Memory Indexing) indexing technique.

# Table of Contents

- [Features](#features)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
- [Indexer Engine](#indexer-engine)
  - [Description of the Tokenizer](#description-of-the-tokenizer)
  - [Optimization Techniques in Indexing](#optimization-techniques-in-indexing)
  - [Index File Format](#index-file-format)
  - [SPIMI Algorithm Implementation](#spimi-algorithm-implementation)
  - [Total Indexing Time](#total-indexing-time)
  - [Missing Features in Indexing](#missing-features-in-indexing)
- [Search Engine](#search-engine)
  - [Implemented Search Algorithms](#implemented-search-algorithms)
  - [Optimization Techniques in Searching](#optimization-techniques-in-searching)
  - [Average Query Processing Time](#average-query-processing-time)
  - [Missing Features in Searching](#missing-features-in-searching)
- [Conclusion](#conclusion)

## Features

- **Tokenizer**: Supports tokenization with options for case normalization, stopword removal, and stemming.
- **Indexing**: Efficiently indexes documents in batches, allowing for scalability.
- **Searching**: Implements the BM25 ranking model for retrieving relevant documents based on user queries.

## Usage

The system can be run from the command line. Use the following command structure:

```bash
python cli.py --index|--search [options]
```

### Command Line Arguments

To showcase the Command Line Interface, run:

```bash
python cli.py --help
```

| Argument               | Type    | Default                        | Description                                                                                  |
|------------------------|---------|--------------------------------|----------------------------------------------------------------------------------------------|
| `--index`              | flag    | -                              | Run the indexer to index the documents in the specified corpus.                            |
| `--search`             | flag    | -                              | Run the searcher to query the indexed documents.                                            |
| `--query`              | string  | -                              | Query for searching documents (required if `--search` is selected).                        |
| `--corpus`             | string  | `../data/MEDLINE_2024_Baseline.jsonl` | Path to the corpus file containing documents to index.                                      |
| `--output_dir`         | string  | `../output`                       | Directory to store partial and merged indexes.                                             |
| `--min_token_length`   | int     | 3                              | Minimum token length for tokenization.                                                     |
| `--lowercase`          | bool    | True                           | Normalize tokens to lowercase if set to True.                                              |
| `--stopwords`          | string  | -                              | String with stopwords (space separated).                                                       |
| `--stemming`           | bool    | False                          | Apply stemming to tokens if set to True.                                                   |
| `--k1`                 | float   | 1.2                            | BM25 parameter k1 (controls term frequency saturation).                                     |
| `--b`                  | float   | 0.75                           | BM25 parameter b (controls document length normalization).                                  |
| `--batch_size`         | int     | 10000                          | Number of documents per batch for indexing.                                                |
| `--load_index`         | string  | -                              | Path to a pre-built index file to load for searching.                                      |

> [!NOTE]
> See `entrypoint.sh` script to see some examples on how to execute this script.

## Indexer Engine

### Description of the Tokenizer

The `Tokenizer` class processes text into tokens suitable for indexing and searching. It includes the following options:

- **Case Normalization**: Converts all tokens to lowercase if `lowercase=True`.
- **Minimum Token Length**: Discards tokens shorter than `min_token_length` (default is 3).
- **Stopword Removal**: Removes common stopwords if a stopword list is provided.
- **Stemming**: Reduces tokens to their stem forms using the Snowball Stemmer if `stem=True`.

### Optimization Techniques in Indexing

- **SPIMI Algorithm**: Implements the Single-Pass In-Memory Indexing algorithm to efficiently handle large datasets by writing partial indexes to disk.
- **Batch Processing**: Indexes documents in batches (`batch_size=10000`) to manage memory usage.
- **Precompiled Regular Expressions**: Uses precompiled regex patterns in the tokenizer to improve tokenization speed.
- **Stemming Cache**: Caches stemmed tokens to avoid redundant computations during tokenization.
- **MessagePack Serialization**: Uses `MessagePack` for efficient binary serialization when writing partial and merged indexes to disk.

### Index File Format
- **Format**: The index is stored as a MessagePack (`.msgpack`) file.

- **Structure**:
  - **Index**: A dictionary where keys are terms and values are dictionaries mapping document IDs to lists of positions where the term occurs.
  - **Document Lengths**: A dictionary mapping document IDs to the total number of tokens in each document.

The (`.msgpack`) file is a binary format, making it impossible to provide a screenshot of its contents. 
However, a sample from the file is shown below:

```json
{
  "index": {
    "ethylhexyl": {
      "25153068": [7, 49],
      "32745991": [3, 26],
      "12437285": [8, 10],
      "15924484": [5, 27],
      "19555962": [150],
      "12270607": [8, 29],
      "23356645": [3, 22],
      "22041199": [9, 17],
      "8333024": [3, 25],
      "20453712": [14, 22],
      "30960727": [97],
      "37536456": [13],
      "14687758": [99],
      "35843048": [21],
      "16956469": [22],
      "34788783": [1, 39, 50, 52, 83, 117, 265, 281],
      "14998748": [14, 17, 28],
      "32610232": [0, 48],
      "14556481": [32],
      "35859238": [35],
      "28661659": [84, 92],
      "31033968": [12],
        .
        .
        .
    },
        .
        .
        .
  },
  "doc_lengths": {
    "2451706": 115,
    "35308048": 192,
    "7660250": 51,
    "28963802": 143,
    "25153068": 231,
    "874026": 101,
    "4001859": 137,
    "10149271": 92,
    "35267334": 190,
    "3656477": 128,
    "30818862": 217,
        .
        .
        .
  }
}
```

### SPIMI Algorithm Implementation

- **Memory Constraints Handling**: The SPIMIIndexer processes documents in batches and writes partial indexes to disk once the batch size is reached, thus preventing memory overflow.
- **Merging Partial Indexes**: After all partial indexes are written, they are merged into a single index file. The merging process combines term postings from all partial indexes.
- **Efficiency**: By handling indexing in this way, the system can index large corpora without requiring large amounts of RAM.

### Total Indexing Time

- **Total Indexing Time**: The total time to index the collection depends on the corpus size and system specifications.
  - **Example**: Indexing the `MEDLINE_2024_Baseline.jsonl` corpus (500 thousand documents) took approximately 133 seconds (with stemming and lowercase normalization).
- **Configurations Tested**:
  - **Without Stemming**: Faster indexing time but less effective retrieval.
  - **With Stemming**: Slightly longer indexing time due to stemming computation but can improve recall in searches.

### Missing Features in Indexing
- **Positional Indexing Limitations**: Does not support proximity searches.
- **No Compression**: Index files are not compressed, leading to larger disk space usage.

## Search Engine

### Implemented Search Algorithms

- **BM25 Ranking Model**: Implements the Okapi BM25 algorithm to rank documents based on query relevance.
- **Parameters**:
  - **k1**: Controls term frequency saturation (default 1.2).
  - **b**: Controls document length normalization (default 0.75).

### Optimization Techniques in Searching

- **Candidate Document Selection**: Only documents containing query terms are considered for scoring.
- **IDF Caching**: Precomputes and caches IDF values for terms to avoid redundant calculations.
- **Efficient Data Structures**: Uses dictionaries and sets for fast lookups and operations.
- **Preprocessing Query Tokens**: Applies the same tokenization process to queries as to documents, ensuring consistency.

### Average Query Processing Time

- **Average Time**: Processing all 100 questions from `questions.jsonl` file takes approximately 135 seconds on average.

### Missing Features in Searching
- **Advanced Query Support**: Does not support Boolean operators and wildcard searches.
- **No Query Expansion**: Does not implement techniques like synonym expansion or spelling correction.

## Conclusion

### Strengths

- **Efficient Indexing**: Utilizes the SPIMI algorithm to handle large datasets efficiently.
- **Customizable Tokenizer**: Offers options for case normalization, stopword removal, and stemming.
- **Effective Retrieval**: Implements the BM25 ranking model, which is widely regarded as effective for information retrieval tasks.

### Weaknesses

- **Limited query capabilities**: does not support advanced query features like boolean operators or proximity searches.
- **Lack of parallelization**: the search algorithm could be optimized further using parallel processing.

### Areas for Improvement

- **Implement Advanced Query Features**: Adding support for phrase queries and Boolean operators.
- **Optimize Search Performance**: Implementing parallel processing or more efficient candidate selection strategies.
- **Enhance Ranking Model**: Exploring BM25 variants or improve the current approach (IDF metric and document scoring).

### Key Challenges Encountered

- **Memory Management**: Handling memory constraints during indexing required careful batch size management and optimization.
- **Balancing Precision and Recall**: Deciding whether to use stemming and stopword removal involved trade-offs between precision and recall.
- **Performance vs. Functionality**: Optimizing for speed sometimes conflicted with adding new features, requiring prioritization.
