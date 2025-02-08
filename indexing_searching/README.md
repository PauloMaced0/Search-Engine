# Information Retrieval System Documentation

## Overview

This Information Retrieval System allows users to index a corpus of documents and perform searches using a BM25 ranking model. The system is designed for processing large datasets efficiently by utilizing a SPIMI (Single Pass In-Memory Indexing) indexing technique.

# Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Indexing](#indexing)
- [Configuration Instructions for Executing the Code](#configuration-instructions-for-executing-the-code)
- [Indexer Engine](#indexer-engine)
  - [Description of the Tokenizer](#description-of-the-tokenizer)
  - [Optimization Techniques in Indexing](#optimization-techniques-in-indexing)
  - [Index File Format](#index-file-format)
  - [SPIMI Algorithm Implementation](#spimi-algorithm-implementation)
  - [Total Indexing Time](#total-indexing-time)
  - [Missing Features in Indexing](#missing-features-in-indexing)
  - [Comparisons and Design Decisions](#comparisons-and-design-decisions)
- [Searching](#searching)
  - [Search Engine](#search-engine)
  - [Implemented Search Algorithms](#implemented-search-algorithms)
  - [Optimization Techniques in Searching](#optimization-techniques-in-searching)
  - [Average Query Processing Time](#average-query-processing-time)
  - [Missing Features in Searching](#missing-features-in-searching)
  - [Comparisons and Experimental Configurations](#comparisons-and-experimental-configurations)
- [Ranking Metrics](#ranking-metrics)
- [Additional Information](#additional-information)
- [Conclusion](#conclusion)
- [Classes](#classes)
  - [CorpusReader](#corpusreader)
  - [SPIMIIndexer](#spimiindexer)
  - [Tokenizer](#tokenizer)
  - [BM25Searcher](#bm25searcher)
- [Dependencies](#dependencies)

## Features

- **Indexing**: Efficiently indexes documents in batches, allowing for scalability.
- **Searching**: Implements the BM25 ranking model for retrieving relevant documents based on user queries.
- **Tokenizer**: Supports tokenization with options for case normalization, stopword removal, and stemming.

## Usage

The system can be run from the command line. Use the following command structure:

```bash
python cli.py --index|--search [options]
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
| `--stopwords`          | string  | -                              | String with stopwords (space separated).                                                       |
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

To search using a list of queries from a file:

```bash
python cli.py --search --questions_file data/questions.jsonl --load_index path/to/merged_index.msgpack
```

### Configuration Instructions for Executing the Code

Ensure you have Python 3.x installed along with the required dependencies listed in the [Dependencies](#dependencies) section.

1. **Indexing**:

```bash
python cli.py --search --questions_file data/questions.jsonl --load_index path/to/merged_index.msgpack
```
2. **Searching**:

```bash
python cli.py --search --query "your search query" --load_index output/merged_index.msgpack
```

Or, to evaluate the system using provided questions:

```bash
python cli.py --search --questions_file data/questions.jsonl --load_index output/merged_index.msgpack
```

4. **Computing Ranking Metrics**:

```bash
python compute_ndcg.py --questions_file data/questions.jsonl --results_file output/ranked_results.jsonl --k 10
```

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
      "30063357": [89],
      "3621371": [118],
      "18639561": [107],
      "32163735": [134],
      "33906035": [27],
      "26484116": [53],
      "4068057": [8, 15, 17],
      "7683943": [23],
      "24245247": [18],
      "28852705": [177],
      "34902399": [63],
      "21932059": [34],
      "37257576": [4, 10],
      "20050724": [29],
      "25058421": [48],
      "10515691": [106],
      "24852702": [132],
      "16332407": [3, 6, 39],
      "26224951": [143],
      "30592400": [50, 56],
      "11312650": [1, 18, 25],
      "24448002": [3, 11],
      "19382104": [12, 41],
      "7757964": [19],
      "31995769": [147],
      "19800281": [52],
      "26628349": [131],
      "5254": [5, 14, 85, 105],
      "33928763": [86, 94],
      "28321989": [143],
      "3131963": [44, 48],
      "29259487": [64, 68],
      "37806616": [35],
      "33659001": [1, 15, 34],
      "1940432": [45],
      "32325186": [13],
      "36263449": [3, 13, 120],
      "20356189": [29],
      "22646993": [44],
      "29377175": [4, 17],
      "34619260": [107],
      "6626564": [35, 38],
      "2346479": [52],
      "26290974": [96],
      "36904310": [37, 46],
      "19912166": [7, 25],
      "25827748": [0, 13],
      "34472344": [112],
      "25461741": [26, 30],
      "24848799": [94],
      "31678724": [97],
      "16246397": [33],
      "21915957": [5, 23],
      "17583543": [9, 13, 26, 34],
      "17502135": [150],
      "36648757": [48],
      "24761783": [160],
      "24349678": [127],
      "20855102": [40, 120],
      "12375655": [94],
      "35917506": [8, 23],
      "24641848": [3, 18],
      "27279583": [104],
      "25149981": [8, 85],
      "2172174": [21, 25],
      "2364944": [96],
      "20943228": [22],
      "32063323": [88],
      "20493477": [7, 20],
      "26268770": [79],
      "1580376": [93],
      "1591274": [62]
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

## Ranking Metrics

The system includes a script to compute the **Normalized Discounted Cumulative Gain (nDCG)** metric, which evaluates the quality of the ranked retrieval results.

#### How nDCG Works

- **DCG (Discounted Cumulative Gain)**: Measures the gain (relevance) of each document in the result list, discounted by its position in the list.
- **IDCG (Ideal DCG)**: The maximum possible DCG achievable, obtained by an ideal ranking of documents.
- **nDCG**: The ratio of DCG to IDCG, normalized to a value between 0 and 1.

#### Script Usage
The nDCG@10 scores were computed using the provided 100 questions:

```bash
python compute_ndcg.py --questions_file path/to/questions.jsonl --results_file path/to/ranked_results.jsonl --k 10
```

> **Note**: When you run `python3 cli.py --index --lowercase True --stemming True && python3 cli.py --search --questions_file data/questions.jsonl && python nDCG.py`, you may see an output similar to the following:
>
> ```
> Query ID: 63f73f1b33942b094c000008, nDCG@10: 1.0000
> Query ID: 643d41e757b1c7a315000037, nDCG@10: 0.6296
> Query ID: 643c88a257b1c7a315000030, nDCG@10: 0.1637
> ...
>
> Average nDCG@10: 0.5722
> ```

- **Interpretation**: The system performs well on some queries but has room for improvement overall.

## Additional Information

### Command Line Interface Help Menu

To showcase the Command Line Interface, run:

```bash
python cli.py --help
```

**Output**:

```bash
usage: cli.py [-h] (--index | --search) [--corpus CORPUS]
              [--output_dir OUTPUT_DIR] [--min_token_length MIN_TOKEN_LENGTH]
              [--lowercase LOWERCASE] [--stopwords STOPWORDS]
              [--stemming STEMMING] [--k1 K1] [--b B] [--batch_size BATCH_SIZE]
              [--load_index LOAD_INDEX] [--num_results NUM_RESULTS]
              [--query QUERY | --questions_file QUESTIONS_FILE | --interactive]

Information Retrieval System

options:
  -h, --help            show this help message and exit
  --index               Run the indexer
  --search              Run the searcher
  --corpus CORPUS       Path to the corpus file
  --output_dir OUTPUT_DIR
                        Directory to store partial and merged indexes
  --min_token_length MIN_TOKEN_LENGTH
                        Minimum token length for tokenization
  --lowercase LOWERCASE
                        Normalize tokens to lowercase
  --stopwords STOPWORDS
                        String with stopwords (space split)
  --stemming STEMMING   Whether to apply stemming to tokens
  --k1 K1               BM25 parameter k1 (controls term frequency saturation)
  --b B                 BM25 parameter b (controls document length normalization)
  --batch_size BATCH_SIZE
                        Number of documents per batch for indexing
  --load_index LOAD_INDEX
                        Path to pre-built index file to load for searching
  --num_results NUM_RESULTS
                        Number of results to return for the query
  --query QUERY         Query for searching documents
  --questions_file QUESTIONS_FILE
                        Path to the questions file (JSONL) for batch processing
  --interactive         Run in interactive mode for manual query input
```

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
- **`stem_token(self, token)`**: Returns the stemmed version of the token using caching.

### BM25Searcher

The `BM25Searcher` class is responsible for searching documents based on the BM25 ranking model.

The `BM25` retrieval function calculates a relevance score for each document based on a specific search query.

The algorithm looks at three things:

1. **Term Frequency (TF)**: How often the query terms appear in the document.
2. **Document Length**: The length of the document compared to the average document length.
3. **Inverse Document Frequency (IDF)**: How common or rare the query terms are across all documents.

#### Methods

- **`__init__(self, index_path, tokenizer_config_path, k1=1.2, b=0.75)`**: Initializes the searcher with paths to the index and tokenizer configuration, along with BM25 parameters.
- **`load_index(self, index_path)`**: Loads the index from a Msgpack file.
- **`load_tokenizer(self, tokenizer_config_path)`**: Loads the tokenizer configuration from a Msgpack file.
- **`calculate_average_doc_length(self)`**: Calculates the average document length.
- **`compute_idf(self)`**: Computes the IDF for each term in the index and caches it.
- **`calculate_bm25(self, query_tokens, doc_id)`**: Computes the BM25 score for a single document given a query.
- **`search(self, query, n_scores=100)`**: Searches for the query and returns the top-ranked documents.

## Dependencies 

- Python 3.x
- **ujson**: For fast JSON reading.
- **msgpack**: For binary serialization of data.
- **nltk**: For natural language processing tasks like stemming.
