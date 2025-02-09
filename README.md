# Information Retrieval System Documentation

## Overview

This Information Retrieval System allows users to index a corpus of documents and perform searches using a BM25 ranking model. The system is designed for processing large datasets efficiently by utilizing a SPIMI (Single Pass In-Memory Indexing) indexing technique.

## Installation
To set up the Information Retrieval System, clone the repository and install the required dependencies for each subdirectory (`indexing_searching` and `reranking`):

```bash
git clone <repository-url>
cd <repository-folder>
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

Make sure to have Python 3.x installed on your machine.
> [!NOTE]
> If you are working with the **reranking** model download [GloVe Pretrained Word Embeddings](https://nlp.stanford.edu/projects/glove/) (`Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download): glove.42B.300d.zip`).
> After being downloaded, unzip it on the `data` folder.

## Computing Ranking Metrics 

The system includes a script to compute the **Normalized Discounted Cumulative Gain (nDCG)** metric, which evaluates the quality of the ranked retrieval results. For this manner, execute the `nDCG.py` script.

#### How nDCG Works

- **DCG (Discounted Cumulative Gain)**: Measures the gain (relevance) of each document in the result list, discounted by its position in the list.
- **IDCG (Ideal DCG)**: The maximum possible DCG achievable, obtained by an ideal ranking of documents.
- **nDCG**: The ratio of DCG to IDCG, normalized to a value between 0 and 1.

**Command Structure**:
```bash
python3 nDCG.py [options]
```

### Command Line Arguments

| Argument               | Type    | Default                        | Description                                                                                  |
|------------------------|---------|--------------------------------|----------------------------------------------------------------------------------------------|
| `--questions_file`              | string    | `data/questions.jsonl`                              | Path to the questions file (JSONL).                            |
| `--results_file`             | string | `output/ranked_questions.jsonl`                              | Path to the ranked results file (JSONL).                                          |
| `--k`              | int | 10                              | Rank cutoff for nDCG computation.                        |

#### BM25 Sample Output
**When you run**:
```bash
python3 cli.py --index --lowercase True --stemming True
python3 cli.py --search --questions_file data/questions.jsonl
python3 nDCG.py
```

**Output:**
```graphql
Query ID: 63f73f1b33942b094c000008, nDCG@10: 1.0000
Query ID: 643d41e757b1c7a315000037, nDCG@10: 0.6296
Query ID: 643c88a257b1c7a315000030, nDCG@10: 0.1637
...

Average nDCG@10: 0.5722
```

#### Reraking Sample Output
**When you run**:
```bash
python3 cli.py --model_checkpoint ../output/<your_trained_cnn_model>.pt
python3 nDCG.py --results_file output/final_ranked_questions.jsonl
```

**Output:**
```graphql
Query ID: 63f73f1b33942b094c000008, nDCG@10: 0.0000
Query ID: 643d41e757b1c7a315000037, nDCG@10: 0.2027
Query ID: 643c88a257b1c7a315000030, nDCG@10: 0.0000
...
Average nDCG@10: 0.1115
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
