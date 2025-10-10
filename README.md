# Information Retrieval System Documentation

## Overview

This Information Retrieval System allows users to index a corpus of documents and perform searches using a BM25 ranking model. The system is designed for processing large datasets efficiently by utilizing a SPIMI (Single Pass In-Memory Indexing) indexing technique.

## Installation
To set up the Information Retrieval System, clone the repository and install the required dependencies.

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

## Downloading Data Files with Git LFS

Some of the large dataset files in the `data/` folder are tracked with **Git LFS**. After cloning this repository for the first time, you need to fetch them explicitly.

#### Required files:
- `MEDLINE_2024_Baseline.jsonl`
- `questions.jsonl`
- `questions_bm25_ranked.jsonl`
- `ranked_questions.jsonl`
- `training_data.jsonl`
- `training_data_bm25_ranked.jsonl`

#### Steps to download:

1. **Install Git LFS** (only once per machine):

```bash
git lfs install
```

2. Pull the LFS Files

```bash 
git lfs pull
```

## Notebooks Overview

This repository contains three main Jupyter notebooks that cover different stages of the information retrieval and reranking pipeline:

### 1. `bm25_analysis.ipynb`
This notebook focuses on **BM25 baseline retrieval**:
- Runs BM25 to retrieve top documents for each query.
- Analyzes the quality of BM25 results.
- Computes evaluation metrics (nDCG) to establish a baseline before reranking.

### 2. `reranking_model_training.ipynb`
This notebook handles **training the reranking model**:
- Prepares training data by combining BM25 results with gold standard labels.
- Implements a **pointwise reranking approach** (positive/negative examples).
- Trains a neural model (CNN-based) to distinguish relevant from non-relevant documents.
- Saves the trained model for later inference.
- Loads the best pretrained model.
- Evaluates reranked results against the baseline (improved nDCG).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
