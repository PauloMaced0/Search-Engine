# Neural Reranker and Evaluation System Documentation

## Overview

The Neural Reranker and Evaluation System enhances the baseline Information Retrieval (IR) system by introducing a CNN-based neural reranker model to improve the relevance of retrieved documents. Additionally, it provides tools to evaluate the system's performance using the Normalized Discounted Cumulative Gain (nDCG) metric.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
- [Trained Models](#trained-models)
  - [Model 1: Positive-Negative Sample Ratio 1:1](#model-1-positive-negative-sample-ratio-11)
  - [Model 2: Positive-Negative Sample Ratio 1:2](#model-2-positive-negative-sample-ratio-12)
  - [Results and Observations](#results-and-observations)
- [Components](#components)
  - [CNNInteractionBasedModel](#cnninteractionbasedmodel)
  - [SimpleDataset](#simpledataset)
  - [Tokenizer](#tokenizer)
- [Enhancements](#enhancements)
- [Conclusion](#conclusions)

## Features

- **Neural Reranking**: Utilizes a CNN-based model to refine and improve the ranking of documents retrieved by the BM25 model.
- **Custom Tokenizer**: Processes text with options for case normalization, stopword removal, and punctuation handling.
- **Pretrained Embeddings**: Supports loading and integrating pretrained word embeddings (GloVe) for enhanced semantic understanding.
- **Efficient Data Handling**: Implements PyTorch `Dataset` and `DataLoader` for scalable and efficient data processing.
- **Evaluation Metrics**: Computes nDCG@10 to assess the quality of retrieval results.

## Usage

The system comprises a main script `cli.py`.

**Command Structure**:
```bash
python3 cli.py [options]
```

### Command Line Arguments

To showcase the Command Line Interface, run:

```bash
python3 cli.py --help
```

| Argument               | Type    | Default                        | Description                                                                                  |
|------------------------|---------|--------------------------------|----------------------------------------------------------------------------------------------|
| `--pretrained_embeddings`              | string    | `../data/glove.42B.300d.txt`                              | Path to the pretrained embeddings file (txt format).                            |
| `--corpus`             | string    | `../data/MEDLINE_2024_Baseline.jsonl`                              | Path to the corpus file (JSONL format).                                          |
| `--questions_file`              | string  | `../data/questions.jsonl`                              | Path to the questions file (JSONL format).                        |
| `--bm25_ranked_file`             | string  | `../data/questions_bm25_ranked.jsonl	` | Path to the BM25-ranked file (JSONL format).                                      |
| `--training_data`         | string  | `../data/training_data.jsonl	`                       | Path to the questions training file (JSONL format).                                             |
| `--training_data_bm25_ranked`   | string     | `../data/training_data_bm25_ranked.jsonl`                              | Path to the BM25-ranked training file (JSONL format).                                                     |
| `--output_file`          | string    | `../output/final_ranked_questions.jsonl`                           | Path to save the reranked results (JSONL format).                                              |
| `--model_checkpoint`          | string  | `../output/trained_cnn_model.pt`                              | Path to the trained model checkpoint (if exists, load it; else train new model).                                                       |
| `--batch_size`           | int    | 64                          | Batch size for reranking.                                                   |
| `--number_documents_ranked`                 | int   | 10                            | Number of top documents retrieved for each question.                                     |
| `--epochs`                  | int   | 5                           | Number of training epochs.                                 |
| `--learning_rate`         | float     | 0.001                          | Learning rate for optimizer.                                                |

> [!NOTE]
> See `entrypoint.sh` script to see some examples on how to execute this script.

## Trained Models

We trained two distinct models to evaluate the impact of different positive-to-negative sample ratios on performance. Both models were based on the `CNNInteractionBasedModel` architecture but differed in their training data composition.

### Model 1: Positive-Negative Sample Ratio 1:1

- **Sample Ratio:** 1:1 (Equal number of positive and negative samples)
- **Description:** This model was trained using a balanced dataset where each positive sample (relevant document) was paired with an equal number of negative samples (non-relevant documents). The balanced ratio was intended to provide the model with an unbiased view of relevance.
- **Training Details:**
  - **Epochs:** 10
  - **Batch Size:** 64
  - **Optimizer:** Adam
  - **Learning Rate:** 0.001

**Model**: `output/trained_cnn_model_1.pt`

### Model 2: Positive-Negative Sample Ratio 1:2

- **Sample Ratio:** 1:2 (Half as many positive samples compared to negative samples)
- **Description:** This model was trained on an imbalanced dataset with twice as many negative samples as positive ones. The intention was to simulate real-world scenarios where non-relevant documents are more prevalent and to assess the model's ability to handle imbalance.
- **Training Details:**
  - **Epochs:** 10
  - **Batch Size:** 64
  - **Optimizer:** Adam
  - **Learning Rate:** 0.001

**Model**: `output/trained_cnn_model_2.pt`

### Results and Observations

After training both models, we evaluated their performance using the nDCG@10 metric. The results are as follows:

- **Model 1 (1:1 Ratio):** Achieved an average nDCG@10 of **0.1033**.
- **Model 2 (1:2 Ratio):** Achieved an average nDCG@10 of **0.1086**.

**Interpretation:** The slight difference in performance between the two models indicates that altering the positive-negative sample ratio from 1:1 to 1:2 did not lead to significant improvements in retrieval effectiveness. Both models poor performance, suggesting that factors other than sample ratio may have a more substantial impact on model effectiveness.

## Components

### CNNInteractionBasedModel

The `CNNInteractionBasedModel` (`model.py`) is a PyTorch neural network model designed to rerank documents based on their relevance to a given query. It leverages convolutional layers to capture interactions between query and document embeddings.

**Key Components:**

- **Embedding Layer:** Converts token IDs into dense vectors. Supports loading pretrained embeddings (e.g., GloVe).
- **Convolutional Layer:** Captures local interactions between query and document embeddings.
- **Activation Function:** Applies ReLU activation to introduce non-linearity.
- **Pooling Layer:** Aggregates features using adaptive max pooling.
- **Fully Connected Layer:** Maps extracted features to a single relevance score.
- **Sigmoid Activation:** Converts logits to probabilities between 0 and 1.

### SimpleDataset

The `SimpleDataset` class (`simple_dataset.py`) is a PyTorch `Dataset` tailored for loading and preparing data for training and reranking tasks. It handles the pairing of queries with documents and assigns labels based on gold-standard relevance.

**Key Features:**

- **Data Pairing:** Associates each query with its corresponding BM25-ranked documents.
- **Label Assignment:** Assigns binary labels (1.0 for relevant documents, 0.0 otherwise) based on gold-standard relevance.
- **Tokenization:** Converts text into sequences of token IDs using the provided tokenizer.
- **Efficient Data Loading:** Filters and maps necessary documents to minimize memory usage.

### Tokenizer

The `Tokenizer` class (`tokenizer.py`) is responsible for converting raw text into a sequence of token IDs, preparing it for embedding and model input. It supports various preprocessing steps like case normalization, punctuation removal, stopword elimination, and stemming.

**Key Features:**

- **Vocabulary Building:** Constructs a mapping from tokens to unique IDs based on the training data.
- **Preprocessing:** Cleans text by removing punctuation, normalizing case, and handling whitespace.
- **Tokenization:** Converts cleaned text into sequences of token IDs for model input.
- **Padding:** Utilizes a special `<PAD>` token (ID 0) for padding shorter sequences.

## Enhancements

1. **Model Training**: The model produces poor predictions, thus we have to change the approach in order to train the model (either change the model architecture or the dataset).
2. **Hyperparameter Tuning**: Explore different hyperparameters (e.g., learning rate, batch size, number of filters) to optimize model performance.
3. **Parallel Processing**: Optimize data loading and model training using parallel processing techniques to speed up computations.

## Conclusion

The CNN-based Neural Reranker did not achieve the expected performance improvements, as indicated by the lower nDCG@10 scores. Further efforts are needed to refine the model and enhance its effectiveness in document retrieval tasks
