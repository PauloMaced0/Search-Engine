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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
