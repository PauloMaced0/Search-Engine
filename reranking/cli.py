import argparse
import os
import ujson
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.tokenizer import Tokenizer
from src.model import CNNInteractionBasedModel
from src.utils import load_pretrained_embeddings, build_collate_fn, get_all_doc_texts, get_questions, load_questions
from torch.utils.data import DataLoader
from src.simple_dataset import SimpleDataset

RESET = "\033[0m"  # Reset all styles
UNDERLINE = "\033[4m"

def train_model(model, tokenizer, training_questions_file, training_ranked_file, corpus_file, device, batch_size=64, epochs=5, lr=1e-3):
    """
    Train the model using a binary classification approach:
    - label=1 if document is in goldstandard_documents
    - label=0 otherwise
    """
    print("Loading training dataset...")
    train_dataset = SimpleDataset(training_questions_file, training_ranked_file, corpus_file, tokenizer, return_label=True)

    # Determine max lengths for padding
    print("Determining max sequence lengths...")
    question_lengths = []
    document_lengths = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        question_lengths.append(len(sample["question_token_ids"]))
        document_lengths.append(len(sample["document_token_ids"]))
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} samples...")

    max_question_length = max(3, max(question_lengths)) if question_lengths else 3
    max_document_length = max(3, max(document_lengths)) if document_lengths else 3
    print(f"Max question length: {max_question_length}, Max document length: {max_document_length}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=build_collate_fn(tokenizer,
                                    max_number_of_question_tokens=max_question_length,
                                    max_number_of_document_tokens=max_document_length),
        shuffle=True,
        pin_memory=(device.type == 'cuda')
    )

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    print("Starting training...")

    model.train(True)
    for epoch in range(epochs):
        running_loss = 0.
        last_loss = 0.
        total_loss = 0.
        for batch_idx, batch in enumerate(train_dataloader):
            query_ids = batch["question_token_ids"].to(device, non_blocking=True)
            doc_ids = batch["document_token_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            scores = model(query_ids, doc_ids)
            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item() 

            if batch_idx % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                running_loss = 0.
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {last_loss:.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")

    print("Training completed.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Neural Reranker CLI")
    
    # Input arguments
    parser.add_argument('--pretrained_embeddings', type=str, help="Path to the pretrained embeddings file (txt format)", default='../data/glove.42B.300d.txt')
    parser.add_argument('--corpus', type=str, help="Path to the corpus file (JSONL format)", default="../data/MEDLINE_2024_Baseline.jsonl")
    parser.add_argument('--questions_file', type=str, help="Path to the questions file (JSONL format)", default='../data/questions.jsonl')
    parser.add_argument('--bm25_ranked_file', type=str, help="Path to the BM25-ranked file (JSONL format)", default='../data/questions_bm25_ranked.jsonl')
    parser.add_argument('--training_data', type=str, help="Path to the questions training file (JSONL format)", default='../data/training_data.jsonl')
    parser.add_argument('--training_data_bm25_ranked', type=str, help="Path to the BM25-ranked training file (JSONL format)", default='../data/training_data_bm25_ranked.jsonl')
    parser.add_argument('--output_file', type=str, help="Path to save the reranked results (JSONL format)", default='../output/final_ranked_questions.jsonl')

    # Model arguments
    parser.add_argument('--model_checkpoint', type=str, help="Path to the trained model checkpoint (if exists, load it; else train new model)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for reranking")
    parser.add_argument('--number_documents_ranked', type=int, default=10, help="Number of top documents retrieved for each question")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for optimizer")

    args = parser.parse_args()

    default_checkpoint = 'output/trained_cnn_model.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    tokenizer = Tokenizer()

    question = get_questions(args.training_data)
    document = get_all_doc_texts(args.training_data, args.training_data_bm25_ranked, args.corpus)
    print(len(question), len(document))

    tokenizer.fit(question + document)

    pretrained_embeddings = None
    if args.pretrained_embeddings and os.path.exists(args.pretrained_embeddings):
        print(f"Loading pretrained embeddings from {args.pretrained_embeddings}...")
        pretrained_embeddings = load_pretrained_embeddings(args.pretrained_embeddings, tokenizer, embedding_dim=300)
    else:
        print("No pretrained embeddings found. Initializing embeddings randomly.")

    model = CNNInteractionBasedModel(vocab_size=tokenizer.vocab_size, pretrained_embeddings=pretrained_embeddings)

    # Load model checkpoint if exists
    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        print(f"Loading model checkpoint from {args.model_checkpoint}...")
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device, weights_only=True))
    else:
        print("No model checkpoint found. Starting training.")
        # Train the model
        model = train_model(
            model, tokenizer, args.training_data, args.training_data_bm25_ranked,
            args.corpus, device, batch_size=args.batch_size, epochs=args.epochs, lr=args.learning_rate
        )
        # Save the trained model
        if args.model_checkpoint:
            print(f"Saving trained model to {args.model_checkpoint}...")
            torch.save(model.state_dict(), args.model_checkpoint)
        else:
            os.makedirs(os.path.dirname(default_checkpoint), exist_ok=True)
            print(f"Saving trained model to {default_checkpoint}...")
            torch.save(model.state_dict(), default_checkpoint)

    # Prepare DataLoader
    print("Preparing data for reranking...")

    dataset = SimpleDataset(args.questions_file, args.bm25_ranked_file, args.corpus, tokenizer)
    question_lengths = []
    document_lengths = []

    for i in range(len(dataset)):
        sample = dataset[i]
        question_lengths.append(len(sample["question_token_ids"]))
        document_lengths.append(len(sample["document_token_ids"]))

    # Dynamically determine the maximum lengths
    max_question_length = max(question_lengths)
    max_document_length = max(document_lengths)
    print("Max doc length:", max_document_length, "Max question length:", max_question_length)

    rerank_dataset = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=build_collate_fn(tokenizer,
                                    max_number_of_question_tokens=max_question_length, 
                                    max_number_of_document_tokens=max_document_length
                                    ), 
        shuffle=False,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Reranking
    print("Starting reranking...")
    model.to(device)

    reranked_results = {}
    with torch.no_grad():
        for batch in rerank_dataset:
            query_ids = batch["question_token_ids"].to(device)
            doc_ids = batch["document_token_ids"].to(device)
            scores = model(query_ids, doc_ids)

            for i, query_id in enumerate(batch["query_ids"]):
                if query_id not in reranked_results:
                    reranked_results[query_id] = []

                reranked_results[query_id].append((batch["document_ids"][i], scores[i]))

    rerank_questions = load_questions(args.questions_file)

    # Sort results
    final_ranking = []
    for query_id, doc_scores in reranked_results.items():
        sorted_docs = [doc_id for doc_id, _ in sorted(doc_scores, key=lambda x: x[1], reverse=True)]
        final_ranking.append({
            "query_id": query_id,
            "question": rerank_questions.get(query_id),
            "retrieved_documents": sorted_docs[:args.number_documents_ranked]
        })

    # Save results
    print(f"Saving reranked results to: {UNDERLINE}{args.output_file}{RESET}")
    with open(args.output_file, 'w') as output_file:
        for entry in final_ranking:
            output_file.write(ujson.dumps(entry) + '\n')

    print("Reranking completed.")

if __name__ == '__main__':
    main()
