import json
import math
import argparse
import os

def compute_dcg(relevances):
    """Compute DCG given a list of relevance scores."""
    dcg = 0.0
    for i, rel in enumerate(relevances):
        position = i + 1
        if position == 1:
            dcg += rel
        else:
            dcg += rel / math.log2(position)
    return dcg

def compute_idcg(relevances):
    """Compute IDCG by sorting the relevance scores in descending order."""
    sorted_relevances = sorted(relevances, reverse=True)
    return compute_dcg(sorted_relevances)

def compute_ndcg_at_k(retrieved_docs, gold_docs, k=10):
    """Compute nDCG@k for a single query."""
    relevances = []
    for doc_id in retrieved_docs[:k]:
        if doc_id in gold_docs:
            relevances.append(1)  # Relevant document
        else:
            relevances.append(0)  # Non-relevant document

    dcg = compute_dcg(relevances)
    
    # Compute IDCG based on ideal ranking
    num_relevant = min(len(gold_docs), k)
    ideal_relevances = [1] * num_relevant + [0] * (k - num_relevant)
    idcg = compute_idcg(ideal_relevances)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

def extract_query_id(entry):
    """Extract query_id from an entry, handling different field names."""
    if 'query_id' in entry:
        return entry['query_id']
    elif 'id' in entry:
        return entry['id']
    else:
        raise KeyError("query_id or id not found in entry.")

def extract_retrieved_docs(entry):
    """Extract retrieved document IDs, handling different formats."""
    if 'retrieved_documents' not in entry:
        raise KeyError("retrieved_documents not found in entry.")
    
    retrieved_docs = entry['retrieved_documents']
    
    if isinstance(retrieved_docs, list):
        if len(retrieved_docs) == 0:
            return []
        first_elem = retrieved_docs[0]
        if isinstance(first_elem, dict):
            # Format 2: list of dicts with 'id' and 'score'
            return [doc['id'].split(':')[1] if ':' in doc['id'] else doc['id'] for doc in retrieved_docs if 'id' in doc]
        elif isinstance(first_elem, str):
            # Format 1 and 3: list of strings
            return [doc.split(':')[1] if ':' in doc else doc for doc in retrieved_docs]
        else:
            raise ValueError("Unsupported type in retrieved_documents list.")
    else:
        raise TypeError("retrieved_documents should be a list.")

def extract_gold_docs(question_entry):
    """Extract goldstandard_documents from a question entry."""
    if 'goldstandard_documents' in question_entry:
        gold_docs = question_entry['goldstandard_documents']
        # Extract PMID numbers
        gold_docs = {doc.split(':')[1] if ':' in doc else doc for doc in gold_docs}
        return gold_docs
    else:
        raise KeyError("goldstandard_documents not found in question entry.")

def compute_average_ndcg(questions_file_path, results_file_path, k=10):
    """Compute average nDCG@k over all queries."""
    # Load gold standard relevance judgements
    gold_data = {}
    with open(questions_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                question_data = json.loads(line)
                query_id = extract_query_id(question_data)
                gold_docs = extract_gold_docs(question_data)
                gold_data[query_id] = gold_docs
            except KeyError as e:
                print(f"Line {line_num}: {e}. Skipping this question.")
            except Exception as e:
                print(f"Line {line_num}: Error processing question. {e}. Skipping this question.")

    # Load retrieved documents
    results_data = {}
    with open(results_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                result_entry = json.loads(line)
                query_id = extract_query_id(result_entry)
                retrieved_docs = extract_retrieved_docs(result_entry)
                results_data[query_id] = retrieved_docs
            except KeyError as e:
                print(f"Line {line_num}: {e}. Skipping this result.")
            except Exception as e:
                print(f"Line {line_num}: Error processing result. {e}. Skipping this result.")

    # Compute nDCG@k for each query
    ndcg_scores = []
    for query_id, gold_docs in gold_data.items():
        retrieved_docs = results_data.get(query_id, [])
        ndcg = compute_ndcg_at_k(retrieved_docs, gold_docs, k)
        ndcg_scores.append(ndcg)
        print(f"Query ID: {query_id}, nDCG@{k}: {ndcg:.4f}")

    # Compute average nDCG@k
    average_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    print(f"\nAverage nDCG@{k}: {average_ndcg:.4f}")
    return average_ndcg

def main():
    parser = argparse.ArgumentParser(description="Compute nDCG@10 for the retrieval system.")
    parser.add_argument('--questions_file', type=str, help="Path to the questions file (JSONL).", default='data/questions.jsonl')
    parser.add_argument('--results_file', type=str, help="Path to the ranked results file (JSONL).", default='output/final_ranked_questions.jsonl')
    parser.add_argument('--k', type=int, default=10, help="Rank cutoff for nDCG computation.")
    args = parser.parse_args()

    if not os.path.exists(args.questions_file):
        print(f"Questions file not found: {args.questions_file}")
        return
    if not os.path.exists(args.results_file):
        print(f"Results file not found: {args.results_file}")
        return

    compute_average_ndcg(args.questions_file, args.results_file, args.k)

if __name__ == '__main__':
    main()
