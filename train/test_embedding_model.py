import argparse
import torch
from sentence_transformers import SentenceTransformer, util


def load_model(model_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_dir, device=device)
    return model


def run_inference(model, query: str, candidate_docs: list, top_k: int = 5):
    emb_query = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    emb_docs = model.encode(
        candidate_docs, convert_to_tensor=True, normalize_embeddings=True
    )
    cos_scores = util.cos_sim(emb_query, emb_docs)[0]
    top_k = min(top_k, len(candidate_docs))
    top_results = torch.topk(cos_scores, k=top_k)
    print(f"\nQuery: {query}\n")
    print("Top-k most similar documents:")
    for score, idx in zip(top_results.values, top_results.indices):
        print(f"{idx.item():>3} | score={score:.4f} | {candidate_docs[idx]}")
    return top_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with trained embedding model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./best_model",
        help="Path to saved model directory",
    )
    parser.add_argument("--query", type=str, required=False, help="Query string")
    parser.add_argument(
        "--candidate_docs",
        type=str,
        nargs="+",
        required=False,
        help="Candidate documents to score",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of top results to return"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model_dir)
    if args.query and args.candidate_docs:
        run_inference(model, args.query, args.candidate_docs, args.top_k)
    else:
        # Sample demonstration
        sample_query = "How can I improve the accuracy of my machine-learning model?"
        sample_docs = [
            "Try hyper-parameter tuning with Optuna to find an optimal learning rate.",
            "Make sure your data is properly cleaned and pre-processed.",
            "Use k-fold cross-validation to get a more reliable performance estimate.",
            "Consider collecting more labeled data to prevent overfitting.",
            "Switching to a larger transformer model might also help.",
        ]
        run_inference(model, sample_query, sample_docs, args.top_k)


if __name__ == "__main__":
    main()
