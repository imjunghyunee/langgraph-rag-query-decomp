import os
import json
import shutil
import tempfile
import uuid
import random
import argparse
from typing import List

import torch
import optuna
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.util import cos_sim, mine_hard_negatives
from torch.utils.data import DataLoader
from datasets import Dataset


def load_pairs(path: str) -> List[InputExample]:
    """Load [[question, answer], …] JSON list into InputExample objects."""
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    return [InputExample(texts=[q, a]) for q, a in pairs]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train embedding model with HPO and hard-negative mining"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/content/paired_data",
        help="Path to paired data JSON file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="./jinaai/jina-embeddings-v3",  # Local path for jina-embeddings-v3
        help="Local path to pretrained student model",
    )
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default="./intfloat/e5-mistral-7b-instruct",  # Local path for teacher model
        help="Local path to teacher model for hard-negative mining",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./best_model",
        help="Directory to save the final model",
    )
    parser.add_argument(
        "--trials", type=int, default=10, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Optuna timeout in seconds"
    )
    parser.add_argument(
        "--tb_logs", type=str, default="./tb_logs", help="TensorBoard logs directory"
    )
    parser.add_argument(
        "--neg_mining_method",
        type=str,
        default="perc",
        choices=["perc", "top1_sampled"],
        help="Hard-negative mining method: 'perc' or 'top1_sampled'",
    )
    return parser.parse_args()


def load_student_model(model_path: str, offline_mode: bool = True):
    """
    Load jina-embeddings-v3 model with specific parameters in offline mode.

    Args:
        model_path: Local path to the model directory
        offline_mode: If True, prevents any internet connection attempts
    """
    # Set offline environment variables to prevent internet access
    if offline_mode:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    return SentenceTransformer(
        model_path,
        trust_remote_code=True,
        model_kwargs={"default_task": "text-matching"},
        local_files_only=True,
        # Ensure local_files_only to prevent internet access
        device=None,  # Let it auto-detect device
    )


def load_teacher_model(model_path: str, offline_mode: bool = True):
    """
    Load teacher model from local path in offline mode.

    Args:
        model_path: Local path to the model directory
        offline_mode: If True, prevents any internet connection attempts
    """
    # Set offline environment variables to prevent internet access
    if offline_mode:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    return SentenceTransformer(model_path, local_files_only=True, device=None)


def perform_hard_negative_mining(
    dataset, teacher_model, method, rel_margin, top_k, batch_size
):
    """
    Perform hard negative mining using the teacher model.

    Hard negative mining process:
    1. Teacher model encodes all queries and positive answers to get embeddings
    2. Computes cosine similarity between each query and ALL possible answers in the dataset
    3. For each query-positive pair, finds answers that are:
       - NOT the true positive answer
       - Similar enough to the query (within relative_margin threshold)
       - But still "hard" examples that could confuse the model
    4. Returns structured data with query, positive, and hard negative examples

    Args:
        dataset: HuggingFace dataset with 'query' and 'positive' columns
        teacher_model: Pre-trained model used for computing similarities
        method: Mining method ('perc' or 'top1_sampled')
        rel_margin: Relative margin threshold (lower = more selective)
        top_k: Number of hard negatives to mine per query
        batch_size: Batch size for encoding

    Returns:
        List of dictionaries containing mined examples
    """

    if method == "perc":
        print(
            f"Mining hard negatives using 'perc' method with relative_margin={rel_margin}, top_k={top_k}"
        )

        # mine_hard_negatives function workflow:
        # 1. Encodes all 'query' texts using teacher_model -> query_embeddings
        # 2. Encodes all 'positive' texts using teacher_model -> positive_embeddings
        # 3. Computes similarity matrix: query_embeddings @ positive_embeddings.T
        # 4. For each query i, finds positive j where similarity(query_i, positive_j) is:
        #    - High enough: similarity > (max_similarity - relative_margin)
        #    - But NOT the true positive pair (j != i)
        # 5. Selects top_k hardest negatives using 'top' sampling strategy
        # 6. Returns n-tuple format: {'query': str, 'positive': str, 'negatives': [str, ...]}
        mined = mine_hard_negatives(
            dataset=dataset,  # Input dataset with query-positive pairs
            model=teacher_model,  # Teacher model for encoding texts
            anchor_column_name="query",  # Column name for queries/anchors
            positive_column_name="positive",  # Column name for positive examples
            relative_margin=rel_margin,  # Similarity threshold for hard negatives
            num_negatives=top_k,  # Number of negatives to mine per query
            sampling_strategy="top",  # Take top-k most similar negatives
            output_format="n-tuple",  # Output format with query, positive, negatives
            batch_size=batch_size,  # Batch size for efficient encoding
            use_faiss=True,  # Use FAISS for fast similarity search
            verbose=False,  # Suppress detailed logging
        )

    else:  # top1_sampled method
        print(
            f"Mining hard negatives using 'top1_sampled' method with relative_margin={rel_margin}"
        )

        # Step 1: Mine the single hardest negative (top-1) for each query
        # This finds the most confusing negative example - the one most similar to query but not the true positive
        print("Mining top-1 hardest negatives...")
        top1 = mine_hard_negatives(
            dataset=dataset,
            model=teacher_model,
            anchor_column_name="query",
            positive_column_name="positive",
            relative_margin=rel_margin,
            num_negatives=1,  # Only get the single hardest negative
            sampling_strategy="top",  # Take the most similar negative
            output_format="n-tuple",
            batch_size=batch_size,
            use_faiss=True,
            verbose=False,
        )

        # Step 2: Mine additional negatives using random sampling
        # This adds diversity by randomly sampling from the pool of hard negatives
        print(f"Mining {top_k-1} randomly sampled hard negatives...")
        sampled = mine_hard_negatives(
            dataset=dataset,
            model=teacher_model,
            anchor_column_name="query",
            positive_column_name="positive",
            relative_margin=rel_margin,
            num_negatives=top_k - 1,  # Remaining negatives after top-1
            sampling_strategy="random",  # Randomly sample from hard negatives pool
            output_format="n-tuple",
            batch_size=batch_size,
            use_faiss=True,
            verbose=False,
        )

        # Step 3: Combine top-1 hardest + randomly sampled hard negatives
        # This creates a balanced set: 1 very hard + (top_k-1) diverse hard negatives
        print("Combining top-1 and sampled negatives...")
        mined = []
        for r1, r2 in zip(top1, sampled):
            # Merge the results: query + positive + top-1_negative + sampled_negatives
            combined_texts = (
                [r1["query"], r1["positive"]]  # Original query-positive pair
                + r1.get("negatives", [])  # Top-1 hardest negative
                + r2.get("negatives", [])  # Additional sampled hard negatives
            )
            mined.append({"texts": combined_texts})

    print(f"Successfully mined {len(mined)} examples with hard negatives")
    return mined


def main():
    args = parse_args()

    # Set environment variables to ensure offline operation and prevent internet access
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    # Critical: Set offline mode to prevent any internet connections
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Optional TensorBoard callback (works offline)
    try:
        from optuna.integration import TensorBoardCallback

        tb_cb = TensorBoardCallback(logdir=args.tb_logs, metric_name="val_mnrl")
        callbacks = [tb_cb]
    except ImportError:
        print("optuna-integration not installed. Running without TensorBoard callback.")
        callbacks = []

    def objective(trial):
        # HPO hyperparams - Optuna suggests these values for each trial
        lr = trial.suggest_categorical("learning_rate", [5e-6, 1e-5, 2e-5])
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        epochs = trial.suggest_int("epochs", 3, 4)
        top_k = trial.suggest_categorical("top_k_hard_neg_samples", [10, 20, 30, 40])
        perc_threshold = trial.suggest_categorical("perc_threshold", [0.90, 0.95])

        # Load student & teacher models from local directories (offline mode)
        student_model = load_student_model(args.model_name, offline_mode=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        student_model.to(device)
        teacher_model = load_teacher_model(args.teacher_model_name, offline_mode=True)
        teacher_model.to(device)

        # Load and split data for training/validation
        all_pairs = load_pairs(args.data_path)
        train_pairs = all_pairs[: int(len(all_pairs) * 0.9)]
        val_pairs = all_pairs[int(len(all_pairs) * 0.9) :]
        if len(val_pairs) < batch_size:
            val_pairs = train_pairs[:batch_size]

        # Prepare dataset for hard-negative mining
        # Convert InputExample objects to HuggingFace Dataset format required by mine_hard_negatives
        hf_ds = Dataset.from_dict(
            {
                "query": [
                    ex.texts[0] for ex in train_pairs
                ],  # Extract queries from training pairs
                "positive": [
                    ex.texts[1] for ex in train_pairs
                ],  # Extract positive answers from training pairs
            }
        )

        # Calculate relative margin for hard negative mining
        # rel_margin = 1.0 - perc_threshold determines how "hard" the negatives should be
        # Lower rel_margin = more selective (only very similar negatives)
        # Higher rel_margin = less selective (allows more distant negatives)
        rel_margin = 1.0 - perc_threshold

        # Perform hard negative mining using the teacher model
        mined = perform_hard_negative_mining(
            dataset=hf_ds,
            teacher_model=teacher_model,
            method=args.neg_mining_method,
            rel_margin=rel_margin,
            top_k=top_k,
            batch_size=batch_size,
        )

        # Convert mined results back to InputExample format for training
        # Each mined example contains: [query, positive, negative1, negative2, ...]
        # This creates triplets/n-tuples for contrastive learning
        train_examples = [InputExample(texts=row["texts"]) for row in mined]

        # Create DataLoader for batch processing during training
        train_dataloader = DataLoader(
            train_examples, batch_size=batch_size, shuffle=True
        )

        # BatchHardTripletLoss:
        # - Creates triplets (anchor, positive, negative) within each batch
        # - Computes triplet loss: max(0, margin + sim(anchor,negative) - sim(anchor,positive))
        # - "Hard" mining within batch: selects hardest positive and negative for each anchor
        train_loss = losses.BatchHardTripletLoss(
            model=student_model,
            metric="cosine",  # Use cosine similarity for computing distances
            margin=0.2,  # Margin for triplet loss (positive should be 0.2 more similar than negative)
        )

        # Training loop with temporary directory for this trial
        out_dir = tempfile.mkdtemp(prefix=f"ckpt_{uuid.uuid4().hex[:6]}_")
        try:
            # Fine-tune the student model using the mined hard negatives
            student_model.fit(
                train_objectives=[
                    (train_dataloader, train_loss)
                ],  # Training data and loss function
                epochs=epochs,  # Number of training epochs
                optimizer_params={"lr": lr},  # Learning rate for optimizer
                output_path=out_dir,  # Temporary save location
                show_progress_bar=False,  # Suppress progress for HPO
            )

            # Validation: Compute validation loss to evaluate this trial
            val_q = [ex.texts[0] for ex in val_pairs]  # Validation queries
            val_p = [ex.texts[1] for ex in val_pairs]  # Validation positive answers

            with torch.no_grad():
                # Encode validation queries and positives
                emb_q = student_model.encode(
                    val_q,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,  # L2 normalize for cosine similarity
                )
                emb_p = student_model.encode(
                    val_p,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,
                )

            # Compute similarity scores between queries and all positives
            scores = cos_sim(emb_q, emb_p) * 1.0
            # Create targets: each query should match its corresponding positive (diagonal)
            targets = torch.arange(scores.size(0), device=scores.device)
            # Cross-entropy loss: measures how well the model ranks correct positives highest
            val_loss = torch.nn.functional.cross_entropy(scores, targets).item()

        except Exception as e:
            print(f"Training failed: {e}")
            return float("inf")  # Return worst possible score for failed trials
        finally:
            # Cleanup: Remove temporary directory and free GPU memory
            shutil.rmtree(out_dir, ignore_errors=True)
            torch.cuda.empty_cache()

        return val_loss  # Return validation loss for Optuna to minimize

    # Run hyperparameter optimization
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction="minimize", study_name="hpo_hardneg_triplet")
    study.optimize(
        objective, n_trials=args.trials, timeout=args.timeout, callbacks=callbacks
    )

    # Final training with best parameters found by HPO
    best = study.best_trial.params
    print("Best hyperparameters found:", best)
    print("Starting final training with full dataset...")

    # Load models for final training (ensure offline mode)
    final_student_model = load_student_model(args.model_name, offline_mode=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_student_model.to(device)
    final_teacher_model = load_teacher_model(args.teacher_model_name, offline_mode=True)
    final_teacher_model.to(device)

    # Load complete dataset for final training (no train/val split)
    all_pairs = load_pairs(args.data_path)

    # Prepare full dataset for hard-negative mining
    hf_ds_full = Dataset.from_dict(
        {
            "query": [ex.texts[0] for ex in all_pairs],
            "positive": [ex.texts[1] for ex in all_pairs],
        }
    )

    # Use best relative margin found during HPO
    rel_margin = 1.0 - best["perc_threshold"]

    # Perform hard negative mining on the complete dataset
    print("Performing hard negative mining on full dataset...")
    mined_full = perform_hard_negative_mining(
        dataset=hf_ds_full,
        teacher_model=final_teacher_model,
        method=args.neg_mining_method,
        rel_margin=rel_margin,
        top_k=best["top_k_hard_neg_samples"],
        batch_size=best["batch_size"],
    )

    # Convert to InputExamples for final training
    train_examples_full = [InputExample(texts=row["texts"]) for row in mined_full]

    # Create DataLoader and loss function with best parameters
    train_dataloader_full = DataLoader(
        train_examples_full, batch_size=best["batch_size"], shuffle=True
    )
    train_loss_full = losses.BatchHardTripletLoss(
        model=final_student_model, metric="cosine", margin=0.2
    )

    # Final training with best hyperparameters
    print("Training final model...")
    try:
        final_student_model.fit(
            train_objectives=[(train_dataloader_full, train_loss_full)],
            epochs=best["epochs"],
            optimizer_params={"lr": best["learning_rate"]},
            output_path=args.output_dir,
            show_progress_bar=True,  # Show progress for final training
        )
        print(f"Final model saved to: {args.output_dir}")

        # Save best hyperparameters for reference
        with open(os.path.join(args.output_dir, "best_params.json"), "w") as f:
            json.dump(best, f, indent=2)
        print(
            f"Best hyperparameters saved to: {os.path.join(args.output_dir, 'best_params.json')}"
        )

    except Exception as e:
        print(f"Final training failed: {e}")
        raise
    finally:
        torch.cuda.empty_cache()

    print("Training completed successfully!")
    print("Model is ready for inference!")


if __name__ == "__main__":
    main()
