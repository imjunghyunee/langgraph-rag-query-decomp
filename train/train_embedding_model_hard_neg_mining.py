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
        default="BAAI/bge-m3",
        help="Pretrained student model identifier",
    )
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default="intfloat/e5-mistral-7b-instruct",
        help="Teacher model for hard-negative mining",
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


def main():
    args = parse_args()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["WANDB_DISABLED"] = "true"

    # Optional TensorBoard callback
    try:
        from optuna.integration import TensorBoardCallback

        tb_cb = TensorBoardCallback(logdir=args.tb_logs, metric_name="val_mnrl")
        callbacks = [tb_cb]
    except ImportError:
        print("optuna-integration not installed. Running without TensorBoard callback.")
        callbacks = []

    def objective(trial):
        # HPO hyperparams
        lr = trial.suggest_categorical("learning_rate", [5e-6, 1e-5, 2e-5])
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        epochs = trial.suggest_int("epochs", 3, 4)
        top_k = trial.suggest_categorical("top_k_hard_neg_samples", [10, 20, 30, 40])
        perc_threshold = trial.suggest_categorical("perc_threshold", [0.90, 0.95])

        # Load student & teacher models
        student_model = SentenceTransformer(args.model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        student_model.to(device)
        teacher_model = SentenceTransformer(args.teacher_model_name)
        teacher_model.to(device)

        # Load and split data
        all_pairs = load_pairs(args.data_path)
        train_pairs = all_pairs[: int(len(all_pairs) * 0.9)]
        val_pairs = all_pairs[int(len(all_pairs) * 0.9) :]
        if len(val_pairs) < batch_size:
            val_pairs = train_pairs[:batch_size]

        # Hard-negative mining
        hf_ds = Dataset.from_dict(
            {
                "query": [ex.texts[0] for ex in train_pairs],
                "positive": [ex.texts[1] for ex in train_pairs],
            }
        )
        rel_margin = 1.0 - perc_threshold
        # Mine negatives
        if args.neg_mining_method == "perc":
            mined = mine_hard_negatives(
                dataset=hf_ds,
                model=teacher_model,
                anchor_column_name="query",
                positive_column_name="positive",
                relative_margin=rel_margin,
                num_negatives=top_k,
                sampling_strategy="top",
                output_format="n-tuple",
                batch_size=batch_size,
                use_faiss=True,
                verbose=False,
            )
        else:
            top1 = mine_hard_negatives(
                dataset=hf_ds,
                model=teacher_model,
                anchor_column_name="query",
                positive_column_name="positive",
                relative_margin=rel_margin,
                num_negatives=1,
                sampling_strategy="top",
                output_format="n-tuple",
                batch_size=batch_size,
                use_faiss=True,
                verbose=False,
            )
            sampled = mine_hard_negatives(
                dataset=hf_ds,
                model=teacher_model,
                anchor_column_name="query",
                positive_column_name="positive",
                relative_margin=rel_margin,
                num_negatives=top_k - 1,
                sampling_strategy="random",
                output_format="n-tuple",
                batch_size=batch_size,
                use_faiss=True,
                verbose=False,
            )
            # Merge top1 + sampled
            mined = []
            for r1, r2 in zip(top1, sampled):
                texts = (
                    [r1["query"], r1["positive"]]
                    + r1.get("negatives", [])
                    + r2.get("negatives", [])
                )
                mined.append({"texts": texts})

        # Convert to InputExamples
        train_examples = [InputExample(texts=row["texts"]) for row in mined]

        # DataLoader and BatchHardTripletLoss
        train_dataloader = DataLoader(
            train_examples, batch_size=batch_size, shuffle=True
        )
        train_loss = losses.BatchHardTripletLoss(  # TripletLoss
            model=student_model, metric="cosine", margin=0.2
        )

        # Training loop
        out_dir = tempfile.mkdtemp(prefix=f"ckpt_{uuid.uuid4().hex[:6]}_")
        try:
            student_model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                optimizer_params={"lr": lr},
                output_path=out_dir,
                show_progress_bar=False,
            )
            # Validation
            val_q = [ex.texts[0] for ex in val_pairs]
            val_p = [ex.texts[1] for ex in val_pairs]
            with torch.no_grad():
                emb_q = student_model.encode(
                    val_q,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,
                )
                emb_p = student_model.encode(
                    val_p,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,
                )
            scores = cos_sim(emb_q, emb_p) * 1.0
            targets = torch.arange(scores.size(0), device=scores.device)
            val_loss = torch.nn.functional.cross_entropy(scores, targets).item()
        except Exception as e:
            print(f"Training failed: {e}")
            return float("inf")
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
            torch.cuda.empty_cache()

        return val_loss

    # Run HPO
    study = optuna.create_study(direction="minimize", study_name="hpo_hardneg_triplet")
    study.optimize(
        objective, n_trials=args.trials, timeout=args.timeout, callbacks=callbacks
    )

    # Final training
    best = study.best_trial.params
    print("Best params:", best)
    print("Training final model...")
    # (Repeat mining & fit with best params)


if __name__ == "__main__":
    main()
