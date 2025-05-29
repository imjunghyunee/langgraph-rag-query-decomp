import os
import json
import shutil
import tempfile
import uuid
import argparse
from typing import List

import torch
import torch.nn.functional as F
import optuna
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader


def load_pairs(path: str) -> List[InputExample]:
    """Load [[query, positive]] JSON list into InputExample objects."""
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    return [InputExample(texts=[q, p]) for q, p in pairs]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train embedding model with GISTEmbedLoss"
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
        "--guide_model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Guide model for false-negative filtering",
    )
    parser.add_argument(
        "--relative_margin",
        type=float,
        default=0.3,
        help="Relative margin threshold for GISTEmbedLoss filtering",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./best_model",
        help="Directory to save the final model",
    )
    parser.add_argument(
        "--trials", type=int, default=10, help="Number of Optuna HPO trials"
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Optuna timeout in seconds"
    )
    parser.add_argument(
        "--tb_logs", type=str, default="./tb_logs", help="TensorBoard logs directory"
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

        tb_cb = TensorBoardCallback(logdir=args.tb_logs, metric_name="val_loss")
        callbacks = [tb_cb]
    except ImportError:
        callbacks = []

    def objective(trial):
        # HPO search space
        lr = trial.suggest_categorical("learning_rate", [5e-6, 1e-5, 2e-5])
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        epochs = trial.suggest_int("epochs", 3, 4)

        # Load student & guide models
        student = SentenceTransformer(args.model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        student.to(device)

        guide = SentenceTransformer(args.guide_model_name)
        guide.to(device)

        # Load data and split
        all_pairs = load_pairs(args.data_path)
        split = int(0.9 * len(all_pairs))
        train_pairs = all_pairs[:split]
        val_pairs = all_pairs[split:]
        if len(val_pairs) < batch_size:
            val_pairs = train_pairs[:batch_size]

        # Prepare DataLoader
        train_dataloader = DataLoader(train_pairs, batch_size=batch_size, shuffle=True)

        # GISTEmbedLoss: filters in-batch negatives via guide
        train_loss = losses.GISTEmbedLoss(
            model=student,
            guide_model=guide,
            filter_fn="relative_margin",
            relative_margin=args.relative_margin,
            normalize_embeddings=True,
        )

        # Training
        ckpt_dir = tempfile.mkdtemp(prefix=f"ckpt_{uuid.uuid4().hex[:6]}_")
        try:
            student.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                optimizer_params={"lr": lr},
                output_path=ckpt_dir,
                show_progress_bar=False,
            )
            # Validation: in-batch softmax (multiple-negatives style)
            queries = [ex.texts[0] for ex in val_pairs]
            positives = [ex.texts[1] for ex in val_pairs]
            with torch.no_grad():
                q_emb = student.encode(
                    queries,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,
                )
                p_emb = student.encode(
                    positives,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,
                )
            scores = cos_sim(q_emb, p_emb)
            targets = torch.arange(scores.size(0), device=scores.device)
            val_loss = F.cross_entropy(scores, targets).item()
        except Exception as e:
            print(f"Error during training: {e}")
            val_loss = float("inf")
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            torch.cuda.empty_cache()

        return val_loss

    # Optuna study
    study = optuna.create_study(direction="minimize", study_name="hpo_gistembed")
    study.optimize(
        objective, n_trials=args.trials, timeout=args.timeout, callbacks=callbacks
    )

    # Final training with best hyperparams
    best = study.best_trial.params
    print("Best HPO params:", best)
    print("Training final model with GISTEmbedLoss...")

    student = SentenceTransformer(args.model_name).to(device)
    guide = SentenceTransformer(args.guide_model_name).to(device)
    all_pairs = load_pairs(args.data_path)
    train_dataloader = DataLoader(
        all_pairs, batch_size=best["batch_size"], shuffle=True
    )
    train_loss = losses.GISTEmbedLoss(
        model=student,
        guide_model=guide,
        filter_fn="relative_margin",
        relative_margin=args.relative_margin,
        normalize_embeddings=True,
    )
    student.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=best["epochs"],
        optimizer_params={"lr": best["learning_rate"]},
        output_path=args.output_dir,
        show_progress_bar=True,
    )
    print(f"Final model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
