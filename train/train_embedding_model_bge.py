import os
import json
import shutil
import tempfile
import uuid
import random
import argparse
from typing import List
import torch
import torch.nn.functional as F
import optuna
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader, Sampler


def load_pairs(path: str) -> List[InputExample]:
    """Load [[question, answer], …] JSON list into InputExample objects."""
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    return [InputExample(texts=[q, a]) for q, a in pairs]


class NoDuplicatesSampler(Sampler):
    """
    Custom sampler that ensures no duplicate samples within a batch.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_samples = len(data_source)

    def __iter__(self):
        indices = list(range(self.num_samples))
        random.shuffle(indices)
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, self.num_samples, self.batch_size)
        ]
        for batch in batches:
            for idx in batch:
                yield idx

    def __len__(self):
        return self.num_samples


def create_no_duplicates_dataloader(dataset, batch_size, shuffle=True):
    if shuffle:
        sampler = NoDuplicatesSampler(dataset, batch_size)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train embedding model with HPO")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/content/paired_data",
        help="Path to paired data JSON file or directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-m3",
        help="Pretrained model identifier",
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

    # Define objective with closure over args
    def objective(trial):
        lr = trial.suggest_categorical("learning_rate", [5e-6, 1e-5, 2e-5])
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        epochs = trial.suggest_int("epochs", 3, 4)

        model = SentenceTransformer(args.model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        all_pairs = load_pairs(args.data_path)
        train_pairs = all_pairs[: int(len(all_pairs) * 0.9)]
        val_pairs = all_pairs[int(len(all_pairs) * 0.9) :]
        if len(val_pairs) < batch_size:
            val_pairs = train_pairs[:batch_size]

        train_loader = create_no_duplicates_dataloader(
            train_pairs, batch_size, shuffle=True
        )
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

        out_dir = tempfile.mkdtemp(prefix=f"ckpt_{uuid.uuid4().hex[:6]}_")
        try:
            model.fit(
                train_objectives=[(train_loader, train_loss)],
                epochs=epochs,
                optimizer_params={"lr": lr},
                output_path=out_dir,
                show_progress_bar=False,
            )
            # Validation
            val_batch = val_pairs[:batch_size]
            queries = [ex.texts[0] for ex in val_batch]
            docs = [ex.texts[1] for ex in val_batch]
            with torch.no_grad():
                emb_q = model.encode(
                    queries,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,
                )
                emb_p = model.encode(
                    docs,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,
                )
            scores = cos_sim(emb_q, emb_p) * train_loss.scale
            targets = torch.arange(scores.size(0), device=scores.device)
            val_loss = F.cross_entropy(scores, targets).item()
        except Exception as e:
            print(f"Training failed: {e}")
            return float("inf")
        finally:
            try:
                del model, emb_q, emb_p
            except:
                pass
            torch.cuda.empty_cache()
            shutil.rmtree(out_dir, ignore_errors=True)
        return val_loss

    # Run Optuna study
    study = optuna.create_study(direction="minimize", study_name="bge_m3_hpo_mnrl")
    print("Starting hyperparameter optimization...")
    study.optimize(
        objective, n_trials=args.trials, timeout=args.timeout, callbacks=callbacks
    )

    print("Best params:", study.best_trial.params)
    print("Lowest validation MNRL:", study.best_trial.value)

    # Train final model
    print("\nTraining final model with best parameters...")
    best_params = study.best_trial.params
    model = SentenceTransformer(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    all_pairs = load_pairs(args.data_path)
    train_loader = create_no_duplicates_dataloader(
        all_pairs, best_params["batch_size"], shuffle=True
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=best_params["epochs"],
        optimizer_params={"lr": best_params["learning_rate"]},
        output_path=args.output_dir,
        show_progress_bar=True,
    )
    print(f"Final model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
