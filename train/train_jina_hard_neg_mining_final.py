import os

os.environ.update(
    {
        "HF_HUB_DISABLE_PROGRESS_BARS": "0",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "WANDB_DISABLED": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",  # Avoid parallelism warnings
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:32",  # Help with memory fragmentation
        "HF_DATASETS_OFFLINE": "1",  # Ensure datasets are loaded offline
        "CUDA_LAUNCH_BLOCKING": "1",  # For more deterministic behavior
    }
)

import torch
import json
import random
import gc
from datetime import datetime
from sentence_transformers import SentenceTransformer, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import (
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
)
from datasets import Dataset
from sentence_transformers.hard_negatives import min_hard_negatives
import optuna
from optuna.integration import Accelerator
from sentence_transformers import BatchSamplers
from sentence_transformers.util import cos_sim
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    return [(entry["query"], entry["passage"]) for entry in pairs]


def load_student_model(model_path):
    return SentenceTransformer(
        model_path,
        trust_remote_code=True,
        model_kwargs={"default_task": "text-matching"},
        local_files_only=True,
    ).to(device)


def load_teacher_model(model_path):
    return SentenceTransformer(
        model_path,
        local_files_only=True,
    ).to(device)


def parse_args():
    parser = ArgumentParser(
        description="Hard Negative Mining with Sentence Transformers"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./train/pairs_data.json",
        help="Dataset path containing query-passage pairs",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="jinaai/jina-embeddings-v3",
        help="Student model name or path",
    )
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default="intfloat/e5-mistral-7b-instruct",
        help="Teacher model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./train/best/hard_neg_mining",
        help="Directory to save the output model and results",
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout for Optuna")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./train/logs",
        help="Directory to save TensorBoard logs",
    )
    return parser.parse_args()


def encode_in_chunks(model, sentences, batch_size=32):
    """Encode sentences in smaller chunks to save memory."""
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        with torch.no_grad():
            embeddings = model.encode(
                chunk, convert_to_tensor=True, normalize_embeddings=True
            )
            all_embeddings.append(embeddings.detach())

    # Concatenate all chunk embeddings
    if len(all_embeddings) == 1:
        return all_embeddings[0]
    return torch.cat(all_embeddings, dim=0)


def perform_hard_negative_mining(dataset, teacher_model, rel_margin, top_k, batch_size):
    anchors, positives = zip(*dataset)

    ds = Dataset.from_dict(
        {
            "anchor": list(anchors),
            "positive": list(positives),
        }
    )

    # Run the hard negative mining with memory cleanup
    raw = min_hard_negatives(
        dataset=ds,
        model=teacher_model,
        anchor_column_name="anchor",
        positive_column_name="positive",
        relative_margin=rel_margin,
        num_negatives=top_k,
        sampling_strategy="top",
        output_format="n-tuple",
        batch_size=batch_size,
        use_faiss=True,
        verbose=False,
    )

    final_mined = []

    for entry in raw:
        q, p = entry["anchor"], entry["positive"]
        negs = [
            entry[key]
            for key in sorted(
                entry, key=lambda k: int(k.split("_")[1]) if "_" in k else 0
            )
            if key.startswith("negative_")
        ]
        if not negs:
            continue
        strongest = negs[0]
        rest = negs[1:]
        sampled = random.sample(rest, k=min(len(rest), top_k - 1))
        negatives = [strongest] + sampled
        final_mined.append((q, p, *negatives))

    anchors, positives, *neg_cols = zip(*final_mined)

    data = {
        "anchor": list(anchors),
        "positive": list(positives),
    }

    for idx, col in enumerate(neg_cols, start=1):
        data[f"negative_{idx}"] = list(col)

    final_ds = Dataset.from_dict(data)

    # Help clear memory
    del raw, anchors, positives, neg_cols, data
    torch.cuda.empty_cache()
    gc.collect()

    return final_ds


class TensorBoardCallback:
    """Custom callback for logging metrics to TensorBoard."""

    def __init__(self, writer, log_steps=10):
        self.writer = writer
        self.log_steps = log_steps
        self.global_step = 0
        self.best_val_loss = float("inf")

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        """Log training metrics after each step."""
        if state.global_step % self.log_steps == 0 and state.global_step > 0:
            if hasattr(state, "train_loss"):
                self.writer.add_scalar(
                    "train/loss", state.train_loss, state.global_step
                )
            if optimizer is not None:
                self.writer.add_scalar(
                    "train/learning_rate",
                    optimizer.param_groups[0]["lr"],
                    state.global_step,
                )
            self.global_step = state.global_step

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics."""
        if metrics:
            step = state.global_step if state.global_step else self.global_step
            # Log all evaluation metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"eval/{key}", value, step)

            # Track best model
            if "eval_loss" in metrics and metrics["eval_loss"] < self.best_val_loss:
                self.best_val_loss = metrics["eval_loss"]
                self.writer.add_scalar("eval/best_loss", self.best_val_loss, step)


def main():
    accelerator = Accelerator()
    print(f"GPU nums: {accelerator.num_processes}")

    args = parse_args()

    # Create TensorBoard log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.log_dir) / f"hard_neg_mining_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"TensorBoard logs will be saved to: {log_dir}")
    print("To view logs during training, open a new terminal and run:")
    print(f"tensorboard --logdir={log_dir}")
    print("Then open a browser and go to http://localhost:6006")

    # Create a TensorBoard writer for the main training process
    main_writer = SummaryWriter(log_dir / "main")

    # Track optuna trials in TensorBoard
    def objective(trial):
        # Force garbage collection at start of each trial
        gc.collect()
        torch.cuda.empty_cache()

        # Create a TensorBoard writer for this specific trial
        trial_writer = SummaryWriter(log_dir / f"trial_{trial.number}")

        lr = trial.suggest_categorical("learning_rate", [5e-5, 3e-5, 2e-5])
        batch_size = 4
        epochs = trial.suggest_int("epochs", 4, 10)
        top_k = trial.suggest_categorical("top_k_hard_neg_samples", [10, 20, 30, 40])
        perc_threshold = trial.suggest_float("perc_threshold", 0.90, 0.95)

        # Log hyperparameters to TensorBoard
        hparam_dict = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "top_k_negatives": top_k,
            "threshold": perc_threshold,
        }
        trial_writer.add_hparams(hparam_dict, {})

        trial_writer.add_text("trial_info", f"Trial {trial.number}: {hparam_dict}")

        student = load_student_model(args.model_name)
        teacher = load_teacher_model(args.teacher_model_name)

        pairs = load_pairs(args.data_path)
        split = int(len(pairs) * 0.7)
        train_pairs, val_pairs = pairs[:split], pairs[split:]

        rel_margin = 1.0 - perc_threshold

        trial_writer.add_scalar("hyperparams/rel_margin", rel_margin, 0)
        trial_writer.add_text(
            "dataset_info",
            f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}",
        )

        train_examples = perform_hard_negative_mining(
            dataset=train_pairs,
            teacher_model=teacher,
            rel_margin=rel_margin,
            top_k=top_k,
            batch_size=batch_size,
        )

        trial_writer.add_scalar("dataset/train_examples", len(train_examples), 0)

        dataset_length = len(train_examples)
        steps_per_epoch = dataset_length // batch_size
        max_steps = epochs * steps_per_epoch

        training_args = SentenceTransformerTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            max_steps=max_steps,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=100,
            load_best_model_at_end=True,
            # Enable TensorBoard reporting in SentenceTransformer
            report_to="tensorboard",
            logging_dir=str(log_dir / f"trial_{trial.number}"),
        )

        train_loss = losses.MultipleNegativesRankingLoss(model=student)

        # Create a TensorBoard callback
        tb_callback = TensorBoardCallback(trial_writer)

        trainer = SentenceTransformerTrainer(
            model=student,
            args=training_args,
            train_dataset=train_examples,
            loss=train_loss,
            callbacks=[tb_callback],
        )

        trainer.train()

        # Free up trainer resources
        del trainer, train_loss, train_examples
        torch.cuda.empty_cache()
        gc.collect()

        # Validation in smaller chunks to reduce memory usage
        student.eval()

        q_list = [ex[0] for ex in val_pairs]
        p_list = [ex[1] for ex in val_pairs]

        # Process in chunks to save memory
        with torch.no_grad():
            emb_q = encode_in_chunks(student, q_list)
            emb_p = encode_in_chunks(student, p_list)

            # Calculate scores and loss
            scores = cos_sim(emb_q, emb_p)
            targets = torch.arange(scores.size(0), device=scores.device)
            val_loss = torch.nn.functional.cross_entropy(scores, targets).item()

            # Log validation results
            trial_writer.add_scalar("validation/loss", val_loss, 0)

            # Clean up tensors immediately
            del scores, targets, emb_q, emb_p

        # Clear student and teacher models from GPU
        student.to("cpu")
        teacher.to("cpu")
        del student, teacher

        # Close the trial's TensorBoard writer
        trial_writer.close()

        torch.cuda.empty_cache()
        gc.collect()

        return val_loss

    print("start optim")

    study = optuna.create_study(direction="minimize", study_name="hpo_hard_neg_triplet")
    study.optimize(
        objective, n_trials=args.trials, timeout=args.timeout, gc_after_trial=True
    )

    best = study.best_trial.params
    print("Best params:", best)

    # Log best parameters to TensorBoard
    main_writer.add_text(
        "best_params", f"Best trial: {study.best_trial.number}\n{best}"
    )
    for param_name, param_value in best.items():
        main_writer.add_scalar(f"best_params/{param_name}", param_value, 0)

    # start fine-tuning with best params
    student = load_student_model(args.model_name)
    student.to(device)
    teacher = load_teacher_model(args.teacher_model_name)
    teacher.to(device)
    pairs = load_pairs(args.data_path)

    # Create final training TensorBoard writer
    final_writer = SummaryWriter(log_dir / "final_training")

    final_writer.add_text(
        "training_info", f"Final training with best parameters: {best}"
    )

    final_train_examples = perform_hard_negative_mining(
        dataset=pairs,
        teacher_model=teacher,
        rel_margin=1.0 - best["perc_threshold"],
        top_k=best["top_k_hard_neg_samples"],
        batch_size=4,
    )
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=best["epochs"],
        learning_rate=best["learning_rate"],
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=10,
        # Enable TensorBoard reporting
        report_to="tensorboard",
        logging_dir=str(log_dir / "final_training"),
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=student)

    # Create TensorBoard callback for final training
    final_tb_callback = TensorBoardCallback(final_writer)

    trainer = SentenceTransformerTrainer(
        model=student,
        args=training_args,
        train_dataset=final_train_examples,
        loss=train_loss,
        callbacks=[final_tb_callback],
    )

    trainer.train()
    student.save(args.output_dir)

    # Add model architecture to TensorBoard
    try:
        dummy_input = torch.zeros((1, 768), device="cpu")  # Example input tensor
        final_writer.add_graph(student, dummy_input)
    except Exception as e:
        print(f"Could not add model graph to TensorBoard: {e}")

    # Close all TensorBoard writers
    final_writer.close()
    main_writer.close()

    torch.cuda.empty_cache()
    print(f"Model saved to {args.output_dir}")
    print(f"TensorBoard logs saved to {log_dir}")
    print("To view the final training logs, run:")
    print(f"tensorboard --logdir={log_dir}")
    print("Then open a browser and go to http://localhost:6006")


if __name__ == "__main__":
    main()
