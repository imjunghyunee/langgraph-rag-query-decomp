import os

os.environ.update(
    {
        "HF_HUB_DISABLE_PROGRESS_BARS": "0",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "WANDB_DISABLED": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
)

import torch
import json
import random
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

# [제안]
# import gc
# [제안] validation용 DataLoader
# from torch.utils.data import DataLoader, Dataset

''' 
[제안] Data Prefetching for Evaluation Implementation -> objective 함수에서 val 부분에 추가

def evaluate_with_prefetching(model, val_pairs, batch_size=16):
    """Evaluate model on validation pairs using efficient data prefetching"""
    from torch.utils.data import DataLoader, Dataset
    
    class PairDataset(Dataset):
        def __init__(self, pairs):
            self.queries = [p[0] for p in pairs]
            self.passages = [p[1] for p in pairs]
            
        def __len__(self):
            return len(self.queries)
            
        def __getitem__(self, idx):
            return self.queries[idx], self.passages[idx]
    
    # Create dataset and dataloader with prefetching
    dataset = PairDataset(val_pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,    # Load data using multiple processes
        pin_memory=True   # Use pinned memory for faster CPU->GPU transfer
    )
    
    # Process in batches with inference_mode for memory efficiency
    all_scores = []
    model.eval()  # Set model to evaluation mode
    
    with torch.inference_mode():  # More efficient than no_grad
        for q_batch, p_batch in dataloader:
            # Process batch
            emb_q = model.encode(q_batch, convert_to_tensor=True, normalize_embeddings=True)
            emb_p = model.encode(p_batch, convert_to_tensor=True, normalize_embeddings=True)
            
            # Calculate scores
            batch_scores = cos_sim(emb_q, emb_p)
            all_scores.append(batch_scores)
            
            # Free immediate batch tensors
            del emb_q
            del emb_p
            torch.cuda.empty_cache()

    # Concatenate results
    scores = torch.cat(all_scores, dim=0)
    targets = torch.arange(scores.size(0), device=scores.device)
    loss = torch.nn.functional.cross_entropy(scores, targets).item()
    
    return loss
'''

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


# [제안]
# teacher_device = "cuda:1" if torch.cuda.device_count() > 1 else "cpu"
"""
def load_teacher_model(model_path):
    return SentenceTransformer(
        model_path,
        local_files_only=True,
    ).to(teacher_device)
"""


def load_teacher_model(model_path):
    return SentenceTransformer(
        model_path,
        local_files_only=True,
    ).to(device)


# [제안] Use Gradient Checkpointing (https://hichoe95.tistory.com/124)
# # -> 성능 저하 가능성 있음.
"""
def load_teacher_model(model_path):
    model = SentenceTransformer(
        model_path,
        local_files_only=True,
    ).to(device)
    
    # Enable activation checkpointing on transformer models
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
    
    return model
"""


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

    return parser.parse_args()


def perform_hard_negative_mining(dataset, teacher_model, rel_margin, top_k, batch_size):
    anchors, positives = zip(*dataset)

    ds = Dataset.from_dict(
        {
            "anchor": list(anchors),
            "positive": list(positives),
        }
    )

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

    return final_ds


# [제안] Chunked Hard Negative Mining
'''
def perform_hard_negative_mining(dataset, teacher_model, rel_margin, top_k, batch_size):
    
    # Process in smaller chunks to prevent OOM errors
    chunk_size = 128  # Adjust based on available memory
    all_raw_results = []
        
    for i in range(0, len(dataset), chunk_size):
        end_idx = min(i + chunk_size, len(dataset))        
        chunk_dataset = dataset[i:end_idx]
        anchors, positives = zip(*chunk_dataset)
        
        ds = Dataset.from_dict({
            "anchor": list(anchors),
            "positive": list(positives),
        })
        
        # Process this chunk with smaller batch size for mining
        chunk_results = min_hard_negatives(
            dataset=ds,
            model=teacher_model,
            anchor_column_name="anchor",
            positive_column_name="positive",
            relative_margin=rel_margin,
            num_negatives=top_k,
            sampling_strategy="top",
            output_format="n-tuple",
            batch_size=min(batch_size, 2),  # Use smaller batch size for mining
            use_faiss=True,
            verbose=False,
        )

        # [제안] Use mixed precision
        """
        with torch.cuda.amp.autocast():  # Use mixed precision
            chunk_results = min_hard_negatives(
                dataset=ds,
                model=teacher_model,
                anchor_column_name="anchor",
                positive_column_name="positive",
                relative_margin=rel_margin,
                num_negatives=top_k,
                sampling_strategy="top",
                output_format="n-tuple",
                batch_size=min(batch_size, 2),  # Reduce batch size for mining
                use_faiss=True,
                verbose=False,
            )
        """
        
        # Add results from this chunk to our collection
        all_raw_results.extend(chunk_results)
        
        # Clear memory after each chunk
        torch.cuda.empty_cache()
        
    # Process the combined results (same logic as original function)
    final_mined = []
    
    for entry in all_raw_results:
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
        sampled = random.sample(rest, k=min(len(rest), top_k - 1)) if rest else []
        negatives = [strongest] + sampled
        final_mined.append((q, p, *negatives))
    
    if not final_mined:
        print("Warning: No hard negatives were found after mining.")
        return Dataset.from_dict({"anchor": [], "positive": []})
    
    anchors, positives, *neg_cols = zip(*final_mined)
    
    data = {
        "anchor": list(anchors),
        "positive": list(positives),
    }
    
    for idx, col in enumerate(neg_cols, start=1):
        data[f"negative_{idx}"] = list(col)
    
    final_ds = Dataset.from_dict(data)
    
    print(f"Final dataset contains {len(final_ds)} examples with hard negatives")
    return final_ds
'''


def main():
    accelerator = Accelerator()
    print(f"GPU nums: {accelerator.num_processes}")

    args = parse_args()

    def objective(trial):
        lr = trial.suggest_categorical("learning_rate", [5e-5, 3e-5, 2e-5])
        batch_size = 4
        epochs = trial.suggest_int("epochs", 4, 10)
        top_k = trial.suggest_categorical("top_k_hard_neg_samples", [10, 20, 30, 40])
        perc_threshold = trial.suggest_float("perc_threshold", 0.90, 0.95)

        student = load_student_model(args.model_name)
        # [제안] Use Gradient Checkpointing (https://hichoe95.tistory.com/124)
        # -> 성능 저하 가능성 있음.
        # student.gradient_checkpointing_enable()
        student.to(device)
        teacher = load_teacher_model(args.teacher_model_name)
        teacher.to(device)

        pairs = load_pairs(args.data_path)
        split = int(len(pairs) * 0.7)
        train_pairs, val_pairs = pairs[:split], pairs[split:]

        rel_margin = 1.0 - perc_threshold

        train_examples = perform_hard_negative_mining(
            dataset=train_pairs,
            teacher_model=teacher,
            rel_margin=rel_margin,
            top_k=top_k,
            batch_size=batch_size,
        )

        dataset_length = len(train_examples)
        steps_per_epoch = dataset_length // batch_size
        max_steps = epochs * steps_per_epoch

        training_args = SentenceTransformerTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=batch_size,
            # [제안] Use gradient accumulation steps to reduce memory usage
            # gradient_accumulation_steps=8,  # Add this
            num_train_epochs=epochs,
            learning_rate=lr,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            max_steps=max_steps,
            # [제안] Enable Mixed Precision Training
            # fp16=True
            # [제안] DeepApeed -> 성능 저하 가능성 있음.
            # deepspeed="ds_config.json"  # Add this line to enable DeepSpeed
        )
        """[제안] DeepSpeed
        위 방법 중 deepspeed를 사용하려면 pip install deepspeed 이후
        추가해야 하는 ds_config.json
        {
        "fp16": {
            "enabled": false
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
            },
            "allgather_partitions": true,
            "allgather_bucket_size": 5e8,
            "contiguous_gradients": true
        },
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 8
        }
        """

        train_loss = losses.MultipleNegativesRankingLoss(model=student)

        trainer = SentenceTransformerTrainer(
            model=student,
            args=training_args,
            train_dataset=train_examples,
            loss=train_loss,
        )
        trainer.train()

        # Validation
        student.eval()
        q_list = [ex[0] for ex in val_pairs]
        p_list = [ex[1] for ex in val_pairs]

        # [제안]
        # with torch.inference_mode():  # Even more efficient than no_grad
        # indentation 주의
        emb_q = student.encode(
            q_list, convert_to_tensor=True, normalize_embeddings=True
        )
        emb_p = student.encode(
            p_list, convert_to_tensor=True, normalize_embeddings=True
        )

        scores = cos_sim(emb_q, emb_p)
        targets = torch.arange(scores.size(0), device=scores.device)
        val_loss = torch.nn.functional.cross_entropy(scores, targets).item()

        # [제안] 위에서 정의한 evaluate_with_prefetching 함수 사용한다면

        # 아래 부분 대신:
        # ------------------------------------
        # student.eval()
        # q_list = [ex[0] for ex in val_pairs]
        # p_list = [ex[1] for ex in val_pairs]
        # emb_q = student.encode(
        #     q_list, convert_to_tensor=True, normalize_embeddings=True
        # )
        # emb_p = student.encode(
        #     p_list, convert_to_tensor=True, normalize_embeddings=True
        # )
        # scores = cos_sim(emb_q, emb_p)
        # targets = torch.arange(scores.size(0), device=scores.device)
        # val_loss = torch.nn.functional.cross_entropy(scores, targets).item()
        # ------------------------------------

        # 아래 코드 사용:
        # val_loss = evaluate_with_prefetching(student, val_pairs, batch_size=16)

        # [제안] Explicitly clear memory
        # del student
        # del teacher
        torch.cuda.empty_cache()
        # gc.collect()
        return val_loss

    print("start optim")

    study = optuna.create_study(direction="minimize", study_name="hpo_hard_neg_triplet")
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    # [제안] optuna gc after trial
    # study.optimize(objective, n_trials=args.trials, timeout=args.timeout, gc_after_trial=True)

    best = study.best_trial.params
    print("Best params:", best)

    # start fine-tuning with best params
    student = load_student_model(args.model_name)
    # [제안] Use Gradient Checkpointing (https://hichoe95.tistory.com/124)
    # -> 성능 저하 가능성 있음.
    # student.gradient_checkpointing_enable()
    student.to(device)
    teacher = load_teacher_model(args.teacher_model_name)
    teacher.to(device)
    pairs = load_pairs(args.data_path)

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
        # [제안] Use gradient accumulation steps to reduce memory usage
        # gradient_accumulation_steps=8,  # Add this
        num_train_epochs=best["epochs"],
        learning_rate=best["learning_rate"],
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # [제안] Enable Mixed Precision Training
        # fp16=True
        # [제안] DeepSpeed # -> 성능 저하 가능성 있음.
        # deepspeed="ds_config.json"  # Add this line to enable DeepSpeed
    )
    """[제안] DeepSpeed
    위 방법 중 deepspeed를 사용하려면 pip install deepspeed 이후

    추가해야 하는 ds_config.json :
    {
    "fp16": {
        "enabled": false
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8
    }
    """
    train_loss = losses.MultipleNegativesRankingLoss(model=student)
    trainer = SentenceTransformerTrainer(
        model=student,
        args=training_args,
        train_dataset=final_train_examples,
        loss=train_loss,
    )
    trainer.train()
    student.save(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    if __name__ == "__main__":
        main()
