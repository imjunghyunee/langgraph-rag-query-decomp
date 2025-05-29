import os
import sys
import json
import copy
import time
import importlib
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from evaluation.eval_utils import math_grader, science_grader
from rag_pipeline import config
from evaluation.eval_vectordb import load_dataset, extract_box_content

INPUT_PATH = Path("./evaluation/eval_utils/eval_neamen.jsonl")
OUTPUT_DIR = Path("./eval_output")
RESULTS_PATH = OUTPUT_DIR / "comparison_results.json"
LOG_PATH = OUTPUT_DIR / "comparison_log.txt"

# Define all setting combinations to test
QUERY_TYPES = [None, "hyde", "summary"]
HYBRID_WEIGHTS = [[0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3]]
TOP_K_VALUES = [1, 2, 3, 4, 5, 10]
SIM_THRESHOLDS = [0.01, 0.65, 0.7, 0.75]
RERANK_OPTIONS = [True, False]

# Make sure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log_message(message: str):
    """Write log message to console and log file"""
    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def update_config(top_k: int, sim_threshold: float, rerank: bool):
    """Update config.py file with new values"""
    os.environ["TOP_K"] = str(top_k)
    os.environ["SIM_THRESHOLD"] = str(sim_threshold)
    os.environ["RERANK"] = str(rerank)

    # Force reload any modules that might have imported config
    importlib.reload(config)

    log_message(
        f"Updated config: TOP_K={top_k}, SIM_THRESHOLD={sim_threshold}, RERANK={rerank}"
    )


def run_evaluation(
    query_type: Optional[str],
    hybrid_weights: Optional[List[float]],
    top_k: int,
    sim_threshold: float,
    rerank: bool,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Run evaluation for a specific configuration
    Returns (score, total_count, detailed_results)
    """
    # Update configuration
    update_config(top_k, sim_threshold, rerank)

    # Import run function after config update to ensure it uses updated settings
    from main import run

    # Load dataset
    data = load_dataset(INPUT_PATH)
    total_records = len(data)
    score = 0
    detailed_results = []

    log_message(
        f"Starting evaluation with settings: query_type={query_type}, "
        f"hybrid_weights={hybrid_weights}, TOP_K={top_k}, "
        f"SIM_THRESHOLD={sim_threshold}, RERANK={rerank}"
    )

    # Loop through each problem in dataset
    for record in tqdm(data, total=total_records, desc="Evaluating"):
        try:
            # Run the RAG pipeline with current settings
            graph_state = run(
                record["problem"],
                pdf_path=None,
                query_type=query_type,
                hybrid_weights=hybrid_weights,
            )

            llm_output = graph_state["answer"]
            predicted_answer = extract_box_content(llm_output)
            gt_answer = record["answer"]

            # Grade the answer
            answer_correct = science_grader.numerical_approx_match(
                predicted_answer,
                gt_answer,
                tolerance=1e-2,
            )

            if answer_correct == 1.0:
                answer_correct_str = "Yes"
                score += 1
            else:
                answer_correct_str = "No"

            # Store detailed result
            detailed_results.append(
                {
                    "id": record["id"],
                    "problem": record["problem"],
                    "explanation": graph_state["explanation"],
                    "context": graph_state["context"],
                    "score": graph_state["score"],
                    "predicted_answer": predicted_answer,
                    "gt_answer": gt_answer,
                    "answer_correct": answer_correct_str,
                }
            )

        except Exception as e:
            tqdm.write(f"[Error id={record['id']}] {str(e)}")
            detailed_results.append(
                {"id": record["id"], "problem": record["problem"], "error": str(e)}
            )

    log_message(f"Evaluation completed with score: {score}/{total_records}")
    return score, total_records, detailed_results


def main():
    # Check if dataset exists
    if not INPUT_PATH.is_file():
        log_message(f"Error: Input file not found at {INPUT_PATH}")
        sys.exit(1)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear log file
    with open(LOG_PATH, "w") as f:
        f.write(f"Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Results will contain all evaluation outcomes
    all_results = []

    # Get total number of combinations to test
    total_combinations = (
        len(QUERY_TYPES)
        * len(HYBRID_WEIGHTS)
        * len(TOP_K_VALUES)
        * len(SIM_THRESHOLDS)
        * len(RERANK_OPTIONS)
    )
    log_message(f"Total combinations to evaluate: {total_combinations}")

    # Track best configuration
    best_score = 0
    best_config = {}

    # Progress counter
    current_combination = 0

    try:
        # Iterate through all combinations
        for query_type in QUERY_TYPES:
            for hybrid_weights in HYBRID_WEIGHTS:
                for top_k in TOP_K_VALUES:
                    for sim_threshold in SIM_THRESHOLDS:
                        for rerank in RERANK_OPTIONS:
                            current_combination += 1
                            log_message(
                                f"\n--- Combination {current_combination}/{total_combinations} ---"
                            )
                            # Run evaluation for current configuration
                            config_name = f"type={query_type}_weights={hybrid_weights}_k={top_k}_sim={sim_threshold}_rerank={rerank}"
                            log_message(f"Testing configuration: {config_name}")

                            start_time = time.time()
                            score, total, details = run_evaluation(
                                query_type, hybrid_weights, top_k, sim_threshold, rerank
                            )
                            elapsed_time = time.time() - start_time

                            # Create result entry
                            result = {
                                "configuration": {
                                    "query_type": query_type,
                                    "hybrid_weights": hybrid_weights,
                                    "top_k": top_k,
                                    "sim_threshold": sim_threshold,
                                    "rerank": rerank,
                                },
                                "score": score,
                                "total": total,
                                "accuracy": (
                                    round(score / total * 100, 2) if total > 0 else 0
                                ),
                                "elapsed_seconds": round(elapsed_time, 2),
                                "detailed_results": details,
                            }

                            # Add to results list
                            all_results.append(result)

                            # Update best configuration if needed
                            if score > best_score:
                                best_score = score
                                best_config = result["configuration"]

                            # Save intermediate results
                            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                                json.dump(
                                    {
                                        "best_configuration": best_config,
                                        "best_score": best_score,
                                        "results": all_results,
                                    },
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )

                            log_message(f"Saved intermediate results to {RESULTS_PATH}")

    except KeyboardInterrupt:
        log_message("\nEvaluation interrupted by user. Saving partial results...")
    except Exception as e:
        log_message(f"\nError during evaluation: {str(e)}")

    # Save final results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_configuration": best_config,
                "best_score": best_score,
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log_message(f"\nEvaluation completed!")
    log_message(f"Best configuration: {best_config} with score {best_score}")
    log_message(f"All results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
