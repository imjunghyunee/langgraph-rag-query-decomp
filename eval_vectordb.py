import os
import sys
import json
import argparse
import subprocess
from tqdm import tqdm
from pathlib import Path
from main import run
import re
import torch
from evaluation import math_grader, science_grader

INPUT_PATH = Path("./eval_neamen.jsonl")
OUTPUT_PATH = Path("./eval_output/eval_results.json")
RAG_SCRIPT = Path("./main.py")


def load_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_box_content(text: str) -> str | None:
    key = r"\box"
    start = text.find(key)
    if start == -1:
        return None

    i = start + len(key)

    while i < len(text) and text[i] != "{":
        i += 1
    if i == len(text):
        return None

    open_cnt = 1
    content_start = i + 1
    i += 1

    while i < len(text):
        ch = text[i]
        if ch == "{":
            open_cnt += 1
        elif ch == "}":
            open_cnt -= 1
            if open_cnt == 0:
                return text[content_start:i]
        i += 1
    return None


def main():
    if not INPUT_PATH.is_file():
        print(f"input file not found", file=sys.stderr)
    if not RAG_SCRIPT.is_file():
        print(f"RAG script not found", file=sys.stderr)
        sys.exit(1)

    data = load_dataset(INPUT_PATH)
    total_records = len(data)
    use_cuda = torch.cuda.is_available()
    score = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for record in tqdm(data, total=total_records, desc="Evaluating"):
            try:
                graph_state = run(record["problem"])
            except Exception as e:
                tqdm.write(f"[Error id={record['id']}] {e}")
                continue

            llm_output = graph_state["answer"]
            predicted_answer = extract_box_content(llm_output)
            gt_answer = record["answer"]

            answer_correct = science_grader.numerical_approx_match(
                predicted_answer,
                gt_answer,
                tolerance=1e-2,
            )

            if answer_correct == 1.0:
                answer_correct = "Yes"
                score += 1
            else:
                answer_correct = "No"

            out_obj = {
                "id": record["id"],
                "problem": record["problem"],
                "context": graph_state["context"],
                "llm_output": llm_output,
                "predicted_answer": predicted_answer,
                "gt_answer": gt_answer,
                "answer_correct": answer_correct,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False, default=str) + "\n")

    print(f"Score: {score}")
    print(f"Finished! Results written at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
