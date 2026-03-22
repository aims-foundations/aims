"""
Curate IgakuQA Japanese medical benchmark into response matrix + item_content.csv.

Rows = models (5 baselines), Columns = items (problem_id across all exam years).
Values = 1 if prediction matches gold answer, 0 otherwise.

For multi-answer questions, prediction must match the full answer set (order-insensitive).
Only text-only questions are included (no image-dependent questions).
"""

import json
import os
import csv
import numpy as np

RAW = "/lfs/skampere2/0/sttruong/aims/src/data/curated_benchmarks/igakuqa_data/raw/IgakuQA"
OUT = "/lfs/skampere2/0/sttruong/aims/src/data/curated_benchmarks/igakuqa_data/processed"

YEARS = ["2018", "2019", "2020", "2021", "2022"]
MODELS = ["gpt3", "chatgpt", "gpt4", "student-majority", "translate_chatgpt-en"]
MODEL_DISPLAY = {
    "gpt3": "GPT-3",
    "chatgpt": "ChatGPT",
    "gpt4": "GPT-4",
    "student-majority": "Student-Majority",
    "translate_chatgpt-en": "Translate-ChatGPT-EN",
}

# Step 1: Load all questions (gold answers + text)
# Only keep text_only questions
questions = {}  # problem_id -> {answer, problem_text, choices, points, exam, text_only}

for year in YEARS:
    data_dir = os.path.join(RAW, "data", year)
    for fn in sorted(os.listdir(data_dir)):
        if "_" in fn:  # skip metadata/translate files
            continue
        exam_name = fn.replace(".jsonl", "")
        with open(os.path.join(data_dir, fn)) as f:
            for line in f:
                d = json.loads(line)
                if not d["text_only"]:
                    continue
                questions[d["problem_id"]] = {
                    "answer": sorted(d["answer"]),
                    "problem_text": d["problem_text"],
                    "choices": d["choices"],
                    "points": d.get("points", "1"),
                    "exam": exam_name,
                    "year": year,
                }

# Sort problem_ids for consistent ordering
problem_ids = sorted(questions.keys())
pid_to_idx = {pid: i for i, pid in enumerate(problem_ids)}

print(f"Total text-only items: {len(problem_ids)}")

# Step 2: Load baseline results and score them
n_models = len(MODELS)
n_items = len(problem_ids)
response_matrix = np.full((n_models, n_items), np.nan)

for year in YEARS:
    results_dir = os.path.join(RAW, "baseline_results", year)
    for fn in sorted(os.listdir(results_dir)):
        # Parse model name from filename like "112-A_gpt4.jsonl"
        # Split on first underscore after exam name
        base = fn.replace(".jsonl", "")
        # exam part is like "112-A", model part is after the first underscore
        parts = base.split("_", 1)
        if len(parts) != 2:
            continue
        exam_name, model_name = parts

        if model_name not in MODELS:
            # Handle "translate_chatgpt-en" which has an extra underscore
            # Try joining with next underscore
            parts2 = base.split("_", 2)
            if len(parts2) == 3:
                model_name = parts2[1] + "_" + parts2[2]
            if model_name not in MODELS:
                print(f"  Unknown model: {model_name} in {fn}")
                continue

        model_idx = MODELS.index(model_name)

        with open(os.path.join(results_dir, fn)) as f:
            for line in f:
                d = json.loads(line)
                pid = d["problem_id"]
                if pid not in pid_to_idx:
                    continue  # non-text-only question, skip

                item_idx = pid_to_idx[pid]
                pred = sorted(d["prediction"].split(","))
                gold = questions[pid]["answer"]
                correct = 1 if pred == gold else 0
                response_matrix[model_idx, item_idx] = correct

# Step 3: Check coverage
for i, model in enumerate(MODELS):
    answered = np.sum(~np.isnan(response_matrix[i]))
    correct = int(np.nansum(response_matrix[i]))
    print(f"  {MODEL_DISPLAY[model]}: {int(answered)}/{n_items} items, {correct} correct ({correct/answered*100:.1f}%)")

# Check for any items with no responses
items_with_responses = np.sum(~np.isnan(response_matrix), axis=0)
missing_items = np.sum(items_with_responses == 0)
if missing_items > 0:
    print(f"  WARNING: {missing_items} items have no model responses")

# Step 4: Save response matrix
# Convert NaN to empty string for CSV
model_names = [MODEL_DISPLAY[m] for m in MODELS]
header = ["model_id"] + problem_ids

with open(os.path.join(OUT, "response_matrix.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i, model in enumerate(model_names):
        row = [model]
        for j in range(n_items):
            val = response_matrix[i, j]
            if np.isnan(val):
                row.append("")
            else:
                row.append(int(val))
        writer.writerow(row)

print(f"\nSaved response_matrix.csv: {n_models} models x {n_items} items")

# Step 5: Save item_content.csv
with open(os.path.join(OUT, "item_content.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["item_id", "exam", "year", "problem_text", "choices", "answer", "points"])
    for pid in problem_ids:
        q = questions[pid]
        choices_str = " | ".join(q["choices"])
        answer_str = ",".join(q["answer"])
        writer.writerow([pid, q["exam"], q["year"], q["problem_text"], choices_str, answer_str, q["points"]])

print(f"Saved item_content.csv: {n_items} items")

# Step 6: Summary
print(f"\n=== Summary ===")
print(f"Benchmark: IgakuQA (Japanese Medical Exam)")
print(f"Items: {n_items} (text-only, across {len(YEARS)} exam years: {', '.join(YEARS)})")
print(f"Models: {n_models} ({', '.join(model_names)})")
print(f"Response matrix shape: {n_models} x {n_items}")
nan_count = int(np.sum(np.isnan(response_matrix)))
total_cells = n_models * n_items
print(f"Missing values: {nan_count}/{total_cells} ({nan_count/total_cells*100:.1f}%)")
print(f"Overall accuracy: {np.nanmean(response_matrix)*100:.1f}%")
