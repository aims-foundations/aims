"""
Curate multimodal VLM benchmarks from OpenVLMRecords into response matrices.

Downloads xlsx results from HuggingFace VLMEval/OpenVLMRecords,
extracts correctness, and builds response matrices (models × items).

Target benchmarks:
- MMBench_V11: MCQ vision understanding (dev split)
- MME: binary yes/no vision perception + cognition
- MMMU_DEV_VAL: college-level multimodal questions (validation split)
- AI2D_TEST: science diagram understanding
- HallusionBench: hallucination detection
- MathVista_MINI: mathematical reasoning with vision
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import list_repo_tree, hf_hub_download

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ID = "VLMEval/OpenVLMRecords"
BASE_DIR = Path("/lfs/skampere2/0/sttruong/aims/src/data/curated_benchmarks")

# Benchmark configurations
BENCHMARKS = {
    "MMBench_V11": {
        "suffix": "MMBench_V11",
        "output_dir": "mmbench_v11_data",
        "type": "mcq",  # multiple choice, answer is a letter
        "split_filter": "dev",  # use dev split (has ground truth)
        "split_col": "split",
    },
    "MME": {
        "suffix": "MME",
        "output_dir": "mme_data",
        "type": "yesno",  # binary yes/no
        "split_filter": None,
        "split_col": None,
    },
    "MMMU_DEV_VAL": {
        "suffix": "MMMU_DEV_VAL",
        "output_dir": "mmmu_dev_val_data",
        "type": "mcq",
        "split_filter": "validation",  # use validation split
        "split_col": "split",
    },
    "AI2D_TEST": {
        "suffix": "AI2D_TEST",
        "output_dir": "ai2d_test_data",
        "type": "mcq",
        "split_filter": None,
        "split_col": None,
    },
    "HallusionBench": {
        "suffix": "HallusionBench",
        "output_dir": "hallusionbench_data",
        "type": "yesno",
        "split_filter": None,
        "split_col": None,
    },
    "MathVista_MINI": {
        "suffix": "MathVista_MINI",
        "output_dir": "mathvista_mini_data",
        "type": "mixed",  # has both MCQ and free-form
        "split_filter": None,
        "split_col": None,
    },
}


def list_all_models():
    """List all model directories in the repo."""
    items = list(list_repo_tree(REPO_ID, path_in_repo="mmeval", repo_type="dataset"))
    models = []
    for item in items:
        name = item.path.replace("mmeval/", "")
        if "/" not in name:  # top-level directory = model name
            models.append(name)
    return sorted(models)


def find_models_for_benchmark(all_models, benchmark_suffix):
    """Find which models have results for a given benchmark."""
    models_with_bench = []
    for model in all_models:
        try:
            items = list(list_repo_tree(REPO_ID, path_in_repo=f"mmeval/{model}", repo_type="dataset"))
            filenames = [item.path.split("/")[-1] for item in items]
            target = f"{model}_{benchmark_suffix}.xlsx"
            if target in filenames:
                models_with_bench.append(model)
        except Exception as e:
            pass
    return models_with_bench


def extract_answer_letter(prediction_text, choices=("A", "B", "C", "D")):
    """Extract the answer letter from a model's free-text prediction."""
    if pd.isna(prediction_text):
        return None
    pred = str(prediction_text).strip()

    # If prediction is just a single letter
    if pred.upper() in choices:
        return pred.upper()

    # Pattern: "The answer is X" or "answer: X"
    m = re.search(r"(?:answer\s*(?:is|:)\s*)([A-D])\b", pred, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern: **X** or **X.** at the start or after newline
    m = re.search(r"\*\*([A-D])\b", pred)
    if m:
        return m.group(1).upper()

    # Pattern: starts with "X." or "X:"  or "(X)"
    m = re.match(r"^[(\s]*([A-D])[.):\s]", pred)
    if m:
        return m.group(1).upper()

    # Look for first standalone letter A-D
    m = re.search(r"\b([A-D])\b", pred)
    if m:
        return m.group(1).upper()

    return None


def extract_yesno(prediction_text):
    """Extract yes/no from a model's prediction."""
    if pd.isna(prediction_text):
        return None
    pred = str(prediction_text).strip().lower()

    if pred in ("yes", "no"):
        return pred.capitalize()

    # Check if starts with yes/no
    if pred.startswith("yes"):
        return "Yes"
    if pred.startswith("no"):
        return "No"

    # Search for yes/no in text
    if re.search(r"\byes\b", pred, re.IGNORECASE):
        if not re.search(r"\bno\b", pred, re.IGNORECASE):
            return "Yes"
        # Both present - check which comes first or last
        yes_pos = re.search(r"\byes\b", pred, re.IGNORECASE).start()
        no_pos = re.search(r"\bno\b", pred, re.IGNORECASE).start()
        # Take last occurrence as final answer
        return "No" if no_pos > yes_pos else "Yes"
    if re.search(r"\bno\b", pred, re.IGNORECASE):
        return "No"

    return None


def score_mcq(row):
    """Score a multiple choice question. Returns 1 if correct, 0 if wrong, NaN if can't parse."""
    pred_letter = extract_answer_letter(str(row["prediction"]))
    if pred_letter is None:
        return np.nan
    return 1 if pred_letter == row["answer"] else 0


def score_yesno(row):
    """Score a yes/no question."""
    pred = extract_yesno(str(row["prediction"]))
    gold = str(row["answer"]).strip().capitalize()
    if pred is None:
        return np.nan
    return 1 if pred == gold else 0


def score_mathvista(row):
    """Score MathVista which has mixed question types."""
    answer_type = row.get("answer_type", "")
    if answer_type == "multi_choice":
        return score_mcq(row)
    else:
        # Free-form: compare normalized answers
        pred = str(row["prediction"]).strip().lower()
        gold = str(row["answer"]).strip().lower()

        # Try numeric comparison
        try:
            pred_num = float(re.search(r"[-+]?\d*\.?\d+", pred).group())
            gold_num = float(gold)
            return 1 if abs(pred_num - gold_num) < 1e-6 else 0
        except:
            pass

        # String comparison
        if gold in pred or pred == gold:
            return 1
        return 0


def process_benchmark(bench_name, config, all_models):
    """Process a single benchmark: download all model results, build response matrix."""
    print(f"\n{'='*60}")
    print(f"Processing: {bench_name}")
    print(f"{'='*60}")

    suffix = config["suffix"]
    output_dir = BASE_DIR / config["output_dir"] / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find models with this benchmark
    print(f"Scanning {len(all_models)} models for {suffix}...")
    models_with_bench = find_models_for_benchmark(all_models, suffix)
    print(f"Found {len(models_with_bench)} models with {suffix}")

    if len(models_with_bench) == 0:
        print(f"No models found for {bench_name}, skipping.")
        return

    # Download and process each model's results
    all_scores = {}
    item_info = None
    failed_models = []

    for i, model in enumerate(models_with_bench):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Downloading {i+1}/{len(models_with_bench)}: {model}")

        try:
            fpath = f"mmeval/{model}/{model}_{suffix}.xlsx"
            local_path = hf_hub_download(REPO_ID, fpath, repo_type="dataset")
            df = pd.read_excel(local_path)

            # Apply split filter if needed
            if config["split_filter"] and config["split_col"] and config["split_col"] in df.columns:
                df = df[df[config["split_col"]] == config["split_filter"]].copy()

            if len(df) == 0:
                failed_models.append((model, "empty after split filter"))
                continue

            # Ensure index column exists
            if "index" not in df.columns:
                df["index"] = range(len(df))

            # Score each item
            if config["type"] == "mcq":
                df["score"] = df.apply(score_mcq, axis=1)
            elif config["type"] == "yesno":
                df["score"] = df.apply(score_yesno, axis=1)
            elif config["type"] == "mixed":
                df["score"] = df.apply(score_mathvista, axis=1)

            # Store scores indexed by item index
            scores = df.set_index("index")["score"]
            all_scores[model] = scores

            # Capture item info from first successful model
            if item_info is None:
                item_cols = ["index", "question"]
                if "category" in df.columns:
                    item_cols.append("category")
                if "l2-category" in df.columns:
                    item_cols.append("l2-category")
                if "answer" in df.columns:
                    item_cols.append("answer")
                if "question_type" in df.columns:
                    item_cols.append("question_type")
                if "answer_type" in df.columns:
                    item_cols.append("answer_type")
                if "hint" in df.columns:
                    item_cols.append("hint")
                if "A" in df.columns:
                    for c in ["A", "B", "C", "D"]:
                        if c in df.columns:
                            item_cols.append(c)
                item_info = df[item_cols].copy()

        except Exception as e:
            failed_models.append((model, str(e)[:100]))

    if not all_scores:
        print(f"No successful downloads for {bench_name}")
        return

    # Build response matrix
    print(f"\nBuilding response matrix...")
    response_matrix = pd.DataFrame(all_scores)

    # Ensure consistent item ordering
    response_matrix = response_matrix.sort_index()

    # Transpose so rows = models, columns = items
    response_matrix_T = response_matrix.T
    response_matrix_T.index.name = "model"

    # Rename columns to item_0, item_1, etc. but keep original index mapping
    item_indices = response_matrix_T.columns.tolist()

    # Calculate statistics
    n_models = len(response_matrix_T)
    n_items = len(response_matrix_T.columns)
    model_acc = response_matrix_T.mean(axis=1)
    item_acc = response_matrix_T.mean(axis=0)
    missing_rate = response_matrix_T.isna().mean().mean()

    print(f"  Models: {n_models}")
    print(f"  Items: {n_items}")
    print(f"  Missing rate: {missing_rate:.3f}")
    print(f"  Model accuracy range: {model_acc.min():.3f} - {model_acc.max():.3f}")
    print(f"  Item accuracy range: {item_acc.min():.3f} - {item_acc.max():.3f}")

    # Save response matrix (models x items)
    response_matrix_T.to_csv(output_dir / "response_matrix.csv")

    # Save item content
    if item_info is not None:
        item_info = item_info.drop_duplicates(subset=["index"]).sort_values("index")
        item_info.to_csv(output_dir / "item_content.csv", index=False)

    # Save metadata
    metadata = {
        "benchmark": bench_name,
        "source": f"HuggingFace VLMEval/OpenVLMRecords",
        "n_models": n_models,
        "n_items": n_items,
        "missing_rate": float(missing_rate),
        "scoring": config["type"],
        "split_filter": config["split_filter"],
        "models": sorted(response_matrix_T.index.tolist()),
        "failed_models": [(m, e) for m, e in failed_models],
        "model_accuracy": {m: float(a) for m, a in model_acc.items()},
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved to {output_dir}")
    if failed_models:
        print(f"  Failed models ({len(failed_models)}): {[m for m,_ in failed_models[:5]]}...")

    return response_matrix_T


def main():
    print("Listing all models in VLMEval/OpenVLMRecords...")
    all_models = list_all_models()
    print(f"Found {len(all_models)} models total")

    results = {}
    for bench_name, config in BENCHMARKS.items():
        try:
            rm = process_benchmark(bench_name, config, all_models)
            if rm is not None:
                results[bench_name] = rm
        except Exception as e:
            print(f"ERROR processing {bench_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, rm in results.items():
        print(f"  {name}: {rm.shape[0]} models × {rm.shape[1]} items, "
              f"mean acc = {rm.mean().mean():.3f}")


if __name__ == "__main__":
    main()
