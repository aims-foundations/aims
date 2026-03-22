"""
Curate multimodal VLM benchmarks from OpenVLMRecords into response matrices.
v2: Uses direct download attempts instead of listing directories to avoid rate limits.
"""

import os
import re
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import list_repo_tree, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ID = "VLMEval/OpenVLMRecords"
BASE_DIR = Path("/lfs/skampere2/0/sttruong/aims/src/data/curated_benchmarks")

BENCHMARKS = {
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
        "type": "mixed",
        "split_filter": None,
        "split_col": None,
    },
    "AI2D_TEST": {
        "suffix": "AI2D_TEST",
        "output_dir": "ai2d_test_data",
        "type": "mcq",
        "split_filter": None,
        "split_col": None,
    },
}


def get_all_models():
    """Get model list (cached from first run)."""
    cache_file = BASE_DIR / "model_list.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    items = list(list_repo_tree(REPO_ID, path_in_repo="mmeval", repo_type="dataset"))
    models = sorted([item.path.replace("mmeval/", "") for item in items if "/" not in item.path.replace("mmeval/", "")])
    with open(cache_file, "w") as f:
        json.dump(models, f)
    return models


def extract_answer_letter(prediction_text, choices=("A", "B", "C", "D")):
    if pd.isna(prediction_text):
        return None
    pred = str(prediction_text).strip()
    if pred.upper() in choices:
        return pred.upper()
    m = re.search(r"(?:answer\s*(?:is|:)\s*)([A-D])\b", pred, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\*\*([A-D])\b", pred)
    if m:
        return m.group(1).upper()
    m = re.match(r"^[(\s]*([A-D])[.):\s]", pred)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", pred)
    if m:
        return m.group(1).upper()
    return None


def extract_yesno(prediction_text):
    if pd.isna(prediction_text):
        return None
    pred = str(prediction_text).strip().lower()
    if pred in ("yes", "no"):
        return pred.capitalize()
    if pred.startswith("yes"):
        return "Yes"
    if pred.startswith("no"):
        return "No"
    yes_match = re.search(r"\byes\b", pred, re.IGNORECASE)
    no_match = re.search(r"\bno\b", pred, re.IGNORECASE)
    if yes_match and not no_match:
        return "Yes"
    if no_match and not yes_match:
        return "No"
    if yes_match and no_match:
        return "No" if no_match.start() > yes_match.start() else "Yes"
    return None


def score_mcq(row):
    pred_letter = extract_answer_letter(str(row["prediction"]))
    if pred_letter is None:
        return np.nan
    return 1 if pred_letter == row["answer"] else 0


def score_yesno(row):
    pred = extract_yesno(str(row["prediction"]))
    gold = str(row["answer"]).strip().capitalize()
    if pred is None:
        return np.nan
    return 1 if pred == gold else 0


def score_mathvista(row):
    answer_type = row.get("answer_type", "")
    if answer_type == "multi_choice":
        return score_mcq(row)
    else:
        pred = str(row["prediction"]).strip().lower()
        gold = str(row["answer"]).strip().lower()
        try:
            pred_num = float(re.search(r"[-+]?\d*\.?\d+", pred).group())
            gold_num = float(gold)
            return 1 if abs(pred_num - gold_num) < 1e-6 else 0
        except:
            pass
        if gold in pred or pred == gold:
            return 1
        return 0


def try_download(model, suffix, max_retries=3):
    """Try to download a benchmark file for a model, with retry on rate limit."""
    fpath = f"mmeval/{model}/{model}_{suffix}.xlsx"
    for attempt in range(max_retries):
        try:
            local_path = hf_hub_download(REPO_ID, fpath, repo_type="dataset")
            return local_path
        except EntryNotFoundError:
            return None  # File doesn't exist
        except HfHubHTTPError as e:
            if "429" in str(e):
                wait_time = 60 * (attempt + 1)
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                return None
        except Exception:
            return None
    return None


def process_benchmark(bench_name, config, all_models):
    print(f"\n{'='*60}")
    print(f"Processing: {bench_name}")
    print(f"{'='*60}")

    suffix = config["suffix"]
    output_dir = BASE_DIR / config["output_dir"] / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scores = {}
    item_info = None
    failed_models = []
    not_found = 0

    for i, model in enumerate(all_models):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Trying {i+1}/{len(all_models)}: {model}")

        local_path = try_download(model, suffix)
        if local_path is None:
            not_found += 1
            continue

        try:
            df = pd.read_excel(local_path)

            if config["split_filter"] and config["split_col"] and config["split_col"] in df.columns:
                df = df[df[config["split_col"]] == config["split_filter"]].copy()

            if len(df) == 0:
                failed_models.append((model, "empty after split filter"))
                continue

            if "index" not in df.columns:
                df["index"] = range(len(df))

            if config["type"] == "mcq":
                df["score"] = df.apply(score_mcq, axis=1)
            elif config["type"] == "yesno":
                df["score"] = df.apply(score_yesno, axis=1)
            elif config["type"] == "mixed":
                df["score"] = df.apply(score_mathvista, axis=1)

            scores = df.set_index("index")["score"]
            all_scores[model] = scores

            if item_info is None:
                item_cols = ["index", "question"]
                for c in ["category", "l2-category", "answer", "question_type", "answer_type", "hint"]:
                    if c in df.columns:
                        item_cols.append(c)
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

    print(f"\n  Found results: {len(all_scores)} models, not found: {not_found}")

    response_matrix = pd.DataFrame(all_scores)
    response_matrix = response_matrix.sort_index()
    response_matrix_T = response_matrix.T
    response_matrix_T.index.name = "model"

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

    response_matrix_T.to_csv(output_dir / "response_matrix.csv")

    if item_info is not None:
        item_info = item_info.drop_duplicates(subset=["index"]).sort_values("index")
        item_info.to_csv(output_dir / "item_content.csv", index=False)

    metadata = {
        "benchmark": bench_name,
        "source": "HuggingFace VLMEval/OpenVLMRecords",
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
    return response_matrix_T


def main():
    all_models = get_all_models()
    print(f"Found {len(all_models)} models total")

    for bench_name, config in BENCHMARKS.items():
        try:
            process_benchmark(bench_name, config, all_models)
        except Exception as e:
            print(f"ERROR processing {bench_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
