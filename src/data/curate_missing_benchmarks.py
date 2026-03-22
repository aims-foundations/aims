#!/usr/bin/env python3
"""
Curate missing benchmarks: download raw data, build response matrices, extract item content.

Targets:
  1. RewardBench      - per-reward-model per-item binary correctness
  2. MT-Bench         - per-model per-question GPT-4 scores (1-10)
  3. UltraFeedback    - per-prompt multi-model multi-dimension scores
  4. LawBench         - 51 models × 10K Chinese legal items (per-question predictions)
  5. FinanceBench     - 16 model configs × 150 financial QA items

Usage:
  python curate_missing_benchmarks.py [benchmark_name]
  python curate_missing_benchmarks.py --all
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("/lfs/skampere2/0/sttruong/aims/src/data/curated_benchmarks")


def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)


def save(bench_name, response_matrix=None, item_content=None, model_summary=None):
    """Save processed data in standard format."""
    out = OUTPUT_DIR / f"{bench_name}_data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    if response_matrix is not None:
        response_matrix.to_csv(out / "response_matrix.csv")
        print(f"  Saved response_matrix.csv: {response_matrix.shape}")

    if item_content is not None:
        item_content.to_csv(out / "item_content.csv", index=False)
        print(f"  Saved item_content.csv: {len(item_content)} items")

    if model_summary is not None:
        model_summary.to_csv(out / "model_summary.csv", index=False)
        print(f"  Saved model_summary.csv: {len(model_summary)} models")


# ---------------------------------------------------------------------------
# 1. RewardBench
# ---------------------------------------------------------------------------
def curate_rewardbench():
    """Download RewardBench per-model per-item scores from HuggingFace."""
    print("\n=== RewardBench ===")

    try:
        from datasets import load_dataset
    except ImportError:
        print("  pip install datasets")
        return

    # Load the results dataset
    print("  Loading reward-bench-results from HuggingFace...")
    try:
        ds = load_dataset("allenai/reward-bench-results", "eval-set-scores", split="train")
        df = ds.to_pandas()
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        print("  Trying alternative: clone from GitHub...")
        raw_dir = OUTPUT_DIR / "rewardbench_data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        result = run(f"git clone --depth 1 https://github.com/allenai/reward-bench.git {raw_dir / 'reward-bench'}")
        if result.returncode != 0:
            print(f"  Clone failed: {result.stderr}")
            return
        # Parse eval-set-scores JSONs
        scores_dir = raw_dir / "reward-bench" / "eval-set-scores"
        if not scores_dir.exists():
            print(f"  No eval-set-scores directory found")
            return
        records = []
        for org_dir in scores_dir.iterdir():
            if not org_dir.is_dir():
                continue
            for model_file in org_dir.glob("*.json"):
                model_name = f"{org_dir.name}/{model_file.stem}"
                with open(model_file) as f:
                    data = json.load(f)
                for item in data:
                    item["model"] = model_name
                    records.append(item)
        df = pd.DataFrame(records)

    if df.empty:
        print("  No data loaded")
        return

    print(f"  Raw data: {len(df)} rows, columns: {list(df.columns)[:10]}")

    # Build response matrix: models × items
    # The dataset has columns like 'id', 'subset', 'correct', and model info
    if "id" in df.columns and "model" in df.columns and "correct" in df.columns:
        pivot = df.pivot_table(index="model", columns="id", values="correct", aggfunc="first")
        pivot = pivot.astype(float)

        # Item content
        item_cols = ["id", "prompt", "subset"]
        avail_cols = [c for c in item_cols if c in df.columns]
        items_df = df[avail_cols].drop_duplicates(subset=["id"])
        if "prompt" in items_df.columns:
            item_content = pd.DataFrame({
                "item_id": items_df["id"].astype(str),
                "content": items_df["prompt"].astype(str).str[:2000],
            })
        else:
            item_content = pd.DataFrame({
                "item_id": items_df["id"].astype(str),
                "content": items_df.apply(lambda r: f"[{r.get('subset', '')}] Item {r['id']}", axis=1),
            })

        save("rewardbench", response_matrix=pivot, item_content=item_content)
    else:
        print(f"  Unexpected columns: {list(df.columns)}")
        print(f"  First row: {df.iloc[0].to_dict()}")


# ---------------------------------------------------------------------------
# 2. MT-Bench
# ---------------------------------------------------------------------------
def curate_mtbench():
    """Download MT-Bench model judgments from FastChat GitHub."""
    print("\n=== MT-Bench ===")

    raw_dir = OUTPUT_DIR / "mtbench_data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Clone FastChat for MT-Bench data
    fastchat_dir = raw_dir / "FastChat"
    if not fastchat_dir.exists():
        print("  Cloning FastChat...")
        result = run(f"git clone --depth 1 https://github.com/lm-sys/FastChat.git {fastchat_dir}")
        if result.returncode != 0:
            print(f"  Clone failed: {result.stderr}")
            return

    # Parse model judgments
    judge_dir = fastchat_dir / "fastchat" / "llm_judge" / "data" / "mt_bench" / "model_judgment"
    if not judge_dir.exists():
        print(f"  No judgment directory at {judge_dir}")
        return

    # Load questions
    q_file = fastchat_dir / "fastchat" / "llm_judge" / "data" / "mt_bench" / "question.jsonl"
    questions = {}
    if q_file.exists():
        with open(q_file) as f:
            for line in f:
                q = json.loads(line)
                questions[q["question_id"]] = q.get("turns", [""])[0][:500]

    # Parse GPT-4 single judgments
    judgment_file = judge_dir / "gpt-4_single.jsonl"
    if not judgment_file.exists():
        # Try other files
        for jf in judge_dir.glob("*.jsonl"):
            judgment_file = jf
            break

    if not judgment_file.exists():
        print("  No judgment files found")
        return

    print(f"  Reading {judgment_file.name}...")
    records = []
    with open(judgment_file) as f:
        for line in f:
            j = json.loads(line)
            records.append({
                "model": j.get("model", ""),
                "question_id": j.get("question_id", ""),
                "score": j.get("score", j.get("rating", None)),
                "turn": j.get("turn", 1),
            })

    df = pd.DataFrame(records)
    df = df[df["score"].notna()]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    if df.empty:
        print("  No valid scores found")
        return

    print(f"  {len(df)} judgments, {df['model'].nunique()} models, {df['question_id'].nunique()} questions")

    # Average across turns for each (model, question)
    avg_scores = df.groupby(["model", "question_id"])["score"].mean().reset_index()
    pivot = avg_scores.pivot(index="model", columns="question_id", values="score")

    # Normalize to [0, 1] (scores are 1-10)
    pivot_norm = (pivot - 1) / 9

    # Item content
    item_content = pd.DataFrame({
        "item_id": [str(qid) for qid in pivot.columns],
        "content": [questions.get(qid, f"MT-Bench Question {qid}") for qid in pivot.columns],
    })

    save("mtbench", response_matrix=pivot_norm, item_content=item_content)


# ---------------------------------------------------------------------------
# 3. UltraFeedback
# ---------------------------------------------------------------------------
def curate_ultrafeedback():
    """Download UltraFeedback from HuggingFace and build response matrix."""
    print("\n=== UltraFeedback ===")

    try:
        from datasets import load_dataset
    except ImportError:
        print("  pip install datasets")
        return

    print("  Loading UltraFeedback from HuggingFace...")
    try:
        ds = load_dataset("openbmb/UltraFeedback", split="train")
    except Exception as e:
        print(f"  Error: {e}")
        return

    print(f"  {len(ds)} prompts loaded")

    # Each item has: instruction, completions (list of {model, output, annotations})
    # Build response matrix: models × prompts, values = overall_score
    records = []
    items = []
    for i, row in enumerate(ds):
        prompt_id = str(i)
        instruction = str(row.get("instruction", ""))[:1000]
        items.append({"item_id": prompt_id, "content": instruction})

        completions = row.get("completions", [])
        for comp in completions:
            model = comp.get("model", "unknown")
            annotations = comp.get("annotations", {})
            # Get overall score
            overall = annotations.get("overall", {})
            if isinstance(overall, dict):
                rating = overall.get("Rating", overall.get("rating", None))
            elif isinstance(overall, (int, float, str)):
                rating = overall
            else:
                rating = None

            if rating is not None:
                try:
                    score = float(rating)
                    records.append({
                        "model": model,
                        "item_id": prompt_id,
                        "score": score,
                    })
                except (ValueError, TypeError):
                    pass

        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i}/{len(ds)} prompts...")

    df = pd.DataFrame(records)
    if df.empty:
        print("  No scores extracted")
        return

    print(f"  {len(df)} scores, {df['model'].nunique()} models, {df['item_id'].nunique()} items")

    pivot = df.pivot_table(index="model", columns="item_id", values="score", aggfunc="mean")
    # Normalize to [0, 1] (scores are typically 1-5)
    pivot_norm = (pivot - 1) / 4
    pivot_norm = pivot_norm.clip(0, 1)

    item_content = pd.DataFrame(items)
    save("ultrafeedback", response_matrix=pivot_norm, item_content=item_content)


# ---------------------------------------------------------------------------
# 4. LawBench
# ---------------------------------------------------------------------------
def curate_lawbench():
    """Download LawBench per-question predictions from GitHub."""
    print("\n=== LawBench ===")

    raw_dir = OUTPUT_DIR / "lawbench_data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    lawbench_dir = raw_dir / "LawBench"
    if not lawbench_dir.exists():
        print("  Cloning LawBench...")
        result = run(f"git clone --depth 1 https://github.com/open-compass/LawBench.git {lawbench_dir}")
        if result.returncode != 0:
            print(f"  Clone failed: {result.stderr}")
            return

    # Find predictions directory
    pred_dir = lawbench_dir / "predictions" / "zero_shot"
    if not pred_dir.exists():
        pred_dir = lawbench_dir / "predictions"
        if not pred_dir.exists():
            print(f"  No predictions directory found")
            # List what's in the repo
            result = run(f"find {lawbench_dir} -maxdepth 3 -type d | head -20")
            print(f"  Dirs: {result.stdout}")
            return

    print(f"  Reading predictions from {pred_dir}...")

    # Each model has a directory with task JSON files
    records = []
    all_items = {}

    for model_dir in sorted(pred_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for task_file in model_dir.glob("*.json"):
            task_name = task_file.stem
            try:
                with open(task_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if isinstance(data, list):
                for j, item in enumerate(data):
                    item_id = f"{task_name}_{j}"
                    pred = item.get("prediction", "")
                    ref = item.get("refr", item.get("reference", ""))
                    prompt = item.get("origin_prompt", "")[:500]

                    # Simple exact match scoring
                    correct = 1.0 if str(pred).strip() == str(ref).strip() else 0.0
                    records.append({
                        "model": model_name,
                        "item_id": item_id,
                        "score": correct,
                    })

                    if item_id not in all_items:
                        all_items[item_id] = f"[{task_name}] {prompt}"

    if not records:
        print("  No predictions found")
        return

    df = pd.DataFrame(records)
    print(f"  {len(df)} predictions, {df['model'].nunique()} models, {df['item_id'].nunique()} items")

    pivot = df.pivot_table(index="model", columns="item_id", values="score", aggfunc="first")

    item_content = pd.DataFrame([
        {"item_id": k, "content": v} for k, v in all_items.items()
    ])

    save("lawbench", response_matrix=pivot, item_content=item_content)


# ---------------------------------------------------------------------------
# 5. FinanceBench
# ---------------------------------------------------------------------------
def curate_financebench():
    """Download FinanceBench per-model results from GitHub."""
    print("\n=== FinanceBench ===")

    raw_dir = OUTPUT_DIR / "financebench_data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fb_dir = raw_dir / "financebench"
    if not fb_dir.exists():
        print("  Cloning FinanceBench...")
        result = run(f"git clone --depth 1 https://github.com/patronus-ai/financebench.git {fb_dir}")
        if result.returncode != 0:
            print(f"  Clone failed: {result.stderr}")
            return

    # Check for results directory
    results_dir = fb_dir / "results"
    if not results_dir.exists():
        print(f"  No results directory found")
        result = run(f"find {fb_dir} -maxdepth 2 -type d")
        print(f"  Dirs: {result.stdout[:500]}")
        return

    print(f"  Reading results from {results_dir}...")

    # Load the benchmark questions
    data_file = fb_dir / "data" / "financebench_open_source.jsonl"
    if not data_file.exists():
        # Try other locations
        for f in fb_dir.glob("**/*.jsonl"):
            data_file = f
            break

    items = {}
    if data_file.exists():
        with open(data_file) as f:
            for line in f:
                item = json.loads(line)
                qid = item.get("financebench_id", item.get("id", ""))
                question = item.get("question", "")
                items[str(qid)] = question[:1000]

    # Parse model results
    records = []
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for result_file in model_dir.glob("*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if isinstance(data, list):
                for item in data:
                    qid = str(item.get("financebench_id", item.get("id", "")))
                    score = item.get("accuracy", item.get("correct", None))
                    if score is not None:
                        records.append({
                            "model": model_name,
                            "item_id": qid,
                            "score": float(score),
                        })
                        if qid not in items:
                            items[qid] = item.get("question", f"FinanceBench {qid}")[:1000]
            elif isinstance(data, dict):
                for qid, item in data.items():
                    score = item.get("accuracy", item.get("correct", None))
                    if score is not None:
                        records.append({
                            "model": model_name,
                            "item_id": str(qid),
                            "score": float(score),
                        })

    if not records:
        print("  No results found")
        return

    df = pd.DataFrame(records)
    print(f"  {len(df)} results, {df['model'].nunique()} models, {df['item_id'].nunique()} items")

    pivot = df.pivot_table(index="model", columns="item_id", values="score", aggfunc="first")

    item_content = pd.DataFrame([
        {"item_id": k, "content": v} for k, v in items.items()
    ])

    save("financebench", response_matrix=pivot, item_content=item_content)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CURATORS = {
    "rewardbench": curate_rewardbench,
    "mtbench": curate_mtbench,
    "ultrafeedback": curate_ultrafeedback,
    "lawbench": curate_lawbench,
    "financebench": curate_financebench,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarks", nargs="*")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    targets = list(CURATORS.keys()) if args.all or not args.benchmarks else args.benchmarks

    for name in targets:
        if name not in CURATORS:
            print(f"Unknown: {name}. Available: {list(CURATORS.keys())}")
            sys.exit(1)
        CURATORS[name]()

    print("\n=== Summary ===")
    for d in sorted(OUTPUT_DIR.glob("*_data/processed")):
        bench = d.parent.name
        rm = list(d.glob("response_matrix*.csv"))
        ic = d / "item_content.csv"
        rm_shape = ""
        if rm:
            df = pd.read_csv(rm[0], index_col=0, nrows=0)
            n_rows = sum(1 for _ in open(rm[0])) - 1
            rm_shape = f"{n_rows} × {len(df.columns)}"
        ic_count = sum(1 for _ in open(ic)) - 1 if ic.exists() else 0
        print(f"  {bench:25s}: rm={rm_shape:>15s}  items={ic_count:>6,}")


if __name__ == "__main__":
    main()
