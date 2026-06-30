"""
Extract item_content.csv for benchmarks that have raw data but no item_content.csv.
Run on skampere1 where data lives.
"""

import csv
import json
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path("/lfs/skampere1/0/sttruong/torch_measure/data")


def save_item_content(bench_dir: str, items: list[dict]):
    """Save item_content.csv with columns: item_id, content."""
    out_path = DATA_DIR / bench_dir / "processed" / "item_content.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(items)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} items to {out_path}")


def extract_terminal_bench():
    """Terminal-Bench: task instruction from metadata."""
    print("=== terminal_bench_data ===")
    meta = pd.read_csv(DATA_DIR / "terminal_bench_data/processed/tasks_complete_metadata.csv")
    items = [{"item_id": row["task_name"], "content": str(row["instruction"])[:2000]}
             for _, row in meta.iterrows() if pd.notna(row.get("instruction"))]
    save_item_content("terminal_bench_data", items)


def extract_livecodebench():
    """LiveCodeBench: problem text from submission eval_all.json files."""
    print("=== livecodebench_data ===")
    submissions_dir = DATA_DIR / "livecodebench_data/raw/submissions"
    if not submissions_dir.exists():
        print("  No submissions dir found")
        return

    # Find one submission with eval data to get problem text
    items = {}
    for model_dir in submissions_dir.iterdir():
        eval_file = model_dir / "eval_all.json"
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
            for entry in data:
                qid = entry.get("question_id", "")
                if qid and qid not in items:
                    # Combine question title + content
                    text_parts = []
                    if entry.get("question_title"):
                        text_parts.append(entry["question_title"])
                    if entry.get("question_content"):
                        text_parts.append(str(entry["question_content"])[:1500])
                    elif entry.get("question"):
                        text_parts.append(str(entry["question"])[:1500])
                    if text_parts:
                        items[qid] = {"item_id": qid, "content": "\n".join(text_parts)}
            if len(items) > 1000:
                break  # got enough

    save_item_content("livecodebench_data", list(items.values()))


def extract_alpacaeval():
    """AlpacaEval: instruction text from item_metadata.csv."""
    print("=== alpacaeval_data ===")
    meta = pd.read_csv(DATA_DIR / "alpacaeval_data/processed/item_metadata.csv")
    items = [{"item_id": str(row.get("item_idx", i)), "content": str(row["instruction"])[:2000]}
             for i, (_, row) in enumerate(meta.iterrows()) if pd.notna(row.get("instruction"))]
    save_item_content("alpacaeval_data", items)


def extract_wildbench():
    """WildBench: intent/primary_tag from task_metadata.csv."""
    print("=== wildbench_data ===")
    meta = pd.read_csv(DATA_DIR / "wildbench_data/raw/task_metadata.csv")
    items = []
    for _, row in meta.iterrows():
        text = str(row.get("intent", ""))
        if row.get("primary_tag"):
            text = f"[{row['primary_tag']}] {text}"
        if len(text) > 10:
            items.append({"item_id": str(row.get("session_id", "")), "content": text})
    save_item_content("wildbench_data", items)


def extract_corebench():
    """CORE-Bench: capsule title + field from task_metadata.csv."""
    print("=== corebench_data ===")
    meta = pd.read_csv(DATA_DIR / "corebench_data/processed/task_metadata.csv")
    items = []
    for _, row in meta.iterrows():
        text_parts = []
        if row.get("capsule_title"):
            text_parts.append(str(row["capsule_title"]))
        if row.get("field"):
            text_parts.append(f"Field: {row['field']}")
        if row.get("language"):
            text_parts.append(f"Language: {row['language']}")
        if text_parts:
            items.append({"item_id": str(row["task_id"]), "content": " | ".join(text_parts)})
    save_item_content("corebench_data", items)


def extract_editbench():
    """EditBench: instruction preview + language from task_metadata.csv."""
    print("=== editbench_data ===")
    meta = pd.read_csv(DATA_DIR / "editbench_data/processed/task_metadata.csv")
    items = []
    for _, row in meta.iterrows():
        text = str(row.get("instruction_preview", ""))
        if row.get("programming_language"):
            text = f"[{row['programming_language']}] {text}"
        if row.get("natural_language") and row["natural_language"] != "english":
            text = f"({row['natural_language']}) {text}"
        if len(text) > 10:
            items.append({"item_id": str(row["task_id"]), "content": text})
    save_item_content("editbench_data", items)


def extract_afrimedqa():
    """AfriMedQA: question text from CSV."""
    print("=== afrimedqa_data ===")
    csv_path = DATA_DIR / "afrimedqa_data/raw/AfriMed-QA/data/afri_med_qa_15k_v2.5_phase_2_15275.csv"
    if not csv_path.exists():
        print("  CSV not found")
        return
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        text_parts = []
        if row.get("question_clean"):
            text_parts.append(str(row["question_clean"]))
        elif row.get("question"):
            text_parts.append(str(row["question"]))
        if row.get("answer_options"):
            text_parts.append(str(row["answer_options"])[:500])
        if text_parts:
            content = "\n".join(text_parts)[:2000]
            items.append({"item_id": str(row.get("sample_id", "")), "content": content})
    save_item_content("afrimedqa_data", items)


def extract_cybench():
    """CyBench: task descriptions from paper tables."""
    print("=== cybench_data ===")
    tables_file = DATA_DIR / "cybench_data/raw/tables_15_to_28.txt"
    if not tables_file.exists():
        print("  Tables file not found")
        return
    # Parse task descriptions from the text file
    with open(tables_file) as f:
        text = f.read()
    # Simple extraction: look for task names and descriptions
    items = []
    leaderboard = DATA_DIR / "cybench_data/raw/leaderboard.csv"
    if leaderboard.exists():
        lb = pd.read_csv(leaderboard)
        for col in lb.columns[1:]:  # first col is model, rest are tasks
            items.append({"item_id": col, "content": f"CTF Challenge: {col}"})
    if items:
        save_item_content("cybench_data", items)
    else:
        print("  Could not extract items")


def extract_dpai():
    """DPAI Arena: task info from results."""
    print("=== dpai_data ===")
    results = DATA_DIR / "dpai_data/processed/all_results_long_format.csv"
    if not results.exists():
        print("  No results file")
        return
    df = pd.read_csv(results)
    if "task_id" in df.columns:
        task_ids = df["task_id"].unique()
        items = [{"item_id": str(tid), "content": f"DPAI Java SE Task: {tid}"} for tid in task_ids]
        save_item_content("dpai_data", items)


def extract_sib200():
    """SIB-200: topic classification text from raw data."""
    print("=== sib200_data ===")
    sib_dir = DATA_DIR / "sib200_data/raw/sib-200/data"
    if not sib_dir.exists():
        print("  No data dir")
        return

    items = []
    # SIB-200 has per-language TSV files
    for f in sorted(sib_dir.glob("*.tsv")):
        try:
            df = pd.read_csv(f, sep="\t", header=None, names=["idx", "category", "text"],
                             on_bad_lines="skip", quoting=csv.QUOTE_NONE)
            lang = f.stem
            for _, row in df.iterrows():
                if pd.notna(row.get("text")) and len(str(row["text"])) > 10:
                    items.append({
                        "item_id": f"{lang}_{row['idx']}",
                        "content": f"[{lang}] [{row.get('category', '')}] {str(row['text'])[:500]}"
                    })
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")
            continue

    if items:
        save_item_content("sib200_data", items)


def extract_taubench():
    """Tau-Bench: task info from JSON files."""
    print("=== taubench_data ===")
    raw_dir = DATA_DIR / "taubench_data/raw"
    items = []
    for jf in raw_dir.glob("*.json"):
        try:
            with open(jf) as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and entry.get("task_id"):
                        text = str(entry.get("instruction", entry.get("task", "")))[:1000]
                        if len(text) > 10:
                            items.append({"item_id": str(entry["task_id"]), "content": text})
            elif isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(val, dict) and val.get("instruction"):
                        items.append({"item_id": key, "content": str(val["instruction"])[:1000]})
        except Exception as e:
            print(f"  Error reading {jf.name}: {e}")

    if items:
        save_item_content("taubench_data", items)
    else:
        print("  Could not extract items")


if __name__ == "__main__":
    extract_terminal_bench()
    extract_livecodebench()
    extract_alpacaeval()
    extract_wildbench()
    extract_corebench()
    extract_editbench()
    extract_afrimedqa()
    extract_cybench()
    extract_dpai()
    extract_sib200()
    extract_taubench()

    # Summary
    print("\n=== Summary ===")
    for d in sorted(DATA_DIR.glob("*_data")):
        ic = d / "processed" / "item_content.csv"
        if ic.exists():
            n = sum(1 for _ in open(ic)) - 1
            print(f"  {d.name:35s}: {n:>8,} items")
