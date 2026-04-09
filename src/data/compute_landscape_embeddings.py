"""
Compute 2D embeddings for the AI evaluation data landscape visualization.

Two outputs:
1. dataset_landscape.json — one point per benchmark (mean of sampled item embeddings)
2. item_landscape.json — sampled items across benchmarks (up to 100 per benchmark)

Requires: sentence-transformers, umap-learn, pandas, numpy

Usage:
  python src/data/compute_landscape_embeddings.py
  python src/data/compute_landscape_embeddings.py --data-dir /path/to/item_cache
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _resolve_path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser() if value else default

# Config
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = _resolve_path(
    "AIMS_BENCH_DATA_DIR",
    REPO_ROOT / "src" / "data" / "item_cache",
)
DEFAULT_OUTPUT_DIR = _resolve_path(
    "AIMS_OUTPUT_DIR",
    REPO_ROOT / "src" / "data",
)
MAX_ITEMS_PER_BENCH = 100  # for item-level plot
MAX_CHARS = 512  # truncate item text for embedding
SEED = 42

# Benchmark metadata: name -> (category, display_name)
BENCH_META = {
    "mmlupro_data": ("Knowledge", "MMLU-Pro"),
    "hle_data": ("Knowledge", "HLE"),
    "livebench_data": ("Knowledge", "LiveBench"),
    "ceval_data": ("Knowledge (Chinese)", "C-Eval"),
    "cmmlu_data": ("Knowledge (Chinese)", "CMMLU"),
    "kmmlu_data": ("Knowledge (Korean)", "KMMLU"),
    "afrieval_data": ("Knowledge (African)", "AfriEval"),
    "asiaeval_data": ("Knowledge (Asian)", "AsiaEval"),
    "culturaleval_data": ("Knowledge (Cultural)", "CulturalEval"),
    "iberbench_data": ("Knowledge (Iberian)", "IberBench"),
    "bigcodebench_data": ("Code Generation", "BigCodeBench"),
    "evalplus_data": ("Code Generation", "EvalPlus"),
    "livecodebench_data": ("Code Generation", "LiveCodeBench"),
    "cruxeval_data": ("Code Reasoning", "CRUXEval"),
    "swebench_data": ("Software Engineering", "SWE-bench"),
    "swebench_full_data": ("Software Engineering", "SWE-bench Full"),
    "swebench_java_data": ("Software Engineering", "SWE-bench Java"),
    "swebench_multilingual_data": ("Software Engineering", "SWE-bench Multilingual"),
    "swepolybench_data": ("Software Engineering", "SWE-PolyBench"),
    "mlebench_data": ("ML Engineering", "MLE-bench"),
    "bfcl_data": ("Tool Use", "BFCL"),
}

random.seed(SEED)
np.random.seed(SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items-per-bench", type=int, default=MAX_ITEMS_PER_BENCH)
    return parser.parse_args()


def load_items(bench_dir: Path, max_items: int = MAX_ITEMS_PER_BENCH) -> List[Dict[str, str]]:
    """Load and sample items from a benchmark's item_content.csv."""
    item_file = bench_dir / "processed" / "item_content.csv"
    if not item_file.exists():
        return []

    try:
        df = pd.read_csv(item_file, nrows=50000)  # cap for huge files
    except Exception as e:
        print(f"  Error reading {item_file}: {e}")
        return []

    if "content" not in df.columns:
        return []

    # Drop empty content
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.len() > 10]

    if len(df) == 0:
        return []

    # Sample
    if len(df) > max_items:
        df = df.sample(n=max_items, random_state=SEED)

    items = []
    for _, row in df.iterrows():
        text = str(row["content"])[:MAX_CHARS]
        items.append({
            "item_id": str(row.get("item_id", row.get("question_id", ""))),
            "text": text,
        })

    return items


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    out_dir = args.output_dir.expanduser()

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory {data_dir} does not exist. Pass --data-dir if your "
            "benchmark cache lives elsewhere."
        )

    available_item_files = [
        data_dir / bench_name / "processed" / "item_content.csv"
        for bench_name in BENCH_META
        if (data_dir / bench_name / "processed" / "item_content.csv").exists()
    ]
    if not available_item_files:
        raise FileNotFoundError(
            f"No benchmark items were found under {data_dir}. Expected per-benchmark "
            "`processed/item_content.csv` files. Pass --data-dir if your cache "
            "lives elsewhere."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sentence-transformers is required to run this script. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    try:
        import umap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "umap-learn is required to run this script. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    print("Loading sentence transformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_items: List[Dict[str, Any]] = []  # for item-level plot
    bench_embeddings: Dict[str, Dict[str, Any]] = {}  # for dataset-level plot

    for bench_name, (category, display_name) in BENCH_META.items():
        bench_dir = data_dir / bench_name
        if not bench_dir.exists():
            print(f"  Skipping {bench_name} (not found)")
            continue

        print(f"Loading {display_name} ({bench_name})...")
        items = load_items(bench_dir, max_items=args.max_items_per_bench)
        if not items:
            print(f"  No items found for {bench_name}")
            continue

        print(f"  {len(items)} items sampled, embedding...")
        texts = [it["text"] for it in items]
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=256)

        # Store for item-level plot
        for it, emb in zip(items, embeddings):
            all_items.append({
                "benchmark": display_name,
                "category": category,
                "item_id": it["item_id"],
                "text_preview": it["text"][:120],
                "embedding": emb.tolist(),
            })

        # Store mean embedding for dataset-level plot
        mean_emb = embeddings.mean(axis=0)
        bench_embeddings[display_name] = {
            "category": category,
            "n_items_sampled": len(items),
            "embedding": mean_emb.tolist(),
        }

    print(f"\nTotal items: {len(all_items)}")
    print(f"Total benchmarks: {len(bench_embeddings)}")
    if len(bench_embeddings) < 2:
        raise RuntimeError("Need at least two benchmarks to compute dataset-level UMAP.")
    if len(all_items) < 2:
        raise RuntimeError("Need at least two items to compute item-level UMAP.")

    # --- UMAP for dataset-level ---
    print("\nComputing dataset-level UMAP...")
    bench_names = list(bench_embeddings.keys())
    bench_embs = np.array([bench_embeddings[n]["embedding"] for n in bench_names])

    n_neighbors = min(5, len(bench_names) - 1)
    reducer_ds = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.3,
                           random_state=SEED, metric="cosine")
    coords_ds = reducer_ds.fit_transform(bench_embs)

    dataset_landscape = []
    for i, name in enumerate(bench_names):
        dataset_landscape.append({
            "benchmark": name,
            "category": bench_embeddings[name]["category"],
            "n_items": bench_embeddings[name]["n_items_sampled"],
            "x": float(coords_ds[i, 0]),
            "y": float(coords_ds[i, 1]),
        })

    # --- UMAP for item-level ---
    print("Computing item-level UMAP...")
    item_embs = np.array([it["embedding"] for it in all_items])
    reducer_it = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                           random_state=SEED, metric="cosine")
    coords_it = reducer_it.fit_transform(item_embs)

    item_landscape = []
    for i, it in enumerate(all_items):
        item_landscape.append({
            "benchmark": it["benchmark"],
            "category": it["category"],
            "item_id": it["item_id"],
            "text_preview": it["text_preview"],
            "x": float(coords_it[i, 0]),
            "y": float(coords_it[i, 1]),
        })

    # --- Save ---
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "dataset_landscape.json", "w") as f:
        json.dump(dataset_landscape, f, indent=2)
    print(f"Saved dataset_landscape.json ({len(dataset_landscape)} benchmarks)")

    with open(out_dir / "item_landscape.json", "w") as f:
        json.dump(item_landscape, f, indent=2)
    print(f"Saved item_landscape.json ({len(item_landscape)} items)")


if __name__ == "__main__":
    main()
