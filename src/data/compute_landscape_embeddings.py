"""
Compute 2D embeddings for the AI evaluation data landscape visualization.

Two outputs:
1. dataset_landscape.json — one point per benchmark (mean of sampled item embeddings)
2. item_landscape.json — sampled items across benchmarks (up to 100 per benchmark)

Requires: sentence-transformers, umap-learn, pandas, numpy
Run on skampere1 where torch_measure data lives.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Config
DATA_DIR = Path("/lfs/skampere1/0/sttruong/torch_measure/data")
OUTPUT_DIR = Path("/lfs/skampere1/0/sttruong/torch_measure/data")
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


def load_items(bench_dir: Path, max_items: int = MAX_ITEMS_PER_BENCH) -> list[dict]:
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
    from sentence_transformers import SentenceTransformer
    import umap

    print("Loading sentence transformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_items = []  # for item-level plot
    bench_embeddings = {}  # for dataset-level plot

    for bench_name, (category, display_name) in BENCH_META.items():
        bench_dir = DATA_DIR / bench_name
        if not bench_dir.exists():
            print(f"  Skipping {bench_name} (not found)")
            continue

        print(f"Loading {display_name} ({bench_name})...")
        items = load_items(bench_dir)
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
    out_dir = Path("/lfs/skampere2/0/sttruong/aims/src/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "dataset_landscape.json", "w") as f:
        json.dump(dataset_landscape, f, indent=2)
    print(f"Saved dataset_landscape.json ({len(dataset_landscape)} benchmarks)")

    with open(out_dir / "item_landscape.json", "w") as f:
        json.dump(item_landscape, f, indent=2)
    print(f"Saved item_landscape.json ({len(item_landscape)} items)")


if __name__ == "__main__":
    main()
