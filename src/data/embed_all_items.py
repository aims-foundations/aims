"""
Embed all benchmark items with gte-large-en-v1.5 and cache results.

Produces per-benchmark cached embeddings:
  item_cache/<bench>_data/processed/item_embeddings.pt

Then runs UMAP on the full dataset and saves coordinates.

Usage:
  CUDA_VISIBLE_DEVICES=5 python src/data/embed_all_items.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CACHE_DIR = Path("/lfs/skampere2/0/sttruong/aims/src/data/item_cache")
OUTPUT_DIR = Path("/lfs/skampere2/0/sttruong/aims/src/data")
MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
MAX_CHARS = 1024
BATCH_SIZE = 512  # gte-large is small enough for big batches

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
    "swebench_multilingual_data": ("Software Engineering", "SWE-bench ML"),
    "swepolybench_data": ("Software Engineering", "SWE-PolyBench"),
    "mlebench_data": ("ML Engineering", "MLE-bench"),
    "bfcl_data": ("Tool Use", "BFCL"),
    "toolbench_data": ("Tool Use", "ToolBench"),
    "arcagi_data": ("Reasoning", "ARC-AGI"),
    "matharena_data": ("Math", "MathArena"),
    "webarena_data": ("Web Agent", "WebArena"),
    "workarena_data": ("Web Agent", "WorkArena"),
    "agentdojo_data": ("Agent", "AgentDojo"),
    "gaia_data": ("Agent", "GAIA"),
    "osworld_data": ("Desktop Agent", "OSWorld"),
    "theagentcompany_data": ("Agent", "TheAgentCompany"),
    "appworld_data": ("Agent", "AppWorld"),
    "androidworld_data": ("Mobile Agent", "AndroidWorld"),
    "clinebench_data": ("Coding Agent", "ClineBench"),
    "paperbench_data": ("Research Agent", "PaperBench"),
    "agentbench_data": ("Agent", "AgentBench"),
    # Newly extracted
    "terminal_bench_data": ("Terminal Agent", "Terminal-Bench"),
    "livecodebench_data": ("Code Generation", "LiveCodeBench"),
    "alpacaeval_data": ("Preference", "AlpacaEval"),
    "wildbench_data": ("Preference", "WildBench"),
    "corebench_data": ("Reproducibility", "CORE-Bench"),
    "editbench_data": ("Code Editing", "EditBench"),
    "afrimedqa_data": ("Medical (African)", "AfriMedQA"),
    "cybench_data": ("Security", "CyBench"),
    "dpai_data": ("Java SWE", "DPAI Arena"),
    "sib200_data": ("Multilingual", "SIB-200"),
    "helm_multilingual_data": ("Multilingual", "HELM Multilingual"),
}


def load_items(bench_dir: str) -> tuple[list[str], list[str]]:
    """Load item texts and IDs from item_content.csv."""
    item_file = CACHE_DIR / bench_dir / "processed" / "item_content.csv"
    if not item_file.exists():
        return [], []

    try:
        df = pd.read_csv(item_file)
    except Exception as e:
        print(f"  Error reading {item_file}: {e}")
        return [], []

    if "content" not in df.columns:
        return [], []

    df = df.dropna(subset=["content"])
    df = df[df["content"].str.len() > 10]

    if len(df) == 0:
        return [], []

    texts = [str(t)[:MAX_CHARS] for t in df["content"]]
    item_id_col = "item_id" if "item_id" in df.columns else ("question_id" if "question_id" in df.columns else None)
    if item_id_col:
        ids = [str(i) for i in df[item_id_col]]
    else:
        ids = [str(i) for i in range(len(df))]

    return texts, ids


def get_cache_path(bench_dir: str) -> Path:
    return CACHE_DIR / bench_dir / "processed" / "item_embeddings.pt"


def main():
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print(f"Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Phase 1: Embed all benchmarks with caching
    bench_order = []  # track which benchmarks we successfully processed
    total_items = 0

    for bench_dir, (category, display_name) in BENCH_META.items():
        cache_path = get_cache_path(bench_dir)

        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            n = cached["embeddings"].shape[0]
            print(f"[CACHED] {display_name}: {n:,} items")
            bench_order.append((bench_dir, display_name, category, n))
            total_items += n
        else:
            texts, item_ids = load_items(bench_dir)
            if not texts:
                print(f"[SKIP]   {display_name}: no items found")
                continue

            n = len(texts)
            print(f"[EMBED]  {display_name}: {n:,} items...", end=" ", flush=True)
            t0 = time.time()
            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            dt = time.time() - t0
            print(f"done in {dt:.1f}s ({n/dt:.0f} items/s)")

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "embeddings": torch.from_numpy(embeddings).half(),  # fp16 to save space
                "item_ids": item_ids,
                "model": MODEL_NAME,
            }, cache_path)

            bench_order.append((bench_dir, display_name, category, n))
            total_items += n

    print(f"\nTotal: {total_items:,} items across {len(bench_order)} benchmarks")

    # Phase 2: Collect all embeddings
    print("\nCollecting embeddings...")
    all_embeddings = []
    all_items = []
    for bench_dir, display_name, category, n in bench_order:
        cache_path = get_cache_path(bench_dir)
        cached = torch.load(cache_path, weights_only=True)
        embs = cached["embeddings"].float().numpy()
        ids = cached["item_ids"]
        all_embeddings.append(embs)
        for i in range(len(ids)):
            all_items.append({
                "benchmark": display_name,
                "category": category,
                "item_id": ids[i],
            })

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embedding matrix: {all_embeddings.shape}")

    # Phase 3: UMAP
    import umap

    print("Running UMAP...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
        metric="cosine",
        verbose=True,
        low_memory=True,
    )
    coords = reducer.fit_transform(all_embeddings)
    print(f"UMAP done in {time.time() - t0:.1f}s, shape: {coords.shape}")

    # Phase 4: Save
    # Round to 2 decimals to keep file size manageable
    item_landscape = {
        "x": np.round(coords[:, 0], 2).tolist(),
        "y": np.round(coords[:, 1], 2).tolist(),
        "benchmark": [it["benchmark"] for it in all_items],
        "category": [it["category"] for it in all_items],
    }
    with open(OUTPUT_DIR / "item_landscape_full.json", "w") as f:
        json.dump(item_landscape, f)
    sz = (OUTPUT_DIR / "item_landscape_full.json").stat().st_size
    print(f"Saved item_landscape_full.json ({len(all_items):,} items, {sz/1e6:.1f} MB)")

    # Dataset centroids
    dataset_landscape = []
    idx = 0
    for bench_dir, display_name, category, n in bench_order:
        bench_coords = coords[idx:idx + n]
        idx += n
        dataset_landscape.append({
            "benchmark": display_name,
            "category": category,
            "n_items": n,
            "x": round(float(bench_coords[:, 0].mean()), 3),
            "y": round(float(bench_coords[:, 1].mean()), 3),
        })

    with open(OUTPUT_DIR / "dataset_landscape_full.json", "w") as f:
        json.dump(dataset_landscape, f, indent=2)
    print(f"Saved dataset_landscape_full.json ({len(dataset_landscape)} benchmarks)")

    # Save coords as torch tensor for reuse
    torch.save({
        "coords": torch.from_numpy(coords).half(),
        "benchmarks": [(d, n, c, nn) for d, n, c, nn in bench_order],
        "model": MODEL_NAME,
    }, OUTPUT_DIR / "landscape_umap.pt")
    print("Saved landscape_umap.pt")


if __name__ == "__main__":
    main()
