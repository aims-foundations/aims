#!/usr/bin/env python3
"""
Pre-compute small data subsets for interactive pyodide visualization in Chapter 1.
Saves subsets as JSON files that can be loaded in the browser.

Usage:
    python scripts/precompute_data_subsets.py
    python scripts/precompute_data_subsets.py --cache-dir /path/to/snapshot
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

# Configuration
DEFAULT_N_TAKERS = 50  # Number of test takers (models) to include
DEFAULT_N_ITEMS = 100  # Number of items (questions) to include
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "src" / "data"
DATASET_REPO_ID = "stair-lab/reeval_fa"
REQUIRED_FILES = (
    "data/HELM_benchmark.pkl",
    "data/benchmark_data_open_llm_full_no_arc.pkl",
)


def numpy_to_list(arr):
    """Convert numpy array to nested list, handling NaN values."""
    result = []
    for row in arr:
        row_data = []
        for val in row:
            if np.isnan(val):
                row_data.append(None)
            else:
                row_data.append(float(val))
        result.append(row_data)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, help="Path to a dataset snapshot containing data/*.pkl.")
    parser.add_argument("--repo-id", default=DATASET_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-takers", type=int, default=DEFAULT_N_TAKERS)
    parser.add_argument("--n-items", type=int, default=DEFAULT_N_ITEMS)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def snapshot_contains_required_files(snapshot_dir: Path) -> bool:
    return all((snapshot_dir / rel_path).exists() for rel_path in REQUIRED_FILES)


def iter_cached_snapshots(repo_id: str) -> Iterable[Path]:
    dataset_cache_name = f"datasets--{repo_id.replace('/', '--')}"
    hub_roots = []

    if os.environ.get("HF_HUB_CACHE"):
        hub_roots.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        hub_roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    hub_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    seen = set()
    for hub_root in hub_roots:
        if hub_root in seen:
            continue
        seen.add(hub_root)
        snapshots_dir = hub_root / dataset_cache_name / "snapshots"
        if not snapshots_dir.exists():
            continue

        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            yield snapshot


def resolve_cache_dir(args: argparse.Namespace) -> Path:
    if args.cache_dir is not None:
        cache_dir = args.cache_dir.expanduser()
        if not snapshot_contains_required_files(cache_dir):
            raise FileNotFoundError(
                f"{cache_dir} does not contain the required dataset files: "
                f"{', '.join(REQUIRED_FILES)}"
            )
        return cache_dir

    for snapshot in iter_cached_snapshots(args.repo_id):
        if snapshot_contains_required_files(snapshot):
            return snapshot

    if args.no_download:
        raise FileNotFoundError(
            f"No cached data found for {args.repo_id}. Set --cache-dir or rerun "
            "without --no-download to fetch the dataset."
        )

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required to download the dataset. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    try:
        download_path = Path(
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                allow_patterns=list(REQUIRED_FILES),
            )
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"No cached data found for {args.repo_id}, and automatic download "
            "failed. Authenticate with Hugging Face if the dataset is gated, "
            "or provide --cache-dir explicitly."
        ) from exc

    if not snapshot_contains_required_files(download_path):
        raise FileNotFoundError(
            f"Downloaded snapshot at {download_path} is missing required files: "
            f"{', '.join(REQUIRED_FILES)}"
        )

    return download_path


def main():
    args = parse_args()
    cache_path = resolve_cache_dir(args)

    print(f"Using cached data from: {cache_path}")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process HELM data
    print("\nProcessing HELM Benchmark data...")
    with open(cache_path / "data/HELM_benchmark.pkl", "rb") as f:
        helm_df = pickle.load(f)

    # DataFrame: rows are models, columns are questions
    helm_matrix = helm_df.values  # Convert to numpy array
    helm_n_takers = min(args.n_takers, helm_matrix.shape[0])
    helm_n_items = min(args.n_items, helm_matrix.shape[1])
    helm_subset = helm_matrix[:helm_n_takers, :helm_n_items]

    # Get model names and item info for subset
    helm_models = list(helm_df.index[:helm_n_takers])

    helm_output = {
        "full_shape": [int(helm_matrix.shape[0]), int(helm_matrix.shape[1])],
        "subset_shape": [helm_n_takers, helm_n_items],
        "models": helm_models,
        "data": numpy_to_list(helm_subset)
    }

    helm_path = output_dir / "helm_subset.json"
    with open(helm_path, "w") as f:
        json.dump(helm_output, f)
    print(f"  Full: {helm_matrix.shape[0]} x {helm_matrix.shape[1]}")
    print(f"  Subset: {helm_n_takers} x {helm_n_items}")
    print(f"  Saved to: {helm_path}")

    # Process Open LLM data
    print("\nProcessing Open LLM Leaderboard data...")
    with open(cache_path / "data/benchmark_data_open_llm_full_no_arc.pkl", "rb") as f:
        openllm_df = pickle.load(f)

    openllm_matrix = openllm_df.values
    openllm_n_takers = min(args.n_takers, openllm_matrix.shape[0])
    openllm_n_items = min(args.n_items, openllm_matrix.shape[1])
    openllm_subset = openllm_matrix[:openllm_n_takers, :openllm_n_items]

    # Get model names for subset
    openllm_models = list(openllm_df.index[:openllm_n_takers])

    openllm_output = {
        "full_shape": [int(openllm_matrix.shape[0]), int(openllm_matrix.shape[1])],
        "subset_shape": [openllm_n_takers, openllm_n_items],
        "models": openllm_models,
        "data": numpy_to_list(openllm_subset)
    }

    openllm_path = output_dir / "openllm_subset.json"
    with open(openllm_path, "w") as f:
        json.dump(openllm_output, f)
    print(f"  Full: {openllm_matrix.shape[0]} x {openllm_matrix.shape[1]}")
    print(f"  Subset: {openllm_n_takers} x {openllm_n_items}")
    print(f"  Saved to: {openllm_path}")

    print("\nDone! Data subsets saved to src/data/")

if __name__ == "__main__":
    main()
