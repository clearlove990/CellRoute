from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from hvg_research import (
    build_default_method_registry,
    evaluate_real_selection,
    load_scrna_dataset,
    subsample_dataset,
)
from hvg_research.eval import timed_call


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RefineMoE-HVG benchmark on real scRNA-seq input.")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "csv", "tsv", "txt", "mtx", "h5ad"])
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts_real")
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--delimiter", type=str, default=None)
    parser.add_argument("--obs-path", type=str, default=None)
    parser.add_argument("--var-path", type=str, default=None)
    parser.add_argument("--genes-path", type=str, default=None)
    parser.add_argument("--cells-path", type=str, default=None)
    parser.add_argument("--labels-col", type=str, default=None)
    parser.add_argument("--batches-col", type=str, default=None)
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--batches-path", type=str, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument("--gate-model-path", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device={device} cuda_available={torch.cuda.is_available()} "
        f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
    )
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    dataset = load_scrna_dataset(
        data_path=args.input_path,
        file_format=args.format,
        transpose=args.transpose,
        obs_path=args.obs_path,
        var_path=args.var_path,
        genes_path=args.genes_path,
        cells_path=args.cells_path,
        labels_col=args.labels_col,
        batches_col=args.batches_col,
        labels_path=args.labels_path,
        batches_path=args.batches_path,
        delimiter=args.delimiter,
        dataset_name=args.dataset_name,
        max_cells=args.max_cells,
        max_genes=args.max_genes,
        random_state=args.seed,
    )

    dataset = subsample_dataset(
        dataset,
        max_cells=args.max_cells,
        max_genes=args.max_genes,
        random_state=args.seed,
    )

    methods = build_default_method_registry(
        top_k=args.top_k,
        refine_epochs=args.epochs,
        random_state=args.seed,
        gate_model_path=args.gate_model_path,
    )
    all_rows = []

    print(
        f"Dataset={dataset.name} cells={dataset.counts.shape[0]} genes={dataset.counts.shape[1]} "
        f"labels={'yes' if dataset.labels is not None else 'no'} batches={'yes' if dataset.batches is not None else 'no'}"
    )

    for method_name, method_fn in methods.items():
        top_k = min(args.top_k, dataset.counts.shape[1])
        scores, elapsed = timed_call(method_fn, dataset.counts, dataset.batches, top_k)
        selected = np.argsort(scores)[-top_k:]

        metrics = evaluate_real_selection(
            counts=dataset.counts,
            selected_genes=selected,
            scorer_fn=lambda subset_counts, subset_batches, fn=method_fn, current_top_k=top_k: fn(
                subset_counts, subset_batches, current_top_k
            ),
            labels=dataset.labels,
            batches=dataset.batches,
            top_k=top_k,
            random_state=args.seed,
        )
        metrics["runtime_sec"] = elapsed
        metrics["dataset"] = dataset.name
        metrics["method"] = method_name
        metrics["top_k"] = top_k
        all_rows.append(metrics)

        save_selected_genes(
            output_dir=args.output_dir,
            method_name=method_name,
            gene_names=dataset.gene_names,
            selected=selected,
            scores=scores,
        )

    df = pd.DataFrame(all_rows)
    df = add_overall_score(df)
    result_path = os.path.join(args.output_dir, f"{dataset.name}_benchmark_results.csv")
    df.to_csv(result_path, index=False)

    summary_cols = [
        col
        for col in [
            "overall_score",
            "ari",
            "nmi",
            "label_silhouette",
            "batch_mixing",
            "cluster_silhouette",
            "neighbor_preservation",
            "stability",
            "runtime_sec",
        ]
        if col in df.columns
    ]
    summary = df.sort_values(by="overall_score", ascending=False)[["method", *summary_cols]]
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved full results to: {result_path}")

    figure_path = os.path.join(args.output_dir, f"{dataset.name}_benchmark_summary.png")
    plot_available_metrics(df, figure_path)
    print(f"Saved summary figure to: {figure_path}")


def add_overall_score(df: pd.DataFrame) -> pd.DataFrame:
    positive_metrics = [
        metric
        for metric in [
            "ari",
            "nmi",
            "label_silhouette",
            "cluster_silhouette",
            "neighbor_preservation",
            "stability",
            "batch_mixing",
        ]
        if metric in df.columns
    ]
    negative_metrics = [metric for metric in ["runtime_sec"] if metric in df.columns]
    if not positive_metrics and not negative_metrics:
        df["overall_score"] = 0.0
        return df

    score = np.zeros(len(df), dtype=np.float64)
    for metric in positive_metrics:
        score += min_max_scale(df[metric].to_numpy(dtype=np.float64))
    for metric in negative_metrics:
        score -= 0.15 * min_max_scale(df[metric].to_numpy(dtype=np.float64))
    df["overall_score"] = score
    return df


def min_max_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values.max(), values.min()):
        return np.zeros_like(values)
    return (values - values.min()) / (values.max() - values.min())


def save_selected_genes(
    *,
    output_dir: str,
    method_name: str,
    gene_names: np.ndarray,
    selected: np.ndarray,
    scores: np.ndarray,
) -> None:
    ordered = selected[np.argsort(scores[selected])[::-1]]
    frame = pd.DataFrame(
        {
            "rank": np.arange(1, len(ordered) + 1),
            "gene": gene_names[ordered],
            "score": scores[ordered],
            "gene_index": ordered,
        }
    )
    path = Path(output_dir) / f"{method_name}_selected_genes.csv"
    frame.to_csv(path, index=False)


def plot_available_metrics(df: pd.DataFrame, output_path: str) -> None:
    metric_order = ["overall_score", "ari", "nmi", "batch_mixing", "cluster_silhouette", "neighbor_preservation", "stability"]
    metrics = [metric for metric in metric_order if metric in df.columns]
    if not metrics:
        return

    n_cols = min(3, len(metrics))
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    ordered = df.sort_values(by="overall_score", ascending=False if "overall_score" in df.columns else True)
    for ax, metric in zip(axes, metrics, strict=False):
        ordered.plot(x="method", y=metric, kind="bar", ax=ax, legend=False)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)

    for ax in axes[len(metrics) :]:
        ax.axis("off")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
