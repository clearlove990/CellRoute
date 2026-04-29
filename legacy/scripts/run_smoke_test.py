from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from hvg_research import build_default_method_registry, evaluate_selection, generate_synthetic_scrna
from hvg_research.eval import timed_call


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RefineMoE-HVG smoke tests on synthetic scRNA-seq data.")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--cells", type=int, default=800)
    parser.add_argument("--genes", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gate-model-path", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    methods = build_default_method_registry(
        top_k=args.top_k,
        refine_epochs=args.epochs,
        random_state=args.seed,
        gate_model_path=args.gate_model_path,
    )

    scenarios = ["discrete", "trajectory", "batch_shift"]
    all_rows = []

    for scenario_idx, scenario in enumerate(scenarios):
        data = generate_synthetic_scrna(
            scenario=scenario,
            n_cells=args.cells,
            n_genes=args.genes,
            random_state=args.seed + scenario_idx,
        )

        for method_name, method_fn in methods.items():
            scores, elapsed = timed_call(method_fn, data.counts, data.batches, args.top_k)
            selected = np.argsort(scores)[-args.top_k:]

            metrics = evaluate_selection(
                counts=data.counts,
                selected_genes=selected,
                cell_types=data.cell_types,
                informative_genes=data.informative_genes,
                batch_genes=data.batch_genes,
                scorer_fn=lambda subset_counts, subset_batches, fn=method_fn: fn(subset_counts, subset_batches, args.top_k),
                batches=data.batches,
                top_k=args.top_k,
                random_state=args.seed,
            )
            metrics["runtime_sec"] = elapsed
            metrics["scenario"] = scenario
            metrics["method"] = method_name
            all_rows.append(metrics)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.output_dir, "smoke_test_results.csv")
    df.to_csv(csv_path, index=False)

    summary = (
        df.groupby("method")[["ari", "nmi", "informative_recall", "batch_gene_fraction", "stability", "runtime_sec"]]
        .mean()
        .sort_values(by=["ari", "informative_recall"], ascending=False)
    )
    print(summary.round(4).to_string())
    print(f"\nSaved full results to: {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    plot_metric(df, axes[0], metric="ari", title="ARI by Scenario")
    plot_metric(df, axes[1], metric="informative_recall", title="Informative Recall by Scenario")
    plot_metric(df, axes[2], metric="batch_gene_fraction", title="Batch-Gene Fraction by Scenario", invert=False)
    png_path = os.path.join(args.output_dir, "smoke_test_summary.png")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Saved summary figure to: {png_path}")


def plot_metric(df: pd.DataFrame, ax, metric: str, title: str, invert: bool = False) -> None:
    pivot = df.pivot(index="method", columns="scenario", values=metric)
    if invert:
        pivot = -pivot
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=25)
    if metric == "batch_gene_fraction":
        ax.set_ylabel("lower is better")


if __name__ == "__main__":
    main()
