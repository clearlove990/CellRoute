from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.14,
            "grid.linestyle": "-",
        }
    )


def _method_color(method_name: str) -> str:
    palette = {
        "baseline_pca": "#8c9aa5",
        "graph_ssl_main": "#e76f51",
        "graph_ssl_no_sample_reg": "#2a9d8f",
        "graph_ssl_no_neighbor": "#3d5a80",
        "graph_ssl_strong_sample_reg": "#e9c46a",
    }
    return palette.get(str(method_name), "#264653")


def plot_motif_layout(
    spot_table: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> None:
    _set_publication_style()
    output_path = Path(output_path)
    frame = spot_table.loc[
        np.isfinite(spot_table["layout_1"].to_numpy(dtype=np.float64))
        & np.isfinite(spot_table["layout_2"].to_numpy(dtype=np.float64))
    ].copy()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    codes, _ = pd.factorize(frame["motif_id"].astype(str))
    ax.scatter(
        frame["layout_1"].to_numpy(dtype=np.float32),
        frame["layout_2"].to_numpy(dtype=np.float32),
        c=codes,
        cmap="tab20",
        s=6,
        alpha=0.80,
        linewidths=0,
    )
    centers = (
        frame.groupby("motif_id", observed=False)[["layout_1", "layout_2"]]
        .median()
        .reset_index()
    )
    for _, row in centers.iterrows():
        if np.isfinite(float(row["layout_1"])) and np.isfinite(float(row["layout_2"])):
            ax.text(float(row["layout_1"]), float(row["layout_2"]), str(row["motif_id"]), fontsize=8, ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("Layout 1")
    ax.set_ylabel("Layout 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_motif_spatial_map(
    spot_table: pd.DataFrame,
    *,
    sample_id: str,
    output_path: str | Path,
    title: str,
) -> None:
    _set_publication_style()
    output_path = Path(output_path)
    frame = spot_table.loc[spot_table["sample_id"].astype(str) == str(sample_id)].copy()
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    codes, _ = pd.factorize(frame["motif_id"].astype(str))
    ax.scatter(
        frame["spatial_x"].to_numpy(dtype=np.float32),
        -frame["spatial_y"].to_numpy(dtype=np.float32),
        c=codes,
        cmap="tab20",
        s=7,
        alpha=0.90,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_condition_abundance(
    abundance_table: pd.DataFrame,
    differential_table: pd.DataFrame,
    *,
    dataset_id: str,
    condition_a: str,
    condition_b: str,
    output_path: str | Path,
    top_n: int = 8,
) -> None:
    _set_publication_style()
    output_path = Path(output_path)
    differential_subset = differential_table.loc[differential_table["dataset_id"] == dataset_id].copy()
    if differential_subset.empty:
        return
    differential_subset = differential_subset.sort_values(
        ["association_call", "mixedlm_pvalue", "permutation_pvalue", "delta_fraction"],
        ascending=[False, True, True, False],
    )
    selected = differential_subset.head(top_n)
    motif_order = selected["motif_id"].astype(str).tolist()
    abundance_subset = abundance_table.loc[
        (abundance_table["dataset_id"] == dataset_id)
        & (abundance_table["motif_id"].astype(str).isin(motif_order))
        & (abundance_table["condition"].astype(str).isin([condition_a, condition_b]))
    ].copy()
    summary = (
        abundance_subset.groupby(["motif_id", "condition"], observed=False)["motif_fraction"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = np.divide(
        summary["std"].fillna(0.0).to_numpy(dtype=np.float64),
        np.sqrt(np.maximum(summary["count"].to_numpy(dtype=np.float64), 1.0)),
        out=np.zeros(summary.shape[0], dtype=np.float64),
        where=summary["count"].to_numpy(dtype=np.float64) > 0,
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(motif_order), dtype=np.float64)
    width = 0.36
    palette = {condition_a: "#3a78a1", condition_b: "#c26b3b"}
    for offset, condition in [(-width / 2, condition_a), (width / 2, condition_b)]:
        condition_frame = summary.loc[summary["condition"].astype(str) == condition].set_index("motif_id")
        means = [float(condition_frame.at[motif_id, "mean"]) if motif_id in condition_frame.index else 0.0 for motif_id in motif_order]
        sems = [float(condition_frame.at[motif_id, "sem"]) if motif_id in condition_frame.index else 0.0 for motif_id in motif_order]
        ax.bar(x + offset, means, width=width, color=palette[condition], label=condition, yerr=sems, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(motif_order, rotation=35, ha="right")
    ax.set_ylabel("Mean sample-level motif fraction")
    ax.set_title(f"{dataset_id}: motif abundance by condition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_differential_volcano(
    differential_table: pd.DataFrame,
    *,
    dataset_id: str,
    output_path: str | Path,
    title: str,
) -> None:
    _set_publication_style()
    output_path = Path(output_path)
    frame = differential_table.loc[differential_table["dataset_id"] == dataset_id].copy()
    if frame.empty:
        return
    significance = frame["permutation_pvalue"].fillna(frame["mixedlm_pvalue"]).fillna(1.0).to_numpy(dtype=np.float64)
    y_value = -np.log10(np.clip(significance, 1.0e-6, 1.0))
    x_value = frame["log2_fold_change"].to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(x_value) & np.isfinite(y_value)
    frame = frame.loc[finite_mask].copy()
    x_value = x_value[finite_mask]
    y_value = y_value[finite_mask]
    if frame.empty:
        return
    color = np.where(frame["association_call"].to_numpy(dtype=bool), "#c26b3b", "#6b8796")
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.scatter(x_value, y_value, c=color, s=52, alpha=0.88, linewidths=0)
    for _, row in frame.sort_values(["association_call", "mixedlm_pvalue"], ascending=[False, True]).head(5).iterrows():
        ax.text(
            float(row["log2_fold_change"]),
            float(-np.log10(max(float(row["mixedlm_pvalue"]), 1.0e-6))),
            str(row["motif_id"]),
            fontsize=8,
        )
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10 permutation p-value")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_representation_comparison(
    baseline_spot_table: pd.DataFrame,
    ssl_spot_table: pd.DataFrame,
    *,
    output_path: str | Path,
    baseline_title: str,
    ssl_title: str,
) -> None:
    _set_publication_style()
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=False, sharey=False)
    for ax, frame, title in zip(axes, [baseline_spot_table, ssl_spot_table], [baseline_title, ssl_title], strict=True):
        valid = frame.loc[
            np.isfinite(frame["layout_1"].to_numpy(dtype=np.float64))
            & np.isfinite(frame["layout_2"].to_numpy(dtype=np.float64))
        ].copy()
        codes, _ = pd.factorize(valid["motif_id"].astype(str))
        ax.scatter(
            valid["layout_1"].to_numpy(dtype=np.float32),
            valid["layout_2"].to_numpy(dtype=np.float32),
            c=codes,
            cmap="tab20",
            s=5,
            alpha=0.82,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("Layout 1")
        ax.set_ylabel("Layout 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metric_boxplot(
    metric_frame: pd.DataFrame,
    *,
    metric_col: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    _set_publication_style()
    output_path = Path(output_path)
    if metric_frame.empty or metric_col not in metric_frame.columns:
        return
    frame = metric_frame.copy()
    method_order = (
        frame.groupby("method", observed=False)[metric_col]
        .mean()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    data = [
        frame.loc[frame["method"].astype(str) == method_name, metric_col].to_numpy(dtype=np.float64)
        for method_name in method_order
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    boxplot = ax.boxplot(
        data,
        labels=method_order,
        patch_artist=True,
        widths=0.58,
        medianprops={"color": "#1f2933", "linewidth": 1.4},
        whiskerprops={"color": "#66788a", "linewidth": 1.0},
        capprops={"color": "#66788a", "linewidth": 1.0},
    )
    for patch, method_name in zip(boxplot["boxes"], method_order, strict=True):
        patch.set_facecolor(_method_color(method_name))
        patch.set_alpha(0.70)
        patch.set_edgecolor("#ffffff")
        patch.set_linewidth(0.8)
    rng = np.random.default_rng(7)
    for x_pos, method_name in enumerate(method_order, start=1):
        subset = frame.loc[frame["method"].astype(str) == method_name, metric_col].to_numpy(dtype=np.float64)
        jitter = rng.normal(0.0, 0.05, size=subset.shape[0])
        ax.scatter(
            np.full(subset.shape[0], float(x_pos)) + jitter,
            subset,
            s=22,
            color="#1f2933",
            alpha=0.42,
            linewidths=0,
            zorder=3,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_signal_vs_leakage(
    metric_frame: pd.DataFrame,
    *,
    output_path: str | Path,
    x_col: str = "batch_sample_leakage",
    y_col: str = "condition_separability",
    title: str = "Sample leakage vs condition signal",
) -> None:
    _set_publication_style()
    output_path = Path(output_path)
    if metric_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    markers = ["o", "s", "^", "D", "v", "P"]
    dataset_order = sorted(metric_frame["dataset_id"].astype(str).unique().tolist())
    marker_lookup = {dataset_id: markers[index % len(markers)] for index, dataset_id in enumerate(dataset_order)}
    for _, row in metric_frame.iterrows():
        method_name = str(row["method"])
        dataset_id = str(row["dataset_id"])
        ax.scatter(
            float(row[x_col]),
            float(row[y_col]),
            s=86,
            color=_method_color(method_name),
            marker=marker_lookup[dataset_id],
            alpha=0.92,
            edgecolors="#ffffff",
            linewidths=0.8,
        )
        ax.text(
            float(row[x_col]) + 0.004,
            float(row[y_col]) + 0.004,
            f"{dataset_id}:{method_name}",
            fontsize=7.2,
            color="#1f2933",
        )
    ax.set_xlabel("Batch / sample leakage (lower is better)")
    ax.set_ylabel("Condition separability (higher is better)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_pareto_frontier(
    metric_frame: pd.DataFrame,
    *,
    output_path: str | Path,
    x_col: str = "batch_sample_leakage",
    y_col: str = "overall_biological_score",
    title: str = "Representation Pareto frontier",
) -> None:
    _set_publication_style()
    output_path = Path(output_path)
    if metric_frame.empty:
        return
    frame = metric_frame.copy()
    if "pareto_optimal" not in frame.columns:
        frame["pareto_optimal"] = False
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    for _, row in frame.iterrows():
        method_name = str(row["method"])
        is_frontier = bool(row.get("pareto_optimal", False))
        ax.scatter(
            float(row[x_col]),
            float(row[y_col]),
            s=102 if is_frontier else 74,
            color=_method_color(method_name),
            alpha=0.95 if is_frontier else 0.55,
            edgecolors="#1f2933" if is_frontier else "#ffffff",
            linewidths=1.1 if is_frontier else 0.8,
            zorder=4 if is_frontier else 2,
        )
        ax.text(
            float(row[x_col]) + 0.004,
            float(row[y_col]) + 0.003,
            f"{row['dataset_id']}:{method_name}",
            fontsize=7.0,
            color="#1f2933",
        )
    frontier = frame.loc[frame["pareto_optimal"].astype(bool)].sort_values(x_col, ascending=True)
    if frontier.shape[0] >= 2:
        ax.plot(
            frontier[x_col].to_numpy(dtype=np.float64),
            frontier[y_col].to_numpy(dtype=np.float64),
            color="#1f2933",
            linewidth=1.2,
            linestyle="--",
            alpha=0.75,
            zorder=3,
        )
    ax.set_xlabel("Batch / sample leakage (lower is better)")
    ax.set_ylabel("Biological signal composite (higher is better)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
