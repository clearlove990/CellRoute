from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    }
)


PALETTE = {
    "baseline_gray": "#B0BEC5",
    "teal": "#264653",
    "green": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "coral": "#E76F51",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "pink": "#CC79A7",
    "red": "#D55E00",
    "slate": "#6B7280",
}


def save_figure(
    fig: plt.Figure,
    output_path: str | Path,
    *,
    extra_formats: Sequence[str] = ("pdf",),
) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path)
    for fmt in extra_formats:
        fig.savefig(target_path.with_suffix(f".{fmt.lstrip('.')}"))
    plt.close(fig)
    return target_path


def plot_metric_heatmap(
    metric_frame: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    group_col: str = "display_method",
    metric_col: str = "metric_name",
    delta_col: str = "delta",
    title: str = "Metric Tradeoff Heatmap",
    metric_order: Sequence[str] | None = None,
    group_order: Sequence[str] | None = None,
) -> plt.Figure:
    grouped = (
        metric_frame.groupby([group_col, metric_col], sort=False)[delta_col]
        .agg(
            positive=lambda series: int((series > 1e-9).sum()),
            negative=lambda series: int((series < -1e-9).sum()),
        )
        .reset_index()
    )
    grouped["directional_score"] = np.where(
        (grouped["positive"] + grouped["negative"]) > 0,
        (grouped["positive"] - grouped["negative"]) / (grouped["positive"] + grouped["negative"]),
        0.0,
    )

    if metric_order is None:
        metric_order = list(dict.fromkeys(grouped[metric_col].tolist()))
    if group_order is None:
        group_order = list(dict.fromkeys(grouped[group_col].tolist()))

    heatmap = grouped.pivot(index=group_col, columns=metric_col, values="directional_score").reindex(
        index=list(group_order),
        columns=list(metric_order),
    )

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    matrix = heatmap.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="#F4F4F4")
    image = ax.imshow(masked, aspect="auto", vmin=-1.0, vmax=1.0, cmap=cmap)

    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels(metric_order, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(group_order)))
    ax.set_yticklabels(group_order)
    ax.set_title(title)

    for row_index in range(len(group_order)):
        for col_index in range(len(metric_order)):
            value = heatmap.iloc[row_index, col_index]
            if pd.isna(value):
                continue
            ax.text(col_index, row_index, f"{value:+.2f}", ha="center", va="center", color="black", fontsize=8)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
    colorbar.set_label("Directional score")

    if output_path is not None:
        save_figure(fig, output_path)
    return fig


def plot_frontier(
    summary_frame: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    x_col: str = "protected_loss_count",
    y_col: str = "target_gain_count",
    hue_col: str = "display_method",
    facet_col: str = "dataset",
    label_col: str = "budget_or_alpha",
    title: str = "Pareto Frontier",
) -> plt.Figure:
    if facet_col in summary_frame.columns:
        facet_values = list(dict.fromkeys(summary_frame[facet_col].tolist()))
    else:
        facet_values = ["all"]

    fig, axes = plt.subplots(1, len(facet_values), figsize=(4.2 * len(facet_values), 3.2), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    hue_values = list(dict.fromkeys(summary_frame[hue_col].tolist()))
    color_cycle = [PALETTE["coral"], PALETTE["blue"], PALETTE["green"], PALETTE["gold"], PALETTE["pink"]]
    color_map = {hue: color_cycle[idx % len(color_cycle)] for idx, hue in enumerate(hue_values)}

    for axis, facet_value in zip(axes, facet_values, strict=False):
        subset = summary_frame.copy()
        if facet_col in summary_frame.columns:
            subset = subset[subset[facet_col] == facet_value]
        for hue_value in hue_values:
            hue_subset = subset[subset[hue_col] == hue_value].copy()
            if hue_subset.empty:
                continue

            if label_col in hue_subset.columns:
                label_numeric = pd.to_numeric(hue_subset[label_col], errors="coerce")
                hue_subset = hue_subset.assign(_label_numeric=label_numeric)
                if hue_subset["_label_numeric"].notna().any():
                    hue_subset = hue_subset.sort_values("_label_numeric", kind="mergesort")

            axis.plot(
                hue_subset[x_col],
                hue_subset[y_col],
                color=color_map[hue_value],
                marker="o",
                label=hue_value,
                zorder=3,
            )

            if label_col in hue_subset.columns:
                for _, row in hue_subset.iterrows():
                    axis.text(
                        float(row[x_col]) + 0.03,
                        float(row[y_col]) + 0.03,
                        str(row[label_col]),
                        fontsize=7,
                        color=color_map[hue_value],
                    )

        axis.axvline(0.0, color=PALETTE["slate"], linestyle="--", linewidth=1.0)
        axis.set_xlabel(x_col.replace("_", " "))
        axis.set_title(str(facet_value))

    axes[0].set_ylabel(y_col.replace("_", " "))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=min(3, len(labels)), loc="upper center", bbox_to_anchor=(0.5, 1.10))
    fig.suptitle(title, y=1.16)

    if output_path is not None:
        save_figure(fig, output_path)
    return fig


def plot_win_loss_barplot(
    summary_frame: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    group_col: str = "experiment_family",
    label_col: str = "pareto_label",
    title: str = "Pareto Label Summary",
) -> plt.Figure:
    counts = (
        summary_frame.groupby([group_col, label_col], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["clean_win", "near_clean", "mixed_tradeoff", "dominated", "flat_or_tied"], fill_value=0)
    )

    labels = counts.index.tolist()
    y = np.arange(len(labels))
    left = np.zeros(len(labels), dtype=float)
    colors = {
        "clean_win": PALETTE["green"],
        "near_clean": PALETTE["gold"],
        "mixed_tradeoff": PALETTE["coral"],
        "dominated": PALETTE["blue"],
        "flat_or_tied": PALETTE["baseline_gray"],
    }

    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    for status in counts.columns:
        values = counts[status].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=colors[status], height=0.62, label=status.replace("_", " "))
        left = left + values

    for idx, total in enumerate(left):
        ax.text(total + 0.1, idx, f"{int(total)}", va="center", fontsize=8, color="#444444")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Configuration count")
    ax.set_title(title)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    if output_path is not None:
        save_figure(fig, output_path)
    return fig

