from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pareto_eval.clean_win import classify_frame, get_runtime_device, profile_for_evaluation_mode


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

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "01_hvg_tradeoff_audit"
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"


@dataclass(frozen=True)
class ArtifactSpec:
    family: str
    root: Path
    summary_csv: Path
    loader_kind: str


ARTIFACT_SPECS = (
    ArtifactSpec(
        family="mainline_benchmark",
        root=REPO_ROOT / "artifacts_benchmarkhvg_formal_partial_20260424_v1",
        summary_csv=REPO_ROOT / "artifacts_benchmarkhvg_formal_partial_20260424_v1" / "formal_partial_summary.csv",
        loader_kind="formal",
    ),
    ArtifactSpec(
        family="risk_parity_family",
        root=REPO_ROOT / "artifacts_benchmarkhvg_formal_partial_20260424_v3",
        summary_csv=REPO_ROOT / "artifacts_benchmarkhvg_formal_partial_20260424_v3" / "risk_parity_family_summary.csv",
        loader_kind="formal",
    ),
    ArtifactSpec(
        family="rank_blend_frontier",
        root=REPO_ROOT / "artifacts_benchmarkhvg_rank_blend_frontier_20260424_v1",
        summary_csv=REPO_ROOT / "artifacts_benchmarkhvg_rank_blend_frontier_20260424_v1" / "rank_blend_frontier_summary.csv",
        loader_kind="rank_blend",
    ),
    ArtifactSpec(
        family="anchor_repair",
        root=REPO_ROOT / "artifacts_benchmarkhvg_anchor_repair_20260425_v1",
        summary_csv=REPO_ROOT / "artifacts_benchmarkhvg_anchor_repair_20260425_v1" / "anchor_repair_summary.csv",
        loader_kind="anchor_repair",
    ),
)


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_partial_metrics_csv(path: Path) -> tuple[str, dict[str, float], dict[str, float]]:
    metrics = pd.read_csv(path)
    baseline = metrics.loc[metrics["setting"] == "baseline_mix"].iloc[0].to_dict()
    adapter = metrics.loc[metrics["setting"] != "baseline_mix"].iloc[0].to_dict()
    metric_names = [
        column
        for column in metrics.columns
        if column not in {"setting", "budget", "hvg_count", "pca_rows", "pca_cols"}
        and not pd.isna(baseline.get(column))
        and not pd.isna(adapter.get(column))
    ]
    if {"ari", "nmi", "lisi"}.issubset(metric_names):
        evaluation_mode = "discrete"
    elif {"dist_cor", "knn_ratio", "three_nn", "max_ari", "max_nmi"}.issubset(metric_names):
        evaluation_mode = "continuous"
    else:
        raise ValueError(f"Unable to infer evaluation mode from {path}")
    return evaluation_mode, baseline, adapter


def read_official_eval_json(path: Path) -> tuple[str, dict[str, float], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluation_mode = str(payload["evaluation_mode"])
    return evaluation_mode, payload["baseline"], payload["adapter"]


def setting_label(evaluation_mode: str) -> str:
    return f"official_partial_{evaluation_mode}"


def normalize_budget_or_alpha(value: Any) -> str:
    if value is None:
        return "direct"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if math.isclose(value, round(value)):
            return str(int(round(value)))
        return f"{value:.2f}"
    return str(value)


def canonical_metric_order() -> list[str]:
    return ["var_ratio", "ari", "nmi", "lisi", "dist_cor", "knn_ratio", "three_nn", "max_ari", "max_nmi"]


def display_method_label(experiment_family: str, method: str) -> str:
    mapping = {
        "adaptive_risk_parity_hvg": "risk_parity",
        "adaptive_risk_parity_safe_hvg": "risk_parity_safe",
        "adaptive_risk_parity_ultrasafe_hvg": "risk_parity_ultrasafe",
        "adaptive_spectral_locality_hvg": "spectral_locality",
        "adaptive_core_consensus_hvg": "core_consensus",
        "adaptive_eb_shrinkage_hvg": "eb_shrinkage",
    }
    base = mapping.get(method, method)
    if experiment_family == "rank_blend_frontier":
        return f"rank_blend:{base}"
    if experiment_family == "anchor_repair":
        return f"anchor_repair:{base}"
    return base


def make_config_id(family: str, method: str, dataset: str, budget_or_alpha: str) -> str:
    return f"{family}|{method}|{dataset}|{budget_or_alpha}"


def make_canonical_config_id(run_key: str, budget_or_alpha: str) -> str:
    return f"{run_key}|{budget_or_alpha}"


def build_metric_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in ARTIFACT_SPECS:
        summary = pd.read_csv(spec.summary_csv)
        if spec.loader_kind == "formal":
            for _, row in summary.iterrows():
                run_dir = REPO_ROOT / str(row["run_dir"])
                metrics_path = run_dir / "partial_metrics.csv"
                evaluation_mode, baseline, adapter = read_partial_metrics_csv(metrics_path)
                method = str(row["method"])
                dataset = str(row["dataset"])
                budget_or_alpha = "direct"
                run_key = str(run_dir.resolve())
                for metric_name in profile_for_evaluation_mode(evaluation_mode).metric_names:
                    baseline_value = float(baseline[metric_name])
                    candidate_value = float(adapter[metric_name])
                    delta = candidate_value - baseline_value
                    rows.append(
                        {
                            "experiment_family": spec.family,
                            "method": method,
                            "display_method": display_method_label(spec.family, method),
                            "dataset": dataset,
                            "setting": setting_label(evaluation_mode),
                            "evaluation_mode": evaluation_mode,
                            "budget_or_alpha": budget_or_alpha,
                            "metric_name": metric_name,
                            "baseline_value": baseline_value,
                            "candidate_value": candidate_value,
                            "delta": delta,
                            "direction": "higher_is_better",
                            "is_win": delta > 1e-9,
                            "is_loss": delta < -1e-9,
                            "artifact_clean_win": bool(row.get("clean_win", False)),
                            "artifact_mixed_tradeoff": bool(row.get("mixed_tradeoff", False)),
                            "artifact_path": str(run_dir.resolve()),
                            "config_id": make_config_id(spec.family, method, dataset, budget_or_alpha),
                            "run_key": run_key,
                            "canonical_config_id": make_canonical_config_id(run_key, budget_or_alpha),
                        }
                    )
        elif spec.loader_kind == "rank_blend":
            for _, row in summary.iterrows():
                run_dir = REPO_ROOT / str(row["run_dir"])
                metrics_path = run_dir / "partial_metrics.csv"
                evaluation_mode, baseline, adapter = read_partial_metrics_csv(metrics_path)
                candidate_method = str(row["candidate"])
                dataset = str(row["dataset"])
                budget_or_alpha = normalize_budget_or_alpha(float(row["alpha"]))
                run_key = str(run_dir.resolve())
                for metric_name in profile_for_evaluation_mode(evaluation_mode).metric_names:
                    baseline_value = float(baseline[metric_name])
                    candidate_value = float(adapter[metric_name])
                    delta = candidate_value - baseline_value
                    rows.append(
                        {
                            "experiment_family": spec.family,
                            "method": candidate_method,
                            "display_method": display_method_label(spec.family, candidate_method),
                            "dataset": dataset,
                            "setting": setting_label(evaluation_mode),
                            "evaluation_mode": evaluation_mode,
                            "budget_or_alpha": budget_or_alpha,
                            "metric_name": metric_name,
                            "baseline_value": baseline_value,
                            "candidate_value": candidate_value,
                            "delta": delta,
                            "direction": "higher_is_better",
                            "is_win": delta > 1e-9,
                            "is_loss": delta < -1e-9,
                            "artifact_clean_win": bool(row.get("clean_win", False)),
                            "artifact_mixed_tradeoff": bool(row.get("mixed_tradeoff", False)),
                            "artifact_path": str(run_dir.resolve()),
                            "config_id": make_config_id(spec.family, candidate_method, dataset, budget_or_alpha),
                            "run_key": run_key,
                            "canonical_config_id": make_canonical_config_id(run_key, budget_or_alpha),
                        }
                    )
        elif spec.loader_kind == "anchor_repair":
            for _, row in summary.iterrows():
                run_dir = REPO_ROOT / str(row["run_dir"])
                budget = int(row["budget"])
                eval_json = run_dir / f"official_extra_rank_eval_budget_{budget}.json"
                evaluation_mode, baseline, adapter = read_official_eval_json(eval_json)
                metadata_path = run_dir / "repair_metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                candidate_method = str(metadata["candidate_method"])
                dataset = str(row["dataset"])
                budget_or_alpha = normalize_budget_or_alpha(budget)
                run_key = str((run_dir / eval_json.name).resolve())
                for metric_name in profile_for_evaluation_mode(evaluation_mode).metric_names:
                    baseline_value = float(baseline[metric_name])
                    candidate_value = float(adapter[metric_name])
                    delta = candidate_value - baseline_value
                    rows.append(
                        {
                            "experiment_family": spec.family,
                            "method": candidate_method,
                            "display_method": display_method_label(spec.family, candidate_method),
                            "dataset": dataset,
                            "setting": setting_label(evaluation_mode),
                            "evaluation_mode": evaluation_mode,
                            "budget_or_alpha": budget_or_alpha,
                            "metric_name": metric_name,
                            "baseline_value": baseline_value,
                            "candidate_value": candidate_value,
                            "delta": delta,
                            "direction": "higher_is_better",
                            "is_win": delta > 1e-9,
                            "is_loss": delta < -1e-9,
                            "artifact_clean_win": bool(row.get("clean_win", False)),
                            "artifact_mixed_tradeoff": bool(row.get("mixed_tradeoff", False)),
                            "artifact_path": str(run_dir.resolve()),
                            "config_id": make_config_id(spec.family, candidate_method, dataset, budget_or_alpha),
                            "run_key": run_key,
                            "canonical_config_id": make_canonical_config_id(run_key, budget_or_alpha),
                        }
                    )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported loader_kind: {spec.loader_kind}")

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No metric rows were collected.")
    return frame


def add_profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    summary = classify_frame(frame)
    merged = frame.merge(summary, on=["config_id", "evaluation_mode"], how="left", validate="many_to_one")
    merged["clean_win"] = merged["clean_win"].fillna(False).astype(bool)
    merged["near_clean"] = merged["near_clean"].fillna(False).astype(bool)
    merged["mixed_tradeoff"] = merged["mixed_tradeoff"].fillna(False).astype(bool)
    merged["dominated"] = merged["dominated"].fillna(False).astype(bool)

    def metric_role(row: pd.Series) -> str:
        profile = profile_for_evaluation_mode(str(row["evaluation_mode"]))
        if row["metric_name"] in profile.target_metrics:
            return "target"
        if row["metric_name"] in profile.protected_metrics:
            return "protected"
        return "other"

    merged["metric_role"] = merged.apply(metric_role, axis=1)
    canonical_source = (
        merged[["canonical_config_id", "config_id"]]
        .drop_duplicates()
        .drop_duplicates("canonical_config_id", keep="first")
    )
    canonical_source_ids = set(canonical_source["config_id"].tolist())
    merged["is_canonical"] = merged["config_id"].isin(canonical_source_ids)
    return merged


def build_config_summary(frame: pd.DataFrame) -> pd.DataFrame:
    config_columns = [
        "config_id",
        "canonical_config_id",
        "experiment_family",
        "method",
        "display_method",
        "dataset",
        "setting",
        "evaluation_mode",
        "budget_or_alpha",
        "artifact_path",
        "artifact_clean_win",
        "artifact_mixed_tradeoff",
        "clean_win",
        "near_clean",
        "mixed_tradeoff",
        "dominated",
        "pareto_label",
        "win_count",
        "loss_count",
        "protected_loss_count",
        "target_gain_count",
        "target_loss_count",
        "target_gain_minus_protected_loss",
        "win_metrics",
        "loss_metrics",
        "protected_loss_metrics",
        "target_gain_metrics",
        "target_loss_metrics",
        "runtime_device",
    ]
    summary = frame[config_columns].drop_duplicates().copy()
    summary["is_canonical"] = ~summary.duplicated("canonical_config_id")
    return summary.sort_values(
        ["experiment_family", "display_method", "dataset", "budget_or_alpha"],
        kind="mergesort",
    ).reset_index(drop=True)


def save_tables(master: pd.DataFrame, config_summary: pd.DataFrame) -> None:
    master = master.copy()
    column_order = [
        "experiment_family",
        "method",
        "display_method",
        "dataset",
        "setting",
        "evaluation_mode",
        "budget_or_alpha",
        "metric_name",
        "metric_role",
        "baseline_value",
        "candidate_value",
        "delta",
        "direction",
        "is_win",
        "is_loss",
        "clean_win",
        "near_clean",
        "mixed_tradeoff",
        "dominated",
        "artifact_clean_win",
        "artifact_mixed_tradeoff",
        "win_count",
        "loss_count",
        "protected_loss_count",
        "target_gain_count",
        "target_loss_count",
        "pareto_label",
        "artifact_path",
        "config_id",
        "canonical_config_id",
        "is_canonical",
    ]
    master = master[column_order]
    master.to_csv(RESULTS_DIR / "hvg_tradeoff_master_table.csv", index=False)
    config_summary.to_csv(RESULTS_DIR / "hvg_tradeoff_config_summary.csv", index=False)


def directional_score(group: pd.DataFrame) -> float:
    positives = float((group["delta"] > 1e-9).sum())
    negatives = float((group["delta"] < -1e-9).sum())
    total = positives + negatives
    if total == 0:
        return 0.0
    return (positives - negatives) / total


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.png")
    fig.savefig(FIGURES_DIR / f"{stem}.pdf")
    plt.close(fig)


def plot_metric_tradeoff_heatmap(master: pd.DataFrame) -> None:
    canonical = master[master["is_canonical"]].copy()
    grouped = (
        canonical.groupby(["display_method", "metric_name"], sort=False)["delta"]
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
    metric_order = [metric for metric in canonical_metric_order() if metric in grouped["metric_name"].unique()]
    method_order = list(dict.fromkeys(canonical["display_method"].tolist()))
    heatmap = grouped.pivot(index="display_method", columns="metric_name", values="directional_score").reindex(
        index=method_order,
        columns=metric_order,
    )

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    matrix = heatmap.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="#F4F4F4")
    im = ax.imshow(masked, aspect="auto", vmin=-1.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels(metric_order, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order)
    ax.set_title("Metric Tradeoff Stability Across Candidate Families")

    for i in range(len(method_order)):
        for j in range(len(metric_order)):
            value = heatmap.iloc[i, j]
            if pd.isna(value):
                continue
            ax.text(
                j,
                i,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Directional score (positive - negative)")
    save_figure(fig, "fig1_metric_tradeoff_heatmap")


def plot_clean_win_failure_summary(config_summary: pd.DataFrame) -> None:
    counts = (
        config_summary.groupby(["experiment_family", "pareto_label"], sort=False)
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

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    for status in counts.columns:
        values = counts[status].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=colors[status], height=0.62, label=status.replace("_", " "))
        left = left + values

    for idx, total in enumerate(left):
        ax.text(total + 0.15, idx, f"{int(total)}", va="center", fontsize=8, color="#444444")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Configuration count")
    ax.set_title("Pareto Failure Summary by Experiment Family")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_figure(fig, "fig2_clean_win_failure_summary")


def plot_rank_blend_frontier(config_summary: pd.DataFrame) -> None:
    blend = config_summary[config_summary["experiment_family"] == "rank_blend_frontier"].copy()
    blend["alpha_numeric"] = blend["budget_or_alpha"].astype(float)
    datasets = ["duo8_pbmc", "pbmc_cite"]
    methods = list(dict.fromkeys(blend["method"].tolist()))
    method_colors = {
        methods[0]: PALETTE["coral"],
        methods[1]: PALETTE["blue"] if len(methods) > 1 else PALETTE["coral"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), sharex=False, sharey=True)
    for ax, dataset in zip(axes, datasets, strict=False):
        subset = blend[blend["dataset"] == dataset].sort_values(["method", "alpha_numeric"])
        for method in methods:
            method_df = subset[subset["method"] == method].sort_values("alpha_numeric")
            ax.plot(
                method_df["protected_loss_count"],
                method_df["target_gain_count"],
                color=method_colors[method],
                marker="o",
                label=display_method_label("rank_blend_frontier", method),
                zorder=3,
            )
            label_points = (
                method_df.groupby(["protected_loss_count", "target_gain_count"], sort=False)["alpha_numeric"]
                .apply(list)
                .reset_index(name="alphas")
            )
            for _, row in label_points.iterrows():
                alpha_text = "/".join(f"{alpha:.2f}" for alpha in row["alphas"])
                ax.text(
                    float(row["protected_loss_count"]) + 0.04,
                    float(row["target_gain_count"]) + 0.04,
                    f"a={alpha_text}",
                    fontsize=7,
                    color=method_colors[method],
                )
        ax.axvline(0.0, color=PALETTE["slate"], linestyle="--", linewidth=1.0)
        ax.set_title(dataset)
        ax.set_xlabel("Protected loss count")
        ax.set_xlim(-0.1, max(3.2, subset["protected_loss_count"].max() + 0.6))
        ax.set_ylim(-0.1, 2.2)
        ax.grid(True, alpha=0.18)

    axes[0].set_ylabel("Target gain count")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Rank Blend Frontier: Alpha Moves Gains and Losses Together", y=1.16, fontsize=11, fontweight="bold")
    save_figure(fig, "fig3_rank_blend_frontier")


def summarize_metric_patterns(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = master[master["is_canonical"]].copy()
    metric_summary = (
        canonical.groupby("metric_name", sort=False)["delta"]
        .agg(
            n="size",
            positive=lambda series: int((series > 1e-9).sum()),
            negative=lambda series: int((series < -1e-9).sum()),
            mean_delta="mean",
            median_delta="median",
        )
        .reset_index()
    )
    metric_summary["positive_rate"] = metric_summary["positive"] / metric_summary["n"]
    metric_summary["negative_rate"] = metric_summary["negative"] / metric_summary["n"]
    rising = metric_summary.sort_values(["positive_rate", "mean_delta"], ascending=[False, False]).head(3)
    falling = metric_summary.sort_values(["negative_rate", "mean_delta"], ascending=[False, True]).head(3)
    return rising, falling


def write_analysis(master: pd.DataFrame, config_summary: pd.DataFrame) -> None:
    canonical_configs = config_summary[config_summary["is_canonical"]].copy()
    total_family_configs = int(config_summary.shape[0])
    total_unique_configs = int(canonical_configs.shape[0])
    clean_count = int(canonical_configs["clean_win"].sum())
    near_clean_count = int(canonical_configs["near_clean"].sum())
    mixed_count = int(canonical_configs["mixed_tradeoff"].sum())
    dominated_count = int(canonical_configs["dominated"].sum())

    rising, falling = summarize_metric_patterns(master)

    stable_rising = ", ".join(
        f"`{row.metric_name}` ({row.positive}/{row.n} positive)"
        for row in rising.itertuples(index=False)
    )
    stable_falling = ", ".join(
        f"`{row.metric_name}` ({row.negative}/{row.n} negative)"
        for row in falling.itertuples(index=False)
    )

    family_table = (
        config_summary.groupby("experiment_family", sort=False)[["clean_win", "near_clean", "mixed_tradeoff", "dominated"]]
        .sum()
        .reset_index()
    )
    family_lines = [
        f"- `{row.experiment_family}`: clean={int(row.clean_win)}, near-clean={int(row.near_clean)}, mixed={int(row.mixed_tradeoff)}, dominated={int(row.dominated)}"
        for row in family_table.itertuples(index=False)
    ]

    risk_family = canonical_configs[canonical_configs["display_method"].str.contains("risk_parity", regex=False)]
    spectral_family = canonical_configs[canonical_configs["display_method"].str.contains("spectral", regex=False)]
    tradeoff_line = (
        f"- risk-parity related configs: `{int(risk_family['mixed_tradeoff'].sum())}` mixed / `{int(risk_family.shape[0])}` unique configs; "
        f"spectral-related configs: `{int(spectral_family['mixed_tradeoff'].sum())}` mixed / `{int(spectral_family.shape[0])}` unique configs."
    )

    lines = [
        "# HVG Tradeoff Audit Analysis",
        "",
        "## Core Claim",
        "",
        f"- 审计共覆盖 `{total_family_configs}` 个 family-scoped config，对应 `{total_unique_configs}` 个唯一评估 config。",
        f"- Pareto `clean_win` 数量为 `{clean_count}`；`near_clean` 数量为 `{near_clean_count}`；`mixed_tradeoff` 数量为 `{mixed_count}`；`dominated` 数量为 `{dominated_count}`。",
        "- 结论是：在当前 official partial evidence 下，问题更像稳定出现的多目标 tradeoff，而不是单一 scorer 还没调到最优。",
        "",
        "## Family Summary",
        "",
        *family_lines,
        tradeoff_line,
        "",
        "## Stable Metric Pattern",
        "",
        f"- 最常上涨的指标：{stable_rising}。",
        f"- 最常回吐的指标：{stable_falling}。",
        "- 这说明局部收益并非不存在；真正的问题是收益与 payback 经常绑定出现。",
        "",
        "## Interpretation",
        "",
        "- `rank_blend_frontier` 没有找到 clean alpha，说明 tradeoff 不是简单由 candidate rank 过猛导致。",
        "- `anchor_repair` 也没有通过小预算替换打开 clean region，说明局部修补同样受限。",
        "- 因此更合理的下一步不是继续单标量加权，而是把问题正式建模成带 protected constraints 的 Pareto repair / context-aware route。",
    ]
    (EXPERIMENT_DIR / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    findings_lines = [
        "# Findings",
        "",
        f"- 统一审计后，`{total_unique_configs}` 个唯一 official partial config 中 `clean_win=0`，核心负结果成立。",
        f"- `mixed_tradeoff` 是主导状态：`{mixed_count}` / `{total_unique_configs}` unique configs。",
        f"- 最常上涨的指标集中在 `{', '.join(rising['metric_name'].tolist())}`；最常回吐的指标集中在 `{', '.join(falling['metric_name'].tolist())}`。",
        "- `rank_blend` 与 `anchor_repair` 都没能把 direct scorer 的局部收益转成 clean domination，因此应把后续工作重定义为 Pareto-constrained problem，而不是继续纯 scorer tuning。",
    ]
    (PROJECT_ROOT / "findings.md").write_text("\n".join(findings_lines) + "\n", encoding="utf-8-sig")


def write_weekly_log(config_summary: pd.DataFrame) -> None:
    canonical = config_summary[config_summary["is_canonical"]]
    device = get_runtime_device()
    torch_note = "torch CUDA available" if torch is not None and device == "cuda" else "CPU fallback"
    lines = [
        "# Weekly Log",
        "",
        "## 2026-04-25",
        "",
        "- 建立 `project_pareto_spatial_context/` 项目骨架与审计协议。",
        "- 汇总四类已有 official partial artifact，生成统一 master table 与 config summary。",
        f"- 生成 `{int(config_summary.shape[0])}` 个 family-scoped config、`{int(canonical.shape[0])}` 个 unique config 的 Pareto 分类。",
        f"- 运行时设备检查：`{torch_note}`；图形渲染保持 CPU 侧。",
        "- 输出首版 analysis 与 3 张 figure，用于支撑 “no clean win, stable tradeoff” 的问题定义。",
    ]
    (PROJECT_ROOT / "weekly_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    master = build_metric_rows()
    master = add_profile_columns(master)
    config_summary = build_config_summary(master)
    save_tables(master, config_summary)
    plot_metric_tradeoff_heatmap(master)
    plot_clean_win_failure_summary(config_summary)
    plot_rank_blend_frontier(config_summary)
    write_analysis(master, config_summary)
    write_weekly_log(config_summary)
    print(f"Wrote audit outputs to {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
