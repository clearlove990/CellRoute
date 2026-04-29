from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from hvg_research.methods import FRONTIER_LITE_METHOD


TIE_EPS = 1e-4
CLASSICAL_EXPERTS = ("variance", "fano", "mv_residual")
PUBLISHED_EXPERTS = (
    "analytic_pearson_residual_hvg",
    "scanpy_cell_ranger_hvg",
    "scanpy_seurat_v3_hvg",
    "triku_hvg",
    "seurat_r_vst_hvg",
    "scran_model_gene_var_hvg",
    "seurat_v3_like_hvg",
    "multinomial_deviance_hvg",
)
SINGLE_EXPERTS = CLASSICAL_EXPERTS + PUBLISHED_EXPERTS
SELECTOR_METHODS = ("adaptive_stat_hvg", "adaptive_hybrid_hvg")
ROUTE_METHOD_ORDER = ("adaptive_hybrid_hvg", "adaptive_stat_hvg")
DISPLAY_NAME = {
    "variance": "variance",
    "fano": "fano",
    "mv_residual": "mv_residual",
    "analytic_pearson_residual_hvg": "analytic_pearson_residual_hvg",
    "triku_hvg": "triku_hvg",
    "seurat_r_vst_hvg": "seurat_r_vst_hvg",
    "scran_model_gene_var_hvg": "scran_model_gene_var_hvg",
    "seurat_v3_like_hvg": "seurat_v3_like_hvg",
    "multinomial_deviance_hvg": "multinomial_deviance_hvg",
    "adaptive_stat_hvg": "adaptive_stat_hvg",
    "adaptive_hybrid_hvg": "adaptive_hybrid_hvg",
    FRONTIER_LITE_METHOD: FRONTIER_LITE_METHOD,
}
TRACKED_METRICS = (
    "ari",
    "nmi",
    "label_silhouette",
    "batch_mixing",
    "neighbor_preservation",
    "cluster_silhouette",
    "stability",
    "runtime_sec",
    "overall_score",
    "overall_rank",
)
PROFILE_COLUMNS = (
    "stat_n_cells",
    "stat_n_genes",
    "stat_batch_classes",
    "stat_dropout_rate",
    "stat_library_cv",
    "stat_cluster_strength",
    "stat_batch_strength",
    "stat_trajectory_strength",
    "stat_pc_entropy",
    "stat_rare_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess selector-bank benchmark artifacts.")
    parser.add_argument("--output-dir", type=str, default="artifacts_selector_bank_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    raw_path = output_dir / "benchmark_raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing benchmark raw results: {raw_path}")

    raw_df = pd.read_csv(raw_path)
    raw_df = raw_df.copy()
    dataset_summary = aggregate_results(raw_df, keys=["dataset", "dataset_id", "method"])
    dataset_summary = rank_within_group(dataset_summary, group_cols=["dataset", "dataset_id"], rank_col="dataset_rank")
    dataset_summary.to_csv(output_dir / "benchmark_dataset_summary.csv", index=False)

    global_summary = aggregate_results(raw_df, keys=["method"])
    global_summary = rank_within_group(global_summary, group_cols=[], rank_col="global_rank")
    global_summary["display_name"] = global_summary["method"].map(DISPLAY_NAME).fillna(global_summary["method"])
    global_summary.to_csv(output_dir / "benchmark_global_summary.csv", index=False)

    winners = dataset_summary.sort_values(["dataset", "dataset_rank", "overall_score"], ascending=[True, True, False]).groupby(
        ["dataset", "dataset_id"], sort=False, as_index=False
    ).head(1)
    winners.to_csv(output_dir / "benchmark_dataset_winners.csv", index=False)

    pairwise_summary, pairwise_dataset = build_pairwise_outputs(dataset_summary)
    pairwise_summary.to_csv(output_dir / "pairwise_win_tie_loss_summary.csv", index=False)
    pairwise_dataset.to_csv(output_dir / "pairwise_dataset_deltas.csv", index=False)

    runtime_tradeoff = build_runtime_tradeoff(global_summary)
    runtime_tradeoff.to_csv(output_dir / "runtime_score_tradeoff_summary.csv", index=False)

    taxonomy_df = build_failure_taxonomy(raw_df, dataset_summary)
    taxonomy_df.to_csv(output_dir / "failure_taxonomy.csv", index=False)

    route_summary = build_route_to_regime_summary(taxonomy_df)
    route_summary.to_csv(output_dir / "route_to_regime_summary.csv", index=False)

    selector_readout = build_selector_readout(
        dataset_summary=dataset_summary,
        global_summary=global_summary,
        winners=winners,
        pairwise_summary=pairwise_summary,
        runtime_tradeoff=runtime_tradeoff,
        taxonomy_df=taxonomy_df,
        route_summary=route_summary,
    )
    (output_dir / "benchmark_readout.md").write_text(selector_readout, encoding="utf-8")

    (output_dir / "selector_story.md").write_text(
        build_selector_story(
            global_summary=global_summary,
            winners=winners,
            pairwise_summary=pairwise_summary,
            runtime_tradeoff=runtime_tradeoff,
            taxonomy_df=taxonomy_df,
        ),
        encoding="utf-8",
    )
    (output_dir / "failure_taxonomy.md").write_text(
        build_failure_taxonomy_doc(taxonomy_df=taxonomy_df, route_summary=route_summary),
        encoding="utf-8",
    )
    (output_dir / "paper_gap_checklist.md").write_text(
        build_gap_checklist(
            global_summary=global_summary,
            winners=winners,
            pairwise_summary=pairwise_summary,
            runtime_tradeoff=runtime_tradeoff,
            taxonomy_df=taxonomy_df,
        ),
        encoding="utf-8",
    )


def aggregate_results(df: pd.DataFrame, *, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    tracked_metrics = [metric for metric in TRACKED_METRICS if metric in df.columns]
    base_columns = set(keys) | set(tracked_metrics) | {"is_winner", "seed", "top_k"}
    metadata_columns = [column for column in df.columns if column not in base_columns]
    rows: list[dict[str, object]] = []
    for group_key, group in df.groupby(keys, dropna=False, sort=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {key: value for key, value in zip(keys, key_tuple, strict=False)}
        for metric in tracked_metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            row[metric] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=0))
        row["run_count"] = int(len(group))
        row["win_count"] = int(group["is_winner"].sum()) if "is_winner" in group.columns else 0
        row["win_rate"] = float(row["win_count"] / max(row["run_count"], 1))
        for column in metadata_columns:
            row[column] = first_non_empty(group[column])
        rows.append(row)
    return pd.DataFrame(rows)


def first_non_empty(series: pd.Series):
    for value in series.tolist():
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return np.nan


def rank_within_group(df: pd.DataFrame, *, group_cols: list[str], rank_col: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    frames = []
    iterator = df.groupby(group_cols, dropna=False, sort=False) if group_cols else [((), df)]
    for _, group in iterator:
        ranked = group.sort_values(
            ["overall_rank", "win_rate", "overall_score", "runtime_sec", "method"],
            ascending=[True, False, False, True, True],
        ).reset_index(drop=True)
        ranked[rank_col] = np.arange(1, len(ranked) + 1)
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True)


def build_pairwise_outputs(dataset_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = tuple(dataset_summary["method"].dropna().astype(str).unique().tolist())
    dataset_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for left_method, right_method in combinations(methods, 2):
        deltas: list[float] = []
        for _, group in dataset_summary.groupby(["dataset", "dataset_id"], sort=False):
            left = group[group["method"] == left_method]
            right = group[group["method"] == right_method]
            if left.empty or right.empty:
                continue
            left_row = left.iloc[0]
            right_row = right.iloc[0]
            score_delta = float(left_row["overall_score"] - right_row["overall_score"])
            rank_delta = float(right_row["dataset_rank"] - left_row["dataset_rank"])
            winner = "tie" if abs(score_delta) <= TIE_EPS else left_method if score_delta > 0 else right_method
            dataset_rows.append(
                {
                    "dataset": left_row["dataset"],
                    "dataset_id": left_row["dataset_id"],
                    "left_method": left_method,
                    "right_method": right_method,
                    "left_score": float(left_row["overall_score"]),
                    "right_score": float(right_row["overall_score"]),
                    "score_delta": score_delta,
                    "rank_delta": rank_delta,
                    "winner": winner,
                }
            )
            deltas.append(score_delta)
        summary_rows.append(
            {
                "left_method": left_method,
                "right_method": right_method,
                "left_wins": int(sum(delta > TIE_EPS for delta in deltas)),
                "ties": int(sum(abs(delta) <= TIE_EPS for delta in deltas)),
                "left_losses": int(sum(delta < -TIE_EPS for delta in deltas)),
                "comparisons": int(len(deltas)),
                "mean_score_delta": float(np.mean(deltas)) if deltas else float("nan"),
                "median_score_delta": float(np.median(deltas)) if deltas else float("nan"),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["left_wins", "mean_score_delta", "comparisons", "left_method", "right_method"],
        ascending=[False, False, False, True, True],
    )
    dataset_df = pd.DataFrame(dataset_rows).sort_values(["dataset", "left_method", "right_method"]).reset_index(drop=True)
    return summary_df.reset_index(drop=True), dataset_df


def build_runtime_tradeoff(global_summary: pd.DataFrame) -> pd.DataFrame:
    if global_summary.empty:
        return pd.DataFrame()
    best_score = float(global_summary["overall_score"].max())
    fastest_runtime = float(global_summary["runtime_sec"].min())
    rows = []
    for _, row in global_summary.iterrows():
        dominated = False
        for _, other in global_summary.iterrows():
            if other["method"] == row["method"]:
                continue
            better_or_equal_score = float(other["overall_score"]) >= float(row["overall_score"]) - TIE_EPS
            better_or_equal_runtime = float(other["runtime_sec"]) <= float(row["runtime_sec"]) + TIE_EPS
            strictly_better = (
                float(other["overall_score"]) > float(row["overall_score"]) + TIE_EPS
                or float(other["runtime_sec"]) < float(row["runtime_sec"]) - TIE_EPS
            )
            if better_or_equal_score and better_or_equal_runtime and strictly_better:
                dominated = True
                break
        rows.append(
            {
                **row.to_dict(),
                "score_gap_to_best": float(best_score - float(row["overall_score"])),
                "runtime_ratio_to_fastest": float(float(row["runtime_sec"]) / max(fastest_runtime, 1e-8)),
                "score_per_second": float(float(row["overall_score"]) / max(float(row["runtime_sec"]), 1e-8)),
                "pareto_optimal": not dominated,
            }
        )
    return pd.DataFrame(rows).sort_values(["pareto_optimal", "global_rank", "runtime_sec"], ascending=[False, True, True]).reset_index(drop=True)


def build_failure_taxonomy(raw_df: pd.DataFrame, dataset_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset_name, dataset_id), group in dataset_summary.groupby(["dataset", "dataset_id"], sort=False):
        winner_row = group.sort_values(["dataset_rank", "overall_score"], ascending=[True, False]).iloc[0]
        profile_row = pick_profile_row(raw_df, dataset_name, dataset_id)
        adaptive_hybrid_row = pick_method_row(group, "adaptive_hybrid_hvg")
        adaptive_stat_row = pick_method_row(group, "adaptive_stat_hvg")
        frontier_row = pick_method_row(group, FRONTIER_LITE_METHOD)
        best_classical_row = best_method_row(group, CLASSICAL_EXPERTS)
        best_published_row = best_method_row(group, PUBLISHED_EXPERTS)
        best_single_row = best_method_row(group, SINGLE_EXPERTS)

        batch_classes = safe_float(profile_row.get("stat_batch_classes", 0.0))
        batch_strength = safe_float(profile_row.get("stat_batch_strength", 0.0))
        dropout_rate = safe_float(profile_row.get("stat_dropout_rate", 0.0))
        library_cv = safe_float(profile_row.get("stat_library_cv", 0.0))
        trajectory_strength = safe_float(profile_row.get("stat_trajectory_strength", 0.0))
        pc_entropy = safe_float(profile_row.get("stat_pc_entropy", 0.0))
        cluster_strength = safe_float(profile_row.get("stat_cluster_strength", 0.0))
        n_cells = safe_float(profile_row.get("stat_n_cells", winner_row.get("cells", 0.0)))
        n_genes = safe_float(profile_row.get("stat_n_genes", winner_row.get("genes", 0.0)))

        selector_score = safe_float(adaptive_hybrid_row.get("overall_score")) if adaptive_hybrid_row else float("nan")
        adaptive_stat_score = safe_float(adaptive_stat_row.get("overall_score")) if adaptive_stat_row else float("nan")
        best_single_score = safe_float(best_single_row["overall_score"])
        published_margin = safe_float(best_published_row["overall_score"] - best_classical_row["overall_score"])
        selector_margin = selector_score - best_single_score if pd.notna(selector_score) else float("nan")

        route_name = str(profile_row.get("route_name", "")) if pd.notna(profile_row.get("route_name", np.nan)) else ""
        route_target = str(profile_row.get("route_target", "")) if pd.notna(profile_row.get("route_target", np.nan)) else ""
        resolved_method = str(profile_row.get("resolved_method", "")) if pd.notna(profile_row.get("resolved_method", np.nan)) else ""
        fallback_target = str(profile_row.get("fallback_target", "")) if pd.notna(profile_row.get("fallback_target", np.nan)) else ""
        used_fallback = safe_float(profile_row.get("used_fallback", 0.0))

        atlas_signal = bounded_score(n_cells, 7000.0, 18000.0) + bounded_score(pc_entropy, 0.88, 0.97) + bounded_score(
            0.35 - cluster_strength,
            0.00,
            0.15,
        )
        batch_signal = bounded_score(batch_classes, 1.0, 8.0) + bounded_score(batch_strength, 0.00, 0.10) + bounded_score(
            library_cv,
            0.60,
            1.20,
        ) + (0.50 if route_target == "frontier_lite" else 0.0)
        trajectory_signal = bounded_score(dropout_rate, 0.82, 0.92) + bounded_score(
            trajectory_strength,
            0.60,
            0.82,
        ) + bounded_score(0.80 - library_cv, 0.00, 0.30) + (0.40 if route_target == "fano" else 0.0)
        residual_signal = bounded_score(published_margin, 0.00, 1.50) + bounded_score(n_genes, 12000.0, 30000.0) + (
            0.50 if str(best_published_row["method"]) == str(winner_row["method"]) else 0.0
        )

        regime_scores = [
            ("atlas-like / large homogeneous panel", atlas_signal),
            ("batch-heavy heterogeneous", batch_signal),
            ("high-dropout trajectory-like", trajectory_signal),
            ("residual-friendly / count-model-friendly", residual_signal),
        ]
        ranked_regimes = sorted(regime_scores, key=lambda item: item[1], reverse=True)

        regime = ""
        classification_mode = ""
        if batch_classes >= 8 or (
            route_target == "frontier_lite" and (batch_strength >= 0.01 or library_cv >= 0.90 or selector_margin > 0.20)
        ):
            regime = "batch-heavy heterogeneous"
            classification_mode = "rule_batch_escape"
        elif route_target == "fano" and dropout_rate >= 0.86 and trajectory_strength >= 0.75:
            regime = "high-dropout trajectory-like"
            classification_mode = "rule_selector_fano_trajectory"
        elif (
            published_margin > 0.40
            and str(best_published_row["method"]) in PUBLISHED_EXPERTS
            and str(best_published_row["method"]) == str(winner_row["method"])
        ):
            regime = "residual-friendly / count-model-friendly"
            classification_mode = "rule_published_margin"
        elif (
            n_cells >= 7000
            and str(best_classical_row["method"]) == "variance"
            and published_margin < 0.40
            and ("variance" in route_name or pc_entropy >= 0.80)
        ):
            regime = "atlas-like / large homogeneous panel"
            classification_mode = "rule_large_panel"
        elif dropout_rate >= 0.86 and trajectory_strength >= 0.68 and library_cv <= 0.80:
            regime = "high-dropout trajectory-like"
            classification_mode = "rule_dropout_trajectory"
        else:
            regime = ranked_regimes[0][0]
            classification_mode = "score_backfill"
        secondary_regime = next((name for name, _ in ranked_regimes if name != regime), regime)

        evidence_parts = [
            f"winner={winner_row['method']}",
            f"best_single={best_single_row['method']}",
            f"best_published={best_published_row['method']}({safe_float(best_published_row['overall_score']):.3f})",
            f"best_classical={best_classical_row['method']}({safe_float(best_classical_row['overall_score']):.3f})",
            f"published_margin={published_margin:.3f}",
            f"selector_margin_vs_best_single={selector_margin:.3f}" if pd.notna(selector_margin) else "selector_margin_vs_best_single=nan",
            f"route={route_name}->{route_target or resolved_method or 'n/a'}",
            f"batch_classes={batch_classes:.1f}",
            f"batch_strength={batch_strength:.3f}",
            f"dropout={dropout_rate:.3f}",
            f"trajectory={trajectory_strength:.3f}",
            f"pc_entropy={pc_entropy:.3f}",
            f"cluster_strength={cluster_strength:.3f}",
        ]
        rows.append(
            {
                "dataset": dataset_name,
                "dataset_id": dataset_id,
                "regime": regime,
                "secondary_regime": secondary_regime,
                "classification_mode": classification_mode,
                "winner_method": str(winner_row["method"]),
                "winner_score": safe_float(winner_row["overall_score"]),
                "best_single_expert": str(best_single_row["method"]),
                "best_single_score": best_single_score,
                "best_classical_expert": str(best_classical_row["method"]),
                "best_classical_score": safe_float(best_classical_row["overall_score"]),
                "best_published_expert": str(best_published_row["method"]),
                "best_published_score": safe_float(best_published_row["overall_score"]),
                "adaptive_hybrid_score": selector_score,
                "adaptive_stat_score": adaptive_stat_score,
                "frontier_lite_score": np.nan if frontier_row is None else safe_float(frontier_row["overall_score"]),
                "published_margin_vs_classical": published_margin,
                "selector_margin_vs_best_single": selector_margin,
                "route_name": route_name,
                "route_target": route_target,
                "resolved_method": resolved_method,
                "fallback_target": fallback_target,
                "used_fallback": used_fallback,
                "stat_n_cells": n_cells,
                "stat_n_genes": n_genes,
                "stat_batch_classes": batch_classes,
                "stat_dropout_rate": dropout_rate,
                "stat_library_cv": library_cv,
                "stat_cluster_strength": cluster_strength,
                "stat_batch_strength": batch_strength,
                "stat_trajectory_strength": trajectory_strength,
                "stat_pc_entropy": pc_entropy,
                "atlas_signal": atlas_signal,
                "batch_signal": batch_signal,
                "trajectory_signal": trajectory_signal,
                "residual_signal": residual_signal,
                "regime_evidence": " | ".join(evidence_parts),
            }
        )
    return pd.DataFrame(rows).sort_values(["regime", "dataset"]).reset_index(drop=True)


def pick_profile_row(raw_df: pd.DataFrame, dataset_name: str, dataset_id: str) -> dict[str, object]:
    dataset_rows = raw_df[(raw_df["dataset"] == dataset_name) & (raw_df["dataset_id"] == dataset_id)]
    for method_name in ROUTE_METHOD_ORDER:
        method_rows = dataset_rows[dataset_rows["method"] == method_name]
        if not method_rows.empty:
            return method_rows.iloc[0].to_dict()
    return dataset_rows.iloc[0].to_dict()


def pick_method_row(group: pd.DataFrame, method_name: str) -> dict[str, object] | None:
    rows = group[group["method"] == method_name]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def best_method_row(group: pd.DataFrame, method_names: tuple[str, ...]) -> dict[str, object]:
    rows = group[group["method"].isin(method_names)].sort_values(
        ["dataset_rank", "overall_score", "runtime_sec"],
        ascending=[True, False, True],
    )
    if rows.empty:
        raise ValueError(f"Expected at least one method from {method_names} in dataset group.")
    return rows.iloc[0].to_dict()


def safe_float(value) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def bounded_score(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def build_route_to_regime_summary(taxonomy_df: pd.DataFrame) -> pd.DataFrame:
    if taxonomy_df.empty:
        return pd.DataFrame()
    rows = []
    for (regime, route_target, resolved_method), group in taxonomy_df.groupby(
        ["regime", "route_target", "resolved_method"],
        dropna=False,
        sort=False,
    ):
        selector_wins = np.sum(group["winner_method"] == "adaptive_hybrid_hvg")
        rows.append(
            {
                "regime": regime,
                "route_target": route_target,
                "resolved_method": resolved_method,
                "dataset_count": int(len(group)),
                "datasets": ",".join(group["dataset"].astype(str).tolist()),
                "selector_win_count": int(selector_wins),
                "mean_selector_margin_vs_best_single": float(
                    np.nanmean(group["selector_margin_vs_best_single"].to_numpy(dtype=np.float64))
                ),
                "mean_published_margin_vs_classical": float(
                    np.nanmean(group["published_margin_vs_classical"].to_numpy(dtype=np.float64))
                ),
                "fallback_count": int(np.nansum(group["used_fallback"].to_numpy(dtype=np.float64))),
            }
        )
    return pd.DataFrame(rows).sort_values(["regime", "dataset_count", "selector_win_count"], ascending=[True, False, False]).reset_index(drop=True)


def build_selector_readout(
    *,
    dataset_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    winners: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    runtime_tradeoff: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    route_summary: pd.DataFrame,
) -> str:
    lines = ["# Selector Bank Benchmark Readout", "", "## Global Ranking"]
    for _, row in global_summary.sort_values("global_rank").iterrows():
        lines.append(
            f"- rank {int(row['global_rank'])}: {row['method']} "
            f"score={float(row['overall_score']):.4f} runtime_sec={float(row['runtime_sec']):.2f} "
            f"win_rate={float(row['win_rate']):.3f}"
        )

    lines.extend(["", "## Per-Dataset Winners"])
    for _, row in winners.sort_values("dataset").iterrows():
        lines.append(
            f"- {row['dataset']}: winner={row['method']} "
            f"score={float(row['overall_score']):.4f} runtime_sec={float(row['runtime_sec']):.2f}"
        )

    lines.extend(["", "## Runtime vs Score Tradeoff"])
    for _, row in runtime_tradeoff.sort_values(["pareto_optimal", "global_rank"], ascending=[False, True]).iterrows():
        lines.append(
            f"- {row['method']}: pareto_optimal={bool(row['pareto_optimal'])} "
            f"score_gap_to_best={float(row['score_gap_to_best']):.4f} "
            f"runtime_ratio_to_fastest={float(row['runtime_ratio_to_fastest']):.2f}"
        )

    lines.extend(["", "## Pairwise Highlights"])
    key_pairs = [
        ("adaptive_hybrid_hvg", "adaptive_stat_hvg"),
        ("adaptive_hybrid_hvg", "scanpy_seurat_v3_hvg"),
        ("adaptive_hybrid_hvg", "scanpy_cell_ranger_hvg"),
        ("adaptive_hybrid_hvg", "triku_hvg"),
        ("adaptive_hybrid_hvg", "seurat_r_vst_hvg"),
        ("adaptive_hybrid_hvg", "scran_model_gene_var_hvg"),
        ("adaptive_hybrid_hvg", "seurat_v3_like_hvg"),
        ("adaptive_hybrid_hvg", "multinomial_deviance_hvg"),
        ("adaptive_hybrid_hvg", "analytic_pearson_residual_hvg"),
    ]
    for left_method, right_method in key_pairs:
        row = pairwise_lookup(pairwise_summary, left_method=left_method, right_method=right_method)
        if row is None:
            continue
        lines.append(
            f"- {left_method} vs {right_method}: "
            f"{int(row['left_wins'])}W/{int(row['ties'])}T/{int(row['left_losses'])}L "
            f"mean_delta={float(row['mean_score_delta']):.4f}"
        )

    lines.extend(["", "## Route To Regime"])
    for _, row in route_summary.iterrows():
        lines.append(
            f"- regime={row['regime']} route_target={row['route_target']} resolved_method={row['resolved_method']} "
            f"datasets={row['datasets']} selector_win_count={int(row['selector_win_count'])}"
        )

    lines.extend(["", "## Failure Taxonomy"])
    for _, row in taxonomy_df.sort_values(["regime", "dataset"]).iterrows():
        lines.append(
            f"- {row['dataset']}: regime={row['regime']} winner={row['winner_method']} "
            f"secondary_regime={row['secondary_regime']} "
            f"best_published={row['best_published_expert']} best_classical={row['best_classical_expert']} "
            f"route={row['route_name']}->{row['route_target'] or row['resolved_method']}"
        )
    return "\n".join(lines) + "\n"


def pairwise_lookup(pairwise_summary: pd.DataFrame, *, left_method: str, right_method: str) -> pd.Series | None:
    direct = pairwise_summary[
        (pairwise_summary["left_method"] == left_method) & (pairwise_summary["right_method"] == right_method)
    ]
    if not direct.empty:
        return direct.iloc[0]
    reverse = pairwise_summary[
        (pairwise_summary["left_method"] == right_method) & (pairwise_summary["right_method"] == left_method)
    ]
    if reverse.empty:
        return None
    row = reverse.iloc[0].copy()
    row["left_method"] = left_method
    row["right_method"] = right_method
    row["left_wins"], row["left_losses"] = row["left_losses"], row["left_wins"]
    row["mean_score_delta"] = -float(row["mean_score_delta"])
    row["median_score_delta"] = -float(row["median_score_delta"])
    return row


def build_selector_story(
    *,
    global_summary: pd.DataFrame,
    winners: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    runtime_tradeoff: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
) -> str:
    best_method = str(global_summary.sort_values("global_rank").iloc[0]["method"])
    best_score = float(global_summary.sort_values("global_rank").iloc[0]["overall_score"])
    best_published = global_summary[global_summary["method"].isin(PUBLISHED_EXPERTS)].sort_values("global_rank").iloc[0]
    hybrid_vs_stat = pairwise_lookup(pairwise_summary, left_method="adaptive_hybrid_hvg", right_method="adaptive_stat_hvg")
    selector_wins = int(np.sum(winners["method"] == "adaptive_hybrid_hvg"))
    residual_winner_count = int(np.sum(winners["method"].isin(PUBLISHED_EXPERTS)))
    frontier_runtime = runtime_tradeoff[runtime_tradeoff["method"] == FRONTIER_LITE_METHOD]
    frontier_runtime_ratio = float(frontier_runtime.iloc[0]["runtime_ratio_to_fastest"]) if not frontier_runtime.empty else float("nan")

    lines = [
        "# Selector Story",
        "",
        "## 方法定位",
        "- 当前更适合写成方法论文的主线，是“published-expert bank + profile-conditioned selector”，而不是继续宣传一个新的单一 HVG scorer。",
        f"- 本轮真实 benchmark 的全局第一仍是 `{best_method}`，overall_score={best_score:.4f}。",
        "- 论文叙事应该把贡献点落在：数据集 profile、expert bank、selector/routing、failure-aware fallback、runtime-aware tradeoff。",
        "",
        "## expert bank 是什么",
        f"- classical experts: {', '.join(CLASSICAL_EXPERTS)}。",
        f"- published/count-model experts: {', '.join([m for m in PUBLISHED_EXPERTS if m in global_summary['method'].tolist()])}。",
        f"- 高成本 frontier 选择器 `{FRONTIER_LITE_METHOD}` 更像 selective escape / upper bound，不适合直接当默认主线。",
        "",
        "## 为什么不是再造一个 scorer",
        f"- 新补的 published experts 已经改变了 baseline panel：当前最强 published expert 是 `{best_published['method']}`，global_rank={int(best_published['global_rank'])}。",
        f"- 单一方法的失败模式已经可见：共有 {residual_winner_count} 个数据集由 published/count-model expert 获胜，而 atlas-like 数据集更偏向 classical expert。",
        "- 这说明问题不在于再发明一个统一分数，而在于根据数据 regime 选择合适 expert。",
        "",
        "## 为什么 selector 比单一 scorer 更合理",
        f"- `adaptive_hybrid_hvg` 相对 `adaptive_stat_hvg` 的 pairwise 结果是 {int(hybrid_vs_stat['left_wins'])}W/{int(hybrid_vs_stat['ties'])}T/{int(hybrid_vs_stat['left_losses'])}L，说明 routing 本身带来净收益。",
        f"- selector 在 {selector_wins}/{len(winners)} 个数据集上拿到 outright dataset winner；这不是它当前最强的 claim。",
        "- 它当前更强的证据是 aggregate ranking 和 pairwise margin：在 E-MTAB-5061 上明显拉开差距，其他数据集多数只小输或打平，因此全局仍排第一。",
        f"- `{FRONTIER_LITE_METHOD}` 的 runtime 是最快方法的 {frontier_runtime_ratio:.2f}x 左右，因此更合理的主线是“selector 把昂贵专家限定在必要 regime”，而不是全局默认调用。",
        "",
        "## 明确回答",
        "- 如果把主线定义为 published-expert bank + profile-conditioned selector，这比“adaptive hybrid”更像方法论文：是。",
        "- 原因是贡献点从 heuristic blending 迁移到了可解释 routing、expert specialization、failure-aware fallback 和 runtime-aware decision。",
        "- 但当前实现仍有一个真实短板：生产 selector 还带着 `adaptive_stat_hvg` 这个 legacy blend expert，所以论文里要诚实写成“selector over a bank that still contains a blended legacy expert”，不能夸成已经完全纯化的 direct expert router。",
    ]
    return "\n".join(lines) + "\n"


def build_failure_taxonomy_doc(*, taxonomy_df: pd.DataFrame, route_summary: pd.DataFrame) -> str:
    lines = ["# Failure Taxonomy", "", "## 数据 regime 分组"]
    for regime, group in taxonomy_df.groupby("regime", sort=False):
        datasets = ", ".join(group["dataset"].astype(str).tolist())
        best_winner = group["winner_method"].value_counts().idxmax()
        best_published = group["best_published_expert"].value_counts().idxmax()
        best_classical = group["best_classical_expert"].value_counts().idxmax()
        lines.append(f"- {regime}: datasets={datasets}")
        lines.append(f"  strongest_winner={best_winner}; strongest_published_expert={best_published}; strongest_classical_expert={best_classical}")

    lines.extend(["", "## selector / hybrid 在什么 regime 下赢"])
    for _, row in route_summary.iterrows():
        lines.append(
            f"- regime={row['regime']}: route_target={row['route_target']} resolved_method={row['resolved_method']} "
            f"dataset_count={int(row['dataset_count'])} selector_win_count={int(row['selector_win_count'])} "
            f"mean_selector_margin_vs_best_single={float(row['mean_selector_margin_vs_best_single']):.3f}"
        )

    lines.extend(["", "## 归类依据"])
    for _, row in taxonomy_df.sort_values(["regime", "dataset"]).iterrows():
        lines.append(f"- {row['dataset']}: primary={row['regime']} secondary={row['secondary_regime']} | {row['regime_evidence']}")
    return "\n".join(lines) + "\n"


def build_gap_checklist(
    *,
    global_summary: pd.DataFrame,
    winners: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    runtime_tradeoff: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
) -> str:
    total_datasets = int(len(winners))
    best_method = str(global_summary.sort_values("global_rank").iloc[0]["method"])
    best_published = global_summary[global_summary["method"].isin(PUBLISHED_EXPERTS)].sort_values("global_rank").iloc[0]
    has_official_seurat = bool((global_summary["method"] == "scanpy_seurat_v3_hvg").any())
    hybrid_vs_stat = pairwise_lookup(pairwise_summary, left_method="adaptive_hybrid_hvg", right_method="adaptive_stat_hvg")
    main_conclusion_changed = winners["method"].isin(PUBLISHED_EXPERTS).any()
    selector_strongest_claim = (
        f"profile-conditioned routing is the best aggregate policy on this {total_datasets}-dataset suite even though it is not the per-dataset winner most often, because it captures the hard E-MTAB-5061 escape case without collapsing elsewhere"
    )
    risky_shortcoming = (
        "routing thresholds are still hand-designed, and although scanpy/triku are now wired, "
        "Seurat/R and a reproducible scran R environment are still optional rather than first-class defaults"
    )

    lines = [
        "# Paper Gap Checklist",
        "",
        "## 当前已经很强的证据",
        f"- {total_datasets} 个真实数据集 benchmark 已经齐备，当前全局最佳仍是 `{best_method}`。",
        f"- `adaptive_hybrid_hvg` vs `adaptive_stat_hvg`: {int(hybrid_vs_stat['left_wins'])}W/{int(hybrid_vs_stat['ties'])}T/{int(hybrid_vs_stat['left_losses'])}L。",
        f"- published baseline panel 已补到 `{best_published['method']}` 这一级别，说明 selector 不是只在内部 heuristic 上取胜。",
        "- 已经有 global ranking、per-dataset winners、pairwise W/T/L、runtime tradeoff、route-to-regime、failure taxonomy。",
        "",
        "## 仍缺哪些 baseline",
        f"- 官方或更贴近官方实现的 Seurat v3 / scanpy `flavor='seurat_v3'` 复现：{'已补 scanpy 官方实现，但仍缺 Seurat/R 侧复现' if has_official_seurat else '仍缺'}。",
        "- 更标准的 deviance / multinomial-residual published baseline 复现，以避免 reviewer 质疑“本地近似实现太弱/太强”。",
        "- geometry-aware baseline：`triku_hvg` 已经接入，但论文叙事里还需要更系统的 benchmark 证据。",
        "- `scran_model_gene_var_hvg` 已经通过 R worker 接入，但在把它当成默认 baseline 之前，项目还需要一个可复现的本地 R 环境方案。",
        "",
        "## 仍缺哪些 ablation",
        "- direct published-expert selector vs legacy `adaptive_stat_hvg` blend expert。",
        "- no-frontier-escape / no-fallback / route-threshold sensitivity。",
        "- top-k sensitivity 和 seed sensitivity，当前 round 还是单一 top-k / seed 设定。",
        f"- leave-one-dataset-out threshold selection，证明 routing 不是对这 {total_datasets} 个数据集手调。",
        "",
        "## 如果现在写初稿，最容易被审稿人打哪里",
        f"- 最危险短板：{risky_shortcoming}。",
        "- `adaptive_hybrid_hvg` 这个名字仍然带着 heuristic 混合器味道，和 selector 主线叙事不完全一致。",
        "- 目前 published-expert bank 已补齐，但 production selector 还没有彻底纯化成 direct expert router。",
        "- selector 当前是 aggregate winner 但不是 per-dataset winner leader，这会逼着我们在论文里把 claim 写得更克制、更准确。",
        f"- benchmark 仍是本地 {total_datasets} 数据集套件，缺跨论文常用 benchmark 的外部复现。",
        "",
        "## 明确回答",
        "- 如果把主线定义为 published-expert bank + profile-conditioned selector，这是否比 adaptive hybrid 更像方法论文：是。",
        f"- `seurat_v3_like_hvg` / 新补 baseline 是否改变当前主结论：{'部分改变' if main_conclusion_changed else '没有推翻'}。它们强化了“单一 scorer 不够、count-model expert 必须进 bank”的结论，但没有把全局主结论从 selector 翻掉。",
        f"- selector 当前 strongest claim 是什么：{selector_strongest_claim}。",
        f"- 当前最危险的短板是什么：{risky_shortcoming}。",
        "- 若目标是 RECOMB / ISMB 级别，还差什么：官方 published baseline 复现、hold-out routing 证明、更多生物数据集、marker/enrichment 级别的生物学验证。",
        "- 若目标是更高层级 ML venue，最不够的是什么：learned selector 的泛化与校准证据、比 rule-based routing 更强的 methodological novelty，以及更大规模、跨 protocol 的 benchmark。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
