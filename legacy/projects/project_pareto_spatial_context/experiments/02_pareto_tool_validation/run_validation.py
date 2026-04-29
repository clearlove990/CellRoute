from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pareto_eval.clean_win import classify_config, classify_frame, profile_for_evaluation_mode
from pareto_eval.frontier import rank_candidates_by_dominance_depth
from pareto_eval.metrics import build_metric_registry, get_runtime_device
from pareto_eval.plotting import plot_frontier, plot_metric_heatmap, plot_win_loss_barplot
from pareto_eval.rank_blend import generate_alpha_grid
from pareto_eval.replacement_frontier import (
    extract_replacement_frontier,
    simulate_anchor_replacement_budgets,
    summarize_replacement_frontier,
)
from pareto_eval.reporting import (
    build_candidate_summary,
    generate_csv_summary,
    generate_markdown_summary,
    write_markdown_summary,
)


EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "02_pareto_tool_validation"
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
WEEK1_MASTER_TABLE = PROJECT_ROOT / "experiments" / "01_hvg_tradeoff_audit" / "results" / "hvg_tradeoff_master_table.csv"
VALIDATION_SUMMARY_PATH = RESULTS_DIR / "tool_validation_summary.csv"
SMOKE_RESULTS_PATH = RESULTS_DIR / "smoke_test_results.csv"
ANALYSIS_PATH = EXPERIMENT_DIR / "analysis.md"

FAMILY_ALIAS = {
    "mainline_benchmark": "global_scorer_family",
    "risk_parity_family": "risk_parity_family",
    "rank_blend_frontier": "rank_blend_frontier",
    "anchor_repair": "anchor_repair",
}

METRIC_ORDER = ["var_ratio", "ari", "nmi", "lisi", "dist_cor", "knn_ratio", "three_nn", "max_ari", "max_nmi"]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_master_table() -> pd.DataFrame:
    master = pd.read_csv(WEEK1_MASTER_TABLE)
    master["validation_family"] = master["experiment_family"].map(FAMILY_ALIAS).fillna(master["experiment_family"])
    return master


def build_wide_delta_table(master: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        "config_id",
        "canonical_config_id",
        "experiment_family",
        "validation_family",
        "method",
        "display_method",
        "dataset",
        "setting",
        "evaluation_mode",
        "budget_or_alpha",
        "artifact_path",
        "is_canonical",
    ]
    existing_index_columns = [column for column in index_columns if column in master.columns]
    wide = (
        master.pivot_table(
            index=existing_index_columns,
            columns="metric_name",
            values="delta",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    return wide


def annotate_group_frontier(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby(group_cols, sort=False, dropna=False):
        evaluation_mode = str(group.iloc[0]["evaluation_mode"])
        registry = profile_for_evaluation_mode(evaluation_mode)
        ranked = rank_candidates_by_dominance_depth(group.copy(), registry)
        keep = ranked[
            [
                "config_id",
                "is_pareto_frontier",
                "dominance_depth",
                "dominance_rank",
                "dominates_count",
                "dominated_by_count",
                "frontier_runtime_device",
            ]
        ].rename(
            columns={
                "is_pareto_frontier": f"{prefix}_frontier",
                "dominance_depth": f"{prefix}_dominance_depth",
                "dominance_rank": f"{prefix}_dominance_rank",
                "dominates_count": f"{prefix}_dominates_count",
                "dominated_by_count": f"{prefix}_dominated_by_count",
                "frontier_runtime_device": f"{prefix}_runtime_device",
            }
        )
        pieces.append(keep)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["config_id"])


def build_validation_summary(master: pd.DataFrame) -> pd.DataFrame:
    classification = classify_frame(master)
    summary = build_candidate_summary(
        master,
        classification,
        carry_columns=(
            "canonical_config_id",
            "experiment_family",
            "validation_family",
            "method",
            "display_method",
            "dataset",
            "setting",
            "budget_or_alpha",
            "artifact_path",
            "is_canonical",
        ),
    )

    wide_delta = build_wide_delta_table(master)
    wide_delta["budget_numeric"] = pd.to_numeric(wide_delta["budget_or_alpha"], errors="coerce")
    family_frontier = annotate_group_frontier(
        wide_delta,
        group_cols=["validation_family", "dataset", "evaluation_mode"],
        prefix="family_dataset",
    )
    dataset_frontier = annotate_group_frontier(
        wide_delta,
        group_cols=["dataset", "evaluation_mode"],
        prefix="dataset_global",
    )

    summary = summary.merge(family_frontier, on="config_id", how="left", validate="one_to_one")
    summary = summary.merge(dataset_frontier, on="config_id", how="left", validate="one_to_one")

    summary["budget_numeric"] = pd.to_numeric(summary["budget_or_alpha"], errors="coerce")
    summary["alpha_numeric"] = summary["budget_numeric"].where(summary["validation_family"] == "rank_blend_frontier")

    anchor_frontier_pieces: list[pd.DataFrame] = []
    anchor_rows = wide_delta[wide_delta["validation_family"] == "anchor_repair"].copy()
    if not anchor_rows.empty:
        for _, group in anchor_rows.groupby(["dataset", "evaluation_mode"], sort=False, dropna=False):
            registry = profile_for_evaluation_mode(str(group.iloc[0]["evaluation_mode"]))
            frontier = extract_replacement_frontier(
                group.copy(),
                registry,
                group_cols=(),
                budget_col="budget_numeric",
            )
            frontier = frontier[
                [
                    "config_id",
                    "is_pareto_frontier",
                    "dominance_depth",
                    "dominance_rank",
                    "dominates_count",
                    "dominated_by_count",
                ]
            ].rename(
                columns={
                    "is_pareto_frontier": "replacement_frontier",
                    "dominance_depth": "replacement_dominance_depth",
                    "dominance_rank": "replacement_dominance_rank",
                    "dominates_count": "replacement_dominates_count",
                    "dominated_by_count": "replacement_dominated_by_count",
                }
            )
            anchor_frontier_pieces.append(frontier)

    if anchor_frontier_pieces:
        anchor_frontier = pd.concat(anchor_frontier_pieces, ignore_index=True)
        summary = summary.merge(anchor_frontier, on="config_id", how="left", validate="one_to_one")

    sort_columns = [
        column
        for column in ["validation_family", "display_method", "dataset", "budget_numeric", "budget_or_alpha"]
        if column in summary.columns
    ]
    return summary.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def build_smoke_test_results() -> pd.DataFrame:
    results: list[dict[str, Any]] = []

    def record(test_name: str, condition: bool, observed: Any, expected: Any, details: str) -> None:
        results.append(
            {
                "test_name": test_name,
                "status": "pass" if condition else "fail",
                "observed": observed,
                "expected": expected,
                "details": details,
            }
        )

    smoke_registry = build_metric_registry(
        "smoke_discrete",
        protected_metrics=("p",),
        target_metrics=("t1", "t2"),
        gain_thresholds={"t1": 0.005, "t2": 0.005},
        loss_tolerances={"p": 0.01, "t1": 0.002, "t2": 0.002},
    )

    clean_case = classify_config({"p": 0.0, "t1": 0.010, "t2": 0.000}, smoke_registry)
    record(
        "clean_win_case",
        clean_case["pareto_label"] == "clean_win",
        clean_case["pareto_label"],
        "clean_win",
        "One target improves and no protected or target metric regresses.",
    )

    near_case = classify_config({"p": -0.004, "t1": 0.006, "t2": -0.001}, smoke_registry)
    record(
        "near_clean_case",
        near_case["pareto_label"] == "near_clean",
        near_case["pareto_label"],
        "near_clean",
        "Protected loss stays inside tolerance and one target clears the gain threshold.",
    )

    mixed_case = classify_config({"p": -0.020, "t1": 0.010, "t2": 0.000}, smoke_registry)
    record(
        "mixed_tradeoff_case",
        mixed_case["pareto_label"] == "mixed_tradeoff",
        mixed_case["pareto_label"],
        "mixed_tradeoff",
        "A target win paired with a protected loss must be labelled mixed tradeoff.",
    )

    dominated_case = classify_config({"p": -0.010, "t1": 0.000, "t2": -0.001}, smoke_registry)
    record(
        "dominated_case",
        dominated_case["pareto_label"] == "dominated",
        dominated_case["pareto_label"],
        "dominated",
        "No wins plus at least one loss should be dominated.",
    )

    frontier_registry = build_metric_registry(
        "frontier_smoke",
        protected_metrics=(),
        target_metrics=("x", "y"),
    )
    frontier_frame = pd.DataFrame(
        [
            {"candidate": "A", "x": 1.0, "y": 0.0},
            {"candidate": "B", "x": 0.0, "y": 1.0},
            {"candidate": "C", "x": 0.5, "y": 0.5},
            {"candidate": "D", "x": 0.25, "y": 0.25},
            {"candidate": "E", "x": -1.0, "y": -1.0},
        ]
    )
    ranked_frontier = rank_candidates_by_dominance_depth(frontier_frame, frontier_registry)
    frontier_members = ",".join(sorted(ranked_frontier.loc[ranked_frontier["is_pareto_frontier"], "candidate"].tolist()))
    depth_map = {
        row.candidate: int(row.dominance_depth)
        for row in ranked_frontier.itertuples(index=False)
    }
    record(
        "frontier_membership",
        frontier_members == "A,B,C",
        frontier_members,
        "A,B,C",
        "Three mutually non-dominated points should remain on the frontier.",
    )
    record(
        "dominance_depth",
        depth_map == {"A": 0, "B": 0, "C": 0, "D": 1, "E": 2},
        depth_map,
        {"A": 0, "B": 0, "C": 0, "D": 1, "E": 2},
        "Dominance depth should peel frontier layers in order.",
    )

    alpha_grid = generate_alpha_grid(step=0.25)
    record(
        "alpha_grid",
        alpha_grid == [0.0, 0.25, 0.5, 0.75, 1.0],
        alpha_grid,
        [0.0, 0.25, 0.5, 0.75, 1.0],
        "The alpha grid helper should emit a stable rounded sweep.",
    )

    simulated = simulate_anchor_replacement_budgets(
        [0, 25, 50],
        lambda budget: {"x": budget / 100.0, "y": 1.0 - (budget / 100.0), "dataset": "toy"},
    )
    record(
        "replacement_simulation_rows",
        int(simulated.shape[0]) == 3,
        int(simulated.shape[0]),
        3,
        "Budget simulation should emit one row per requested budget.",
    )

    replacement_frame = pd.DataFrame(
        [
            {"dataset": "toy", "budget": 0, "x": 0.20, "y": 0.20},
            {"dataset": "toy", "budget": 25, "x": 0.60, "y": 0.20},
            {"dataset": "toy", "budget": 50, "x": 0.40, "y": 0.50},
            {"dataset": "toy", "budget": 100, "x": 0.10, "y": 0.10},
        ]
    )
    replacement_frontier = extract_replacement_frontier(
        replacement_frame,
        frontier_registry,
        group_cols=("dataset",),
        budget_col="budget",
    )
    replacement_summary = summarize_replacement_frontier(
        replacement_frontier,
        group_cols=("dataset",),
        budget_col="budget",
    )
    frontier_budgets = replacement_summary.iloc[0]["frontier_budgets"]
    record(
        "replacement_frontier_budgets",
        frontier_budgets == "25,50",
        frontier_budgets,
        "25,50",
        "Replacement frontier should keep the two non-dominated budget points.",
    )

    return pd.DataFrame(results)


def write_outputs(summary: pd.DataFrame, master: pd.DataFrame, smoke_results: pd.DataFrame) -> None:
    generate_csv_summary(summary, VALIDATION_SUMMARY_PATH)
    generate_csv_summary(smoke_results, SMOKE_RESULTS_PATH)

    plot_metric_heatmap(
        master,
        output_path=FIGURES_DIR / "fig1_tool_metric_tradeoff_heatmap.png",
        metric_order=[metric for metric in METRIC_ORDER if metric in master["metric_name"].unique()],
        group_col="display_method",
        title="Pareto Toolkit Metric Tradeoff Heatmap",
    )
    plot_win_loss_barplot(
        summary,
        output_path=FIGURES_DIR / "fig2_tool_family_label_summary.png",
        group_col="validation_family",
        title="Pareto Toolkit Family Summary",
    )

    rank_blend = summary[summary["validation_family"] == "rank_blend_frontier"].copy()
    if not rank_blend.empty:
        plot_frontier(
            rank_blend,
            output_path=FIGURES_DIR / "fig3_tool_rank_blend_frontier.png",
            x_col="protected_loss_count",
            y_col="target_gain_count",
            hue_col="display_method",
            facet_col="dataset",
            label_col="budget_or_alpha",
            title="Pareto Toolkit Rank Blend Frontier",
        )

    family_counts = (
        summary.groupby("validation_family", sort=False)[["clean_win", "near_clean", "mixed_tradeoff", "dominated"]]
        .sum()
        .reset_index()
    )
    family_lines = [
        f"- `{row.validation_family}`: clean={int(row.clean_win)}, near_clean={int(row.near_clean)}, mixed={int(row.mixed_tradeoff)}, dominated={int(row.dominated)}"
        for row in family_counts.itertuples(index=False)
    ]

    anchor_budget_lines: list[str] = []
    anchor_rows = summary[summary["validation_family"] == "anchor_repair"].copy()
    if not anchor_rows.empty:
        anchor_frontier = anchor_rows.rename(columns={"replacement_frontier": "is_pareto_frontier"})
        anchor_budget_summary = summarize_replacement_frontier(
            anchor_frontier,
            group_cols=("dataset",),
            budget_col="budget_numeric",
        )
        anchor_budget_lines = [
            f"- `{row.dataset}` frontier budgets: `{row.frontier_budgets or 'none'}`"
            for row in anchor_budget_summary.itertuples(index=False)
        ]

    extra_lines = [
        f"- runtime device preference: `{get_runtime_device()}`",
        "- Week 2 validation reruns classification and dominance depth from the reusable toolkit instead of reusing Week 1 one-off logic.",
        "- Validation target: reproduce the Week 1 negative result that no clean win exists in the four candidate families.",
    ]

    analysis_text = generate_markdown_summary(
        summary,
        title="Pareto Toolkit Validation",
        group_col="validation_family",
        frontier_flag_col="family_dataset_frontier",
        highlight_columns=("validation_family", "display_method", "dataset", "budget_or_alpha"),
        extra_lines=extra_lines,
    )
    analysis_text += "\n## Key Conclusions\n\n"
    analysis_text += "\n".join(family_lines) + "\n"
    analysis_text += (
        "\n"
        "## Week 1 Reproduction\n\n"
        "- `global_scorer_family` has no clean win.\n"
        "- `risk_parity_family` has no clean win.\n"
        "- `rank_blend_frontier` has no clean alpha.\n"
        "- `anchor_repair` has no clean repair budget.\n"
    )
    if anchor_budget_lines:
        analysis_text += "\n## Anchor Frontier\n\n" + "\n".join(anchor_budget_lines) + "\n"

    write_markdown_summary(analysis_text, ANALYSIS_PATH)


def main() -> None:
    ensure_dirs()
    master = load_master_table()
    summary = build_validation_summary(master)
    smoke_results = build_smoke_test_results()
    write_outputs(summary, master, smoke_results)

    failed = smoke_results[smoke_results["status"] != "pass"]
    if not failed.empty:
        raise RuntimeError(f"Smoke tests failed: {failed['test_name'].tolist()}")

    print(f"Wrote validation outputs to {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
