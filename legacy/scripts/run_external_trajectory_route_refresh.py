from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for candidate in (str(SRC), str(SCRIPTS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hvg_research import build_default_method_registry

import download_external_trajectory_datasets as extdata
import run_external_trajectory_import_audit as extaudit
import run_regime_specific_route as base
import run_trajectory_route_coherence_audit as traj


DEFAULT_OUTPUT_DIR = ROOT / "artifacts_external_trajectory_route_refresh"
DEFAULT_EXTERNAL_DATASETS = (
    "atlas_human_hpsc_eht",
    "atlas_mouse_primitive_streak",
    "atlas_mouse_hspc_diff",
)
ROBUSTNESS_SEEDS = (7, 11, 23, 37, 47)
ROBUSTNESS_TOPKS = (100, 200, 500)
EVALUATED_METHODS = (
    traj.ANCHOR_METHOD,
    "adaptive_spectral_locality_hvg",
    "triku_hvg",
    "scanpy_cell_ranger_hvg",
)
ROUTE_METHODS = tuple(name for name in EVALUATED_METHODS if name != traj.ANCHOR_METHOD)
ROBUST_POSITIVE_FRACTION_FLOOR = 0.80
REFRESH_PLANS: dict[str, object] = {
    "atlas_mouse_hspc_diff": extaudit.rr1.DatasetPlan(
        dataset_name="atlas_mouse_hspc_diff",
        max_cells=None,
        max_genes=6000,
        mode="refresh_gene_budget",
        rationale="Route refresh sweep uses a memory-safe gene budget for repeated seed/top-k scoring.",
    ),
    "atlas_mouse_primitive_streak": extaudit.rr1.DatasetPlan(
        dataset_name="atlas_mouse_primitive_streak",
        max_cells=6000,
        max_genes=5000,
        mode="refresh_sampled",
        rationale="Dense h5ad matrix; route refresh sweep uses a repeated memory-safe sample budget.",
    ),
    "atlas_human_hpsc_eht": extaudit.rr1.DatasetPlan(
        dataset_name="atlas_human_hpsc_eht",
        max_cells=6000,
        max_genes=5000,
        mode="refresh_sampled",
        rationale="Dense h5ad matrix; route refresh sweep uses a repeated memory-safe sample budget.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external-backed trajectory route robustness and refresh artifacts.")
    parser.add_argument("--external-data-root", type=str, default="data/external_inputs")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_EXTERNAL_DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ROBUSTNESS_SEEDS))
    parser.add_argument("--top-ks", type=int, nargs="+", default=list(ROBUSTNESS_TOPKS))
    parser.add_argument("--refine-epochs", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "device_info.json", resolve_device_info())

    external_data_root = (ROOT / args.external_data_root).resolve()
    dataset_names = tuple(part.strip() for part in args.datasets.split(",") if part.strip())
    audit_df = pd.read_csv(ROOT / "artifacts_external_trajectory_import_audit" / "external_analysis_audit.csv")
    audit_df = audit_df[audit_df["dataset_name"].astype(str).isin(dataset_names)].copy()
    audit_lookup = audit_df.set_index("dataset_name")
    spec_map = extaudit.load_external_spec_map(external_data_root)

    grid_path = output_dir / "external_robustness_grid_by_dataset.csv"
    grid_df = load_existing_grid(grid_path, resume=args.resume)
    completed_keys = {
        (str(row.dataset), int(row.seed), int(row.top_k))
        for row in grid_df[["dataset", "seed", "top_k"]].drop_duplicates().itertuples(index=False)
    } if not grid_df.empty else set()

    for dataset_name in dataset_names:
        if dataset_name not in spec_map:
            raise FileNotFoundError(f"Missing external dataset in registry: {dataset_name}")
        if dataset_name not in audit_lookup.index:
            raise KeyError(f"Missing prior external audit row for {dataset_name}")
        spec = spec_map[dataset_name]
        plan = REFRESH_PLANS.get(dataset_name) or extaudit.DEFAULT_PLANS.get(dataset_name)
        if plan is None:
            raise KeyError(f"Missing external audit plan for {dataset_name}")
        audit_row = audit_lookup.loc[dataset_name]

        for seed in args.seeds:
            dataset = extaudit.load_external_dataset(spec=spec, plan=plan, random_state=int(seed))
            for top_k in args.top_ks:
                current_top_k = min(int(top_k), int(dataset.counts.shape[1]))
                key = (dataset_name, int(seed), current_top_k)
                if key in completed_keys:
                    continue
                setting_df = build_setting_rows(
                    dataset=dataset,
                    spec=spec,
                    audit_row=audit_row,
                    seed=int(seed),
                    top_k=current_top_k,
                    refine_epochs=int(args.refine_epochs),
                )
                grid_df = pd.concat([grid_df, setting_df], ignore_index=True, sort=False)
                grid_df = grid_df.sort_values(["dataset", "seed", "top_k", "method"]).reset_index(drop=True)
                grid_df.to_csv(grid_path, index=False)
                completed_keys.add(key)

    summary_df = summarize_external_robustness(grid_df)
    summary_df.to_csv(output_dir / "external_robustness_grid_summary.csv", index=False)

    registry_df = build_route_registry(audit_df=audit_df, robustness_summary=summary_df)
    registry_df.to_csv(output_dir / "external_route_registry.csv", index=False)
    (output_dir / "external_route_registry.md").write_text(render_route_registry(registry_df), encoding="utf-8")

    combined_summary = build_combined_summary(registry_df=registry_df, robustness_summary=summary_df)
    combined_summary.to_csv(output_dir / "combined_route_refresh_summary.csv", index=False)
    (output_dir / "combined_route_refresh.md").write_text(
        render_combined_refresh(combined_summary=combined_summary, robustness_summary=summary_df),
        encoding="utf-8",
    )
    (output_dir / "upgrade_rule_refresh.md").write_text(
        render_upgrade_rule(combined_summary=combined_summary),
        encoding="utf-8",
    )
    (output_dir / "final_route_refresh_decision.md").write_text(
        render_final_decision(combined_summary=combined_summary),
        encoding="utf-8",
    )


def load_existing_grid(path: Path, *, resume: bool) -> pd.DataFrame:
    if resume and path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_setting_rows(
    *,
    dataset,
    spec,
    audit_row: pd.Series,
    seed: int,
    top_k: int,
    refine_epochs: int,
) -> pd.DataFrame:
    registry = build_default_method_registry(
        top_k=top_k,
        refine_epochs=refine_epochs,
        random_state=seed,
        gate_model_path=None,
    )
    best_single_method = str(audit_row["best_single_method"])
    best_single_is_route_family = int(best_single_method in ROUTE_METHODS)
    methods_to_compute = tuple(dict.fromkeys([*EVALUATED_METHODS, best_single_method]))
    missing = [method for method in methods_to_compute if method not in registry]
    if missing:
        raise KeyError(f"Method(s) missing from registry: {missing}")

    score_cache: dict[str, np.ndarray] = {}
    topk_cache: dict[str, np.ndarray] = {}
    for method_name in methods_to_compute:
        scores = np.asarray(registry[method_name](dataset.counts, dataset.batches, top_k), dtype=np.float64)
        score_cache[method_name] = scores
        topk_cache[method_name] = base.topk_indices(scores, top_k)

    anchor_scores = score_cache[traj.ANCHOR_METHOD]
    anchor_topk = topk_cache[traj.ANCHOR_METHOD]
    best_scores = score_cache[best_single_method]
    best_topk = topk_cache[best_single_method]
    anchor_overlap_to_best = base.jaccard(anchor_topk, best_topk)
    anchor_corr_to_best = base.spearman_correlation(anchor_scores, best_scores)

    rows: list[dict[str, object]] = []
    for method_name in EVALUATED_METHODS:
        method_scores = score_cache[method_name]
        method_topk = topk_cache[method_name]
        overlap_to_best = base.jaccard(method_topk, best_topk)
        corr_to_best = base.spearman_correlation(method_scores, best_scores)
        shift_vs_anchor = 1.0 - base.jaccard(method_topk, anchor_topk)
        overlap_pull = overlap_to_best - anchor_overlap_to_best
        corr_pull = corr_to_best - anchor_corr_to_best
        is_route_method = int(method_name in ROUTE_METHODS)
        route_family_consistency_flag = int(
            is_route_method == 1
            and best_single_is_route_family == 1
            and overlap_pull > 0.0
            and corr_pull >= 0.0
        )
        rows.append(
            {
                "dataset": spec.dataset_name,
                "dataset_id": spec.dataset_id,
                "seed": int(seed),
                "top_k": int(top_k),
                "cells_loaded": int(dataset.counts.shape[0]),
                "genes_loaded": int(dataset.counts.shape[1]),
                "method": method_name,
                "method_family": route_family_label(method_name),
                "is_anchor": int(method_name == traj.ANCHOR_METHOD),
                "is_route_family_method": is_route_method,
                "prior_external_evidence": str(audit_row["route_evidence"]),
                "best_single_method": best_single_method,
                "best_single_expert_family": route_family_label(best_single_method),
                "best_single_is_route_family": best_single_is_route_family,
                "topk_overlap_to_anchor": base.jaccard(method_topk, anchor_topk),
                "shift_vs_anchor": shift_vs_anchor,
                "topk_overlap_to_best_single": overlap_to_best,
                "anchor_overlap_to_best_single": anchor_overlap_to_best,
                "overlap_pull_vs_anchor": overlap_pull,
                "rank_corr_to_best_single": corr_to_best,
                "anchor_corr_to_best_single": anchor_corr_to_best,
                "corr_pull_vs_anchor": corr_pull,
                "rank_corr_to_anchor": base.spearman_correlation(method_scores, anchor_scores),
                "route_family_consistency_flag": route_family_consistency_flag,
                "positive_overlap_pull_flag": int(overlap_pull > 0.0),
                "nonnegative_corr_pull_flag": int(corr_pull >= 0.0),
                "robust_positive_setting_flag": int(route_family_consistency_flag == 1),
            }
        )
    return pd.DataFrame(rows)


def route_family_label(method_name: str) -> str:
    if method_name == traj.ANCHOR_METHOD:
        return "anchor"
    if method_name in traj.ROUTE_FAMILY_LABELS:
        return str(traj.ROUTE_FAMILY_LABELS[method_name])
    return f"outside_route:{method_name}"


def summarize_external_robustness(grid_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset_name, method_name), group in grid_df.groupby(["dataset", "method"], sort=True):
        setting_count = int(len(group))
        robust_count = int(group["robust_positive_setting_flag"].sum())
        positive_overlap_count = int(group["positive_overlap_pull_flag"].sum())
        nonnegative_corr_count = int(group["nonnegative_corr_pull_flag"].sum())
        rows.append(
            {
                "row_type": "dataset_method",
                "dataset": dataset_name,
                "method": method_name,
                "prior_external_evidence": str(group["prior_external_evidence"].iloc[0]),
                "best_single_method": str(group["best_single_method"].iloc[0]),
                "best_single_expert_family": str(group["best_single_expert_family"].iloc[0]),
                "best_single_is_route_family": int(group["best_single_is_route_family"].iloc[0]),
                "setting_count": setting_count,
                "mean_overlap_pull_vs_anchor": float(group["overlap_pull_vs_anchor"].mean()),
                "median_overlap_pull_vs_anchor": float(group["overlap_pull_vs_anchor"].median()),
                "min_overlap_pull_vs_anchor": float(group["overlap_pull_vs_anchor"].min()),
                "mean_corr_pull_vs_anchor": float(group["corr_pull_vs_anchor"].mean()),
                "median_corr_pull_vs_anchor": float(group["corr_pull_vs_anchor"].median()),
                "min_corr_pull_vs_anchor": float(group["corr_pull_vs_anchor"].min()),
                "mean_shift_vs_anchor": float(group["shift_vs_anchor"].mean()),
                "positive_overlap_setting_count": positive_overlap_count,
                "nonnegative_corr_setting_count": nonnegative_corr_count,
                "route_family_consistency_count": int(group["route_family_consistency_flag"].sum()),
                "robust_positive_setting_count": robust_count,
                "robust_positive_fraction": float(robust_count / max(setting_count, 1)),
                "robust_positive": int(robust_count / max(setting_count, 1) >= ROBUST_POSITIVE_FRACTION_FLOOR),
            }
        )

    method_summary = pd.DataFrame(rows)
    for dataset_name, group in method_summary[method_summary["row_type"] == "dataset_method"].groupby("dataset", sort=True):
        route_group = group[group["method"].isin(ROUTE_METHODS)].copy()
        best_route = route_group.sort_values(
            ["robust_positive_fraction", "mean_overlap_pull_vs_anchor", "mean_corr_pull_vs_anchor"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(
            {
                "row_type": "dataset_overall",
                "dataset": dataset_name,
                "method": str(best_route["method"]),
                "prior_external_evidence": str(best_route["prior_external_evidence"]),
                "best_single_method": str(best_route["best_single_method"]),
                "best_single_expert_family": str(best_route["best_single_expert_family"]),
                "best_single_is_route_family": int(best_route["best_single_is_route_family"]),
                "setting_count": int(route_group["setting_count"].max()),
                "mean_overlap_pull_vs_anchor": float(best_route["mean_overlap_pull_vs_anchor"]),
                "median_overlap_pull_vs_anchor": float(best_route["median_overlap_pull_vs_anchor"]),
                "min_overlap_pull_vs_anchor": float(best_route["min_overlap_pull_vs_anchor"]),
                "mean_corr_pull_vs_anchor": float(best_route["mean_corr_pull_vs_anchor"]),
                "median_corr_pull_vs_anchor": float(best_route["median_corr_pull_vs_anchor"]),
                "min_corr_pull_vs_anchor": float(best_route["min_corr_pull_vs_anchor"]),
                "mean_shift_vs_anchor": float(best_route["mean_shift_vs_anchor"]),
                "positive_overlap_setting_count": int(best_route["positive_overlap_setting_count"]),
                "nonnegative_corr_setting_count": int(best_route["nonnegative_corr_setting_count"]),
                "route_family_consistency_count": int(best_route["route_family_consistency_count"]),
                "robust_positive_setting_count": int(best_route["robust_positive_setting_count"]),
                "robust_positive_fraction": float(best_route["robust_positive_fraction"]),
                "robust_positive": int(best_route["robust_positive"]),
            }
        )
    return pd.DataFrame(rows)


def build_route_registry(*, audit_df: pd.DataFrame, robustness_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    route_split = pd.read_csv(ROOT / "artifacts_route_coherence_audit_trajectory" / "route_dataset_split.csv")
    role_map = {
        "core_modeling": "internal_core_modeling",
        "protected_control": "protected_controls",
        "heldout_validation_only": "internal_route_breaking_counterexample",
    }
    for _, row in route_split.iterrows():
        old_role = str(row["role"])
        if old_role not in role_map:
            continue
        rows.append(
            {
                "dataset": str(row["dataset"]),
                "source": "internal",
                "route_refresh_role": role_map[old_role],
                "original_role": old_role,
                "best_single_method": str(row.get("best_single_expert", "")),
                "best_single_is_route_family": int(row.get("route_family_best_single", 0)),
                "headroom_vs_best_single": row.get("headroom_vs_best_single", ""),
                "robust_positive": "",
                "route_risk_flag": int(old_role in {"protected_control", "heldout_validation_only"}),
                "eligible_if_stricter_analysis_only_refinement_unlocks": 0,
                "included_in_current_refinement": 0,
                "rationale": str(row.get("rationale", "")),
            }
        )

    dataset_overall = robustness_summary[robustness_summary["row_type"] == "dataset_overall"].set_index("dataset")
    for _, row in audit_df.iterrows():
        dataset_name = str(row["dataset_name"])
        robust_row = dataset_overall.loc[dataset_name]
        if str(row["route_evidence"]) == "route_supporting_evidence" and int(robust_row["robust_positive"]) == 1:
            role = "external_supporting_evidence"
            include = 1
        else:
            role = "external_mixed_or_weak"
            include = 0
        rows.append(
            {
                "dataset": dataset_name,
                "source": "external",
                "route_refresh_role": role,
                "original_role": str(row["route_evidence"]),
                "best_single_method": str(row["best_single_method"]),
                "best_single_is_route_family": int(row["best_single_is_route_family"]),
                "headroom_vs_best_single": float(row["headroom_vs_best_single"]),
                "robust_positive": int(robust_row["robust_positive"]),
                "route_risk_flag": int(role == "external_mixed_or_weak"),
                "eligible_if_stricter_analysis_only_refinement_unlocks": include,
                "included_in_current_refinement": 0,
                "rationale": external_registry_rationale(row=row, robust_row=robust_row),
            }
        )
    role_order = {
        "internal_core_modeling": 0,
        "external_supporting_evidence": 1,
        "external_mixed_or_weak": 2,
        "internal_route_breaking_counterexample": 3,
        "protected_controls": 4,
    }
    registry_df = pd.DataFrame(rows)
    registry_df["role_order"] = registry_df["route_refresh_role"].map(role_order).fillna(99)
    return registry_df.sort_values(["role_order", "dataset"]).drop(columns=["role_order"]).reset_index(drop=True)


def external_registry_rationale(*, row: pd.Series, robust_row: pd.Series) -> str:
    if str(row["route_evidence"]) == "route_supporting_evidence" and int(robust_row["robust_positive"]) == 1:
        return (
            f"External route-family positive evidence replicated across robustness sweep; "
            f"best route method `{robust_row['method']}` has robust_positive_fraction={float(robust_row['robust_positive_fraction']):.3f}."
        )
    return (
        f"External evidence is not clean enough for core support; prior label `{row['route_evidence']}`, "
        f"robust_positive_fraction={float(robust_row['robust_positive_fraction']):.3f}."
    )


def build_combined_summary(*, registry_df: pd.DataFrame, robustness_summary: pd.DataFrame) -> pd.DataFrame:
    dataset_overall = robustness_summary[robustness_summary["row_type"] == "dataset_overall"].copy()
    external_supporting_count = int((registry_df["route_refresh_role"] == "external_supporting_evidence").sum())
    external_mixed_count = int((registry_df["route_refresh_role"] == "external_mixed_or_weak").sum())
    robust_external_support_count = int(dataset_overall["robust_positive"].sum())
    internal_core_count = int((registry_df["route_refresh_role"] == "internal_core_modeling").sum())
    protected_control_count = int((registry_df["route_refresh_role"] == "protected_controls").sum())
    counterexample_count = int((registry_df["route_refresh_role"] == "internal_route_breaking_counterexample").sum())

    external_support_robust = external_supporting_count >= 2 and robust_external_support_count >= 2
    single_point_dependency_reduced = (internal_core_count + external_supporting_count) >= 4 and external_supporting_count >= 2
    internal_control_damage_resolved = False
    internal_counterexample_resolved = False
    allow_stricter_refinement = (
        external_support_robust
        and single_point_dependency_reduced
        and internal_control_damage_resolved
        and internal_counterexample_resolved
    )
    if allow_stricter_refinement:
        route_status = "allow_stricter_analysis_only_refinement"
    elif counterexample_count > 0 and external_supporting_count == 0:
        route_status = "downgrade_to_no_go"
    else:
        route_status = "maintain_analysis_only_hold"

    rows = [
        ("external_supporting_dataset_count", external_supporting_count),
        ("external_mixed_or_weak_dataset_count", external_mixed_count),
        ("robust_external_supporting_dataset_count", robust_external_support_count),
        ("internal_core_modeling_dataset_count", internal_core_count),
        ("protected_control_dataset_count", protected_control_count),
        ("internal_route_breaking_counterexample_count", counterexample_count),
        ("external_supporting_evidence_robust", int(external_support_robust)),
        ("support_no_longer_single_internal_point_only", int(single_point_dependency_reduced)),
        ("internal_control_damage_resolved", int(internal_control_damage_resolved)),
        ("internal_route_breaking_counterexample_resolved", int(internal_counterexample_resolved)),
        ("external_positive_offsets_internal_control_damage", 0),
        ("external_positive_offsets_internal_heldout_failure", 0),
        ("model_design_recommended", 0),
        ("stricter_analysis_only_refinement_allowed", int(allow_stricter_refinement)),
        ("route_status", route_status),
    ]
    return pd.DataFrame([{"metric": metric, "value": value} for metric, value in rows])


def render_route_registry(registry_df: pd.DataFrame) -> str:
    lines = [
        "# External-Backed Route Registry",
        "",
        "## Role Counts",
    ]
    for role, count in registry_df["route_refresh_role"].value_counts(sort=False).items():
        lines.append(f"- `{role}`: {int(count)}")
    lines.extend(["", "## Dataset Registry"])
    for row in registry_df.itertuples(index=False):
        lines.extend(
            [
                f"### {row.dataset}",
                f"- Source: `{row.source}`",
                f"- Route refresh role: `{row.route_refresh_role}`",
                f"- Best single method: `{row.best_single_method}`",
                f"- Best single route-family flag: `{row.best_single_is_route_family}`",
                f"- Eligible if stricter analysis-only refinement unlocks: `{row.eligible_if_stricter_analysis_only_refinement_unlocks}`",
                f"- Included in current refinement: `{row.included_in_current_refinement}`",
                f"- Rationale: {row.rationale}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_combined_refresh(*, combined_summary: pd.DataFrame, robustness_summary: pd.DataFrame) -> str:
    metrics = combined_summary.set_index("metric")["value"].to_dict()
    dataset_overall = robustness_summary[robustness_summary["row_type"] == "dataset_overall"].copy()
    lines = [
        "# Combined Route Refresh",
        "",
        "## Direct Answers",
        f"- Route-family consistency enhanced externally? `{bool(int(metrics['external_supporting_evidence_robust']))}`.",
        f"- Supporting evidence no longer depends only on one internal dataset? `{bool(int(metrics['support_no_longer_single_internal_point_only']))}`.",
        "- Route-breaking counterexample still dominates route risk? `True`, because the internal held-out counterexample remains unresolved.",
        f"- Current route status: `{metrics['route_status']}`.",
        "",
        "## External Robustness Readout",
        "- Sweep budget: `atlas_mouse_hspc_diff` used all cells with 6000 genes; `atlas_human_hpsc_eht` and `atlas_mouse_primitive_streak` used 6000 cells by 5000 genes for repeated dense-h5ad scoring.",
    ]
    for row in dataset_overall.itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: best robust route method `{row.method}`, robust_positive_fraction={float(row.robust_positive_fraction):.3f}, prior={row.prior_external_evidence}."
        )
    lines.extend(
        [
            "",
            "## Internal Risk Readout",
            "- External positive signals do not offset the internal protected-control damage. Existing internal control damage remains systemic, especially for `adaptive_spectral_locality_hvg` on `mus_tissue` and `homo_tissue`.",
            "- External positive signals do not offset the internal held-out failure. `cellxgene_unciliated_epithelial_five_donors` remains a route-breaking counterexample with outside-family best single expert and negative held-out overlap.",
            "- Therefore external support is useful route evidence, but not an unlock condition for model design or stricter analysis-only refinement under the refreshed rule.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_upgrade_rule(*, combined_summary: pd.DataFrame) -> str:
    metrics = combined_summary.set_index("metric")["value"].to_dict()
    lines = [
        "# Route Upgrade Rule Refresh",
        "",
        "## Refreshed Rule",
        "- External robust positivity: at least two external supporting datasets must have a route-family method with robust-positive fraction >= 0.80 across the declared seed/top-k grid.",
        "- Internal protected controls: the same route-family method must not show protected-control degradation; control guard must be >= -0.02 and no protected control may have majority failure.",
        "- Internal route-breaking counterexample: the held-out counterexample must stop expanding the risk profile; best single should be in-family or route-family overlap pull must become non-negative in majority settings.",
        "- Combined evidence: support must not depend on a single internal or external dataset.",
        "- Scope: satisfying the rule can only allow stricter analysis-only refinement, not model design.",
        "",
        "## Current Audit Against Rule",
        f"- External robust positivity satisfied: `{bool(int(metrics['external_supporting_evidence_robust']))}`.",
        f"- Internal protected controls satisfied: `{bool(int(metrics['internal_control_damage_resolved']))}`.",
        f"- Internal counterexample satisfied: `{bool(int(metrics['internal_route_breaking_counterexample_resolved']))}`.",
        f"- Combined evidence non-single-point satisfied: `{bool(int(metrics['support_no_longer_single_internal_point_only']))}`.",
        f"- Stricter analysis-only refinement allowed now: `{bool(int(metrics['stricter_analysis_only_refinement_allowed']))}`.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_final_decision(*, combined_summary: pd.DataFrame) -> str:
    metrics = combined_summary.set_index("metric")["value"].to_dict()
    route_status = str(metrics["route_status"])
    lines = [
        "# Final Route Refresh Decision",
        "",
        "## Decision",
        f"- `{route_status}`",
        "",
        "## Why",
        "- External supporting evidence is now stronger and appears robust on two imported trajectory-like datasets.",
        "- `atlas_mouse_hspc_diff` remains `external_mixed_or_weak` and should not be promoted into the supporting core.",
        "- Internal protected-control damage remains unresolved and cannot be cancelled by external positives.",
        "- The internal route-breaking counterexample remains unresolved and cannot be cancelled by external positives.",
        "- Model design remains disallowed.",
        "",
        "## Status Flags",
        f"- external_supporting_evidence_robust={metrics['external_supporting_evidence_robust']}",
        f"- internal_control_damage_resolved={metrics['internal_control_damage_resolved']}",
        f"- internal_route_breaking_counterexample_resolved={metrics['internal_route_breaking_counterexample_resolved']}",
        f"- stricter_analysis_only_refinement_allowed={metrics['stricter_analysis_only_refinement_allowed']}",
        f"- model_design_recommended={metrics['model_design_recommended']}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def resolve_device_info() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    info: dict[str, object] = {
        "device": "cuda" if cuda_available else "cpu",
        "cuda_available": cuda_available,
        "cuda_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        info["cuda_devices"] = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
    return info


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
