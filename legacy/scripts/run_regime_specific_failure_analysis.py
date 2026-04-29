from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for candidate in (str(SRC), str(SCRIPTS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hvg_research import build_default_method_registry

import run_real_inputs_round1 as rr1
import run_regime_specific_route as route


DEFAULT_OUTPUT_DIR = ROOT / "artifacts_regime_specific_failure_analysis"
BASE_METHOD = route.CANDIDATE_METHOD
ROBUSTNESS_SEEDS = (7, 11, 23, 37, 47)
ROBUSTNESS_TOPKS = (100, 200, 500)
ABLATION_METHODS = (
    BASE_METHOD,
    "real_batch_only_invariant",
    "stronger_atlas_guard_invariant",
    "reduced_deviance_aggression_invariant",
)
TARGET_DATASETS = tuple(route.TARGET_MODELING_DATASETS)
CONTROL_DATASETS = tuple(route.PROTECTED_CONTROL_DATASETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze donor-aware invariant-residual route failure and robustness.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--real-data-root", type=str, default=str(ROOT / "data" / "real_inputs"))
    parser.add_argument("--refine-epochs", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=1)
    return parser.parse_args()


def mean_or_nan(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.mean())


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def load_existing_analysis_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis_summary = pd.read_csv(ROOT / "artifacts_regime_specific_route" / "analysis_gate_summary.csv")
    analysis_dataset = pd.read_csv(ROOT / "artifacts_regime_specific_route" / "analysis_dataset_metrics.csv")
    split_df = pd.read_csv(ROOT / "artifacts_regime_specific_route" / "dataset_split.csv")
    return analysis_summary, analysis_dataset, split_df


def build_analysis_rows(
    *,
    dataset_cache,
    headroom_df: pd.DataFrame,
    candidate_methods: tuple[str, ...],
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
    headroom_lookup = headroom_df.set_index("dataset")
    rows: list[dict[str, object]] = []
    for dataset_name in (*TARGET_DATASETS, *CONTROL_DATASETS):
        dataset = dataset_cache.get(dataset_name, seed)
        current_top_k = min(int(top_k), int(dataset.counts.shape[1]))
        best_single_method = str(headroom_lookup.loc[dataset_name, "best_single_expert"])
        methods_to_compute = tuple(dict.fromkeys([route.ANCHOR_METHOD, best_single_method, *candidate_methods]))
        score_cache: dict[str, np.ndarray] = {}
        topk_cache: dict[str, np.ndarray] = {}
        metadata_cache: dict[str, dict[str, object]] = {}

        for method_name in methods_to_compute:
            method_fn = registry[method_name]
            scores = np.asarray(method_fn(dataset.counts, dataset.batches, current_top_k), dtype=np.float64)
            score_cache[method_name] = scores
            topk_cache[method_name] = route.topk_indices(scores, current_top_k)
            metadata_cache[method_name] = rr1.extract_method_metadata(method_fn)

        anchor_scores = score_cache[route.ANCHOR_METHOD]
        anchor_topk = topk_cache[route.ANCHOR_METHOD]
        best_scores = score_cache[best_single_method]
        best_topk = topk_cache[best_single_method]
        anchor_overlap_to_best = route.jaccard(anchor_topk, best_topk)
        anchor_corr_to_best = route.spearman_correlation(anchor_scores, best_scores)
        group_name = "positive_headroom" if dataset_name in TARGET_DATASETS else "atlas_control"

        for method_name in candidate_methods:
            candidate_scores = score_cache[method_name]
            candidate_topk = topk_cache[method_name]
            row = {
                "seed": int(seed),
                "top_k": int(current_top_k),
                "method": method_name,
                "dataset": dataset_name,
                "group_name": group_name,
                "best_single_method": best_single_method,
                "rank_corr_to_anchor": route.spearman_correlation(candidate_scores, anchor_scores),
                "topk_overlap_to_anchor": route.jaccard(candidate_topk, anchor_topk),
                "topk_shift_vs_anchor": 1.0 - route.jaccard(candidate_topk, anchor_topk),
                "topk_overlap_to_best_single": route.jaccard(candidate_topk, best_topk),
                "delta_overlap_to_best_single_vs_anchor": route.jaccard(candidate_topk, best_topk) - anchor_overlap_to_best,
                "rank_corr_to_best_single": route.spearman_correlation(candidate_scores, best_scores),
                "delta_rank_corr_to_best_single_vs_anchor": route.spearman_correlation(candidate_scores, best_scores) - anchor_corr_to_best,
                "score_dispersion_ratio_vs_anchor": float(np.std(candidate_scores) / max(np.std(anchor_scores), 1e-8)),
            }
            for key, value in metadata_cache[method_name].items():
                row[f"meta_{key}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_analysis_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method_name, seed, top_k), group in df.groupby(["method", "seed", "top_k"], sort=True):
        positive_group = group[group["group_name"] == "positive_headroom"]
        control_group = group[group["group_name"] == "atlas_control"]
        targeted_shift_gap = mean_or_nan(positive_group["topk_shift_vs_anchor"]) - mean_or_nan(control_group["topk_shift_vs_anchor"])
        winner_overlap_pull_positive = mean_or_nan(positive_group["delta_overlap_to_best_single_vs_anchor"])
        control_guard = mean_or_nan(control_group["delta_overlap_to_best_single_vs_anchor"])
        winner_overlap_pull_gap = winner_overlap_pull_positive - control_guard
        winner_corr_pull_positive = mean_or_nan(positive_group["delta_rank_corr_to_best_single_vs_anchor"])
        control_corr_pull = mean_or_nan(control_group["delta_rank_corr_to_best_single_vs_anchor"])
        winner_corr_pull_gap = winner_corr_pull_positive - control_corr_pull
        conditions = {
            "targeted_shift_gap": targeted_shift_gap >= 0.03,
            "winner_overlap_pull_positive": winner_overlap_pull_positive > 0.0,
            "winner_overlap_pull_gap": winner_overlap_pull_gap >= 0.015,
            "winner_corr_pull_gap": winner_corr_pull_gap >= 0.01,
            "control_guard": control_guard >= -0.02,
        }
        row = {
            "method": method_name,
            "seed": int(seed),
            "top_k": int(top_k),
            "targeted_shift_gap": float(targeted_shift_gap),
            "winner_overlap_pull_positive": float(winner_overlap_pull_positive),
            "winner_overlap_pull_gap": float(winner_overlap_pull_gap),
            "winner_corr_pull_gap": float(winner_corr_pull_gap),
            "control_guard": float(control_guard),
            "positive_shift_vs_anchor": mean_or_nan(positive_group["topk_shift_vs_anchor"]),
            "control_shift_vs_anchor": mean_or_nan(control_group["topk_shift_vs_anchor"]),
            "positive_corr_pull_vs_anchor": winner_corr_pull_positive,
            "control_corr_pull_vs_anchor": control_corr_pull,
            "analysis_condition_count": int(sum(bool(value) for value in conditions.values())),
        }
        row["analysis_pass"] = bool(row["analysis_condition_count"] >= 4)
        for name, value in conditions.items():
            row[f"condition_{name}"] = bool(value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["method", "top_k", "seed"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_failure_decomposition(
    *,
    analysis_dataset_df: pd.DataFrame,
    split_df: pd.DataFrame,
    base_metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    df = analysis_dataset_df.copy()
    if not base_metadata_df.empty:
        meta_cols = [column for column in base_metadata_df.columns if column.startswith("meta_")]
        df = df.merge(
            base_metadata_df[["dataset", *meta_cols]],
            on="dataset",
            how="left",
        )

    split_lookup = split_df.set_index("dataset")
    target_count = max(int((df["group_name"] == "positive_headroom").sum()), 1)
    control_count = max(int((df["group_name"] == "atlas_control").sum()), 1)
    control_damage_rank = (
        df[df["group_name"] == "atlas_control"]
        .sort_values("delta_overlap_to_best_single_vs_anchor")
        .assign(control_damage_rank=np.arange(1, int((df["group_name"] == "atlas_control").sum()) + 1))
        [["dataset", "control_damage_rank"]]
    )
    df = df.merge(control_damage_rank, on="dataset", how="left")

    df["targeted_shift_gap_contribution"] = np.where(
        df["group_name"] == "positive_headroom",
        df["topk_shift_vs_anchor"] / target_count,
        -df["topk_shift_vs_anchor"] / control_count,
    )
    df["winner_overlap_pull_positive_contribution"] = np.where(
        df["group_name"] == "positive_headroom",
        df["delta_overlap_to_best_single_vs_anchor"] / target_count,
        0.0,
    )
    df["winner_overlap_pull_gap_contribution"] = np.where(
        df["group_name"] == "positive_headroom",
        df["delta_overlap_to_best_single_vs_anchor"] / target_count,
        -df["delta_overlap_to_best_single_vs_anchor"] / control_count,
    )
    df["winner_corr_pull_gap_contribution"] = np.where(
        df["group_name"] == "positive_headroom",
        df["delta_rank_corr_to_best_single_vs_anchor"] / target_count,
        -df["delta_rank_corr_to_best_single_vs_anchor"] / control_count,
    )
    df["control_guard_contribution"] = np.where(
        df["group_name"] == "atlas_control",
        df["delta_overlap_to_best_single_vs_anchor"] / control_count,
        0.0,
    )

    target_roles: list[str] = []
    for row in df.itertuples(index=False):
        if row.group_name != "positive_headroom":
            target_roles.append("")
            continue
        if row.dataset == "cellxgene_mouse_kidney_aging_10x":
            target_roles.append("transfer_stress_test")
        elif float(row.delta_overlap_to_best_single_vs_anchor) > -0.10 and float(row.delta_rank_corr_to_best_single_vs_anchor) > 0.0:
            target_roles.append("partial_signal")
        else:
            target_roles.append("hard_counterexample")
    df["target_role"] = target_roles

    control_roles: list[str] = []
    for row in df.itertuples(index=False):
        if row.group_name != "atlas_control":
            control_roles.append("")
            continue
        if int(row.control_damage_rank) <= 2:
            control_roles.append("primary_control_damage")
        else:
            control_roles.append("secondary_control_damage")
    df["control_role"] = control_roles

    df["winner_family_mismatch_flag"] = np.where(
        (df["group_name"] == "positive_headroom") & (df["best_single_method"].isin(["variance", "scanpy_seurat_v3_hvg"])),
        1,
        0,
    )
    for dataset_name in df["dataset"].astype(str):
        if dataset_name in split_lookup.index:
            df.loc[df["dataset"] == dataset_name, "route_role"] = str(split_lookup.loc[dataset_name, "role"])
            df.loc[df["dataset"] == dataset_name, "route_rationale"] = str(split_lookup.loc[dataset_name, "rationale"])
    return df.sort_values(["group_name", "dataset"]).reset_index(drop=True)


def render_failure_decomposition_md(
    *,
    failure_df: pd.DataFrame,
    base_summary_row: pd.Series,
) -> str:
    gate_lookup = {
        "targeted_shift_gap": base_summary_row.get("targeted_shift_gap", base_summary_row.get("positive_minus_control_shift")),
        "winner_overlap_pull_positive": base_summary_row.get("winner_overlap_pull_positive", base_summary_row.get("positive_overlap_pull_vs_anchor")),
        "winner_overlap_pull_gap": base_summary_row.get("winner_overlap_pull_gap", base_summary_row.get("overlap_pull_gap")),
        "winner_corr_pull_gap": base_summary_row.get("winner_corr_pull_gap", base_summary_row.get("corr_pull_gap")),
        "control_guard": base_summary_row.get("control_guard", base_summary_row.get("control_overlap_pull_vs_anchor")),
    }
    target_df = failure_df[failure_df["group_name"] == "positive_headroom"].copy()
    control_df = failure_df[failure_df["group_name"] == "atlas_control"].copy()
    target_sorted = target_df.sort_values("delta_overlap_to_best_single_vs_anchor")
    control_sorted = control_df.sort_values("delta_overlap_to_best_single_vs_anchor")

    lines = [
        "# Failure Decomposition",
        "",
        "## Gate-Level Result",
        f"- `targeted_shift_gap` = {fmt_float(gate_lookup['targeted_shift_gap'])}.",
        f"- `winner_overlap_pull_positive` = {fmt_float(gate_lookup['winner_overlap_pull_positive'])}.",
        f"- `winner_overlap_pull_gap` = {fmt_float(gate_lookup['winner_overlap_pull_gap'])}.",
        f"- `winner_corr_pull_gap` = {fmt_float(gate_lookup['winner_corr_pull_gap'])}.",
        f"- `control_guard` = {fmt_float(gate_lookup['control_guard'])}.",
        "",
        "## Target Split: Partial Signal Vs Counterexample",
    ]
    for row in target_sorted.itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: target_role={row.target_role}, best_single={row.best_single_method}, "
            f"overlap_pull={fmt_float(row.delta_overlap_to_best_single_vs_anchor)}, "
            f"corr_pull={fmt_float(row.delta_rank_corr_to_best_single_vs_anchor)}, "
            f"shift={fmt_float(row.topk_shift_vs_anchor)}."
        )
    lines.extend(
        [
            "",
            "## Control Damage Ranking",
        ]
    )
    for row in control_sorted.itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: control_role={row.control_role}, overlap_pull={fmt_float(row.delta_overlap_to_best_single_vs_anchor)}, "
            f"guard_contribution={fmt_float(row.control_guard_contribution)}."
        )
    lines.extend(
        [
            "",
            "## Why Corr Pull Can Be Positive While Overlap Pull Is Negative",
            "- The scorer often becomes more rank-correlated with the winner across the full gene list, but its top-k set still moves away from the winner's top genes.",
            "- In practice this means broad score reordering without recovering the winner-like selected genes that matter for the gate.",
            "- The pattern is clearest on `cellxgene_human_kidney_nonpt` and `E-MTAB-4888`, where corr pull is strongly positive but overlap pull remains negative.",
            "",
            "## Specific Audit",
            "- `cellxgene_mouse_kidney_aging_10x` should be treated as a transfer stress test, not as the core modeling evidence for a donor-aware count-residual route.",
            "- The target split is materially heterogeneous: the three target datasets point to three different winner families (`variance`, `multinomial_deviance_hvg`, `scanpy_seurat_v3_hvg`).",
        ]
    )
    return "\n".join(lines) + "\n"


def render_regime_audit_md(
    *,
    failure_df: pd.DataFrame,
    base_summary_row: pd.Series,
) -> str:
    target_df = failure_df[failure_df["group_name"] == "positive_headroom"].copy()
    control_df = failure_df[failure_df["group_name"] == "atlas_control"].copy()

    def subset_metrics(target_keep: tuple[str, ...]) -> dict[str, float]:
        positive_group = target_df[target_df["dataset"].isin(target_keep)]
        return {
            "targeted_shift_gap": mean_or_nan(positive_group["topk_shift_vs_anchor"]) - mean_or_nan(control_df["topk_shift_vs_anchor"]),
            "winner_overlap_pull_positive": mean_or_nan(positive_group["delta_overlap_to_best_single_vs_anchor"]),
            "winner_overlap_pull_gap": mean_or_nan(positive_group["delta_overlap_to_best_single_vs_anchor"]) - mean_or_nan(control_df["delta_overlap_to_best_single_vs_anchor"]),
        }

    full_metrics = subset_metrics(TARGET_DATASETS)
    drop_mouse_metrics = subset_metrics(("cellxgene_human_kidney_nonpt", "E-MTAB-4888"))

    lines = [
        "# Regime Audit",
        "",
        "## Split Heterogeneity",
        "- The target split is not cleanly one-route evidence because each target dataset points to a different winner family.",
        "- `cellxgene_mouse_kidney_aging_10x` is especially misaligned with the donor-aware story because its winner is `variance`, not a donor-aware residual scorer.",
        "",
        "## Mouse Kidney Audit",
        f"- With the full target split, `targeted_shift_gap` is {fmt_float(full_metrics['targeted_shift_gap'])}.",
        f"- Dropping `cellxgene_mouse_kidney_aging_10x`, `targeted_shift_gap` becomes {fmt_float(drop_mouse_metrics['targeted_shift_gap'])}.",
        f"- Dropping `cellxgene_mouse_kidney_aging_10x`, `winner_overlap_pull_positive` remains {fmt_float(drop_mouse_metrics['winner_overlap_pull_positive'])}.",
        "- This means mouse kidney is currently acting more like a perturbation amplifier than like clean donor-aware evidence.",
        "",
        "## Audit Conclusion",
        "- Yes, the current target split is probably too heterogeneous to serve as clean single-route core evidence.",
        "- But split heterogeneity is not the whole story: even after mentally removing the mouse stress test, positive overlap stays negative and control damage still fails the gate.",
        "- So the current failure is a mix of scope mismatch and implementation aggression, not only a bad split definition.",
    ]
    return "\n".join(lines) + "\n"


def summarize_pass_counts(summary_df: pd.DataFrame, *, methods: tuple[str, ...], top_k: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method_name in methods:
        group = summary_df[(summary_df["method"] == method_name) & (summary_df["top_k"] == top_k)].copy()
        if group.empty:
            continue
        rows.append(
            {
                "method": method_name,
                "top_k": int(top_k),
                "seed_pass_count": int(group["analysis_pass"].sum()),
                "seed_count": int(len(group)),
                "mean_targeted_shift_gap": float(group["targeted_shift_gap"].mean()),
                "mean_winner_overlap_pull_positive": float(group["winner_overlap_pull_positive"].mean()),
                "mean_control_guard": float(group["control_guard"].mean()),
                "all_overlap_positive": bool((group["winner_overlap_pull_positive"] > 0).all()),
                "all_control_guard_safe": bool((group["control_guard"] >= -0.02).all()),
            }
        )
    return pd.DataFrame(rows).sort_values(["seed_pass_count", "method"], ascending=[False, True]).reset_index(drop=True)


def render_next_decision_md(
    *,
    robustness_summary_df: pd.DataFrame,
    ablation_summary_df: pd.DataFrame,
    robustness_pass_counts: pd.DataFrame,
    ablation_pass_counts: pd.DataFrame,
) -> str:
    base_topk200 = robustness_summary_df[(robustness_summary_df["method"] == BASE_METHOD) & (robustness_summary_df["top_k"] == 200)].copy()
    base_pass_count = int(base_topk200["analysis_pass"].sum())
    total_base_settings = int(len(robustness_summary_df[robustness_summary_df["method"] == BASE_METHOD]))
    base_any_pass = bool(robustness_summary_df[robustness_summary_df["method"] == BASE_METHOD]["analysis_pass"].any())

    unlock_candidates = ablation_pass_counts[
        (ablation_pass_counts["seed_pass_count"] >= 3)
        & (ablation_pass_counts["all_overlap_positive"] == True)  # noqa: E712
        & (ablation_pass_counts["all_control_guard_safe"] == True)  # noqa: E712
    ].copy()

    lines = [
        "# Next Decision",
        "",
        "## Robustness Decision",
        f"- Base method top_k=200 pass count: {base_pass_count}/5 seeds.",
        f"- Base method any analysis pass across the 15 robustness settings: {base_any_pass}.",
        f"- Total evaluated base settings: {total_base_settings}.",
    ]
    if not base_any_pass:
        lines.append("- Current no-go is robust: the base candidate does not unlock analysis under any tested seed/top_k setting.")
    else:
        lines.append("- Current no-go is not fully absolute across all settings, but the base candidate is still not stable enough to unlock the route.")

    lines.extend(
        [
            "",
            "## Same-Family Unlock Check",
        ]
    )
    for row in ablation_pass_counts.itertuples(index=False):
        lines.append(
            f"- `{row.method}`: seed_pass_count={row.seed_pass_count}/5, "
            f"mean_overlap_positive={fmt_float(row.mean_winner_overlap_pull_positive)}, "
            f"mean_control_guard={fmt_float(row.mean_control_guard)}."
        )
    if unlock_candidates.empty:
        lines.extend(
            [
                "",
                "## Final Decision",
                "- No same-family variant satisfies the strict unlock rule.",
                "- `adaptive_invariant_residual_hvg` and its current same-family diagnostic ablations should be closed on this route.",
                "- The next move, if any, must be regime redefinition or a different paper claim, not more patching of this line.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Final Decision",
                "- A same-family variant satisfied the strict unlock rule and may proceed to a minimal target+control benchmark in the next round.",
                f"- Unlock candidate(s): {', '.join(unlock_candidates['method'].astype(str).tolist())}.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device_info = rr1.resolve_device_info()
    (output_dir / "compute_context.json").write_text(
        json.dumps(device_info, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    analysis_summary_existing, analysis_dataset_existing, split_df = load_existing_analysis_artifacts()
    headroom_df = route.load_headroom_table()
    resources = route.load_dataset_resources(Path(args.real_data_root))
    dataset_cache = route.pkg.DatasetCache(resources)

    robustness_by_dataset_path = output_dir / "robustness_grid_by_dataset.csv"
    robustness_summary_path = output_dir / "robustness_grid_summary.csv"
    if robustness_by_dataset_path.exists() and robustness_summary_path.exists():
        robustness_by_dataset_df = pd.read_csv(robustness_by_dataset_path)
        robustness_summary_df = pd.read_csv(robustness_summary_path)
    else:
        robustness_rows: list[pd.DataFrame] = []
        robustness_summary_rows: list[pd.DataFrame] = []
        for top_k in ROBUSTNESS_TOPKS:
            for seed in ROBUSTNESS_SEEDS:
                by_dataset = build_analysis_rows(
                    dataset_cache=dataset_cache,
                    headroom_df=headroom_df,
                    candidate_methods=(BASE_METHOD,),
                    seed=seed,
                    top_k=top_k,
                    refine_epochs=args.refine_epochs,
                )
                robustness_rows.append(by_dataset)
                robustness_summary_rows.append(summarize_analysis_rows(by_dataset))

        robustness_by_dataset_df = pd.concat(robustness_rows, ignore_index=True, sort=False)
        robustness_summary_df = pd.concat(robustness_summary_rows, ignore_index=True, sort=False)
        robustness_by_dataset_df.to_csv(robustness_by_dataset_path, index=False)
        robustness_summary_df.to_csv(robustness_summary_path, index=False)

    base_metadata_df = robustness_by_dataset_df[
        (robustness_by_dataset_df["method"] == BASE_METHOD)
        & (robustness_by_dataset_df["seed"] == 7)
        & (robustness_by_dataset_df["top_k"] == 200)
    ].copy()
    failure_df = build_failure_decomposition(
        analysis_dataset_df=analysis_dataset_existing,
        split_df=split_df,
        base_metadata_df=base_metadata_df,
    )
    failure_df.to_csv(output_dir / "failure_decomposition.csv", index=False)
    (output_dir / "failure_decomposition.md").write_text(
        render_failure_decomposition_md(
            failure_df=failure_df,
            base_summary_row=analysis_summary_existing.iloc[0],
        ),
        encoding="utf-8",
    )
    (output_dir / "regime_audit.md").write_text(
        render_regime_audit_md(
            failure_df=failure_df,
            base_summary_row=analysis_summary_existing.iloc[0],
        ),
        encoding="utf-8",
    )

    ablation_by_dataset_path = output_dir / "ablation_gate_by_dataset.csv"
    ablation_summary_path = output_dir / "ablation_gate_summary.csv"
    if ablation_by_dataset_path.exists() and ablation_summary_path.exists():
        ablation_by_dataset_df = pd.read_csv(ablation_by_dataset_path)
        ablation_summary_df = pd.read_csv(ablation_summary_path)
    else:
        ablation_rows: list[pd.DataFrame] = []
        ablation_summary_rows: list[pd.DataFrame] = []
        for seed in ROBUSTNESS_SEEDS:
            by_dataset = build_analysis_rows(
                dataset_cache=dataset_cache,
                headroom_df=headroom_df,
                candidate_methods=ABLATION_METHODS,
                seed=seed,
                top_k=200,
                refine_epochs=args.refine_epochs,
            )
            ablation_rows.append(by_dataset)
            ablation_summary_rows.append(summarize_analysis_rows(by_dataset))

        ablation_by_dataset_df = pd.concat(ablation_rows, ignore_index=True, sort=False)
        ablation_summary_df = pd.concat(ablation_summary_rows, ignore_index=True, sort=False)
        ablation_by_dataset_df.to_csv(ablation_by_dataset_path, index=False)
        ablation_summary_df.to_csv(ablation_summary_path, index=False)

    robustness_pass_counts = summarize_pass_counts(
        robustness_summary_df,
        methods=(BASE_METHOD,),
        top_k=200,
    )
    ablation_pass_counts = summarize_pass_counts(
        ablation_summary_df,
        methods=ABLATION_METHODS,
        top_k=200,
    )
    (output_dir / "next_decision.md").write_text(
        render_next_decision_md(
            robustness_summary_df=robustness_summary_df,
            ablation_summary_df=ablation_summary_df,
            robustness_pass_counts=robustness_pass_counts,
            ablation_pass_counts=ablation_pass_counts,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
