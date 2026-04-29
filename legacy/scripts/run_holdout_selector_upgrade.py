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

from hvg_research import (
    build_default_method_registry,
    choose_torch_device,
    discover_scrna_input_specs,
    evaluate_real_selection,
    load_scrna_dataset,
)
from hvg_research.eval import timed_call
from hvg_research.holdout_selector import RiskControlledSelectorConfig, RiskControlledSelectorPolicy
from hvg_research.methods import HOLDOUT_RISK_SELECTOR_METHOD

import build_topconf_selector_round2_package as pkg
import run_real_inputs_round1 as rr1


DEFAULT_SOURCE_DIR = ROOT / "artifacts_topconf_selector_round2"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts_codex_selector_mvp"
DEFAULT_TOPK_GRID = (100, 200, 500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the codex holdout-selector MVP upgrade.")
    parser.add_argument("--source-dir", type=str, default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--real-data-root", type=str, default="data/real_inputs")
    parser.add_argument("--gate-model-path", type=str, default=rr1.DEFAULT_GATE_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-k-grid", type=str, default="100,200,500")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--refine-epochs", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required_paths = {
        "benchmark_readout": source_dir / "benchmark_readout.md",
        "benchmark_dataset_summary": source_dir / "benchmark_dataset_summary.csv",
        "benchmark_raw_results": source_dir / "benchmark_raw_results.csv",
        "selector_holdout_summary": source_dir / "selector_holdout_summary.csv",
        "official_baseline_audit": source_dir / "official_baseline_audit.csv",
        "dataset_manifest": source_dir / "dataset_manifest.csv",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required round2 artifacts: {missing}")

    readout_md = required_paths["benchmark_readout"].read_text(encoding="utf-8")
    dataset_summary = pd.read_csv(required_paths["benchmark_dataset_summary"])
    raw_df = pd.read_csv(required_paths["benchmark_raw_results"])
    source_holdout_summary = pd.read_csv(required_paths["selector_holdout_summary"])
    official_audit = pd.read_csv(required_paths["official_baseline_audit"])
    manifest_df = pd.read_csv(required_paths["dataset_manifest"])

    profile_df = build_profile_df(raw_df)
    overall_df = dataset_summary.pivot(index="dataset", columns="method", values="overall_score").sort_index()
    runtime_df = dataset_summary.pivot(index="dataset", columns="method", values="runtime_sec").sort_index()

    device_info = resolve_device_info()
    save_json(output_dir / "device_info.json", device_info)

    write_diagnosis_docs(
        output_dir=output_dir,
        readout_md=readout_md,
        source_holdout_summary=source_holdout_summary,
        official_audit=official_audit,
    )

    audit_df = build_official_vs_fallback_audit(official_audit=official_audit)
    audit_df.to_csv(output_dir / "official_vs_fallback_audit.csv", index=False)

    policy_config = RiskControlledSelectorConfig()
    biology_observed_df, biology_summary = compute_or_load_biology_proxy(
        output_dir=output_dir,
        real_data_root=(ROOT / args.real_data_root).resolve(),
        top_k=int(args.top_k),
        gate_model_path=args.gate_model_path,
        refine_epochs=int(args.refine_epochs),
        seed=int(args.seed),
        methods=tuple(
            sorted(
                {
                    "adaptive_hybrid_hvg",
                    *RiskControlledSelectorConfig().candidate_methods,
                }
            )
        ),
    )
    biology_summary.to_csv(output_dir / "biology_proxy_summary.csv", index=False)
    biology_fit_df = build_biology_model_df(
        observed_biology_df=biology_observed_df,
        overall_df=overall_df,
        safe_anchor=policy_config.safe_anchor,
    )

    holdout_results, holdout_summary = run_leave_one_dataset_out(
        feature_df=profile_df,
        overall_df=overall_df,
        biology_df=biology_fit_df,
        runtime_df=runtime_df,
        official_audit=audit_df,
        config=policy_config,
    )
    holdout_results.to_csv(output_dir / "holdout_results.csv", index=False)
    holdout_summary.to_csv(output_dir / "holdout_summary.csv", index=False)

    full_policy = RiskControlledSelectorPolicy.fit_from_tables(
        feature_df=profile_df,
        overall_df=overall_df,
        biology_df=biology_fit_df,
        official_audit_df=audit_df,
        config=policy_config,
    )
    full_policy_path = output_dir / "selector_policy.json"
    full_policy.save_json(full_policy_path)

    topk_sensitivity = run_topk_sensitivity(
        output_dir=output_dir,
        real_data_root=(ROOT / args.real_data_root).resolve(),
        topk_values=parse_int_list(args.top_k_grid),
        gate_model_path=args.gate_model_path,
        refine_epochs=int(args.refine_epochs),
        seed=int(args.seed),
        routed_methods=holdout_results.set_index("dataset")["selected_method"].to_dict(),
    )
    topk_sensitivity.to_csv(output_dir / "topk_sensitivity.csv", index=False)

    go_no_go_text = build_go_no_go(
        holdout_summary=holdout_summary,
        holdout_results=holdout_results,
        audit_df=audit_df,
    )
    (output_dir / "go_no_go.md").write_text(go_no_go_text, encoding="utf-8")

    verify_registry_integration(
        policy_path=full_policy_path,
        gate_model_path=args.gate_model_path,
        refine_epochs=int(args.refine_epochs),
        seed=int(args.seed),
    )


def parse_int_list(spec: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in spec.split(",") if part.strip())


def resolve_device_info() -> dict[str, object]:
    device = choose_torch_device()
    cuda_available = torch.cuda.is_available()
    payload: dict[str, object] = {
        "device": str(device),
        "cuda_available": bool(cuda_available),
        "cuda_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        payload["cuda_devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return payload


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_profile_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for dataset_name, group in raw_df.groupby("dataset", sort=False):
        selector_rows = group[group["method"] == "adaptive_hybrid_hvg"]
        if selector_rows.empty:
            selector_rows = group[group["method"] == "adaptive_stat_hvg"]
        if selector_rows.empty:
            raise ValueError(f"Missing selector profile rows for dataset '{dataset_name}'.")
        row = selector_rows.iloc[0]
        rows.append(
            {
                "dataset": str(dataset_name),
                "n_cells": float(row["stat_n_cells"]),
                "n_genes": float(row["stat_n_genes"]),
                "batch_classes": float(row["stat_batch_classes"]),
                "dropout_rate": float(row["stat_dropout_rate"]),
                "library_cv": float(row["stat_library_cv"]),
                "cluster_strength": float(row["stat_cluster_strength"]),
                "batch_strength": float(row["stat_batch_strength"]),
                "trajectory_strength": float(row["stat_trajectory_strength"]),
                "pc_entropy": float(row["stat_pc_entropy"]),
                "rare_fraction": float(row["stat_rare_fraction"]),
            }
        )
    profile_df = pd.DataFrame(rows).set_index("dataset").sort_index()
    return profile_df


def write_diagnosis_docs(
    *,
    output_dir: Path,
    readout_md: str,
    source_holdout_summary: pd.DataFrame,
    official_audit: pd.DataFrame,
) -> None:
    rule_row = source_holdout_summary[source_holdout_summary["policy"] == "rule_based_selector"].iloc[0]
    learned_row = source_holdout_summary[source_holdout_summary["policy"] == "holdout_profile_knn_selector"].iloc[0]
    fallback_risks = official_audit[official_audit["official_fallback_rate"].fillna(0.0) >= 0.50]["method"].astype(str).tolist()

    diagnosis_lines = [
        "# Diagnosis",
        "",
        "## Keep",
        "- `compute_adaptive_stat_profile` and the selector metadata plumbing are worth keeping because they expose a compact dataset profile that is already reused across routing, failure taxonomy, and hold-out analysis.",
        "- The round2 evaluation stack is worth keeping because it already separates per-dataset winners, pairwise deltas, route-to-regime summaries, and official baseline audit signals in a reviewer-readable way.",
        "- `adaptive_hybrid_hvg` is worth keeping only as a safe anchor because the round2 readout still ranks it global #1, even though its internal routing story is heuristic-heavy.",
        "",
        "## Freeze",
        "- Do not keep extending `resolve_adaptive_stat_decision` / `resolve_adaptive_hybrid_decision` with more thresholds, route names, or escape cases; that path is already too close to heuristic engineering.",
        "- Do not expand the frontier-lite branch into the main story; it is high-cost, dataset-fragile, and the existing learned hold-out selector already underperforms the rule-based anchor on mean held-out score.",
        f"- Do not mix fallback-heavy official baselines into the claimed expert bank without explicit audit. Current major fallback risks: {', '.join(fallback_risks) if fallback_risks else 'none detected'}.",
        "",
        "## Promote",
        "- The most paper-worthy core component is a held-out, risk-controlled dataset-level selector that uses the existing dataset profile as input, routes over a published/official expert bank, and abstains to a strong safe anchor when confidence is low.",
        "",
        "## Evidence",
        f"- Existing hold-out KNN selector mean score={float(learned_row['mean_score']):.4f}, rule-based selector mean score={float(rule_row['mean_score']):.4f}, mean delta vs rule={float(learned_row['mean_delta_vs_rule']):.4f}.",
        "- That gap says direct best-expert prediction is not yet a publishable story, but it also points to the exact missing piece: explicit uncertainty plus abstention.",
    ]
    (output_dir / "diagnosis.md").write_text("\n".join(diagnosis_lines), encoding="utf-8")

    candidate_lines = [
        "# Candidate Directions",
        "",
        "1. Safe-anchor selective release over the published/official expert bank.",
        "   Train a dataset-level policy to predict expert gains relative to `adaptive_hybrid_hvg`, require uncertainty-aware support, and abstain back to the anchor when release risk is high.",
        "",
        "2. Direct held-out winner prediction over the full method bank.",
        "   Predict the single best method directly from dataset profile features.",
        "   Reject because the existing round2 hold-out KNN selector is already a version of this idea and loses clearly to the rule-based anchor on mean held-out score.",
        "",
        "3. Soft score fusion over heterogeneous experts with confidence shrinkage.",
        "   Learn a confidence-aware blend of expert gene scores.",
        "   Reject because score scales are method-specific, blending them reintroduces heuristic score normalization, and the story drifts back toward engineering instead of selective decision theory.",
    ]
    (output_dir / "candidate_directions.md").write_text("\n".join(candidate_lines), encoding="utf-8")

    selected_lines = [
        "# Selected Direction",
        "",
        "Chosen direction: safe-anchor selective release over a published/official expert bank.",
        "",
        "Why this is the single best shot:",
        "- It keeps the strongest empirical asset we already have, namely `adaptive_hybrid_hvg` as a safe default, while moving the new contribution into a cleaner selector-policy layer.",
        "- It adds exactly the missing method ingredients the current story lacks: explicit uncertainty, abstention, held-out calibration, and official-vs-fallback admissibility.",
        "- It avoids further threshold proliferation inside the heuristic router itself.",
    ]
    (output_dir / "selected_direction.md").write_text("\n".join(selected_lines), encoding="utf-8")

    experiment_lines = [
        "# Experiment Plan",
        "",
        "Primary question: can a hold-out calibrated, uncertainty-aware release policy beat or at least match `adaptive_hybrid_hvg` on held-out mean score while staying biology-aware and audit-clean?",
        "",
        "Plan:",
        "- Build a risk-controlled selector that predicts per-expert improvement over `adaptive_hybrid_hvg` from dataset profile features.",
        "- Use weighted marker recovery as the biology-aware proxy in both the decision rule and the final evaluation summary.",
        "- Exclude or penalize fallback-heavy official baselines using the existing official audit file rather than ad-hoc exceptions.",
        "- Evaluate leave-one-dataset-out on the round2 benchmark tables, then run top-k sensitivity and direct biology checks on the routed expert methods.",
        "- Emit a strict go / no-go recommendation based on held-out mean score first, biology proxy second, and aggregate readouts only as support.",
    ]
    (output_dir / "experiment_plan.md").write_text("\n".join(experiment_lines), encoding="utf-8")


def build_official_vs_fallback_audit(*, official_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    official_methods = set(official_audit["method"].astype(str).tolist())
    for method_name in RiskControlledSelectorConfig().candidate_methods:
        if method_name not in official_methods:
            rows.append(
                {
                    "method": method_name,
                    "source_type": "published_local",
                    "official_fallback_rate": np.nan,
                    "admissible_for_selector": True,
                    "admissibility_reason": "published non-official expert",
                }
            )
            continue
        row = official_audit[official_audit["method"] == method_name].iloc[0]
        fallback_rate = float(row["official_fallback_rate"]) if np.isfinite(row["official_fallback_rate"]) else np.nan
        admissible = not np.isfinite(fallback_rate) or fallback_rate <= 0.30
        rows.append(
            {
                "method": method_name,
                "source_type": "official_or_wrapped",
                "official_fallback_rate": fallback_rate,
                "resolved_backend": row.get("resolved_backend", ""),
                "package_versions": row.get("package_versions", ""),
                "admissible_for_selector": bool(admissible),
                "admissibility_reason": (
                    "fallback rate within selector budget"
                    if admissible
                    else f"fallback rate too high ({fallback_rate:.3f})"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["admissible_for_selector", "method"], ascending=[False, True]).reset_index(drop=True)


def compute_or_load_biology_proxy(
    *,
    output_dir: Path,
    real_data_root: Path,
    top_k: int,
    gate_model_path: str,
    refine_epochs: int,
    seed: int,
    methods: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = output_dir / "biology_proxy_raw.csv"
    summary_path = output_dir / "biology_proxy_method_summary.csv"
    if raw_path.exists() and summary_path.exists():
        raw_df = pd.read_csv(raw_path)
        summary_df = pd.read_csv(summary_path)
        pivot_df = raw_df.pivot(index="dataset", columns="method", values="weighted_marker_recall_at_50").sort_index()
        return pivot_df, summary_df

    specs = {spec.dataset_name: spec for spec in discover_scrna_input_specs(real_data_root)}
    registry = build_default_method_registry(
        top_k=top_k,
        refine_epochs=refine_epochs,
        random_state=seed,
        gate_model_path=gate_model_path,
    )
    rows: list[dict[str, object]] = []
    for dataset_name in sorted(specs):
        spec = specs[dataset_name]
        plan = rr1.ROUND2_PLANS.get(dataset_name, rr1.ROUND1_PLANS.get(dataset_name))
        if plan is None:
            continue
        dataset = load_scrna_dataset(
            data_path=spec.input_path,
            file_format=spec.file_format,
            transpose=spec.transpose,
            obs_path=spec.obs_path,
            var_path=spec.var_path,
            genes_path=spec.genes_path,
            cells_path=spec.cells_path,
            labels_col=spec.labels_col,
            batches_col=spec.batches_col,
            dataset_name=spec.dataset_name,
            max_cells=plan.max_cells,
            max_genes=plan.max_genes,
            random_state=seed,
        )
        if dataset.labels is None:
            continue
        labels = np.asarray(dataset.labels, dtype=object)
        markers, class_weights = pkg.compute_one_vs_rest_markers(
            counts=dataset.counts,
            labels=labels,
            top_n=50,
        )
        if not markers:
            continue
        current_top_k = min(top_k, dataset.counts.shape[1])
        for method_name in methods:
            if method_name not in registry:
                continue
            error_type = ""
            error_message = ""
            recall = np.nan
            weighted = np.nan
            rare = np.nan
            try:
                method_fn = registry[method_name]
                scores = method_fn(dataset.counts, dataset.batches, current_top_k)
                selected = np.argsort(scores)[-current_top_k:]
                recall, weighted, rare = pkg.marker_recovery(
                    selected=selected,
                    marker_sets=markers,
                    class_weights=class_weights,
                )
            except Exception as exc:  # pragma: no cover - experiment-side fault capture
                error_type = type(exc).__name__
                error_message = str(exc)
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "marker_recall_at_50": float(recall),
                    "weighted_marker_recall_at_50": float(weighted),
                    "rare_marker_recall_at_50": float(rare),
                    "class_count": int(len(markers)),
                    "error_type": error_type,
                    "error_message": error_message,
                }
            )

    raw_df = pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)
    raw_df.to_csv(raw_path, index=False)
    summary_df = raw_df.groupby("method", sort=False).agg(
        marker_recall_at_50=("marker_recall_at_50", "mean"),
        weighted_marker_recall_at_50=("weighted_marker_recall_at_50", "mean"),
        rare_marker_recall_at_50=("rare_marker_recall_at_50", "mean"),
        dataset_count=("dataset", "nunique"),
    ).reset_index()
    summary_df = summary_df.sort_values(
        ["weighted_marker_recall_at_50", "rare_marker_recall_at_50", "marker_recall_at_50"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary_df["biology_rank"] = np.arange(1, len(summary_df) + 1)
    summary_df.to_csv(summary_path, index=False)

    pivot_df = raw_df.pivot(index="dataset", columns="method", values="weighted_marker_recall_at_50").sort_index()
    return pivot_df, summary_df


def build_biology_model_df(
    *,
    observed_biology_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    safe_anchor: str,
) -> pd.DataFrame:
    biology_df = observed_biology_df.reindex(index=overall_df.index, columns=overall_df.columns)
    if safe_anchor not in biology_df.columns:
        raise KeyError(f"Safe anchor '{safe_anchor}' missing from biology table.")
    anchor_mean = float(np.nanmean(biology_df[safe_anchor].to_numpy(dtype=np.float64)))
    for column in biology_df.columns:
        column_values = biology_df[column].to_numpy(dtype=np.float64)
        fill_value = float(np.nanmean(column_values))
        if not np.isfinite(fill_value):
            fill_value = anchor_mean
        biology_df[column] = biology_df[column].fillna(fill_value)
    return biology_df.sort_index()


def run_leave_one_dataset_out(
    *,
    feature_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    biology_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    official_audit: pd.DataFrame,
    config: RiskControlledSelectorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = tuple(sorted(feature_df.index.astype(str).tolist()))
    rows: list[dict[str, object]] = []
    for heldout in datasets:
        train_names = [name for name in datasets if name != heldout]
        policy = RiskControlledSelectorPolicy.fit_from_tables(
            feature_df=feature_df.loc[train_names],
            overall_df=overall_df.loc[train_names.union([heldout]) if isinstance(train_names, pd.Index) else train_names + [heldout]],
            biology_df=biology_df.loc[train_names.union([heldout]) if isinstance(train_names, pd.Index) else train_names + [heldout]],
            official_audit_df=official_audit,
            config=config,
        )
        decision = policy.predict(feature_df.loc[heldout].to_dict())
        selected_method = decision.selected_method
        candidate_bank = list(policy.config.candidate_methods)
        best_single_train_method = (
            overall_df.loc[train_names, candidate_bank].mean(axis=0).sort_values(ascending=False).index[0]
        )
        best_single_actual_score = float(overall_df.loc[heldout, best_single_train_method])
        best_single_actual_biology = float(biology_df.loc[heldout, best_single_train_method])
        row = {
            "dataset": heldout,
            "selected_method": selected_method,
            "proposed_method": decision.proposed_method,
            "safe_anchor": decision.safe_anchor,
            "abstained": float(decision.abstained),
            "decision_reason": decision.decision_reason,
            "route_threshold": float(decision.route_threshold),
            "route_score": float(decision.route_score),
            "confidence": float(decision.confidence),
            "predicted_utility_delta": float(decision.predicted_utility_delta),
            "predicted_utility_uncertainty": float(decision.predicted_utility_uncertainty),
            "utility_lcb": float(decision.utility_lcb),
            "predicted_biology_delta": float(decision.predicted_biology_delta),
            "predicted_biology_uncertainty": float(decision.predicted_biology_uncertainty),
            "biology_lcb": float(decision.biology_lcb),
            "selected_overall_score": float(overall_df.loc[heldout, selected_method]),
            "anchor_overall_score": float(overall_df.loc[heldout, decision.safe_anchor]),
            "selected_biology_proxy": float(biology_df.loc[heldout, selected_method]),
            "anchor_biology_proxy": float(biology_df.loc[heldout, decision.safe_anchor]),
            "selected_runtime_sec": float(runtime_df.loc[heldout, selected_method]),
            "anchor_runtime_sec": float(runtime_df.loc[heldout, decision.safe_anchor]),
            "best_single_train_method": best_single_train_method,
            "best_single_actual_score": best_single_actual_score,
            "best_single_actual_biology": best_single_actual_biology,
            "best_single_runtime_sec": float(runtime_df.loc[heldout, best_single_train_method]),
            "selected_vs_anchor_score_delta": float(
                overall_df.loc[heldout, selected_method] - overall_df.loc[heldout, decision.safe_anchor]
            ),
            "selected_vs_anchor_biology_delta": float(
                biology_df.loc[heldout, selected_method] - biology_df.loc[heldout, decision.safe_anchor]
            ),
            "selected_vs_best_single_score_delta": float(
                overall_df.loc[heldout, selected_method] - best_single_actual_score
            ),
            "selected_vs_best_single_biology_delta": float(
                biology_df.loc[heldout, selected_method] - best_single_actual_biology
            ),
            "neighbor_names": ",".join(decision.neighbor_names),
            "neighbor_weights": ",".join(f"{value:.6f}" for value in decision.neighbor_weights),
            "official_reliability": float(decision.official_reliability),
        }
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    summary_rows = []
    summary_rows.append(
        summarize_policy_row(
            policy_name="holdout_risk_selector",
            overall=results_df["selected_overall_score"],
            biology=results_df["selected_biology_proxy"],
            runtime=results_df["selected_runtime_sec"],
        )
    )
    summary_rows.append(
        summarize_policy_row(
            policy_name="adaptive_hybrid_anchor",
            overall=results_df["anchor_overall_score"],
            biology=results_df["anchor_biology_proxy"],
            runtime=results_df["anchor_runtime_sec"],
        )
    )
    summary_rows.append(
        summarize_policy_row(
            policy_name="best_single_train_bank",
            overall=results_df["best_single_actual_score"],
            biology=results_df["best_single_actual_biology"],
            runtime=results_df["best_single_runtime_sec"],
        )
    )
    summary_df = pd.DataFrame(summary_rows)
    selector_row = summary_df[summary_df["policy"] == "holdout_risk_selector"].iloc[0]
    anchor_row = summary_df[summary_df["policy"] == "adaptive_hybrid_anchor"].iloc[0]
    best_single_row = summary_df[summary_df["policy"] == "best_single_train_bank"].iloc[0]
    summary_df["wins_vs_anchor"] = [
        int(np.sum(results_df["selected_vs_anchor_score_delta"] > 0.0)) if policy == "holdout_risk_selector" else np.nan
        for policy in summary_df["policy"]
    ]
    summary_df["wins_vs_best_single"] = [
        int(np.sum(results_df["selected_vs_best_single_score_delta"] > 0.0)) if policy == "holdout_risk_selector" else np.nan
        for policy in summary_df["policy"]
    ]
    summary_df["mean_score_delta_vs_anchor"] = [
        float(selector_row["mean_overall_score"] - anchor_row["mean_overall_score"])
        if policy == "holdout_risk_selector"
        else np.nan
        for policy in summary_df["policy"]
    ]
    summary_df["mean_biology_delta_vs_anchor"] = [
        float(selector_row["mean_biology_proxy"] - anchor_row["mean_biology_proxy"])
        if policy == "holdout_risk_selector"
        else np.nan
        for policy in summary_df["policy"]
    ]
    summary_df["mean_score_delta_vs_best_single"] = [
        float(selector_row["mean_overall_score"] - best_single_row["mean_overall_score"])
        if policy == "holdout_risk_selector"
        else np.nan
        for policy in summary_df["policy"]
    ]
    summary_df["override_rate"] = [
        float(1.0 - results_df["abstained"].mean()) if policy == "holdout_risk_selector" else np.nan
        for policy in summary_df["policy"]
    ]
    return results_df, summary_df


def summarize_policy_row(
    *,
    policy_name: str,
    overall: pd.Series,
    biology: pd.Series,
    runtime: pd.Series,
) -> dict[str, object]:
    return {
        "policy": policy_name,
        "mean_overall_score": float(np.nanmean(overall.to_numpy(dtype=np.float64))),
        "std_overall_score": float(np.nanstd(overall.to_numpy(dtype=np.float64), ddof=0)),
        "mean_biology_proxy": float(np.nanmean(biology.to_numpy(dtype=np.float64))),
        "std_biology_proxy": float(np.nanstd(biology.to_numpy(dtype=np.float64), ddof=0)),
        "mean_runtime_sec": float(np.nanmean(runtime.to_numpy(dtype=np.float64))),
        "dataset_count": int(len(overall)),
    }


def run_topk_sensitivity(
    *,
    output_dir: Path,
    real_data_root: Path,
    topk_values: tuple[int, ...],
    gate_model_path: str,
    refine_epochs: int,
    seed: int,
    routed_methods: dict[str, str],
) -> pd.DataFrame:
    raw_path = output_dir / "topk_sensitivity_raw.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path)

    specs = {spec.dataset_name: spec for spec in discover_scrna_input_specs(real_data_root)}
    rows: list[dict[str, object]] = []
    for dataset_name in sorted(routed_methods):
        if dataset_name not in specs:
            continue
        spec = specs[dataset_name]
        plan = rr1.ROUND2_PLANS.get(dataset_name, rr1.ROUND1_PLANS.get(dataset_name))
        if plan is None:
            continue
        dataset = load_scrna_dataset(
            data_path=spec.input_path,
            file_format=spec.file_format,
            transpose=spec.transpose,
            obs_path=spec.obs_path,
            var_path=spec.var_path,
            genes_path=spec.genes_path,
            cells_path=spec.cells_path,
            labels_col=spec.labels_col,
            batches_col=spec.batches_col,
            dataset_name=spec.dataset_name,
            max_cells=plan.max_cells,
            max_genes=plan.max_genes,
            random_state=seed,
        )
        labels = None if dataset.labels is None else np.asarray(dataset.labels, dtype=object)
        markers: dict[str, set[int]] = {}
        class_weights: dict[str, float] = {}
        if labels is not None:
            markers, class_weights = pkg.compute_one_vs_rest_markers(
                counts=dataset.counts,
                labels=labels,
                top_n=50,
            )
        for top_k in topk_values:
            registry = build_default_method_registry(
                top_k=int(top_k),
                refine_epochs=refine_epochs,
                random_state=seed,
                gate_model_path=gate_model_path,
            )
            current_top_k = min(int(top_k), dataset.counts.shape[1])
            for method_name in sorted({routed_methods[dataset_name], "adaptive_hybrid_hvg"}):
                method_fn = registry[method_name]
                scores, runtime_sec = timed_call(method_fn, dataset.counts, dataset.batches, current_top_k)
                selected = np.argsort(scores)[-current_top_k:]
                metrics = evaluate_real_selection(
                    counts=dataset.counts,
                    selected_genes=selected,
                    scorer_fn=lambda subset_counts, subset_batches, fn=method_fn, local_top_k=current_top_k: fn(
                        subset_counts,
                        subset_batches,
                        local_top_k,
                    ),
                    labels=dataset.labels,
                    batches=dataset.batches,
                    top_k=current_top_k,
                    random_state=seed,
                    n_bootstrap=2,
                )
                marker_recall = np.nan
                weighted_marker_recall = np.nan
                rare_marker_recall = np.nan
                if markers:
                    marker_recall, weighted_marker_recall, rare_marker_recall = pkg.marker_recovery(
                        selected=selected,
                        marker_sets=markers,
                        class_weights=class_weights,
                    )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "top_k": int(current_top_k),
                        "method": method_name,
                        "runtime_sec": float(runtime_sec),
                        "ari": float(metrics.get("ari", np.nan)),
                        "nmi": float(metrics.get("nmi", np.nan)),
                        "neighbor_preservation": float(metrics.get("neighbor_preservation", np.nan)),
                        "cluster_silhouette": float(metrics.get("cluster_silhouette", np.nan)),
                        "batch_mixing": float(metrics.get("batch_mixing", np.nan)),
                        "weighted_marker_recall_at_50": float(weighted_marker_recall),
                        "marker_recall_at_50": float(marker_recall),
                        "rare_marker_recall_at_50": float(rare_marker_recall),
                        "routed_method_at_200": routed_methods[dataset_name],
                    }
                )
    raw_df = pd.DataFrame(rows).sort_values(["dataset", "top_k", "method"]).reset_index(drop=True)
    raw_df.to_csv(raw_path, index=False)
    return raw_df


def build_go_no_go(
    *,
    holdout_summary: pd.DataFrame,
    holdout_results: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> str:
    selector_row = holdout_summary[holdout_summary["policy"] == "holdout_risk_selector"].iloc[0]
    anchor_row = holdout_summary[holdout_summary["policy"] == "adaptive_hybrid_anchor"].iloc[0]
    score_delta = float(selector_row["mean_overall_score"] - anchor_row["mean_overall_score"])
    biology_delta = float(selector_row["mean_biology_proxy"] - anchor_row["mean_biology_proxy"])
    fallback_risky = audit_df[
        (audit_df["source_type"] == "official_or_wrapped")
        & (~audit_df["admissible_for_selector"].astype(bool))
    ]["method"].astype(str).tolist()

    if score_delta < 0.0:
        verdict = "stop / pivot"
    elif biology_delta < 0.0:
        verdict = "continue with caution"
    elif fallback_risky:
        verdict = "continue with caution"
    else:
        verdict = "continue"

    next_line = (
        "Next main line: keep `adaptive_hybrid_hvg` as the production-safe method and demote the learned selector to a calibrated analysis layer unless a substantially larger multi-dataset training pool is added."
        if verdict != "continue"
        else "Next main line: finalize the risk-controlled selector story around selective release from the safe anchor, then tighten the official-baseline section for reviewers."
    )

    lines = [
        "# Go / No-Go",
        "",
        f"Decision: `{verdict}`",
        "",
        "## Evidence",
        f"- Hold-out selector mean overall score: {float(selector_row['mean_overall_score']):.4f}",
        f"- Rule-based safe anchor mean overall score: {float(anchor_row['mean_overall_score']):.4f}",
        f"- Mean score delta vs anchor: {score_delta:.4f}",
        f"- Mean biology proxy delta vs anchor: {biology_delta:.4f}",
        f"- Override count: {int(np.sum(holdout_results['abstained'] < 1.0))} / {int(len(holdout_results))}",
        (
            f"- Official fallback methods still unsafe for inclusion in the selector bank: {', '.join(fallback_risky)}"
            if fallback_risky
            else "- No selector-bank method failed the official/fallback admissibility screen."
        ),
        "",
        "## Conclusion",
        next_line,
    ]
    return "\n".join(lines)


def verify_registry_integration(
    *,
    policy_path: Path,
    gate_model_path: str,
    refine_epochs: int,
    seed: int,
) -> None:
    registry = build_default_method_registry(
        top_k=200,
        refine_epochs=refine_epochs,
        random_state=seed,
        gate_model_path=gate_model_path,
        holdout_selector_policy_path=str(policy_path),
    )
    if HOLDOUT_RISK_SELECTOR_METHOD not in registry:
        raise RuntimeError("Holdout selector method was not registered.")


if __name__ == "__main__":
    main()
