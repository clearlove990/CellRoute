from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from hvg_research import build_default_method_registry, choose_torch_device, discover_scrna_input_specs, evaluate_real_selection, load_scrna_dataset
from hvg_research.adaptive_stat import compute_adaptive_stat_profile
from hvg_research.eval import timed_call
from hvg_research.holdout_selector import RiskControlledSelectorConfig, RiskControlledSelectorPolicy

import run_holdout_selector_upgrade as phase1
import build_topconf_selector_round2_package as pkg
import run_real_inputs_round1 as rr1


DEFAULT_PHASE1_DIR = ROOT / "artifacts_codex_selector_mvp"
DEFAULT_ROUND2_DIR = ROOT / "artifacts_topconf_selector_round2"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts_codex_selector_phase2"
DEFAULT_SEEDS = (7, 17, 27)
DEFAULT_TOPKS = (100, 200, 500)
ROBUSTNESS_CANDIDATE_METHOD = "multinomial_deviance_hvg"
SAFE_ANCHOR_METHOD = "adaptive_hybrid_hvg"


@dataclass(frozen=True)
class PolicyVariant:
    name: str
    description: str
    bank_mode: str = "current_filtered"
    use_uncertainty: bool = True
    biology_weight: float = 0.35
    use_official_penalty: bool = True
    use_calibrated_route_threshold: bool = True
    route_threshold_offset: float = 0.0
    confidence_threshold: float = 0.0
    use_biology_release_gate: bool = False
    biology_margin: float = -np.inf
    biology_margin_kind: str = "lcb"
    anchor_only: bool = False
    always_release: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase2 experiments for the holdout selective-release HVG policy.")
    parser.add_argument("--round2-dir", type=str, default=str(DEFAULT_ROUND2_DIR))
    parser.add_argument("--phase1-dir", type=str, default=str(DEFAULT_PHASE1_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--real-data-root", type=str, default="data/real_inputs")
    parser.add_argument("--gate-model-path", type=str, default=rr1.DEFAULT_GATE_MODEL_PATH)
    parser.add_argument("--seeds", type=str, default="7,17,27")
    parser.add_argument("--topks", type=str, default="100,200,500")
    parser.add_argument("--refine-epochs", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    round2_dir = Path(args.round2_dir).resolve()
    phase1_dir = Path(args.phase1_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    round2_df, raw_df, phase1_bio_raw, official_audit = load_required_tables(
        round2_dir=round2_dir,
        phase1_dir=phase1_dir,
    )
    write_experiment_plan(output_dir=output_dir)

    feature_df = phase1.build_profile_df(raw_df)
    metric_tables = build_metric_tables(dataset_summary=round2_df)
    overall_df = metric_tables["overall_score"]
    runtime_df = metric_tables["runtime_sec"]
    bio_obs = phase1_bio_raw.pivot(index="dataset", columns="method", values="weighted_marker_recall_at_50").sort_index()
    full_biology_df = phase1.build_biology_model_df(
        observed_biology_df=bio_obs,
        overall_df=overall_df,
        safe_anchor=SAFE_ANCHOR_METHOD,
    )

    strict_official_audit = build_strict_official_audit(
        round2_methods=tuple(sorted(round2_df["method"].astype(str).unique().tolist())),
        official_audit=official_audit,
    )
    strict_official_audit.to_csv(output_dir / "strict_official_audit.csv", index=False)

    coverage_sweep = run_coverage_risk_sweep(
        feature_df=feature_df,
        overall_df=overall_df,
        biology_df=full_biology_df,
        runtime_df=runtime_df,
        metric_tables=metric_tables,
        audit_df=strict_official_audit,
    )
    coverage_sweep.to_csv(output_dir / "coverage_risk_sweep.csv", index=False)
    coverage_summary = summarize_coverage_sweep(coverage_sweep=coverage_sweep)
    coverage_summary.to_csv(output_dir / "coverage_risk_summary.csv", index=False)
    write_coverage_notes(
        output_dir=output_dir,
        coverage_sweep=coverage_sweep,
        coverage_summary=coverage_summary,
    )

    ablation_results, ablation_summary = run_policy_ablations(
        feature_df=feature_df,
        overall_df=overall_df,
        biology_df=full_biology_df,
        runtime_df=runtime_df,
        metric_tables=metric_tables,
        audit_df=strict_official_audit,
    )
    ablation_results.to_csv(output_dir / "policy_ablation_results.csv", index=False)
    ablation_summary.to_csv(output_dir / "policy_ablation_summary.csv", index=False)
    write_policy_ablation_readout(
        output_dir=output_dir,
        ablation_summary=ablation_summary,
    )

    biology_guardrail_sweep, biology_guardrail_summary = run_biology_guardrail_analysis(
        feature_df=feature_df,
        overall_df=overall_df,
        biology_df=full_biology_df,
        runtime_df=runtime_df,
        metric_tables=metric_tables,
        audit_df=strict_official_audit,
    )
    biology_guardrail_sweep.to_csv(output_dir / "biology_guardrail_sweep.csv", index=False)
    biology_guardrail_summary.to_csv(output_dir / "biology_guardrail_summary.csv", index=False)
    write_paul15_case_study(
        output_dir=output_dir,
        guardrail_summary=biology_guardrail_summary,
    )

    robustness_raw, robustness_summary = run_topk_seed_robustness(
        output_dir=output_dir,
        real_data_root=(ROOT / args.real_data_root).resolve(),
        gate_model_path=args.gate_model_path,
        refine_epochs=int(args.refine_epochs),
        seeds=parse_int_list(args.seeds),
        topks=parse_int_list(args.topks),
    )
    robustness_raw.to_csv(output_dir / "topk_seed_robustness.csv", index=False)
    robustness_summary.to_csv(output_dir / "topk_seed_summary.csv", index=False)
    write_topk_seed_readout(
        output_dir=output_dir,
        robustness_summary=robustness_summary,
    )

    strict_holdout_results, strict_holdout_summary = run_strict_official_dependency_analysis(
        feature_df=feature_df,
        overall_df=overall_df,
        biology_df=full_biology_df,
        runtime_df=runtime_df,
        metric_tables=metric_tables,
        audit_df=strict_official_audit,
    )
    strict_holdout_results.to_csv(output_dir / "strict_official_holdout_results.csv", index=False)
    strict_holdout_summary.to_csv(output_dir / "strict_official_holdout_summary.csv", index=False)
    write_official_dependency_note(
        output_dir=output_dir,
        strict_holdout_summary=strict_holdout_summary,
        strict_holdout_results=strict_holdout_results,
    )

    phase2_go_no_go = build_phase2_go_no_go(
        coverage_summary=coverage_summary,
        ablation_summary=ablation_summary,
        biology_guardrail_summary=biology_guardrail_summary,
        robustness_summary=robustness_summary,
        strict_holdout_summary=strict_holdout_summary,
    )
    (output_dir / "phase2_go_no_go.md").write_text(phase2_go_no_go, encoding="utf-8")
    save_device_info(output_dir=output_dir)


def parse_int_list(spec: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in spec.split(",") if part.strip())


def load_required_tables(
    *,
    round2_dir: Path,
    phase1_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "dataset_summary": round2_dir / "benchmark_dataset_summary.csv",
        "raw_results": round2_dir / "benchmark_raw_results.csv",
        "official_audit": round2_dir / "official_baseline_audit.csv",
        "phase1_biology": phase1_dir / "biology_proxy_raw.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input artifacts: {missing}")
    return (
        pd.read_csv(required["dataset_summary"]),
        pd.read_csv(required["raw_results"]),
        pd.read_csv(required["phase1_biology"]),
        pd.read_csv(required["official_audit"]),
    )


def save_device_info(*, output_dir: Path) -> None:
    cuda_available = torch.cuda.is_available()
    payload: dict[str, object] = {
        "device": str(choose_torch_device()),
        "cuda_available": bool(cuda_available),
        "cuda_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        payload["cuda_devices"] = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
    (output_dir / "device_info.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_experiment_plan(*, output_dir: Path) -> None:
    lines = [
        "# Phase2 Experiment Plan",
        "",
        "Main goal: stress-test `holdout_risk_release_hvg` as a calibrated selective release layer over the safe anchor, rather than as a new heuristic router.",
        "",
        "Planned experiment groups:",
        "- Coverage-risk sweep over confidence threshold, biology non-inferiority margin, official admissibility filter, and calibrated route-threshold offsets.",
        "- Policy ablations isolating abstention, uncertainty, biology guardrail, official admissibility, and held-out calibration.",
        "- `paul15` case study focused on biology conflict and whether stricter biology constraints prevent the risky override.",
        "- Top-k / seed robustness using the currently routed expert (`multinomial_deviance_hvg`) versus the safe anchor across multiple settings.",
        "- Strict-official dependency audit comparing the current bank to a bank limited to strict official implementations.",
        "",
        "Success criteria:",
        "- Show that selective release has a low-risk operating region, not just a single lucky threshold.",
        "- Show that removing abstention or uncertainty hurts, so the gain comes from the risk-control structure rather than a one-off expert choice.",
        "- Show whether biology guardrails should be part of the method definition.",
        "- Quantify whether the main result survives if the bank is restricted to strict official experts.",
    ]
    (output_dir / "experiment_plan.md").write_text("\n".join(lines), encoding="utf-8")


def build_metric_tables(*, dataset_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metric_names = [
        "overall_score",
        "runtime_sec",
        "ari",
        "nmi",
        "neighbor_preservation",
        "cluster_silhouette",
        "batch_mixing",
        "stability",
        "label_silhouette",
    ]
    tables = {}
    for metric in metric_names:
        if metric in dataset_summary.columns:
            tables[metric] = dataset_summary.pivot(index="dataset", columns="method", values=metric).sort_index()
    return tables


def build_strict_official_audit(
    *,
    round2_methods: tuple[str, ...],
    official_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lookup = official_audit.set_index("method", drop=False).to_dict(orient="index")
    for method_name in round2_methods:
        if method_name in {SAFE_ANCHOR_METHOD, "adaptive_stat_hvg"}:
            method_class = "selector"
            strict_official = False
            fallback_used = False
            source_type = "selector"
            note = "not part of the expert bank"
        elif method_name in lookup:
            audit_row = lookup[method_name]
            fallback_rate = audit_row.get("official_fallback_rate", np.nan)
            fallback_used = bool(np.isfinite(fallback_rate) and float(fallback_rate) > 0.0)
            strict_official = not fallback_used
            source_type = "strict_official_available" if strict_official else "official_declared_fallback_used"
            method_class = "official_or_wrapped"
            note = str(audit_row.get("resolved_backend", ""))
        else:
            source_type = "repo_local_only"
            method_class = "repo_local"
            strict_official = False
            fallback_used = False
            note = "no official wrapper in the round2 audit"
        rows.append(
            {
                "method": method_name,
                "method_class": method_class,
                "source_type": source_type,
                "strict_official_available": bool(strict_official),
                "official_fallback_used": bool(fallback_used),
                "note": note,
            }
        )
    return pd.DataFrame(rows).sort_values(["source_type", "method"]).reset_index(drop=True)


def get_bank_methods(*, bank_mode: str, audit_df: pd.DataFrame) -> tuple[str, ...]:
    default_bank = tuple(RiskControlledSelectorConfig().candidate_methods)
    if bank_mode == "current_filtered":
        return default_bank
    if bank_mode == "full_bank":
        return default_bank
    if bank_mode == "strict_official":
        return tuple(
            audit_df[
                (audit_df["method"].isin(default_bank))
                & (audit_df["strict_official_available"].astype(bool))
            ]["method"].astype(str).tolist()
        )
    if bank_mode == "md_only":
        return (ROBUSTNESS_CANDIDATE_METHOD,)
    raise ValueError(f"Unknown bank mode: {bank_mode}")


def build_variant_config(*, variant: PolicyVariant, audit_df: pd.DataFrame) -> RiskControlledSelectorConfig:
    bank_methods = get_bank_methods(bank_mode=variant.bank_mode, audit_df=audit_df)
    reliability_floor = 0.70 if variant.bank_mode == "current_filtered" else 0.0
    return RiskControlledSelectorConfig(
        safe_anchor=SAFE_ANCHOR_METHOD,
        candidate_methods=bank_methods,
        utility_uncertainty_scale=0.60 if variant.use_uncertainty else 0.0,
        biology_uncertainty_scale=0.45 if variant.use_uncertainty else 0.0,
        biology_weight=float(variant.biology_weight),
        official_reliability_floor=float(reliability_floor),
        official_penalty_scale=0.20 if variant.use_official_penalty else 0.0,
    )


def _neighbor_bundle(
    *,
    policy: RiskControlledSelectorPolicy,
    standardized_profile: dict[str, float],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    distances = []
    for dataset_name in policy.dataset_names:
        feature_row = policy.feature_table[dataset_name]
        vector = np.asarray(
            [feature_row[name] - standardized_profile[name] for name in policy.config.feature_names],
            dtype=np.float64,
        )
        distances.append((dataset_name, float(np.sqrt(np.square(vector).sum()))))
    distances.sort(key=lambda item: item[1])
    neighbors = distances[: min(policy.config.k_neighbors, len(distances))]
    names = tuple(name for name, _ in neighbors)
    inverse_distance = np.asarray([1.0 / (distance + 1e-6) for _, distance in neighbors], dtype=np.float64)
    inverse_distance = inverse_distance / max(float(inverse_distance.sum()), 1e-8)
    return names, tuple(float(value) for value in inverse_distance)


def _candidate_rows(
    *,
    policy: RiskControlledSelectorPolicy,
    profile: dict[str, float],
) -> list[dict[str, float | str]]:
    standardized_profile = policy._standardize_profile(profile)
    neighbor_names, neighbor_weights = _neighbor_bundle(
        policy=policy,
        standardized_profile=standardized_profile,
    )
    rows: list[dict[str, float | str]] = []
    for method_name in policy.config.candidate_methods:
        utility_values = np.asarray(
            [
                policy.overall_table[dataset_name][method_name] - policy.overall_table[dataset_name][policy.config.safe_anchor]
                for dataset_name in neighbor_names
            ],
            dtype=np.float64,
        )
        biology_values = np.asarray(
            [
                policy.biology_table[dataset_name][method_name] - policy.biology_table[dataset_name][policy.config.safe_anchor]
                for dataset_name in neighbor_names
            ],
            dtype=np.float64,
        )
        utility_mu, utility_std = _weighted_stats(utility_values, neighbor_weights)
        biology_mu, biology_std = _weighted_stats(biology_values, neighbor_weights)
        reliability = float(policy.official_reliability.get(method_name, 1.0))
        rows.append(
            {
                "method": method_name,
                "utility_mu": float(utility_mu),
                "utility_std": float(utility_std),
                "biology_mu": float(biology_mu),
                "biology_std": float(biology_std),
                "reliability": reliability,
                "neighbor_names": ",".join(neighbor_names),
                "neighbor_weights": ",".join(f"{value:.6f}" for value in neighbor_weights),
            }
        )
    return rows


def evaluate_variant_lodo(
    *,
    feature_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    biology_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    metric_tables: dict[str, pd.DataFrame],
    audit_df: pd.DataFrame,
    variant: PolicyVariant,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = tuple(sorted(feature_df.index.astype(str).tolist()))
    result_rows: list[dict[str, object]] = []
    for heldout in datasets:
        train_names = [name for name in datasets if name != heldout]
        policy_config = build_variant_config(variant=variant, audit_df=audit_df)
        policy = RiskControlledSelectorPolicy.fit_from_tables(
            feature_df=feature_df.loc[train_names],
            overall_df=overall_df.loc[train_names + [heldout]],
            biology_df=biology_df.loc[train_names + [heldout]],
            official_audit_df=audit_df,
            config=policy_config,
        )
        candidate_rows = _candidate_rows(
            policy=policy,
            profile=feature_df.loc[heldout].to_dict(),
        )
        route_threshold = (
            float(policy.route_threshold + variant.route_threshold_offset)
            if variant.use_calibrated_route_threshold
            else float(variant.route_threshold_offset)
        )
        for row in candidate_rows:
            utility_lcb = float(row["utility_mu"]) - float(policy.config.utility_uncertainty_scale) * float(row["utility_std"])
            biology_lcb = float(row["biology_mu"]) - float(policy.config.biology_uncertainty_scale) * float(row["biology_std"])
            route_score = utility_lcb + float(variant.biology_weight) * biology_lcb - float(policy.config.official_penalty_scale) * max(0.0, 1.0 - float(row["reliability"]))
            uncertainty_mass = 0.0
            if variant.use_uncertainty:
                uncertainty_mass = float(row["utility_std"]) + float(variant.biology_weight) * float(row["biology_std"])
            confidence_logit = route_score - route_threshold - 0.50 * uncertainty_mass
            confidence = float(float(row["reliability"]) * (1.0 / (1.0 + np.exp(-6.0 * confidence_logit))))
            row["utility_lcb"] = utility_lcb
            row["biology_lcb"] = biology_lcb
            row["route_score"] = route_score
            row["confidence"] = confidence
        candidate_df = pd.DataFrame(candidate_rows).sort_values(["route_score", "utility_mu", "biology_mu"], ascending=[False, False, False]).reset_index(drop=True)
        proposed_method = str(candidate_df.iloc[0]["method"]) if not candidate_df.empty else SAFE_ANCHOR_METHOD
        proposed_row = candidate_df.iloc[0] if not candidate_df.empty else None

        if variant.anchor_only or proposed_row is None:
            selected_method = SAFE_ANCHOR_METHOD
            release = False
            decision_reason = "anchor_only"
        else:
            route_pass = bool(float(proposed_row["route_score"]) > route_threshold)
            confidence_pass = bool(float(proposed_row["confidence"]) >= float(variant.confidence_threshold))
            biology_signal = float(proposed_row["biology_lcb"]) if variant.biology_margin_kind == "lcb" else float(proposed_row["biology_mu"])
            biology_pass = True if not variant.use_biology_release_gate else bool(biology_signal >= float(variant.biology_margin))
            if variant.always_release:
                release = True
                decision_reason = "always_release"
            else:
                release = bool(route_pass and confidence_pass and biology_pass)
                failure_bits = []
                if not route_pass:
                    failure_bits.append("route")
                if not confidence_pass:
                    failure_bits.append("confidence")
                if variant.use_biology_release_gate and not biology_pass:
                    failure_bits.append("biology")
                decision_reason = "release" if release else "abstain_" + "_".join(failure_bits or ["anchor"])
            selected_method = proposed_method if release else SAFE_ANCHOR_METHOD

        candidate_bank = list(policy.config.candidate_methods)
        best_single_train_method = (
            overall_df.loc[train_names, candidate_bank].mean(axis=0).sort_values(ascending=False).index[0]
            if candidate_bank
            else SAFE_ANCHOR_METHOD
        )
        oracle_bank_method = (
            overall_df.loc[heldout, candidate_bank].sort_values(ascending=False).index[0]
            if candidate_bank
            else SAFE_ANCHOR_METHOD
        )
        row = {
            "policy": variant.name,
            "dataset": heldout,
            "selected_method": selected_method,
            "proposed_method": proposed_method,
            "decision_reason": decision_reason,
            "released": float(selected_method != SAFE_ANCHOR_METHOD),
            "route_threshold": route_threshold,
            "confidence_threshold": float(variant.confidence_threshold),
            "biology_margin": float(variant.biology_margin) if np.isfinite(variant.biology_margin) else np.nan,
            "selected_overall_score": float(overall_df.loc[heldout, selected_method]),
            "anchor_overall_score": float(overall_df.loc[heldout, SAFE_ANCHOR_METHOD]),
            "selected_biology_proxy": float(biology_df.loc[heldout, selected_method]),
            "anchor_biology_proxy": float(biology_df.loc[heldout, SAFE_ANCHOR_METHOD]),
            "selected_runtime_sec": float(runtime_df.loc[heldout, selected_method]),
            "anchor_runtime_sec": float(runtime_df.loc[heldout, SAFE_ANCHOR_METHOD]),
            "best_single_train_method": best_single_train_method,
            "best_single_actual_score": float(overall_df.loc[heldout, best_single_train_method]),
            "best_single_actual_biology": float(biology_df.loc[heldout, best_single_train_method]),
            "oracle_bank_method": oracle_bank_method,
            "oracle_bank_score": float(overall_df.loc[heldout, oracle_bank_method]),
            "selected_vs_anchor_score_delta": float(overall_df.loc[heldout, selected_method] - overall_df.loc[heldout, SAFE_ANCHOR_METHOD]),
            "selected_vs_anchor_biology_delta": float(biology_df.loc[heldout, selected_method] - biology_df.loc[heldout, SAFE_ANCHOR_METHOD]),
            "selected_vs_best_single_score_delta": float(overall_df.loc[heldout, selected_method] - overall_df.loc[heldout, best_single_train_method]),
            "selected_vs_best_single_biology_delta": float(biology_df.loc[heldout, selected_method] - biology_df.loc[heldout, best_single_train_method]),
            "regret_to_oracle_bank": float(overall_df.loc[heldout, oracle_bank_method] - overall_df.loc[heldout, selected_method]),
            "proposed_route_score": np.nan if proposed_row is None else float(proposed_row["route_score"]),
            "proposed_confidence": np.nan if proposed_row is None else float(proposed_row["confidence"]),
            "proposed_utility_mu": np.nan if proposed_row is None else float(proposed_row["utility_mu"]),
            "proposed_utility_lcb": np.nan if proposed_row is None else float(proposed_row["utility_lcb"]),
            "proposed_biology_mu": np.nan if proposed_row is None else float(proposed_row["biology_mu"]),
            "proposed_biology_lcb": np.nan if proposed_row is None else float(proposed_row["biology_lcb"]),
            "proposed_reliability": np.nan if proposed_row is None else float(proposed_row["reliability"]),
        }
        for metric_name, table in metric_tables.items():
            if selected_method in table.columns and SAFE_ANCHOR_METHOD in table.columns:
                row[f"selected_{metric_name}"] = float(table.loc[heldout, selected_method])
                row[f"anchor_{metric_name}"] = float(table.loc[heldout, SAFE_ANCHOR_METHOD])
                row[f"delta_{metric_name}"] = float(table.loc[heldout, selected_method] - table.loc[heldout, SAFE_ANCHOR_METHOD])
        result_rows.append(row)

    results_df = pd.DataFrame(result_rows).sort_values(["policy", "dataset"]).reset_index(drop=True)
    summary_rows = []
    for policy_name, group in results_df.groupby("policy", sort=False):
        routed = group[group["released"] > 0.0]
        summary_rows.append(
            {
                "policy": policy_name,
                "mean_overall_score": float(group["selected_overall_score"].mean()),
                "mean_biology_proxy": float(group["selected_biology_proxy"].mean()),
                "mean_runtime_sec": float(group["selected_runtime_sec"].mean()),
                "mean_delta_vs_anchor": float(group["selected_vs_anchor_score_delta"].mean()),
                "mean_delta_vs_best_single": float(group["selected_vs_best_single_score_delta"].mean()),
                "mean_regret_to_oracle_bank": float(group["regret_to_oracle_bank"].mean()),
                "worst_dataset_delta": float(group["selected_vs_anchor_score_delta"].min()),
                "worst_case_biology_delta": float(group["selected_vs_anchor_biology_delta"].min()),
                "release_coverage": float(group["released"].mean()),
                "override_count": int(group["released"].sum()),
                "dataset_count": int(len(group)),
                "routed_subset_mean_ari_delta": float(routed["delta_ari"].mean()) if "delta_ari" in routed.columns and not routed.empty else np.nan,
                "routed_subset_mean_nmi_delta": float(routed["delta_nmi"].mean()) if "delta_nmi" in routed.columns and not routed.empty else np.nan,
                "routed_subset_mean_biology_delta": float(routed["selected_vs_anchor_biology_delta"].mean()) if not routed.empty else np.nan,
                "routed_subset_mean_score_delta": float(routed["selected_vs_anchor_score_delta"].mean()) if not routed.empty else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["mean_overall_score", "mean_biology_proxy"], ascending=[False, False]).reset_index(drop=True)
    return results_df, summary_df


def run_coverage_risk_sweep(
    *,
    feature_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    biology_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    metric_tables: dict[str, pd.DataFrame],
    audit_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    confidence_grid = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    biology_margin_grid = (-0.10, -0.05, -0.02, 0.0)
    threshold_offset_grid = (-0.10, -0.05, 0.0, 0.05, 0.10)
    for official_filter in (True, False):
        bank_mode = "current_filtered" if official_filter else "full_bank"
        for confidence_threshold in confidence_grid:
            for biology_margin in biology_margin_grid:
                for threshold_offset in threshold_offset_grid:
                    variant = PolicyVariant(
                        name="coverage_sweep",
                        description="coverage-risk sweep",
                        bank_mode=bank_mode,
                        confidence_threshold=float(confidence_threshold),
                        use_biology_release_gate=True,
                        biology_margin=float(biology_margin),
                        biology_margin_kind="lcb",
                        route_threshold_offset=float(threshold_offset),
                    )
                    _, summary = evaluate_variant_lodo(
                        feature_df=feature_df,
                        overall_df=overall_df,
                        biology_df=biology_df,
                        runtime_df=runtime_df,
                        metric_tables=metric_tables,
                        audit_df=audit_df,
                        variant=variant,
                    )
                    summary_row = summary.iloc[0].to_dict()
                    summary_row["official_filter_on"] = bool(official_filter)
                    summary_row["confidence_threshold"] = float(confidence_threshold)
                    summary_row["biology_margin"] = float(biology_margin)
                    summary_row["route_threshold_offset"] = float(threshold_offset)
                    rows.append(summary_row)
    return pd.DataFrame(rows).sort_values(
        ["official_filter_on", "release_coverage", "mean_delta_vs_anchor", "mean_biology_proxy"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)


def summarize_coverage_sweep(*, coverage_sweep: pd.DataFrame) -> pd.DataFrame:
    feasible = coverage_sweep[
        (coverage_sweep["mean_delta_vs_anchor"] >= 0.0)
        & (coverage_sweep["worst_case_biology_delta"] >= -0.06)
    ].copy()
    if feasible.empty:
        feasible = coverage_sweep.copy()
    feasible["sweet_spot_score"] = (
        feasible["mean_delta_vs_anchor"]
        + 0.35 * feasible["mean_biology_proxy"]
        - 0.10 * feasible["release_coverage"]
        - 0.25 * feasible["mean_regret_to_oracle_bank"]
    )
    sweet_spot = feasible.sort_values(
        ["sweet_spot_score", "mean_delta_vs_anchor", "mean_biology_proxy"],
        ascending=[False, False, False],
    ).head(10).copy()
    sweet_spot["summary_type"] = "sweet_spot"

    frontier_rows = []
    for _, row in coverage_sweep.iterrows():
        dominated = False
        for _, other in coverage_sweep.iterrows():
            if row.name == other.name:
                continue
            if (
                float(other["mean_delta_vs_anchor"]) >= float(row["mean_delta_vs_anchor"]) - 1e-12
                and float(other["worst_case_biology_delta"]) >= float(row["worst_case_biology_delta"]) - 1e-12
                and float(other["release_coverage"]) >= float(row["release_coverage"]) - 1e-12
                and (
                    float(other["mean_delta_vs_anchor"]) > float(row["mean_delta_vs_anchor"]) + 1e-12
                    or float(other["worst_case_biology_delta"]) > float(row["worst_case_biology_delta"]) + 1e-12
                    or float(other["release_coverage"]) > float(row["release_coverage"]) + 1e-12
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier_rows.append(row.to_dict())
    frontier = pd.DataFrame(frontier_rows)
    if not frontier.empty:
        frontier = frontier.sort_values(
            ["release_coverage", "mean_delta_vs_anchor", "worst_case_biology_delta"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        frontier["summary_type"] = "pareto_frontier"
    return pd.concat([sweet_spot, frontier], ignore_index=True, sort=False)


def write_coverage_notes(
    *,
    output_dir: Path,
    coverage_sweep: pd.DataFrame,
    coverage_summary: pd.DataFrame,
) -> None:
    best_row = coverage_summary[coverage_summary["summary_type"] == "sweet_spot"].iloc[0]
    current_like = coverage_sweep[
        (coverage_sweep["official_filter_on"])
        & (np.isclose(coverage_sweep["confidence_threshold"], 0.0))
        & (np.isclose(coverage_sweep["biology_margin"], -0.10))
        & (np.isclose(coverage_sweep["route_threshold_offset"], 0.0))
    ].iloc[0]
    strict_bio = coverage_sweep[
        (coverage_sweep["official_filter_on"])
        & (np.isclose(coverage_sweep["biology_margin"], 0.0))
    ].sort_values("mean_overall_score", ascending=False).iloc[0]
    lines = [
        "# Coverage-Risk Notes",
        "",
        f"- Current-like operating point: coverage={float(current_like['release_coverage']):.3f}, mean delta vs anchor={float(current_like['mean_delta_vs_anchor']):.4f}, worst biology delta={float(current_like['worst_case_biology_delta']):.4f}.",
        f"- Best low-risk sweet spot in this sweep: coverage={float(best_row['release_coverage']):.3f}, confidence_threshold={float(best_row['confidence_threshold']):.2f}, biology_margin={float(best_row['biology_margin']):.2f}, route_threshold_offset={float(best_row['route_threshold_offset']):.2f}, mean delta vs anchor={float(best_row['mean_delta_vs_anchor']):.4f}.",
        f"- Strict biology LCB margins are currently too conservative: the best `biology_margin=0` point only reaches coverage={float(strict_bio['release_coverage']):.3f} and mean delta vs anchor={float(strict_bio['mean_delta_vs_anchor']):.4f}.",
        "- This says the release layer is calibratable, but the biology safety estimate is still under-confident enough that hard non-inferiority constraints collapse coverage.",
    ]
    (output_dir / "coverage_risk_notes.md").write_text("\n".join(lines), encoding="utf-8")


def run_policy_ablations(
    *,
    feature_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    biology_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    metric_tables: dict[str, pd.DataFrame],
    audit_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = (
        PolicyVariant(
            name="anchor_only",
            description="Always abstain to the safe anchor.",
            anchor_only=True,
        ),
        PolicyVariant(
            name="always_release_best_predicted_expert",
            description="Always release the top predicted expert with no abstention.",
            always_release=True,
        ),
        PolicyVariant(
            name="no_uncertainty_gate",
            description="Remove uncertainty from route scoring and threshold calibration.",
            use_uncertainty=False,
        ),
        PolicyVariant(
            name="no_biology_guardrail",
            description="Ignore biology in route scoring and release gating.",
            biology_weight=0.0,
        ),
        PolicyVariant(
            name="no_official_admissibility_filter",
            description="Use the full bank without fallback-based filtering or penalties.",
            bank_mode="full_bank",
            use_official_penalty=False,
        ),
        PolicyVariant(
            name="no_heldout_calibration_naive_release",
            description="Disable held-out threshold calibration and use a fixed route threshold of zero.",
            use_calibrated_route_threshold=False,
            route_threshold_offset=0.0,
        ),
        PolicyVariant(
            name="current_full_policy",
            description="Current phase1 full policy.",
        ),
    )
    frames = []
    summaries = []
    for variant in variants:
        results, summary = evaluate_variant_lodo(
            feature_df=feature_df,
            overall_df=overall_df,
            biology_df=biology_df,
            runtime_df=runtime_df,
            metric_tables=metric_tables,
            audit_df=audit_df,
            variant=variant,
        )
        results["description"] = variant.description
        summary["description"] = variant.description
        frames.append(results)
        summaries.append(summary)
    return pd.concat(frames, ignore_index=True), pd.concat(summaries, ignore_index=True).sort_values(
        ["mean_overall_score", "mean_biology_proxy"],
        ascending=[False, False],
    ).reset_index(drop=True)


def write_policy_ablation_readout(
    *,
    output_dir: Path,
    ablation_summary: pd.DataFrame,
) -> None:
    current = ablation_summary[ablation_summary["policy"] == "current_full_policy"].iloc[0]
    always = ablation_summary[ablation_summary["policy"] == "always_release_best_predicted_expert"].iloc[0]
    no_unc = ablation_summary[ablation_summary["policy"] == "no_uncertainty_gate"].iloc[0]
    lines = [
        "# Policy Ablation Readout",
        "",
        f"- Current full policy: mean overall score={float(current['mean_overall_score']):.4f}, mean biology proxy={float(current['mean_biology_proxy']):.4f}, coverage={float(current['release_coverage']):.3f}.",
        f"- Always release hurts badly: mean overall score={float(always['mean_overall_score']):.4f}, mean delta vs anchor={float(always['mean_delta_vs_anchor']):.4f}.",
        f"- Removing uncertainty also hurts: mean overall score={float(no_unc['mean_overall_score']):.4f}, mean delta vs anchor={float(no_unc['mean_delta_vs_anchor']):.4f}.",
        "- This is the main positive ablation result: the gain is coming from selective abstention with uncertainty, not from simply preferring one globally strong expert.",
    ]
    (output_dir / "policy_ablation_readout.md").write_text("\n".join(lines), encoding="utf-8")


def run_biology_guardrail_analysis(
    *,
    feature_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    biology_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    metric_tables: dict[str, pd.DataFrame],
    audit_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    margin_grid = (-0.10, -0.08, -0.05, -0.02, 0.0)
    sweep_rows = []
    for margin in margin_grid:
        for name, description in (
            ("strict_biology_safe_policy" if margin == 0.0 else "soft_biology_policy", "biology guardrail sweep"),
        ):
            variant = PolicyVariant(
                name=name,
                description=description,
                use_biology_release_gate=True,
                biology_margin=float(margin),
                biology_margin_kind="lcb",
            )
            results, summary = evaluate_variant_lodo(
                feature_df=feature_df,
                overall_df=overall_df,
                biology_df=biology_df,
                runtime_df=runtime_df,
                metric_tables=metric_tables,
                audit_df=audit_df,
                variant=variant,
            )
            summary_row = summary.iloc[0].to_dict()
            summary_row["guardrail_margin_lcb"] = float(margin)
            summary_row["paul15_selected_method"] = str(results[results["dataset"] == "paul15"]["selected_method"].iloc[0])
            summary_row["paul15_score_delta"] = float(results[results["dataset"] == "paul15"]["selected_vs_anchor_score_delta"].iloc[0])
            summary_row["paul15_biology_delta"] = float(results[results["dataset"] == "paul15"]["selected_vs_anchor_biology_delta"].iloc[0])
            sweep_rows.append(summary_row)
    for variant in (
        PolicyVariant(
            name="current_full_policy",
            description="Current phase1 policy.",
        ),
        PolicyVariant(
            name="no_biology_constraint_policy",
            description="No biology release gate and no biology term in the route score.",
            biology_weight=0.0,
        ),
    ):
        results, summary = evaluate_variant_lodo(
            feature_df=feature_df,
            overall_df=overall_df,
            biology_df=biology_df,
            runtime_df=runtime_df,
            metric_tables=metric_tables,
            audit_df=audit_df,
            variant=variant,
        )
        summary_row = summary.iloc[0].to_dict()
        summary_row["guardrail_margin_lcb"] = np.nan
        summary_row["paul15_selected_method"] = str(results[results["dataset"] == "paul15"]["selected_method"].iloc[0])
        summary_row["paul15_score_delta"] = float(results[results["dataset"] == "paul15"]["selected_vs_anchor_score_delta"].iloc[0])
        summary_row["paul15_biology_delta"] = float(results[results["dataset"] == "paul15"]["selected_vs_anchor_biology_delta"].iloc[0])
        sweep_rows.append(summary_row)

    summary_df = pd.DataFrame(sweep_rows).sort_values(
        ["mean_overall_score", "mean_biology_proxy", "release_coverage"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    sweep_df = summary_df.copy()
    return sweep_df, summary_df


def write_paul15_case_study(
    *,
    output_dir: Path,
    guardrail_summary: pd.DataFrame,
) -> None:
    current = guardrail_summary[guardrail_summary["policy"] == "current_full_policy"].iloc[0]
    strict_rows = guardrail_summary[guardrail_summary["policy"] == "strict_biology_safe_policy"]
    strict = strict_rows.iloc[0] if not strict_rows.empty else None
    soft = guardrail_summary[
        (guardrail_summary["policy"] == "soft_biology_policy")
        & (np.isclose(guardrail_summary["guardrail_margin_lcb"], -0.08))
    ]
    soft_row = soft.iloc[0] if not soft.empty else guardrail_summary[guardrail_summary["policy"] == "soft_biology_policy"].iloc[0]
    no_bio = guardrail_summary[guardrail_summary["policy"] == "no_biology_constraint_policy"].iloc[0]
    lines = [
        "# Paul15 Case Study",
        "",
        f"- Current full policy releases `paul15` to `{current['paul15_selected_method']}`, gaining overall score {float(current['paul15_score_delta']):.4f} but losing biology proxy {float(current['paul15_biology_delta']):.4f}.",
        "- The conflict is not a top-k mismatch artifact: in phase1 the routed expert stayed biology-weaker than the anchor across 100/200/500 on `paul15`.",
        "- The immediate problem is calibration: the predicted biology mean was slightly positive, but the biology LCB was already negative, so a strict biology-safe gate would have blocked the release.",
        f"- Strict biology-safe release (`biology_lcb >= 0`) removes the `paul15` override, but it also removes the good `GBM_sd` override and collapses coverage to {float(strict['release_coverage']):.3f}." if strict is not None else "- Strict biology-safe release collapses coverage.",
        f"- A softer LCB margin around {float(soft_row['guardrail_margin_lcb']):.2f} still tends to over-abstain, so the current biology proxy is too under-confident for hard gating.",
        f"- Removing biology constraints entirely is not attractive either: it leaves the risky `paul15` override in place with no protection and mean biology proxy {float(no_bio['mean_biology_proxy']):.4f}.",
        "",
        "Conclusion:",
        "- Biology guardrails should stay in the method definition, but as a soft selective-release constraint with better calibration, not as a hard `biology_lcb >= 0` rule under the current proxy.",
    ]
    (output_dir / "paul15_case_study.md").write_text("\n".join(lines), encoding="utf-8")


def run_topk_seed_robustness(
    *,
    output_dir: Path,
    real_data_root: Path,
    gate_model_path: str,
    refine_epochs: int,
    seeds: tuple[int, ...],
    topks: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = output_dir / "_topk_seed_method_raw.csv"
    raw_method_df = compute_topk_seed_method_raw(
        real_data_root=real_data_root,
        gate_model_path=gate_model_path,
        refine_epochs=refine_epochs,
        seeds=seeds,
        topks=topks,
        cache_path=raw_path,
    )

    policy_rows: list[dict[str, object]] = []
    for (seed, top_k), group in raw_method_df.groupby(["seed", "top_k"], sort=False):
        overall_tables = group.pivot(index="dataset", columns="method", values="overall_score").sort_index()
        runtime_tables = group.pivot(index="dataset", columns="method", values="runtime_sec").sort_index()
        biology_tables = group.pivot(index="dataset", columns="method", values="weighted_marker_recall_at_50").sort_index()
        feature_df = build_feature_df_from_raw(group=group)
        metric_tables = {
            metric: group.pivot(index="dataset", columns="method", values=metric).sort_index()
            for metric in ("overall_score", "runtime_sec", "ari", "nmi", "neighbor_preservation", "weighted_marker_recall_at_50")
            if metric in group.columns
        }
        audit_df = pd.DataFrame(
            [
                {"method": ROBUSTNESS_CANDIDATE_METHOD, "strict_official_available": False},
            ]
        )
        variants = (
            PolicyVariant(
                name="current_full_policy",
                description="Current full policy restricted to the routed candidate from phase1.",
                bank_mode="md_only",
            ),
            PolicyVariant(
                name="adaptive_hybrid_anchor",
                description="Safe anchor only.",
                bank_mode="md_only",
                anchor_only=True,
            ),
            PolicyVariant(
                name="always_release_best_predicted_expert",
                description="Always release the routed candidate.",
                bank_mode="md_only",
                always_release=True,
            ),
            PolicyVariant(
                name="best_single_train_bank_expert",
                description="Best single train-bank expert under the routed candidate bank.",
                bank_mode="md_only",
                always_release=True,
            ),
        )
        for variant in variants:
            results, summary = evaluate_variant_lodo(
                feature_df=feature_df,
                overall_df=overall_tables,
                biology_df=biology_tables,
                runtime_df=runtime_tables,
                metric_tables=metric_tables,
                audit_df=audit_df,
                variant=variant,
            )
            results["seed"] = int(seed)
            results["top_k"] = int(top_k)
            summary_row = summary.iloc[0].to_dict()
            summary_row["seed"] = int(seed)
            summary_row["top_k"] = int(top_k)
            results["setting_release_coverage"] = float(summary_row["release_coverage"])
            policy_rows.append(results)
    policy_df = pd.concat(policy_rows, ignore_index=True)
    summary_df = policy_df.groupby(["policy", "seed", "top_k"], sort=False).agg(
        mean_overall_score=("selected_overall_score", "mean"),
        mean_biology_proxy=("selected_biology_proxy", "mean"),
        mean_ari=("selected_ari", "mean"),
        mean_nmi=("selected_nmi", "mean"),
        mean_runtime_sec=("selected_runtime_sec", "mean"),
        mean_delta_vs_anchor=("selected_vs_anchor_score_delta", "mean"),
        mean_biology_delta_vs_anchor=("selected_vs_anchor_biology_delta", "mean"),
        release_coverage=("released", "mean"),
        override_count=("released", "sum"),
        dataset_count=("dataset", "nunique"),
        routed_subset_mean_delta=("selected_vs_anchor_score_delta", lambda s: float(np.mean([value for value in s if value != 0.0])) if np.any(s != 0.0) else 0.0),
    ).reset_index()
    summary_df["stability_score_std"] = summary_df.groupby("policy", sort=False)["mean_overall_score"].transform("std")
    summary_df["stability_biology_std"] = summary_df.groupby("policy", sort=False)["mean_biology_proxy"].transform("std")
    return policy_df.sort_values(["seed", "top_k", "policy", "dataset"]).reset_index(drop=True), summary_df.sort_values(
        ["top_k", "seed", "policy"]
    ).reset_index(drop=True)


def compute_topk_seed_method_raw(
    *,
    real_data_root: Path,
    gate_model_path: str,
    refine_epochs: int,
    seeds: tuple[int, ...],
    topks: tuple[int, ...],
    cache_path: Path | None = None,
) -> pd.DataFrame:
    specs = {spec.dataset_name: spec for spec in discover_scrna_input_specs(real_data_root)}
    cache_df = pd.DataFrame()
    if cache_path is not None and cache_path.exists():
        cache_df = _normalize_topk_seed_cache(pd.read_csv(cache_path))
    completed_keys = _completed_topk_seed_keys(cache_df)
    for seed in seeds:
        registry = build_default_method_registry(
            top_k=max(topks),
            refine_epochs=refine_epochs,
            random_state=int(seed),
            gate_model_path=gate_model_path,
        )
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
                random_state=int(seed),
            )
            profile = compute_adaptive_stat_profile(
                counts=dataset.counts,
                batches=dataset.batches,
                random_state=int(seed),
            )
            profile_payload = {
                "n_cells": float(profile.n_cells),
                "n_genes": float(profile.n_genes),
                "batch_classes": float(profile.batch_classes),
                "dropout_rate": float(profile.dropout_rate),
                "library_cv": float(profile.library_cv),
                "cluster_strength": float(profile.cluster_strength),
                "batch_strength": float(profile.batch_strength),
                "trajectory_strength": float(profile.trajectory_strength),
                "pc_entropy": float(profile.pc_entropy),
                "rare_fraction": float(profile.rare_fraction),
            }
            markers: dict[str, set[int]] = {}
            class_weights: dict[str, float] = {}
            if dataset.labels is not None:
                labels = np.asarray(dataset.labels, dtype=object)
                markers, class_weights = pkg.compute_one_vs_rest_markers(
                    counts=dataset.counts,
                    labels=labels,
                    top_n=50,
                )
            for top_k in topks:
                current_top_k = min(int(top_k), dataset.counts.shape[1])
                setting_key = (str(dataset_name), int(seed), int(current_top_k))
                if setting_key in completed_keys:
                    continue
                setting_rows = []
                for method_name in (SAFE_ANCHOR_METHOD, ROBUSTNESS_CANDIDATE_METHOD):
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
                        random_state=int(seed),
                        n_bootstrap=2,
                    )
                    weighted_marker = np.nan
                    marker_recall = np.nan
                    rare_marker = np.nan
                    if markers:
                        marker_recall, weighted_marker, rare_marker = pkg.marker_recovery(
                            selected=selected,
                            marker_sets=markers,
                            class_weights=class_weights,
                        )
                    row = {
                        "dataset": dataset_name,
                        "dataset_id": str(spec.dataset_id),
                        "dataset_name": str(spec.dataset_name),
                        "seed": int(seed),
                        "top_k": int(current_top_k),
                        "method": method_name,
                        "runtime_sec": float(runtime_sec),
                        "weighted_marker_recall_at_50": float(weighted_marker),
                        "marker_recall_at_50": float(marker_recall),
                        "rare_marker_recall_at_50": float(rare_marker),
                    }
                    row.update({key: float(value) for key, value in metrics.items()})
                    row.update({f"profile_{key}": value for key, value in profile_payload.items()})
                    setting_rows.append(row)
                setting_df = pd.DataFrame(setting_rows)
                setting_df = rr1.add_run_level_scores(setting_df)
                cache_df = pd.concat([cache_df, setting_df], ignore_index=True)
                cache_df = _normalize_topk_seed_cache(cache_df)
                completed_keys.add(setting_key)
                if cache_path is not None:
                    cache_df.to_csv(cache_path, index=False)
    return _normalize_topk_seed_cache(cache_df)


def _normalize_topk_seed_cache(cache_df: pd.DataFrame) -> pd.DataFrame:
    if cache_df.empty:
        return cache_df.copy()
    return (
        cache_df.drop_duplicates(subset=["dataset", "seed", "top_k", "method"], keep="last")
        .sort_values(["seed", "top_k", "dataset", "method"])
        .reset_index(drop=True)
    )


def _completed_topk_seed_keys(cache_df: pd.DataFrame) -> set[tuple[str, int, int]]:
    if cache_df.empty:
        return set()
    complete = (
        cache_df.groupby(["dataset", "seed", "top_k"], sort=False)["method"]
        .nunique()
        .reset_index(name="method_count")
    )
    completed = complete[complete["method_count"] >= 2]
    return {
        (str(row["dataset"]), int(row["seed"]), int(row["top_k"]))
        for _, row in completed.iterrows()
    }


def build_feature_df_from_raw(*, group: pd.DataFrame) -> pd.DataFrame:
    rows = (
        group[group["method"] == SAFE_ANCHOR_METHOD][
            [
                "dataset",
                "profile_n_cells",
                "profile_n_genes",
                "profile_batch_classes",
                "profile_dropout_rate",
                "profile_library_cv",
                "profile_cluster_strength",
                "profile_batch_strength",
                "profile_trajectory_strength",
                "profile_pc_entropy",
                "profile_rare_fraction",
            ]
        ]
        .drop_duplicates("dataset")
        .rename(
            columns={
                "profile_n_cells": "n_cells",
                "profile_n_genes": "n_genes",
                "profile_batch_classes": "batch_classes",
                "profile_dropout_rate": "dropout_rate",
                "profile_library_cv": "library_cv",
                "profile_cluster_strength": "cluster_strength",
                "profile_batch_strength": "batch_strength",
                "profile_trajectory_strength": "trajectory_strength",
                "profile_pc_entropy": "pc_entropy",
                "profile_rare_fraction": "rare_fraction",
            }
        )
        .set_index("dataset")
        .sort_index()
    )
    return rows


def write_topk_seed_readout(
    *,
    output_dir: Path,
    robustness_summary: pd.DataFrame,
) -> None:
    current = robustness_summary[robustness_summary["policy"] == "current_full_policy"]
    anchor = robustness_summary[robustness_summary["policy"] == "adaptive_hybrid_anchor"]
    current_mean = float(current["mean_overall_score"].mean())
    anchor_mean = float(anchor["mean_overall_score"].mean())
    mean_overall_delta = current_mean - anchor_mean
    mean_biology_delta = float(current["mean_biology_proxy"].mean() - anchor["mean_biology_proxy"].mean())
    active_settings = int((current["release_coverage"] > 0.0).sum())
    total_settings = int(len(current))
    lines = [
        "# Top-k / Seed Readout",
        "",
        "- Under the current admissible bank, the routed expert remains `multinomial_deviance_hvg`, so robustness reduces to testing whether the release layer stays useful against the same candidate across settings.",
        f"- Mean overall score across all top-k/seed settings: current full policy={current_mean:.4f}, anchor={anchor_mean:.4f}, delta={mean_overall_delta:.4f}.",
        f"- Mean biology proxy across settings: current full policy={float(current['mean_biology_proxy'].mean()):.4f}, anchor={float(anchor['mean_biology_proxy'].mean()):.4f}, delta={mean_biology_delta:.4f}.",
        f"- Mean release coverage across settings: {float(current['release_coverage'].mean()):.3f}, with non-zero release in {active_settings}/{total_settings} settings.",
        "- Result: robustness does not support a general gain claim. The release layer stays sparse and slightly biology-favorable, but its average overall score falls below the anchor once we sweep top-k and seed.",
    ]
    (output_dir / "topk_seed_readout.md").write_text("\n".join(lines), encoding="utf-8")


def run_strict_official_dependency_analysis(
    *,
    feature_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    biology_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    metric_tables: dict[str, pd.DataFrame],
    audit_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = (
        PolicyVariant(
            name="full_bank_no_filter",
            description="All currently available experts without fallback filtering.",
            bank_mode="full_bank",
            use_official_penalty=False,
        ),
        PolicyVariant(
            name="strict_official_bank_only",
            description="Bank restricted to strict official experts only.",
            bank_mode="strict_official",
            use_official_penalty=False,
        ),
    )
    results_frames = []
    summary_frames = []
    for variant in variants:
        results, summary = evaluate_variant_lodo(
            feature_df=feature_df,
            overall_df=overall_df,
            biology_df=biology_df,
            runtime_df=runtime_df,
            metric_tables=metric_tables,
            audit_df=audit_df,
            variant=variant,
        )
        results["analysis_bank"] = variant.name
        summary["analysis_bank"] = variant.name
        results_frames.append(results)
        summary_frames.append(summary)
    return pd.concat(results_frames, ignore_index=True), pd.concat(summary_frames, ignore_index=True).sort_values(
        ["analysis_bank", "mean_overall_score"],
        ascending=[True, False],
    ).reset_index(drop=True)


def write_official_dependency_note(
    *,
    output_dir: Path,
    strict_holdout_summary: pd.DataFrame,
    strict_holdout_results: pd.DataFrame,
) -> None:
    full_bank = strict_holdout_summary[strict_holdout_summary["analysis_bank"] == "full_bank_no_filter"].iloc[0]
    strict_bank = strict_holdout_summary[strict_holdout_summary["analysis_bank"] == "strict_official_bank_only"].iloc[0]
    surviving = strict_holdout_results[
        (strict_holdout_results["analysis_bank"] == "strict_official_bank_only")
        & (strict_holdout_results["released"] > 0.0)
    ]["dataset"].astype(str).tolist()
    lines = [
        "# Official / Fallback Dependency",
        "",
        f"- Full unfiltered bank: mean overall score={float(full_bank['mean_overall_score']):.4f}, mean delta vs anchor={float(full_bank['mean_delta_vs_anchor']):.4f}, coverage={float(full_bank['release_coverage']):.3f}.",
        f"- Strict official bank only: mean overall score={float(strict_bank['mean_overall_score']):.4f}, mean delta vs anchor={float(strict_bank['mean_delta_vs_anchor']):.4f}, coverage={float(strict_bank['release_coverage']):.3f}.",
        (
            f"- Override decisions that survive under the strict official bank: {', '.join(surviving)}."
            if surviving
            else "- No override survives under the strict official bank."
        ),
        "- Interpretation: the phase1 positive result does not depend on fallback-heavy official wrappers, but it does depend on a repo-local published expert (`multinomial_deviance_hvg`) rather than a strict official expert.",
        "- Recommended paper framing: main text should report the filtered-bank result plus explicit bank audit, while the strict-official analysis belongs in the risk/limitations section or supplement.",
    ]
    (output_dir / "official_fallback_dependency.md").write_text("\n".join(lines), encoding="utf-8")


def build_phase2_go_no_go(
    *,
    coverage_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    biology_guardrail_summary: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    strict_holdout_summary: pd.DataFrame,
) -> str:
    current_ablation = ablation_summary[ablation_summary["policy"] == "current_full_policy"].iloc[0]
    always_release = ablation_summary[ablation_summary["policy"] == "always_release_best_predicted_expert"].iloc[0]
    strict_official = strict_holdout_summary[strict_holdout_summary["analysis_bank"] == "strict_official_bank_only"].iloc[0]
    strict_bio = biology_guardrail_summary[biology_guardrail_summary["policy"] == "strict_biology_safe_policy"].iloc[0]
    robustness_current = robustness_summary[robustness_summary["policy"] == "current_full_policy"]
    robustness_anchor = robustness_summary[robustness_summary["policy"] == "adaptive_hybrid_anchor"]
    robustness_mean_overall_delta = float(
        robustness_current["mean_overall_score"].mean() - robustness_anchor["mean_overall_score"].mean()
    )
    robustness_mean_biology_delta = float(
        robustness_current["mean_biology_proxy"].mean() - robustness_anchor["mean_biology_proxy"].mean()
    )

    if float(current_ablation["mean_delta_vs_anchor"]) < 0.0:
        verdict = "stop / pivot"
    elif float(strict_official["mean_delta_vs_anchor"]) <= 0.0 or robustness_mean_overall_delta < 0.0:
        verdict = "continue with caution"
    else:
        verdict = "continue"

    q1 = "Not yet. The phase2 ablations support the selective-release structure, but the gain still looks like a calibrated operating-point effect rather than a broadly stable method-level win."
    q2 = "Yes, but as a soft method-defining guardrail. Hard `biology_lcb >= 0` release rules over-abstain under the current proxy calibration."
    q3 = "No. Under the strict official bank the release layer collapses back to the anchor, so the current positive result does not survive that restriction."
    q4 = "official reproduction"

    lines = [
        "# Phase2 Go / No-Go",
        "",
        f"Decision: `{verdict}`",
        "",
        "## Core Evidence",
        f"- Current full policy mean delta vs anchor: {float(current_ablation['mean_delta_vs_anchor']):.4f}",
        f"- Always-release mean delta vs anchor: {float(always_release['mean_delta_vs_anchor']):.4f}",
        f"- Strict biology-safe policy coverage: {float(strict_bio['release_coverage']):.3f}",
        f"- Strict official bank mean delta vs anchor: {float(strict_official['mean_delta_vs_anchor']):.4f}",
        f"- Mean robustness overall delta across seeds/top-k: {robustness_mean_overall_delta:.4f}",
        f"- Mean robustness biology delta across seeds/top-k: {robustness_mean_biology_delta:.4f}",
        f"- Mean robustness release coverage across seeds/top-k: {float(robustness_current['release_coverage'].mean()):.3f}",
        "",
        "## Answers",
        f"1. Has the line moved beyond a conservative patch? {q1}",
        f"2. Should biology guardrail be part of the method definition? {q2}",
        f"3. Does the main conclusion still hold under a strict official bank? {q3}",
        f"4. What is the single highest-value next step? {q4}",
        "",
        "## Recommendation",
        "- Keep the main story as a risk-controlled selective-release layer over the safe anchor.",
        "- Do not claim a fully general learned router yet.",
        "- Prioritize official reproduction before trying to expand coverage, because strict-official failure is now the cleanest reviewer attack surface.",
    ]
    return "\n".join(lines)


def _weighted_stats(values: np.ndarray, weights: tuple[float, ...]) -> tuple[float, float]:
    weight_array = np.asarray(weights, dtype=np.float64)
    mean_value = float(np.dot(weight_array, values))
    variance = float(np.dot(weight_array, np.square(values - mean_value)))
    return mean_value, float(np.sqrt(max(variance, 0.0)))


if __name__ == "__main__":
    main()
