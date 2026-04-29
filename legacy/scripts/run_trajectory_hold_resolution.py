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

import build_topconf_selector_round2_package as pkg
import run_real_inputs_round1 as rr1
import run_regime_specific_route as base
import run_trajectory_route_coherence_audit as traj


DEFAULT_OUTPUT_DIR = ROOT / "artifacts_route_hold_resolution_trajectory"
ROBUSTNESS_SEEDS = (7, 11, 23, 37, 47)
ROBUSTNESS_TOPKS = (100, 200, 500)
AUDIT_DATASETS = (
    *traj.CORE_MODELING_DATASETS,
    *traj.PROTECTED_CONTROL_DATASETS,
    *traj.HELDOUT_VALIDATION_DATASETS,
)
PROTECTED_CONTROL_COUNT = len(traj.PROTECTED_CONTROL_DATASETS)
CORE_TARGET_COUNT = len(traj.CORE_MODELING_DATASETS)
SETTING_ROW_COUNT = len(AUDIT_DATASETS) * len(traj.ROUTE_METHODS)
SEPARABILITY_FEATURES = (
    "headroom_vs_best_single",
    "stat_trajectory_strength",
    "stat_dropout_rate",
    "stat_cluster_strength",
    "stat_pc_entropy",
    "stat_batch_classes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve whether the trajectory/locality route should stay on hold or downgrade to no-go.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--real-data-root", type=str, default=str(ROOT / "data" / "real_inputs"))
    parser.add_argument("--refine-epochs", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ROBUSTNESS_SEEDS))
    parser.add_argument("--top-ks", type=int, nargs="+", default=list(ROBUSTNESS_TOPKS))
    return parser.parse_args()


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def load_route_split() -> pd.DataFrame:
    path = ROOT / "artifacts_route_coherence_audit_trajectory" / "route_dataset_split.csv"
    return pd.read_csv(path)


def load_headroom_table() -> pd.DataFrame:
    return base.load_headroom_table()


def load_dataset_cache(real_data_root: Path) -> pkg.DatasetCache:
    resources = base.load_dataset_resources(real_data_root)
    return pkg.DatasetCache(resources)


def build_analysis_rows(
    *,
    dataset_cache: pkg.DatasetCache,
    headroom_df: pd.DataFrame,
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
    for dataset_name in AUDIT_DATASETS:
        dataset = dataset_cache.get(dataset_name, seed)
        current_top_k = min(int(top_k), int(dataset.counts.shape[1]))
        best_single_method = str(headroom_lookup.loc[dataset_name, "best_single_expert"])
        methods_to_compute = tuple(dict.fromkeys([traj.ANCHOR_METHOD, best_single_method, *traj.ROUTE_METHODS]))
        score_cache: dict[str, np.ndarray] = {}
        topk_cache: dict[str, np.ndarray] = {}
        for method_name in methods_to_compute:
            score_cache[method_name] = np.asarray(
                registry[method_name](dataset.counts, dataset.batches, current_top_k),
                dtype=np.float64,
            )
            topk_cache[method_name] = base.topk_indices(score_cache[method_name], current_top_k)

        anchor_scores = score_cache[traj.ANCHOR_METHOD]
        anchor_topk = topk_cache[traj.ANCHOR_METHOD]
        best_scores = score_cache[best_single_method]
        best_topk = topk_cache[best_single_method]
        anchor_overlap_to_best = base.jaccard(anchor_topk, best_topk)
        anchor_corr_to_best = base.spearman_correlation(anchor_scores, best_scores)

        if dataset_name in traj.CORE_MODELING_DATASETS:
            group_name = "core_modeling"
        elif dataset_name in traj.PROTECTED_CONTROL_DATASETS:
            group_name = "protected_control"
        else:
            group_name = "heldout_validation_only"

        for method_name in traj.ROUTE_METHODS:
            candidate_scores = score_cache[method_name]
            candidate_topk = topk_cache[method_name]
            rows.append(
                {
                    "seed": int(seed),
                    "top_k": int(current_top_k),
                    "dataset": dataset_name,
                    "group_name": group_name,
                    "method": method_name,
                    "route_family_label": traj.ROUTE_FAMILY_LABELS[method_name],
                    "best_single_method": best_single_method,
                    "rank_corr_to_anchor": base.spearman_correlation(candidate_scores, anchor_scores),
                    "topk_overlap_to_anchor": base.jaccard(candidate_topk, anchor_topk),
                    "topk_shift_vs_anchor": 1.0 - base.jaccard(candidate_topk, anchor_topk),
                    "topk_overlap_to_best_single": base.jaccard(candidate_topk, best_topk),
                    "delta_overlap_to_best_single_vs_anchor": base.jaccard(candidate_topk, best_topk) - anchor_overlap_to_best,
                    "rank_corr_to_best_single": base.spearman_correlation(candidate_scores, best_scores),
                    "delta_rank_corr_to_best_single_vs_anchor": base.spearman_correlation(candidate_scores, best_scores) - anchor_corr_to_best,
                    "score_dispersion_ratio_vs_anchor": float(
                        np.std(candidate_scores) / max(np.std(anchor_scores), 1e-8)
                    ),
                }
            )
    df = pd.DataFrame(rows)
    df["targeted_shift_gap_contribution"] = np.where(
        df["group_name"] == "core_modeling",
        df["topk_shift_vs_anchor"] / max(CORE_TARGET_COUNT, 1),
        np.where(
            df["group_name"] == "protected_control",
            -df["topk_shift_vs_anchor"] / max(PROTECTED_CONTROL_COUNT, 1),
            0.0,
        ),
    )
    df["winner_overlap_pull_positive_contribution"] = np.where(
        df["group_name"] == "core_modeling",
        df["delta_overlap_to_best_single_vs_anchor"] / max(CORE_TARGET_COUNT, 1),
        0.0,
    )
    df["winner_overlap_pull_gap_contribution"] = np.where(
        df["group_name"] == "core_modeling",
        df["delta_overlap_to_best_single_vs_anchor"] / max(CORE_TARGET_COUNT, 1),
        np.where(
            df["group_name"] == "protected_control",
            -df["delta_overlap_to_best_single_vs_anchor"] / max(PROTECTED_CONTROL_COUNT, 1),
            0.0,
        ),
    )
    df["winner_corr_pull_gap_contribution"] = np.where(
        df["group_name"] == "core_modeling",
        df["delta_rank_corr_to_best_single_vs_anchor"] / max(CORE_TARGET_COUNT, 1),
        np.where(
            df["group_name"] == "protected_control",
            -df["delta_rank_corr_to_best_single_vs_anchor"] / max(PROTECTED_CONTROL_COUNT, 1),
            0.0,
        ),
    )
    df["control_guard_contribution"] = np.where(
        df["group_name"] == "protected_control",
        df["delta_overlap_to_best_single_vs_anchor"] / max(PROTECTED_CONTROL_COUNT, 1),
        0.0,
    )
    df["heldout_overlap_pull_contribution"] = np.where(
        df["group_name"] == "heldout_validation_only",
        df["delta_overlap_to_best_single_vs_anchor"],
        0.0,
    )
    df["control_damage_flag"] = np.where(
        (df["group_name"] == "protected_control") & (df["delta_overlap_to_best_single_vs_anchor"] < -0.02),
        1,
        0,
    )
    df["control_severe_damage_flag"] = np.where(
        (df["group_name"] == "protected_control") & (df["delta_overlap_to_best_single_vs_anchor"] < -0.10),
        1,
        0,
    )
    df["heldout_nonnegative_flag"] = np.where(
        (df["group_name"] == "heldout_validation_only") & (df["delta_overlap_to_best_single_vs_anchor"] >= 0.0),
        1,
        0,
    )
    return df


def build_robustness_grid(
    *,
    output_dir: Path,
    dataset_cache: pkg.DatasetCache,
    headroom_df: pd.DataFrame,
    refine_epochs: int,
    seeds: tuple[int, ...],
    top_ks: tuple[int, ...],
) -> pd.DataFrame:
    output_path = output_dir / "robustness_grid_by_dataset.csv"
    if output_path.exists():
        df = pd.read_csv(output_path)
    else:
        df = pd.DataFrame()

    for top_k in top_ks:
        for seed in seeds:
            key_mask = (df.get("seed", pd.Series(dtype=int)) == seed) & (df.get("top_k", pd.Series(dtype=int)) == top_k)
            existing_rows = int(key_mask.sum()) if not df.empty else 0
            if existing_rows == SETTING_ROW_COUNT:
                continue
            if not df.empty:
                df = df.loc[~key_mask].copy()
            setting_df = build_analysis_rows(
                dataset_cache=dataset_cache,
                headroom_df=headroom_df,
                seed=seed,
                top_k=top_k,
                refine_epochs=refine_epochs,
            )
            df = pd.concat([df, setting_df], ignore_index=True, sort=False)
            df = df.sort_values(["method", "top_k", "seed", "group_name", "dataset"]).reset_index(drop=True)
            df.to_csv(output_path, index=False)
    return df


def summarize_robustness(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method_name, seed, top_k), group in df.groupby(["method", "seed", "top_k"], sort=True):
        core_df = group[group["group_name"] == "core_modeling"].copy()
        control_df = group[group["group_name"] == "protected_control"].copy()
        heldout_df = group[group["group_name"] == "heldout_validation_only"].copy()
        targeted_shift_gap = base.mean_or_nan(core_df["topk_shift_vs_anchor"]) - base.mean_or_nan(control_df["topk_shift_vs_anchor"])
        winner_overlap_pull_positive = base.mean_or_nan(core_df["delta_overlap_to_best_single_vs_anchor"])
        control_guard = base.mean_or_nan(control_df["delta_overlap_to_best_single_vs_anchor"])
        winner_overlap_pull_gap = winner_overlap_pull_positive - control_guard
        winner_corr_pull_gap = base.mean_or_nan(core_df["delta_rank_corr_to_best_single_vs_anchor"]) - base.mean_or_nan(
            control_df["delta_rank_corr_to_best_single_vs_anchor"]
        )
        heldout_overlap_pull = base.mean_or_nan(heldout_df["delta_overlap_to_best_single_vs_anchor"])
        heldout_corr_pull = base.mean_or_nan(heldout_df["delta_rank_corr_to_best_single_vs_anchor"])
        conditions = {
            "targeted_shift_gap": targeted_shift_gap >= 0.03,
            "winner_overlap_pull_positive": winner_overlap_pull_positive > 0.0,
            "winner_overlap_pull_gap": winner_overlap_pull_gap >= 0.015,
            "winner_corr_pull_gap": winner_corr_pull_gap >= 0.01,
            "control_guard": control_guard >= -0.02,
            "heldout_overlap_pull": heldout_overlap_pull >= 0.0,
        }
        rows.append(
            {
                "method": method_name,
                "seed": int(seed),
                "top_k": int(top_k),
                "core_positive_overlap_dataset_count": int((core_df["delta_overlap_to_best_single_vs_anchor"] > 0.0).sum()),
                "protected_control_damage_dataset_count": int((control_df["delta_overlap_to_best_single_vs_anchor"] < -0.02).sum()),
                "targeted_shift_gap": float(targeted_shift_gap),
                "winner_overlap_pull_positive": float(winner_overlap_pull_positive),
                "winner_overlap_pull_gap": float(winner_overlap_pull_gap),
                "winner_corr_pull_gap": float(winner_corr_pull_gap),
                "control_guard": float(control_guard),
                "heldout_overlap_pull": float(heldout_overlap_pull),
                "heldout_corr_pull": float(heldout_corr_pull),
                "analysis_signal_present": bool(
                    conditions["targeted_shift_gap"]
                    and conditions["winner_overlap_pull_positive"]
                    and conditions["winner_overlap_pull_gap"]
                    and conditions["winner_corr_pull_gap"]
                ),
                "strict_unlock_ready": bool(all(conditions.values())),
                "condition_targeted_shift_gap": bool(conditions["targeted_shift_gap"]),
                "condition_winner_overlap_pull_positive": bool(conditions["winner_overlap_pull_positive"]),
                "condition_winner_overlap_pull_gap": bool(conditions["winner_overlap_pull_gap"]),
                "condition_winner_corr_pull_gap": bool(conditions["winner_corr_pull_gap"]),
                "condition_control_guard": bool(conditions["control_guard"]),
                "condition_heldout_overlap_pull": bool(conditions["heldout_overlap_pull"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "top_k", "seed"]).reset_index(drop=True)


def build_control_damage_views(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    control_df = df[df["group_name"] == "protected_control"].copy()
    if control_df.empty:
        return control_df, pd.DataFrame()

    control_df["abs_guard_contribution"] = control_df["control_guard_contribution"].abs()
    control_df["setting_abs_guard_total"] = control_df.groupby(["method", "seed", "top_k"])["abs_guard_contribution"].transform("sum")
    control_df["setting_abs_guard_share"] = np.where(
        control_df["setting_abs_guard_total"] > 0.0,
        control_df["abs_guard_contribution"] / control_df["setting_abs_guard_total"],
        0.0,
    )
    control_df["setting_damage_rank"] = (
        control_df.groupby(["method", "seed", "top_k"])["abs_guard_contribution"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    method_dataset_rows: list[dict[str, object]] = []
    for (method_name, dataset_name), group in control_df.groupby(["method", "dataset"], sort=True):
        method_dataset_rows.append(
            {
                "row_type": "method_dataset",
                "method": method_name,
                "dataset": dataset_name,
                "setting_count": int(len(group)),
                "mean_overlap_pull": float(group["delta_overlap_to_best_single_vs_anchor"].mean()),
                "median_overlap_pull": float(group["delta_overlap_to_best_single_vs_anchor"].median()),
                "mean_guard_contribution": float(group["control_guard_contribution"].mean()),
                "mean_abs_guard_share": float(group["setting_abs_guard_share"].mean()),
                "lead_damage_count": int((group["setting_damage_rank"] == 1).sum()),
                "fail_count": int((group["delta_overlap_to_best_single_vs_anchor"] < -0.02).sum()),
                "severe_fail_count": int((group["delta_overlap_to_best_single_vs_anchor"] < -0.10).sum()),
                "majority_fail": bool((group["delta_overlap_to_best_single_vs_anchor"] < -0.02).sum() > len(group) / 2.0),
            }
        )

    method_overall_rows: list[dict[str, object]] = []
    for method_name, group in control_df.groupby("method", sort=True):
        setting_fail = group.groupby(["seed", "top_k"])["delta_overlap_to_best_single_vs_anchor"].apply(lambda s: int((s < -0.02).sum()))
        setting_share = group.groupby(["seed", "top_k"])["setting_abs_guard_share"].max()
        leading_dataset = (
            group[group["setting_damage_rank"] == 1]["dataset"]
            .astype(str)
            .value_counts()
            .index.tolist()
        )
        method_overall_rows.append(
            {
                "row_type": "method_overall",
                "method": method_name,
                "dataset": "",
                "setting_count": int(setting_fail.shape[0]),
                "mean_overlap_pull": float(group["delta_overlap_to_best_single_vs_anchor"].mean()),
                "median_overlap_pull": float(group["delta_overlap_to_best_single_vs_anchor"].median()),
                "mean_guard_contribution": float(group["control_guard_contribution"].mean()),
                "mean_abs_guard_share": float(setting_share.mean()),
                "lead_damage_count": int((group["setting_damage_rank"] == 1).sum()),
                "fail_count": int((group["delta_overlap_to_best_single_vs_anchor"] < -0.02).sum()),
                "severe_fail_count": int((group["delta_overlap_to_best_single_vs_anchor"] < -0.10).sum()),
                "settings_any_control_fail": int((setting_fail >= 1).sum()),
                "settings_all_controls_fail": int((setting_fail == PROTECTED_CONTROL_COUNT).sum()),
                "settings_systemic_fail": int((setting_fail >= 2).sum()),
                "dominant_control_dataset": leading_dataset[0] if leading_dataset else "",
                "majority_fail": bool((setting_fail == PROTECTED_CONTROL_COUNT).sum() > len(setting_fail) / 2.0),
            }
        )

    summary_df = pd.concat(
        [
            pd.DataFrame(method_dataset_rows),
            pd.DataFrame(method_overall_rows),
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(["row_type", "method", "dataset"]).reset_index(drop=True)
    return control_df, summary_df


def build_heldout_validity_summary(
    *,
    robustness_by_dataset_df: pd.DataFrame,
    route_split_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    heldout_df = robustness_by_dataset_df[robustness_by_dataset_df["group_name"] == "heldout_validation_only"].copy()
    heldout_split = route_split_df[route_split_df["role"] == "heldout_validation_only"].copy().iloc[0]
    rows: list[dict[str, object]] = []
    for method_name, group in heldout_df.groupby("method", sort=True):
        rows.append(
            {
                "row_type": "method_summary",
                "method": method_name,
                "heldout_dataset": str(heldout_split["dataset"]),
                "best_single_method": str(heldout_split["best_single_expert"]),
                "best_single_in_route_family": bool(heldout_split["route_family_best_single"]),
                "headroom_vs_best_single": float(heldout_split["headroom_vs_best_single"]),
                "setting_count": int(len(group)),
                "nonnegative_overlap_count": int((group["delta_overlap_to_best_single_vs_anchor"] >= 0.0).sum()),
                "positive_overlap_count": int((group["delta_overlap_to_best_single_vs_anchor"] > 0.0).sum()),
                "mean_overlap_pull": float(group["delta_overlap_to_best_single_vs_anchor"].mean()),
                "median_overlap_pull": float(group["delta_overlap_to_best_single_vs_anchor"].median()),
                "min_overlap_pull": float(group["delta_overlap_to_best_single_vs_anchor"].min()),
                "max_overlap_pull": float(group["delta_overlap_to_best_single_vs_anchor"].max()),
                "mean_corr_pull": float(group["delta_rank_corr_to_best_single_vs_anchor"].mean()),
                "all_negative_overlap": bool((group["delta_overlap_to_best_single_vs_anchor"] < 0.0).all()),
                "majority_nonnegative": bool((group["delta_overlap_to_best_single_vs_anchor"] >= 0.0).sum() > len(group) / 2.0),
            }
        )

    method_summary_df = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    stable_positive_method_count = int((method_summary_df["majority_nonnegative"] == True).sum())  # noqa: E712
    headroom_ok = float(heldout_split["headroom_vs_best_single"]) >= 0.10
    best_single_in_family = bool(heldout_split["route_family_best_single"])
    should_reclassify = (not best_single_in_family) and (not headroom_ok) and stable_positive_method_count == 0
    global_row = pd.DataFrame(
        [
            {
                "row_type": "global_decision",
                "method": "",
                "heldout_dataset": str(heldout_split["dataset"]),
                "best_single_method": str(heldout_split["best_single_expert"]),
                "best_single_in_route_family": best_single_in_family,
                "headroom_vs_best_single": float(heldout_split["headroom_vs_best_single"]),
                "setting_count": int(heldout_df[["seed", "top_k"]].drop_duplicates().shape[0]),
                "nonnegative_overlap_count": int((heldout_df["delta_overlap_to_best_single_vs_anchor"] >= 0.0).sum()),
                "positive_overlap_count": int((heldout_df["delta_overlap_to_best_single_vs_anchor"] > 0.0).sum()),
                "mean_overlap_pull": float(heldout_df["delta_overlap_to_best_single_vs_anchor"].mean()),
                "median_overlap_pull": float(heldout_df["delta_overlap_to_best_single_vs_anchor"].median()),
                "min_overlap_pull": float(heldout_df["delta_overlap_to_best_single_vs_anchor"].min()),
                "max_overlap_pull": float(heldout_df["delta_overlap_to_best_single_vs_anchor"].max()),
                "mean_corr_pull": float(heldout_df["delta_rank_corr_to_best_single_vs_anchor"].mean()),
                "all_negative_overlap": bool((heldout_df["delta_overlap_to_best_single_vs_anchor"] < 0.0).all()),
                "majority_nonnegative": bool((heldout_df["delta_overlap_to_best_single_vs_anchor"] >= 0.0).sum() > len(heldout_df) / 2.0),
                "stable_positive_method_count": stable_positive_method_count,
                "should_reclassify_to_route_breaking_counterexample": should_reclassify,
            }
        ]
    )
    summary_df = pd.concat([method_summary_df, global_row], ignore_index=True, sort=False)

    if should_reclassify:
        decision = "route_breaking_counterexample"
        rationale = [
            f"- `{heldout_split['dataset']}` should be reclassified from held-out validation to route-breaking counterexample.",
            f"- Its best single expert is `{heldout_split['best_single_expert']}`, which is outside the declared route family.",
            f"- Held-out headroom is only {fmt_float(heldout_split['headroom_vs_best_single'])}, below the minimal stability floor used in the unlock rule.",
            "- Across the full robustness sweep, no audited route-family method produced majority non-negative held-out overlap pull.",
            f"- Aggregate held-out overlap remains negative at {fmt_float(heldout_df['delta_overlap_to_best_single_vs_anchor'].mean())}.",
        ]
    else:
        decision = "keep_as_heldout_validation"
        rationale = [
            f"- `{heldout_split['dataset']}` remains a held-out validation candidate for now.",
        ]

    lines = [
        "# Held-Out Validity Audit",
        "",
        "## Decision",
        f"- `{decision}`",
        "",
        "## Evidence",
        *rationale,
    ]
    return summary_df, "\n".join(lines) + "\n"


def train_linear_probe(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    device: torch.device,
    epochs: int = 800,
    lr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    x_train_t = torch.from_numpy(x_train.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available())
    y_train_t = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1)).to(device, non_blocking=torch.cuda.is_available())
    x_eval_t = torch.from_numpy(x_eval.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available())

    model = torch.nn.Linear(x_train.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    pos_count = float(max(y_train.sum(), 1.0))
    neg_count = float(max((1.0 - y_train).sum(), 1.0))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_count / pos_count], device=device))

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        eval_probs = torch.sigmoid(model(x_eval_t)).detach().to("cpu").numpy().reshape(-1)
        coef = model.weight.detach().to("cpu").numpy().reshape(-1)
    return eval_probs, coef


def build_route_separability_summary(route_split_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    analysis_df = route_split_df[route_split_df["role"].isin(["core_modeling", "protected_control", "heldout_validation_only"])].copy()
    numeric_df = analysis_df.loc[:, SEPARABILITY_FEATURES].copy().fillna(0.0)
    means = numeric_df.mean(axis=0)
    stds = numeric_df.std(axis=0, ddof=0).replace(0.0, 1.0)
    z_df = (numeric_df - means) / stds
    analysis_df.loc[:, [f"z_{name}" for name in SEPARABILITY_FEATURES]] = z_df.to_numpy(dtype=np.float64)

    target_df = analysis_df[analysis_df["role"] == "core_modeling"].copy()
    control_df = analysis_df[analysis_df["role"] == "protected_control"].copy()
    heldout_df = analysis_df[analysis_df["role"] == "heldout_validation_only"].copy()

    rows: list[dict[str, object]] = []
    for feature_name in SEPARABILITY_FEATURES:
        target_values = target_df[feature_name].astype(float)
        control_values = control_df[feature_name].astype(float)
        heldout_value = float(heldout_df.iloc[0][feature_name])
        pooled_std = float(
            np.sqrt(
                (
                    np.var(target_values.to_numpy(dtype=np.float64), ddof=0)
                    + np.var(control_values.to_numpy(dtype=np.float64), ddof=0)
                )
                / 2.0
            )
        )
        if pooled_std <= 1e-8:
            pooled_std = 1.0
        rows.append(
            {
                "row_type": "feature_gap",
                "feature": feature_name,
                "target_mean": float(target_values.mean()),
                "control_mean": float(control_values.mean()),
                "heldout_value": heldout_value,
                "target_minus_control": float(target_values.mean() - control_values.mean()),
                "standardized_gap": float((target_values.mean() - control_values.mean()) / pooled_std),
            }
        )

    class_df = analysis_df[analysis_df["role"].isin(["core_modeling", "protected_control"])].copy().reset_index(drop=True)
    z_cols = [f"z_{name}" for name in SEPARABILITY_FEATURES]
    x = class_df[z_cols].to_numpy(dtype=np.float64)
    y = np.where(class_df["role"] == "core_modeling", 1.0, 0.0).astype(np.float32)

    dist_matrix = np.sqrt(np.maximum(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2), 0.0))
    within_target_dist = dist_matrix[np.ix_(class_df["role"] == "core_modeling", class_df["role"] == "core_modeling")]
    within_control_dist = dist_matrix[np.ix_(class_df["role"] == "protected_control", class_df["role"] == "protected_control")]
    cross_dist = dist_matrix[np.ix_(class_df["role"] == "core_modeling", class_df["role"] == "protected_control")]
    within_target_vals = within_target_dist[np.triu_indices(within_target_dist.shape[0], k=1)]
    within_control_vals = within_control_dist[np.triu_indices(within_control_dist.shape[0], k=1)]
    within_vals = np.concatenate([within_target_vals, within_control_vals])

    rows.append(
        {
            "row_type": "pairwise_distance_summary",
            "feature": "z_profile_space",
            "target_mean": float(within_target_vals.mean()) if within_target_vals.size else float("nan"),
            "control_mean": float(within_control_vals.mean()) if within_control_vals.size else float("nan"),
            "heldout_value": float("nan"),
            "target_minus_control": float(cross_dist.mean() - within_vals.mean()),
            "standardized_gap": float(cross_dist.min() - within_vals.max()),
        }
    )

    for idx, row in class_df.iterrows():
        same_mask = class_df["role"] == row["role"]
        opp_mask = ~same_mask
        same_dist = dist_matrix[idx, same_mask.to_numpy()]
        same_dist = same_dist[same_dist > 0.0]
        opp_dist = dist_matrix[idx, opp_mask.to_numpy()]
        rows.append(
            {
                "row_type": "nearest_margin",
                "feature": str(row["dataset"]),
                "target_mean": float(same_dist.min()) if same_dist.size else float("nan"),
                "control_mean": float(opp_dist.min()) if opp_dist.size else float("nan"),
                "heldout_value": float("nan"),
                "target_minus_control": float(opp_dist.min() - same_dist.min()) if same_dist.size and opp_dist.size else float("nan"),
                "standardized_gap": float(y[idx]),
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loo_rows: list[dict[str, object]] = []
    correct = 0
    for holdout_idx in range(x.shape[0]):
        train_mask = np.ones(x.shape[0], dtype=bool)
        train_mask[holdout_idx] = False
        probs, coef = train_linear_probe(
            x_train=x[train_mask],
            y_train=y[train_mask],
            x_eval=x[~train_mask],
            device=device,
        )
        pred = int(probs[0] >= 0.5)
        truth = int(y[holdout_idx])
        correct += int(pred == truth)
        loo_rows.append(
            {
                "row_type": "leave_one_out",
                "feature": str(class_df.iloc[holdout_idx]["dataset"]),
                "target_mean": float(probs[0]),
                "control_mean": float(truth),
                "heldout_value": float(pred),
                "target_minus_control": float(probs[0] - truth),
                "standardized_gap": float(pred == truth),
            }
        )

    heldout_probs, all_coef = train_linear_probe(
        x_train=x,
        y_train=y,
        x_eval=heldout_df[z_cols].to_numpy(dtype=np.float64),
        device=device,
    )
    rows.append(
        {
            "row_type": "linear_probe_summary",
            "feature": "target_vs_control_linear_probe",
            "target_mean": float(correct / max(x.shape[0], 1)),
            "control_mean": float(heldout_probs[0]),
            "heldout_value": float("nan"),
            "target_minus_control": float(cross_dist.mean()),
            "standardized_gap": float(cross_dist.min() - within_vals.max()),
            "device_name": str(device),
        }
    )
    for feature_name, coef_value in zip(SEPARABILITY_FEATURES, all_coef.tolist()):
        rows.append(
            {
                "row_type": "linear_probe_coef",
                "feature": feature_name,
                "target_mean": float(coef_value),
                "control_mean": float("nan"),
                "heldout_value": float("nan"),
                "target_minus_control": float("nan"),
                "standardized_gap": float("nan"),
            }
        )

    summary_df = pd.DataFrame(rows)
    feature_gap_df = summary_df[summary_df["row_type"] == "feature_gap"].copy()
    pairwise_row = summary_df[summary_df["row_type"] == "pairwise_distance_summary"].iloc[0]
    probe_row = summary_df[summary_df["row_type"] == "linear_probe_summary"].iloc[0]
    strongest_gap = feature_gap_df.reindex(feature_gap_df["standardized_gap"].abs().sort_values(ascending=False).index).iloc[0]

    if float(probe_row["target_mean"]) >= 0.80 and float(pairwise_row["standardized_gap"]) > 0.0:
        separability_label = "moderate_but_not_clean"
    elif float(probe_row["target_mean"]) >= 0.60:
        separability_label = "weak"
    else:
        separability_label = "very_weak"

    lines = [
        "# Route Separability Audit",
        "",
        "## Direct Answer",
        f"- The current benchmark-level trajectory/locality route separability is `{separability_label}` rather than clean.",
        "",
        "## Feature Gap Summary",
        f"- Largest standardized target-vs-control gap is `{strongest_gap['feature']}` with standardized_gap={fmt_float(strongest_gap['standardized_gap'])}.",
        f"- Pairwise cross-vs-within margin is {fmt_float(pairwise_row['standardized_gap'])}; values at or below zero mean the classes overlap in profile space.",
        "",
        "## Linear Separability",
        f"- Leave-one-out linear probe accuracy is {fmt_float(probe_row['target_mean'])} on core targets vs protected controls.",
        f"- When trained on the target/control split, the held-out dataset gets target-like probability {fmt_float(probe_row['control_mean'])}.",
        "- This is a diagnostic only; it is not a new model claim.",
        "",
        "## Interpretation",
        "- If pairwise margin is non-positive or linear accuracy is only modest, then safe unlock is structurally difficult because the protected controls are too close to the target side in existing profile space.",
    ]
    return summary_df, "\n".join(lines) + "\n"


def decide_final_resolution(
    *,
    robustness_summary_df: pd.DataFrame,
    control_damage_summary_df: pd.DataFrame,
    heldout_summary_df: pd.DataFrame,
    separability_summary_df: pd.DataFrame,
) -> tuple[str, str]:
    spectral_df = robustness_summary_df[robustness_summary_df["method"] == "adaptive_spectral_locality_hvg"].copy()
    spectral_positive_count = int((spectral_df["winner_overlap_pull_positive"] > 0.0).sum())
    spectral_total = int(len(spectral_df))
    spectral_control_safe_count = int((spectral_df["control_guard"] >= -0.02).sum())
    spectral_heldout_nonnegative_count = int((spectral_df["heldout_overlap_pull"] >= 0.0).sum())

    control_overall = control_damage_summary_df[
        (control_damage_summary_df["row_type"] == "method_overall")
        & (control_damage_summary_df["method"] == "adaptive_spectral_locality_hvg")
    ].iloc[0]
    heldout_global = heldout_summary_df[heldout_summary_df["row_type"] == "global_decision"].iloc[0]
    separability_row = separability_summary_df[separability_summary_df["row_type"] == "linear_probe_summary"].iloc[0]
    pairwise_row = separability_summary_df[separability_summary_df["row_type"] == "pairwise_distance_summary"].iloc[0]

    weak_signal_not_robust = spectral_positive_count <= spectral_total / 2.0
    control_failure_systemic = int(control_overall["settings_systemic_fail"]) > spectral_total / 2.0
    heldout_failure_stable = bool(heldout_global["all_negative_overlap"]) and bool(
        heldout_global["should_reclassify_to_route_breaking_counterexample"]
    )
    separability_weak = float(separability_row["target_mean"]) < 0.80 or float(pairwise_row["standardized_gap"]) <= 0.0

    if weak_signal_not_robust and control_failure_systemic and heldout_failure_stable and separability_weak:
        decision = "downgrade_to_no_go"
        reason = (
            "The only current evidence anchor is not robust across the 15-setting sweep, control damage is systemic rather than local, "
            "the held-out dataset behaves like a route-breaking counterexample, and target/control separability remains weak."
        )
    elif (not weak_signal_not_robust) and (not control_failure_systemic) and (not heldout_failure_stable):
        decision = "allow_stricter_analysis_only_refinement"
        reason = (
            "Target-side signal survives the robustness sweep without systemic control failure or stable held-out collapse, "
            "so stricter analysis-only refinement is still defensible."
        )
    else:
        decision = "maintain_analysis_only_hold"
        reason = (
            "The route still has some analysis-only signal, but it remains too unstable or unsafe to move forward."
        )
    return decision, reason


def render_final_resolution(
    *,
    route_split_df: pd.DataFrame,
    robustness_summary_df: pd.DataFrame,
    control_damage_summary_df: pd.DataFrame,
    heldout_summary_df: pd.DataFrame,
    separability_summary_df: pd.DataFrame,
    decision: str,
    reason: str,
) -> str:
    spectral_df = robustness_summary_df[robustness_summary_df["method"] == "adaptive_spectral_locality_hvg"].copy()
    spectral_positive_count = int((spectral_df["winner_overlap_pull_positive"] > 0.0).sum())
    spectral_control_safe_count = int((spectral_df["control_guard"] >= -0.02).sum())
    spectral_heldout_nonnegative_count = int((spectral_df["heldout_overlap_pull"] >= 0.0).sum())
    spectral_total = int(len(spectral_df))

    control_method_overall = control_damage_summary_df[
        (control_damage_summary_df["row_type"] == "method_overall")
        & (control_damage_summary_df["method"] == "adaptive_spectral_locality_hvg")
    ].iloc[0]
    control_by_dataset = control_damage_summary_df[
        (control_damage_summary_df["row_type"] == "method_dataset")
        & (control_damage_summary_df["method"] == "adaptive_spectral_locality_hvg")
    ].copy().sort_values("mean_overlap_pull")
    heldout_global = heldout_summary_df[heldout_summary_df["row_type"] == "global_decision"].iloc[0]
    separability_row = separability_summary_df[separability_summary_df["row_type"] == "linear_probe_summary"].iloc[0]
    pairwise_row = separability_summary_df[separability_summary_df["row_type"] == "pairwise_distance_summary"].iloc[0]
    heldout_dataset = route_split_df[route_split_df["role"] == "heldout_validation_only"].iloc[0]

    lines = [
        "# Final Resolution",
        "",
        "## Direct Answers",
        f"- `adaptive_spectral_locality_hvg` weak positive signal robust? No. Positive `winner_overlap_pull_positive` appears in {spectral_positive_count}/{spectral_total} settings; control-safe settings are {spectral_control_safe_count}/{spectral_total}; held-out non-negative settings are {spectral_heldout_nonnegative_count}/{spectral_total}.",
        f"- Control damage local or systemic? Systemic. `adaptive_spectral_locality_hvg` has {int(control_method_overall['settings_all_controls_fail'])}/{spectral_total} settings where all protected controls fail together and {int(control_method_overall['settings_systemic_fail'])}/{spectral_total} settings where at least two controls fail together.",
        f"- `cellxgene_unciliated_epithelial_five_donors` status: route-breaking counterexample. Best single expert is `{heldout_dataset['best_single_expert']}`, route-family winner flag is {bool(heldout_dataset['route_family_best_single'])}, headroom is {fmt_float(heldout_dataset['headroom_vs_best_single'])}, aggregate held-out overlap is {fmt_float(heldout_global['mean_overlap_pull'])}.",
        f"- Current benchmark route separable? Weakly at best. Leave-one-out linear separability is {fmt_float(separability_row['target_mean'])} and cross-vs-within profile margin is {fmt_float(pairwise_row['standardized_gap'])}.",
        f"- Final decision: `{decision}`.",
        "",
        "## Reason",
        f"- {reason}",
        "",
        "## Protected Control Breakdown For `adaptive_spectral_locality_hvg`",
    ]
    for row in control_by_dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: mean_overlap_pull={fmt_float(row.mean_overlap_pull)}, fail_count={int(row.fail_count)}/{int(row.setting_count)}, lead_damage_count={int(row.lead_damage_count)}."
        )
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "- `robustness_grid_summary.csv`",
            "- `robustness_grid_by_dataset.csv`",
            "- `control_damage_summary.csv`",
            "- `control_damage_by_dataset.csv`",
            "- `heldout_validity_audit.md`",
            "- `heldout_validity_summary.csv`",
            "- `route_separability_audit.md`",
            "- `route_separability_summary.csv`",
            "- `final_resolution.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in args.seeds)
    top_ks = tuple(int(top_k) for top_k in args.top_ks)

    device_info = rr1.resolve_device_info()
    compute_context = {
        **device_info,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "robustness_seeds": list(seeds),
        "robustness_topks": list(top_ks),
        "route_methods": list(traj.ROUTE_METHODS),
    }
    (output_dir / "compute_context.json").write_text(
        json.dumps(compute_context, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    route_split_df = load_route_split()
    headroom_df = load_headroom_table()
    dataset_cache = load_dataset_cache(Path(args.real_data_root))

    robustness_by_dataset_df = build_robustness_grid(
        output_dir=output_dir,
        dataset_cache=dataset_cache,
        headroom_df=headroom_df,
        refine_epochs=args.refine_epochs,
        seeds=seeds,
        top_ks=top_ks,
    )
    robustness_summary_df = summarize_robustness(robustness_by_dataset_df)
    robustness_summary_df.to_csv(output_dir / "robustness_grid_summary.csv", index=False)

    control_damage_by_dataset_df, control_damage_summary_df = build_control_damage_views(robustness_by_dataset_df)
    control_damage_by_dataset_df.to_csv(output_dir / "control_damage_by_dataset.csv", index=False)
    control_damage_summary_df.to_csv(output_dir / "control_damage_summary.csv", index=False)

    heldout_summary_df, heldout_md = build_heldout_validity_summary(
        robustness_by_dataset_df=robustness_by_dataset_df,
        route_split_df=route_split_df,
    )
    heldout_summary_df.to_csv(output_dir / "heldout_validity_summary.csv", index=False)
    (output_dir / "heldout_validity_audit.md").write_text(heldout_md, encoding="utf-8")

    separability_summary_df, separability_md = build_route_separability_summary(route_split_df)
    separability_summary_df.to_csv(output_dir / "route_separability_summary.csv", index=False)
    (output_dir / "route_separability_audit.md").write_text(separability_md, encoding="utf-8")

    decision, reason = decide_final_resolution(
        robustness_summary_df=robustness_summary_df,
        control_damage_summary_df=control_damage_summary_df,
        heldout_summary_df=heldout_summary_df,
        separability_summary_df=separability_summary_df,
    )
    (output_dir / "final_resolution.md").write_text(
        render_final_resolution(
            route_split_df=route_split_df,
            robustness_summary_df=robustness_summary_df,
            control_damage_summary_df=control_damage_summary_df,
            heldout_summary_df=heldout_summary_df,
            separability_summary_df=separability_summary_df,
            decision=decision,
            reason=reason,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
