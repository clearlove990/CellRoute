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


DEFAULT_OUTPUT_DIR = ROOT / "artifacts_route_coherence_audit_trajectory"
ANCHOR_METHOD = "adaptive_hybrid_hvg"
QUESTION_ONE_LINER = (
    "Is there a trajectory/locality/dropout-neighborhood HVG route that is "
    "mechanistically coherent enough to justify a formal model-design cycle "
    "beyond `adaptive_hybrid_hvg`?"
)

CORE_MODELING_DATASETS = (
    "GBM_sd",
    "cellxgene_immune_five_donors",
)
PROTECTED_CONTROL_DATASETS = (
    "FBM_cite",
    "homo_tissue",
    "mus_tissue",
)
HELDOUT_VALIDATION_DATASETS = (
    "cellxgene_unciliated_epithelial_five_donors",
)
OUT_OF_SCOPE_DATASETS = (
    "cellxgene_human_kidney_nonpt",
    "cellxgene_mouse_kidney_aging_10x",
    "E-MTAB-4888",
    "E-MTAB-5061",
    "paul15",
    "E-MTAB-4388",
)

ROUTE_METHODS = (
    "adaptive_spectral_locality_hvg",
    "triku_hvg",
    "scanpy_cell_ranger_hvg",
)
ROUTE_FAMILY_LABELS = {
    "adaptive_spectral_locality_hvg": "adaptive graph-locality fusion",
    "triku_hvg": "published local-neighborhood enrichment",
    "scanpy_cell_ranger_hvg": "dropout-structured dispersion baseline",
}
ROUTE_FAMILY_WINNERS = {
    "adaptive_spectral_locality_hvg",
    "triku_hvg",
    "scanpy_cell_ranger_hvg",
}

ROLE_CONFIG = {
    "GBM_sd": {
        "role": "core_modeling",
        "rationale": "Strongest headroom signal in the repo for the trajectory/locality story; best single expert is `triku_hvg` and the regime label is already high-dropout trajectory-like.",
    },
    "cellxgene_immune_five_donors": {
        "role": "core_modeling",
        "rationale": "Positive-headroom donor panel that still sits in the high-dropout trajectory-like regime and points to `scanpy_cell_ranger_hvg`, so it is the cleanest second target-side route-evidence dataset.",
    },
    "cellxgene_unciliated_epithelial_five_donors": {
        "role": "heldout_validation_only",
        "rationale": "Held out because the headroom is weaker and the current best single winner is `multinomial_deviance_hvg`, which is outside the intended trajectory/locality family.",
    },
    "FBM_cite": {
        "role": "protected_control",
        "rationale": "Anchor-safe atlas-like control that should veto any candidate method paying for target pull by broad anchor rewriting.",
    },
    "homo_tissue": {
        "role": "protected_control",
        "rationale": "Largest human atlas-like panel; if a route is credible it must not collapse here.",
    },
    "mus_tissue": {
        "role": "protected_control",
        "rationale": "Second large atlas-like panel used to measure whether locality-biased movement is just generic control damage.",
    },
    "cellxgene_human_kidney_nonpt": {
        "role": "out_of_scope",
        "rationale": "Despite a secondary trajectory-like profile, the target winner is `scanpy_seurat_v3_hvg`, so it is better treated as donor/batch-heavy evidence than as core trajectory/locality evidence.",
    },
    "cellxgene_mouse_kidney_aging_10x": {
        "role": "out_of_scope",
        "rationale": "Best single expert is `variance`, which makes it a poor fit for a locality-family route claim.",
    },
    "E-MTAB-4888": {
        "role": "out_of_scope",
        "rationale": "The regime is batch-heavy heterogeneous and the winner is `multinomial_deviance_hvg`, so forcing it into the route would mix mechanisms.",
    },
    "E-MTAB-5061": {
        "role": "out_of_scope",
        "rationale": "The anchor already wins decisively, so this is route-limiting counterevidence rather than model-design headroom for trajectory/locality.",
    },
    "paul15": {
        "role": "out_of_scope",
        "rationale": "Residual-friendly count-model dataset whose best single expert is `scran_model_gene_var_hvg`, not a trajectory/locality winner.",
    },
    "E-MTAB-4388": {
        "role": "out_of_scope",
        "rationale": "Atlas-like dataset with limited headroom and no route-family winner; useful background evidence only.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trajectory/locality route coherence audit.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--real-data-root", type=str, default=str(ROOT / "data" / "real_inputs"))
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--refine-epochs", type=int, default=6)
    return parser.parse_args()


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def load_failure_taxonomy() -> pd.DataFrame:
    return pd.read_csv(ROOT / "artifacts_topconf_selector_round2" / "failure_taxonomy.csv")


def load_donor_route_comparison() -> dict[str, object]:
    split_df = pd.read_csv(ROOT / "artifacts_regime_specific_route" / "dataset_split.csv")
    analysis_df = pd.read_csv(ROOT / "artifacts_regime_specific_route" / "analysis_gate_summary.csv")
    donor_targets = split_df[split_df["role"] == "target_model_dev"].copy()
    analysis_row = analysis_df.iloc[0]
    return {
        "target_dataset_count": int(len(donor_targets)),
        "winner_family_count": int(donor_targets["best_single_expert"].nunique()),
        "winner_family_list": donor_targets["best_single_expert"].astype(str).tolist(),
        "winner_overlap_pull_positive": float(analysis_row["positive_overlap_pull_vs_anchor"]),
        "control_guard": float(analysis_row["control_overlap_pull_vs_anchor"]),
    }


def build_route_dataset_split(*, headroom_df: pd.DataFrame, failure_df: pd.DataFrame) -> pd.DataFrame:
    headroom_lookup = headroom_df.set_index("dataset")
    failure_lookup = failure_df.set_index("dataset")
    rows: list[dict[str, object]] = []
    for dataset_name, config in ROLE_CONFIG.items():
        row = {
            "dataset": dataset_name,
            "role": config["role"],
            "use_for_modeling": int(config["role"] == "core_modeling"),
            "use_for_analysis_gate": int(config["role"] in {"core_modeling", "protected_control"}),
            "use_for_final_validation_only": int(config["role"] == "heldout_validation_only"),
            "rationale": config["rationale"],
        }
        if dataset_name in headroom_lookup.index:
            headroom_row = headroom_lookup.loc[dataset_name]
            for column in (
                "regime",
                "secondary_regime",
                "anchor_score",
                "best_single_expert",
                "best_single_score",
                "headroom_vs_best_single",
                "anchor_minus_adaptive_stat",
            ):
                row[column] = headroom_row.get(column)
        if dataset_name in failure_lookup.index:
            failure_row = failure_lookup.loc[dataset_name]
            for column in (
                "best_published_expert",
                "selector_margin_vs_best_single",
                "stat_batch_classes",
                "stat_cluster_strength",
                "stat_trajectory_strength",
                "stat_dropout_rate",
                "stat_pc_entropy",
            ):
                row[column] = failure_row.get(column)
        info = base.load_dataset_info(dataset_name)
        row["dataset_id"] = info.get("dataset_id", "")
        row["input_path"] = info.get("input_path", "")
        row["labels_col"] = info.get("labels_col", "")
        row["batches_col"] = info.get("batches_col", "")
        row["cells_loaded"] = info.get("cells_loaded")
        row["genes_loaded"] = info.get("genes_loaded")
        row["batch_classes_loaded"] = info.get("batch_classes_loaded")
        row["label_classes_loaded"] = info.get("label_classes_loaded")
        row["route_family_best_single"] = int(str(row.get("best_single_expert", "")) in ROUTE_FAMILY_WINNERS)
        rows.append(row)

    df = pd.DataFrame(rows)
    role_order = {
        "core_modeling": 0,
        "protected_control": 1,
        "heldout_validation_only": 2,
        "out_of_scope": 3,
    }
    df["role_order"] = df["role"].map(role_order).fillna(99)
    df = df.sort_values(["role_order", "dataset"]).drop(columns=["role_order"]).reset_index(drop=True)
    return df


def build_route_coherence_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    detail_df = split_df.copy()
    detail_df["row_type"] = "dataset"

    rows: list[dict[str, object]] = []
    for split_name, role_name in (
        ("core_modeling", "core_modeling"),
        ("protected_control", "protected_control"),
        ("heldout_validation_only", "heldout_validation_only"),
    ):
        frame = split_df[split_df["role"] == role_name].copy()
        if frame.empty:
            continue
        rows.append(
            {
                "row_type": "split_summary",
                "split": split_name,
                "dataset_count": int(len(frame)),
                "mean_headroom_vs_best_single": float(frame["headroom_vs_best_single"].mean()),
                "min_headroom_vs_best_single": float(frame["headroom_vs_best_single"].min()),
                "max_headroom_vs_best_single": float(frame["headroom_vs_best_single"].max()),
                "mean_trajectory_strength": float(frame["stat_trajectory_strength"].mean()),
                "mean_dropout_rate": float(frame["stat_dropout_rate"].mean()),
                "route_family_best_single_count": int(frame["route_family_best_single"].sum()),
                "best_single_family_count": int(frame["best_single_expert"].astype(str).nunique()),
            }
        )

    target_df = split_df[split_df["role"] == "core_modeling"].copy()
    control_df = split_df[split_df["role"] == "protected_control"].copy()
    heldout_df = split_df[split_df["role"] == "heldout_validation_only"].copy()
    rows.append(
        {
            "row_type": "route_summary",
            "split": "trajectory_locality_route",
            "dataset_count": int(len(target_df) + len(control_df) + len(heldout_df)),
            "target_dataset_count": int(len(target_df)),
            "protected_control_count": int(len(control_df)),
            "heldout_dataset_count": int(len(heldout_df)),
            "target_winner_family_count": int(target_df["best_single_expert"].astype(str).nunique()),
            "target_all_best_single_in_route_family": bool((target_df["route_family_best_single"] == 1).all()),
            "heldout_all_best_single_in_route_family": bool((heldout_df["route_family_best_single"] == 1).all()),
            "target_all_positive_headroom": bool((target_df["headroom_vs_best_single"] > 0).all()),
            "heldout_all_positive_headroom": bool((heldout_df["headroom_vs_best_single"] > 0).all()),
            "trajectory_strength_gap_vs_controls": float(
                target_df["stat_trajectory_strength"].mean() - control_df["stat_trajectory_strength"].mean()
            ),
            "dropout_gap_vs_controls": float(
                target_df["stat_dropout_rate"].mean() - control_df["stat_dropout_rate"].mean()
            ),
        }
    )
    return pd.concat([detail_df, pd.DataFrame(rows)], ignore_index=True, sort=False)


def render_route_definition(split_df: pd.DataFrame) -> str:
    target_df = split_df[split_df["role"] == "core_modeling"].copy()
    control_df = split_df[split_df["role"] == "protected_control"].copy()
    heldout_df = split_df[split_df["role"] == "heldout_validation_only"].copy()
    lines = [
        "# Route Definition",
        "",
        "## One-Sentence Question",
        f"- {QUESTION_ONE_LINER}",
        "",
        "## Route Scope",
        "- Route family: trajectory/locality/dropout-neighborhood scorers only.",
        f"- Safety anchor: `{ANCHOR_METHOD}` remains the default-safe anchor.",
        "- This audit is analysis-only and does not authorize a new scorer family or a full benchmark cycle.",
        "",
        "## Core Modeling Datasets",
    ]
    for row in target_df.itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: regime={row.regime}; best_single={row.best_single_expert}; headroom={fmt_float(row.headroom_vs_best_single)}; rationale={row.rationale}"
        )
    lines.extend(
        [
            "",
            "## Protected Controls",
        ]
    )
    for row in control_df.itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: anchor headroom={fmt_float(row.headroom_vs_best_single)}; rationale={row.rationale}"
        )
    lines.extend(
        [
            "",
            "## Held-Out Validation Only",
        ]
    )
    for row in heldout_df.itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: best_single={row.best_single_expert}; headroom={fmt_float(row.headroom_vs_best_single)}; rationale={row.rationale}"
        )
    lines.extend(
        [
            "",
            "## Existing Methods Audited",
        ]
    )
    for method_name in ROUTE_METHODS:
        lines.append(f"- `{method_name}`: {ROUTE_FAMILY_LABELS[method_name]}.")
    lines.extend(
        [
            "",
            "## Out Of Scope",
        ]
    )
    for row in split_df[split_df["role"] == "out_of_scope"].itertuples(index=False):
        lines.append(f"- `{row.dataset}`: {row.rationale}")
    return "\n".join(lines) + "\n"


def render_route_dataset_split_md(split_df: pd.DataFrame) -> str:
    lines = [
        "# Route Dataset Split",
        "",
        "## Core Modeling",
    ]
    for row in split_df[split_df["role"] == "core_modeling"].itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: best_single={row.best_single_expert}, headroom={fmt_float(row.headroom_vs_best_single)}, trajectory_strength={fmt_float(row.stat_trajectory_strength)}, dropout={fmt_float(row.stat_dropout_rate)}"
        )
    lines.extend(
        [
            "",
            "## Protected Controls",
        ]
    )
    for row in split_df[split_df["role"] == "protected_control"].itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: best_single={row.best_single_expert}, anchor headroom={fmt_float(row.headroom_vs_best_single)}, trajectory_strength={fmt_float(row.stat_trajectory_strength)}"
        )
    lines.extend(
        [
            "",
            "## Held-Out Validation Only",
        ]
    )
    for row in split_df[split_df["role"] == "heldout_validation_only"].itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: best_single={row.best_single_expert}, route_family_best_single={bool(row.route_family_best_single)}, headroom={fmt_float(row.headroom_vs_best_single)}"
        )
    lines.extend(
        [
            "",
            "## Out Of Scope",
        ]
    )
    for row in split_df[split_df["role"] == "out_of_scope"].itertuples(index=False):
        lines.append(f"- `{row.dataset}`: {row.rationale}")
    return "\n".join(lines) + "\n"


def build_feasibility_rows(
    *,
    dataset_cache: pkg.DatasetCache,
    headroom_df: pd.DataFrame,
    top_k: int,
    seed: int,
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
    audit_datasets = (*CORE_MODELING_DATASETS, *PROTECTED_CONTROL_DATASETS, *HELDOUT_VALIDATION_DATASETS)
    for dataset_name in audit_datasets:
        dataset = dataset_cache.get(dataset_name, seed)
        current_top_k = min(int(top_k), int(dataset.counts.shape[1]))
        best_single_method = str(headroom_lookup.loc[dataset_name, "best_single_expert"])
        methods_to_compute = tuple(dict.fromkeys([ANCHOR_METHOD, best_single_method, *ROUTE_METHODS]))
        score_cache: dict[str, np.ndarray] = {}
        topk_cache: dict[str, np.ndarray] = {}
        for method_name in methods_to_compute:
            score_cache[method_name] = np.asarray(
                registry[method_name](dataset.counts, dataset.batches, current_top_k),
                dtype=np.float64,
            )
            topk_cache[method_name] = base.topk_indices(score_cache[method_name], current_top_k)

        anchor_scores = score_cache[ANCHOR_METHOD]
        anchor_topk = topk_cache[ANCHOR_METHOD]
        best_scores = score_cache[best_single_method]
        best_topk = topk_cache[best_single_method]
        anchor_overlap_to_best = base.jaccard(anchor_topk, best_topk)
        anchor_corr_to_best = base.spearman_correlation(anchor_scores, best_scores)

        if dataset_name in CORE_MODELING_DATASETS:
            group_name = "core_modeling"
        elif dataset_name in PROTECTED_CONTROL_DATASETS:
            group_name = "protected_control"
        else:
            group_name = "heldout_validation_only"

        for method_name in ROUTE_METHODS:
            candidate_scores = score_cache[method_name]
            candidate_topk = topk_cache[method_name]
            rows.append(
                {
                    "dataset": dataset_name,
                    "group_name": group_name,
                    "method": method_name,
                    "route_family_label": ROUTE_FAMILY_LABELS[method_name],
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
    return pd.DataFrame(rows)


def summarize_feasibility(dataset_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method_name, frame in dataset_df.groupby("method", sort=True):
        core_df = frame[frame["group_name"] == "core_modeling"].copy()
        control_df = frame[frame["group_name"] == "protected_control"].copy()
        heldout_df = frame[frame["group_name"] == "heldout_validation_only"].copy()

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
            "heldout_overlap_nonnegative": heldout_overlap_pull >= 0.0,
        }
        row = {
            "row_type": "method_summary",
            "method": method_name,
            "route_family_label": ROUTE_FAMILY_LABELS[method_name],
            "core_target_count": int(len(core_df)),
            "protected_control_count": int(len(control_df)),
            "heldout_count": int(len(heldout_df)),
            "positive_target_dataset_count": int((core_df["delta_overlap_to_best_single_vs_anchor"] > 0.0).sum()),
            "damaged_control_dataset_count": int((control_df["delta_overlap_to_best_single_vs_anchor"] < -0.02).sum()),
            "targeted_shift_gap": float(targeted_shift_gap),
            "winner_overlap_pull_positive": float(winner_overlap_pull_positive),
            "winner_overlap_pull_gap": float(winner_overlap_pull_gap),
            "winner_corr_pull_gap": float(winner_corr_pull_gap),
            "control_guard": float(control_guard),
            "heldout_overlap_pull": float(heldout_overlap_pull),
            "heldout_corr_pull": float(heldout_corr_pull),
            "legacy_smoke_condition_count": int(sum(bool(value) for name, value in conditions.items() if name != "heldout_overlap_nonnegative")),
            "legacy_smoke_style_pass": bool(sum(bool(value) for name, value in conditions.items() if name != "heldout_overlap_nonnegative") >= 4),
            "analysis_signal_present": bool(
                conditions["targeted_shift_gap"]
                and conditions["winner_overlap_pull_positive"]
                and conditions["winner_overlap_pull_gap"]
                and conditions["winner_corr_pull_gap"]
            ),
            "strict_unlock_ready": bool(
                conditions["targeted_shift_gap"]
                and conditions["winner_overlap_pull_positive"]
                and conditions["winner_overlap_pull_gap"]
                and conditions["winner_corr_pull_gap"]
                and conditions["control_guard"]
                and conditions["heldout_overlap_nonnegative"]
            ),
        }
        for condition_name, value in conditions.items():
            row[f"condition_{condition_name}"] = bool(value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def render_existing_method_feasibility_md(
    *,
    summary_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
) -> str:
    lines = [
        "# Existing Method Feasibility",
        "",
        "## Route-Level Summary",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- `{row.method}`: target_shift_gap={fmt_float(row.targeted_shift_gap)}, overlap_positive={fmt_float(row.winner_overlap_pull_positive)}, control_guard={fmt_float(row.control_guard)}, heldout_overlap={fmt_float(row.heldout_overlap_pull)}, strict_unlock_ready={bool(row.strict_unlock_ready)}."
        )

    spectral_row = summary_df[summary_df["method"] == "adaptive_spectral_locality_hvg"].iloc[0]
    lines.extend(
        [
            "",
            "## Main Readout",
            f"- `adaptive_spectral_locality_hvg` is the only audited route-family method with target-side positive signal on all four analysis-only pull checks: targeted_shift_gap={fmt_float(spectral_row.targeted_shift_gap)}, winner_overlap_pull_positive={fmt_float(spectral_row.winner_overlap_pull_positive)}, winner_overlap_pull_gap={fmt_float(spectral_row.winner_overlap_pull_gap)}, winner_corr_pull_gap={fmt_float(spectral_row.winner_corr_pull_gap)}.",
            f"- That signal is still not unlock-grade because control_guard is {fmt_float(spectral_row.control_guard)} and the held-out overlap pull is {fmt_float(spectral_row.heldout_overlap_pull)}.",
            "- `triku_hvg` and `scanpy_cell_ranger_hvg` pull hard toward the target winners on one core dataset each, but both behave like unsafe broad rewrites on protected controls.",
            "",
            "## Dataset-Level Notes",
        ]
    )
    for dataset_name in (*CORE_MODELING_DATASETS, *PROTECTED_CONTROL_DATASETS, *HELDOUT_VALIDATION_DATASETS):
        subset = dataset_df[dataset_df["dataset"] == dataset_name].copy()
        best_method = subset.sort_values("delta_overlap_to_best_single_vs_anchor", ascending=False).iloc[0]
        lines.append(
            f"- `{dataset_name}`: strongest overlap pull came from `{best_method['method']}` with delta_overlap={fmt_float(best_method['delta_overlap_to_best_single_vs_anchor'])} against best_single={best_method['best_single_method']}."
        )
    return "\n".join(lines) + "\n"


def render_route_coherence_audit_md(
    *,
    split_df: pd.DataFrame,
    coherence_summary_df: pd.DataFrame,
    feasibility_summary_df: pd.DataFrame,
    donor_comparison: dict[str, object],
) -> str:
    target_df = split_df[split_df["role"] == "core_modeling"].copy()
    heldout_df = split_df[split_df["role"] == "heldout_validation_only"].copy()
    route_row = coherence_summary_df[coherence_summary_df["row_type"] == "route_summary"].iloc[0]
    best_signal_row = feasibility_summary_df.sort_values(
        ["analysis_signal_present", "winner_overlap_pull_positive", "targeted_shift_gap"],
        ascending=[False, False, False],
    ).iloc[0]

    lines = [
        "# Route Coherence Audit",
        "",
        "## Direct Answer",
        "- The trajectory/locality route is cleaner than the closed donor-aware route as a target-side candidate, but it is still not coherent enough to unlock model design.",
        "",
        "## Target Split Coherence",
        f"- Core modeling datasets: {', '.join(target_df['dataset'].astype(str).tolist())}.",
        f"- Their best single experts are {', '.join(target_df['best_single_expert'].astype(str).tolist())}, and all of them stay inside the declared trajectory/locality/dropout-neighborhood family.",
        f"- Held-out validation set: {', '.join(heldout_df['dataset'].astype(str).tolist())}; its best single expert is {', '.join(heldout_df['best_single_expert'].astype(str).tolist())}, which breaks the family-level story.",
        f"- Target winner-family count is {int(route_row['target_winner_family_count'])}; donor-aware route target winner-family count was {int(donor_comparison['winner_family_count'])}.",
        "",
        "## Headroom And Regime Separation",
        f"- Core target headroom stays positive on all modeling datasets, but it is uneven: mean={fmt_float(target_df['headroom_vs_best_single'].mean())}, min={fmt_float(target_df['headroom_vs_best_single'].min())}, max={fmt_float(target_df['headroom_vs_best_single'].max())}.",
        f"- Held-out headroom remains positive but weak at {fmt_float(heldout_df['headroom_vs_best_single'].mean())}.",
        f"- The route-profile separation itself is weak: target minus control trajectory_strength gap is {fmt_float(route_row['trajectory_strength_gap_vs_controls'])} and target minus control dropout gap is {fmt_float(route_row['dropout_gap_vs_controls'])}.",
        "",
        "## Existing Method Evidence Anchor",
        f"- Best current evidence anchor is `{best_signal_row['method']}` because it is the strongest existing route-family method with target-side positive pull: overlap_positive={fmt_float(best_signal_row['winner_overlap_pull_positive'])}, targeted_shift_gap={fmt_float(best_signal_row['targeted_shift_gap'])}.",
        f"- That evidence is not safe enough for unlock: control_guard={fmt_float(best_signal_row['control_guard'])}, heldout_overlap={fmt_float(best_signal_row['heldout_overlap_pull'])}.",
        "",
        "## Comparison To The Closed Donor-Aware Route",
        f"- Donor-aware route analysis ended with winner_overlap_pull_positive={fmt_float(donor_comparison['winner_overlap_pull_positive'])} and control_guard={fmt_float(donor_comparison['control_guard'])}.",
        f"- This trajectory/locality audit does better on target-side signal through `{best_signal_row['method']}`, but not on safety: control damage is still large and the held-out dataset points outside the intended family.",
        "- So the route is more self-consistent as a candidate dataset slice, but not more unlock-ready as a paper route.",
    ]
    return "\n".join(lines) + "\n"


def render_unlock_rule_md(
    *,
    split_df: pd.DataFrame,
    coherence_summary_df: pd.DataFrame,
    feasibility_summary_df: pd.DataFrame,
) -> str:
    route_row = coherence_summary_df[coherence_summary_df["row_type"] == "route_summary"].iloc[0]
    best_signal_row = feasibility_summary_df.sort_values(
        ["analysis_signal_present", "winner_overlap_pull_positive", "targeted_shift_gap"],
        ascending=[False, False, False],
    ).iloc[0]

    target_consistency = bool(route_row["target_all_best_single_in_route_family"]) and bool(
        route_row["heldout_all_best_single_in_route_family"]
    )
    stable_headroom = bool(route_row["target_all_positive_headroom"]) and float(
        split_df[split_df["role"] == "heldout_validation_only"]["headroom_vs_best_single"].mean()
    ) >= 0.10
    method_unlock = bool(best_signal_row["condition_winner_overlap_pull_positive"]) and bool(
        best_signal_row["condition_control_guard"]
    )
    heldout_unlock = bool(best_signal_row["condition_heldout_overlap_nonnegative"]) and bool(
        route_row["heldout_all_best_single_in_route_family"]
    )

    lines = [
        "# Unlock Rule",
        "",
        "## Required Unlock Conditions",
        "- Target split must stay mechanism-consistent: core modeling datasets and held-out validation should all point to the declared trajectory/locality/dropout-neighborhood family.",
        "- Anchor headroom must stay stable across the core split and remain meaningfully positive on held-out validation, rather than being driven by one flagship dataset only.",
        "- At least one existing route-family method must show analysis-only positive signal with `winner_overlap_pull_positive > 0` and `control_guard >= -0.02`.",
        "- Held-out validation must be feasible: the held-out dataset should not immediately point outside the route family, and at least one existing route-family method should have non-negative held-out overlap pull.",
        "",
        "## Current Readout",
        f"- target_split_consistency={target_consistency} (core winners stay in-family, but held-out winner family consistency is {bool(route_row['heldout_all_best_single_in_route_family'])}).",
        f"- stable_anchor_headroom={stable_headroom} (held-out headroom={fmt_float(split_df[split_df['role'] == 'heldout_validation_only']['headroom_vs_best_single'].mean())}).",
        f"- existing_method_unlock={method_unlock} (best method `{best_signal_row['method']}` has winner_overlap_pull_positive={fmt_float(best_signal_row['winner_overlap_pull_positive'])} and control_guard={fmt_float(best_signal_row['control_guard'])}).",
        f"- heldout_validation_feasible={heldout_unlock} (best method heldout_overlap={fmt_float(best_signal_row['heldout_overlap_pull'])}).",
        "",
        "## Unlock Decision",
        "- The route is not unlocked for model design.",
        "- Any further work must stay analysis-only until safety and held-out consistency are both fixed without broadening the route definition.",
    ]
    return "\n".join(lines) + "\n"


def render_route_decision_md(
    *,
    split_df: pd.DataFrame,
    feasibility_summary_df: pd.DataFrame,
) -> str:
    best_signal_row = feasibility_summary_df.sort_values(
        ["analysis_signal_present", "winner_overlap_pull_positive", "targeted_shift_gap"],
        ascending=[False, False, False],
    ).iloc[0]

    decision = "analysis_only_hold"
    lines = [
        "# Route Decision",
        "",
        "## Final Classification",
        f"- `{decision}`",
        "",
        "## Why",
        "- The route now has a narrow core split that is cleaner than the donor-aware route and is not just one-dataset-only evidence.",
        f"- There is at least one existing route-family method with target-side positive signal: `{best_signal_row['method']}`.",
        "- The route still fails the two requirements that matter most before modeling: protected-control safety and held-out consistency.",
        "- No truly external trajectory-like validation dataset exists in the repo, so even a cleaner internal signal would still be below paper-grade evidence.",
        "",
        "## What Is Allowed Next",
        "- Analysis-only split refinement if it makes the route stricter rather than broader.",
        "- Analysis-only acquisition or import of a truly external trajectory/locality validation dataset.",
        "",
        "## What Is Not Allowed Next",
        "- No new scorer family.",
        "- No full benchmark run for this route.",
        "- No model-design cycle until the unlock rule is satisfied.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device_info = rr1.resolve_device_info()
    compute_context = {
        **device_info,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "route_methods": list(ROUTE_METHODS),
        "anchor_method": ANCHOR_METHOD,
    }
    (output_dir / "compute_context.json").write_text(
        json.dumps(compute_context, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    headroom_df = base.load_headroom_table()
    failure_df = load_failure_taxonomy()
    split_df = build_route_dataset_split(headroom_df=headroom_df, failure_df=failure_df)
    split_df.to_csv(output_dir / "route_dataset_split.csv", index=False)

    coherence_summary_df = build_route_coherence_summary(split_df)
    coherence_summary_df.to_csv(output_dir / "route_coherence_summary.csv", index=False)

    donor_comparison = load_donor_route_comparison()

    resources = base.load_dataset_resources(Path(args.real_data_root))
    dataset_cache = pkg.DatasetCache(resources)
    feasibility_dataset_df = build_feasibility_rows(
        dataset_cache=dataset_cache,
        headroom_df=headroom_df,
        top_k=args.top_k,
        seed=args.seed,
        refine_epochs=args.refine_epochs,
    )
    feasibility_summary_df = summarize_feasibility(feasibility_dataset_df)
    existing_method_feasibility_df = pd.concat(
        [feasibility_dataset_df.assign(row_type="dataset"), feasibility_summary_df],
        ignore_index=True,
        sort=False,
    )
    existing_method_feasibility_df.to_csv(output_dir / "existing_method_feasibility.csv", index=False)

    (output_dir / "route_definition.md").write_text(
        render_route_definition(split_df),
        encoding="utf-8",
    )
    (output_dir / "route_dataset_split.md").write_text(
        render_route_dataset_split_md(split_df),
        encoding="utf-8",
    )
    (output_dir / "route_coherence_audit.md").write_text(
        render_route_coherence_audit_md(
            split_df=split_df,
            coherence_summary_df=coherence_summary_df,
            feasibility_summary_df=feasibility_summary_df,
            donor_comparison=donor_comparison,
        ),
        encoding="utf-8",
    )
    (output_dir / "existing_method_feasibility.md").write_text(
        render_existing_method_feasibility_md(
            summary_df=feasibility_summary_df,
            dataset_df=feasibility_dataset_df,
        ),
        encoding="utf-8",
    )
    (output_dir / "unlock_rule.md").write_text(
        render_unlock_rule_md(
            split_df=split_df,
            coherence_summary_df=coherence_summary_df,
            feasibility_summary_df=feasibility_summary_df,
        ),
        encoding="utf-8",
    )
    (output_dir / "route_decision.md").write_text(
        render_route_decision_md(
            split_df=split_df,
            feasibility_summary_df=feasibility_summary_df,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
