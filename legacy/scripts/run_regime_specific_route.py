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


DEFAULT_OUTPUT_DIR = ROOT / "artifacts_regime_specific_route"
ANCHOR_METHOD = "adaptive_hybrid_hvg"
CANDIDATE_METHOD = "adaptive_invariant_residual_hvg"
QUESTION_ONE_LINER = (
    "Can a donor-aware, batch-aware count-residual HVG scorer outperform "
    "`adaptive_hybrid_hvg` on explicit donor/batch-heavy clustering datasets "
    "without degrading protected atlas-like controls?"
)

TARGET_MODELING_DATASETS = (
    "cellxgene_human_kidney_nonpt",
    "cellxgene_mouse_kidney_aging_10x",
    "E-MTAB-4888",
)
PROTECTED_CONTROL_DATASETS = (
    "FBM_cite",
    "homo_tissue",
    "mus_tissue",
)
HELDOUT_VALIDATION_DATASETS = (
    "cellxgene_immune_five_donors",
    "cellxgene_unciliated_epithelial_five_donors",
)
AUXILIARY_CONTROL_DATASETS = (
    "E-MTAB-4388",
)
COUNTEREXAMPLE_DATASETS = (
    "E-MTAB-5061",
)
OUT_OF_SCOPE_DATASETS = (
    "GBM_sd",
    "paul15",
)

ROLE_CONFIG = {
    "cellxgene_human_kidney_nonpt": {
        "role": "target_model_dev",
        "rationale": "High-headroom human kidney donor panel with explicit donor_id batches; primary regime example.",
    },
    "cellxgene_mouse_kidney_aging_10x": {
        "role": "target_model_dev",
        "rationale": "Explicit donor-aware mouse kidney panel with positive headroom; useful to test whether the route transfers across species.",
    },
    "E-MTAB-4888": {
        "role": "target_model_dev",
        "rationale": "Strong batch-heavy dataset with 21 individuals and positive anchor headroom; included to keep the route batch-aware rather than kidney-only.",
    },
    "FBM_cite": {
        "role": "protected_control",
        "rationale": "Stable atlas-like control where the anchor is already strong; must not regress if the route is credible.",
    },
    "homo_tissue": {
        "role": "protected_control",
        "rationale": "Largest atlas-like anchor-safe control; used to guard against broad anchor rewriting.",
    },
    "mus_tissue": {
        "role": "protected_control",
        "rationale": "Second atlas-like control to check whether any gain is paying for itself by harming broad tissue panels.",
    },
    "cellxgene_immune_five_donors": {
        "role": "heldout_validation_only",
        "rationale": "Internal donor-heavy holdout kept out of modeling because its primary regime is trajectory-like rather than batch-heavy.",
    },
    "cellxgene_unciliated_epithelial_five_donors": {
        "role": "heldout_validation_only",
        "rationale": "Second internal donor-heavy holdout reserved for the final validation stage only.",
    },
    "E-MTAB-4388": {
        "role": "auxiliary_control",
        "rationale": "Anchor-stable residual-friendly dataset kept as a soft control, but not promoted to the protected atlas set.",
    },
    "E-MTAB-5061": {
        "role": "counterexample_not_target",
        "rationale": "Batch-heavy by taxonomy, but the anchor already wins decisively; keep it out of model design and use it as evidence that the route is not universally valid.",
    },
    "GBM_sd": {
        "role": "out_of_scope",
        "rationale": "High-dropout trajectory dataset; useful background evidence but not aligned with the donor/batch-heavy route.",
    },
    "paul15": {
        "role": "out_of_scope",
        "rationale": "Residual-friendly count-model dataset without donor structure; not part of the target regime.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build regime-specific donor/batch-heavy route artifacts.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--real-data-root", type=str, default=str(ROOT / "data" / "real_inputs"))
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--refine-epochs", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=1)
    return parser.parse_args()


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size == 0:
        return np.asarray([], dtype=np.int64)
    k = max(1, min(int(k), int(scores.size)))
    idx = np.argpartition(scores, -k)[-k:]
    return np.asarray(idx, dtype=np.int64)


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_set = set(np.asarray(left, dtype=np.int64).tolist())
    right_set = set(np.asarray(right, dtype=np.int64).tolist())
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set) / len(union))


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=np.float64).ravel()
    right_arr = np.asarray(right, dtype=np.float64).ravel()
    if left_arr.size != right_arr.size or left_arr.size < 2:
        return 0.0
    left_rank = pd.Series(left_arr).rank(method="average").to_numpy(dtype=np.float64)
    right_rank = pd.Series(right_arr).rank(method="average").to_numpy(dtype=np.float64)
    corr = np.corrcoef(left_rank, right_rank)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def mean_or_nan(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.mean())


def load_headroom_table() -> pd.DataFrame:
    path = ROOT / "artifacts_next_direction" / "anchor_headroom_tables.csv"
    df = pd.read_csv(path)
    return df[df["row_type"] == "dataset"].copy().reset_index(drop=True)


def load_failure_taxonomy() -> pd.DataFrame:
    path = ROOT / "artifacts_topconf_selector_round2" / "failure_taxonomy.csv"
    return pd.read_csv(path)


def load_dataset_info(dataset_name: str) -> dict[str, object]:
    path = ROOT / "artifacts_recomb_ismb_benchmark" / "datasets" / dataset_name / "dataset_info.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataset_split(*, headroom_df: pd.DataFrame, failure_df: pd.DataFrame) -> pd.DataFrame:
    headroom_lookup = headroom_df.set_index("dataset")
    failure_lookup = failure_df.set_index("dataset")
    rows: list[dict[str, object]] = []
    for dataset_name, config in ROLE_CONFIG.items():
        row = {
            "dataset": dataset_name,
            "role": config["role"],
            "use_for_modeling": int(config["role"] == "target_model_dev"),
            "use_for_analysis_gate": int(config["role"] in {"target_model_dev", "protected_control"}),
            "use_for_full_benchmark": int(config["role"] in {"target_model_dev", "protected_control"}),
            "use_for_final_validation_only": int(config["role"] == "heldout_validation_only"),
            "rationale": config["rationale"],
        }
        if dataset_name in headroom_lookup.index:
            headroom_row = headroom_lookup.loc[dataset_name]
            for column in (
                "regime",
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
            ):
                row[column] = failure_row.get(column)
        info = load_dataset_info(dataset_name)
        row["dataset_id"] = info.get("dataset_id", "")
        row["input_path"] = info.get("input_path", "")
        row["labels_col"] = info.get("labels_col", "")
        row["batches_col"] = info.get("batches_col", "")
        row["cells_loaded"] = info.get("cells_loaded")
        row["genes_loaded"] = info.get("genes_loaded")
        row["batch_classes_loaded"] = info.get("batch_classes_loaded")
        row["label_classes_loaded"] = info.get("label_classes_loaded")
        rows.append(row)
    df = pd.DataFrame(rows)
    role_order = {
        "target_model_dev": 0,
        "protected_control": 1,
        "heldout_validation_only": 2,
        "auxiliary_control": 3,
        "counterexample_not_target": 4,
        "out_of_scope": 5,
    }
    df["role_order"] = df["role"].map(role_order).fillna(99)
    df = df.sort_values(["role_order", "dataset"]).drop(columns=["role_order"]).reset_index(drop=True)
    return df


def render_regime_definition(split_df: pd.DataFrame) -> str:
    target_rows = split_df[split_df["role"] == "target_model_dev"].copy()
    control_rows = split_df[split_df["role"] == "protected_control"].copy()
    heldout_rows = split_df[split_df["role"] == "heldout_validation_only"].copy()
    counterexample_rows = split_df[split_df["role"] == "counterexample_not_target"].copy()

    lines = [
        "# Regime Definition",
        "",
        "## One-Sentence Question",
        f"- {QUESTION_ONE_LINER}",
        "",
        "## Target Regime",
        "- Primary regime: explicit donor/batch-heavy clustering datasets with positive anchor headroom and non-trivial batch structure.",
        "- Inclusion rule: dataset must have a donor/individual batch column, positive headroom over the anchor, and be used for clustering-style evaluation.",
        "- Exclusion rule: datasets are excluded from the target modeling split if the anchor is already decisively strong or if the primary regime is trajectory-like rather than batch-heavy.",
        "",
        "## Target Modeling Split",
    ]
    for row in target_rows.itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: regime={row.regime}; best_single={row.best_single_expert}; "
            f"headroom={fmt_float(row.headroom_vs_best_single)}; batches={row.batch_classes_loaded}."
        )
    lines.extend(
        [
            "",
            "## Protected Controls",
        ]
    )
    for row in control_rows.itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: regime={row.regime}; anchor headroom={fmt_float(row.headroom_vs_best_single)}; "
            f"role is safety control, not optimization target."
        )
    lines.extend(
        [
            "",
            "## Held-Out Validation",
            "- No truly external donor-heavy dataset exists in the current repo beyond the benchmark pool.",
            "- The current held-out validation set is therefore an internal donor-heavy reserve only; paper-grade external validation remains a gap.",
        ]
    )
    for row in heldout_rows.itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: internal donor-heavy holdout with `use_for_final_validation_only=1`; kept out of modeling and analysis gates."
        )
    if not counterexample_rows.empty:
        lines.extend(
            [
                "",
                "## Counterexample Kept Out Of Scope",
            ]
        )
        for row in counterexample_rows.itertuples(index=False):
            lines.append(
                f"- `{row.dataset}` stays outside the target split because anchor headroom is {fmt_float(row.headroom_vs_best_single)} and the route would be over-claimed if this dataset were optimized against."
            )
    lines.extend(
        [
            "",
            "## Candidate Family",
            f"- Candidate family: `{CANDIDATE_METHOD}`.",
            "- Mechanism scope: donor-aware or batch-aware count-residual scoring only.",
            "- Safety anchor: `adaptive_hybrid_hvg` remains the only default-safe anchor.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_dataset_split_md(split_df: pd.DataFrame) -> str:
    lines = [
        "# Dataset Split",
        "",
        "## Modeling / Debugging Allowed",
    ]
    for row in split_df[split_df["use_for_modeling"] == 1].itertuples(index=False):
        lines.append(
            f"- `{row.dataset}` ({row.role}): best_single={row.best_single_expert}, headroom={fmt_float(row.headroom_vs_best_single)}, rationale={row.rationale}"
        )
    lines.extend(
        [
            "",
            "## Protected Controls",
        ]
    )
    for row in split_df[split_df["role"] == "protected_control"].itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: anchor-safe control, headroom={fmt_float(row.headroom_vs_best_single)}, rationale={row.rationale}"
        )
    lines.extend(
        [
            "",
            "## Final Validation Only",
        ]
    )
    for row in split_df[split_df["role"] == "heldout_validation_only"].itertuples(index=False):
        lines.append(f"- `{row.dataset}`: kept untouched until after analysis and full benchmark gates.")
    lines.extend(
        [
            "",
            "## Not Used For This Route",
        ]
    )
    for row in split_df[split_df["role"].isin(["auxiliary_control", "counterexample_not_target", "out_of_scope"])].itertuples(index=False):
        lines.append(f"- `{row.dataset}` ({row.role}): {row.rationale}")
    return "\n".join(lines) + "\n"


def build_headroom_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    target_df = split_df[split_df["role"] == "target_model_dev"].copy()
    control_df = split_df[split_df["role"] == "protected_control"].copy()
    rows = []
    for frame_name, frame in (("target_model_dev", target_df), ("protected_control", control_df)):
        if frame.empty:
            continue
        rows.append(
            {
                "row_type": "split_summary",
                "split": frame_name,
                "dataset_count": int(len(frame)),
                "mean_headroom_vs_best_single": float(frame["headroom_vs_best_single"].mean()),
                "median_headroom_vs_best_single": float(frame["headroom_vs_best_single"].median()),
                "min_headroom_vs_best_single": float(frame["headroom_vs_best_single"].min()),
                "max_headroom_vs_best_single": float(frame["headroom_vs_best_single"].max()),
                "positive_headroom_count": int((frame["headroom_vs_best_single"] > 0).sum()),
            }
        )
    detail_rows = split_df[
        split_df["role"].isin(
            ["target_model_dev", "protected_control", "heldout_validation_only", "counterexample_not_target"]
        )
    ].copy()
    detail_rows["row_type"] = "dataset"
    keep_cols = [
        "row_type",
        "dataset",
        "role",
        "regime",
        "best_single_expert",
        "headroom_vs_best_single",
        "anchor_minus_adaptive_stat",
        "batch_classes_loaded",
        "selector_margin_vs_best_single",
    ]
    return pd.concat(
        [
            detail_rows[keep_cols],
            pd.DataFrame(rows),
        ],
        ignore_index=True,
        sort=False,
    )


def render_headroom_diagnosis(split_df: pd.DataFrame) -> str:
    target_df = split_df[split_df["role"] == "target_model_dev"].copy()
    control_df = split_df[split_df["role"] == "protected_control"].copy()
    heldout_df = split_df[split_df["role"] == "heldout_validation_only"].copy()
    counterexample_df = split_df[split_df["role"] == "counterexample_not_target"].copy()

    target_mean = float(target_df["headroom_vs_best_single"].mean())
    control_mean = float(control_df["headroom_vs_best_single"].mean())
    winners = target_df["best_single_expert"].astype(str).value_counts().to_dict()

    lines = [
        "# Regime Headroom Diagnosis",
        "",
        "## Target Split Signal",
        f"- Target modeling datasets: {', '.join(target_df['dataset'].astype(str).tolist())}.",
        f"- Mean target headroom vs best single expert: {fmt_float(target_mean)}.",
        f"- Target headroom range: {fmt_float(target_df['headroom_vs_best_single'].min())} to {fmt_float(target_df['headroom_vs_best_single'].max())}.",
        f"- Protected control mean headroom: {fmt_float(control_mean)}.",
        "",
        "## Interpretation",
        "- The route has enough target-only headroom to justify an analysis-stage probe because all modeling datasets have positive headroom over the anchor.",
        "- The remaining gap is still heterogeneous across winners, so the route should be treated as a narrow hypothesis test rather than as a universal replacement story.",
        f"- Target winners are distributed across {len(winners)} expert families: {json.dumps(winners, ensure_ascii=True)}.",
        "",
        "## Guardrails",
        "- `adaptive_hybrid_hvg` remains the safe anchor and is not being replaced globally.",
        "- Protected controls stay outside optimization and are only used to veto unsafe movement.",
        "- Held-out donor panels remain untouched until after both the analysis gate and the target+control benchmark gate.",
    ]
    if not counterexample_df.empty:
        lines.extend(
            [
                "",
                "## Route-Limiting Counterexample",
                f"- `{counterexample_df.iloc[0]['dataset']}` is excluded because the anchor already beats the best single expert by {fmt_float(-counterexample_df.iloc[0]['headroom_vs_best_single'])}.",
                "- This is positive evidence that the donor-aware route, if it works at all, is not universal even inside the broader batch-heavy taxonomy.",
            ]
        )
    if not heldout_df.empty:
        lines.extend(
            [
                "",
                "## External Validation Gap",
                "- The repo does not currently contain a truly external donor-heavy dataset outside the benchmark pool.",
                "- Internal held-out donor panels exist, but a paper-grade external validation step still needs new data acquisition.",
            ]
        )
    return "\n".join(lines) + "\n"


def render_candidate_spec() -> str:
    lines = [
        "# Candidate Method Spec",
        "",
        "## Selected Candidate",
        f"- Candidate: `{CANDIDATE_METHOD}`.",
        "- Implementation status: already present in `src/hvg_research/adaptive_stat.py`; this route reuses the existing scorer rather than inventing a new family.",
        "",
        "## Mechanism",
        "- Start from count-aware core signals already available in the repo.",
        "- Build pseudo-environments from real batches when a meaningful batch split exists; otherwise fall back to sampled k-means pseudo-environments.",
        "- Aggregate per-environment multinomial-deviance and MV-residual ranks with a worst-group plus mean-group style summary.",
        "- Penalize cross-environment instability and lightly blend back toward fano / variance only when profile signals indicate trajectory pressure or atlas guard pressure.",
        "",
        "## Current Formula Family",
        "- Score uses worst-group deviance, mean deviance, MV residual support, global deviance support, and a small Pearson residual term.",
        "- A cross-environment standard-deviation penalty discourages donor-specific shortcuts.",
        "",
        "## Why This Candidate And Not A New One",
        "- The prompt explicitly forbids reopening a new longlist or parallel family search.",
        "- This scorer is already the repo's closest donor-aware count-residual candidate.",
        "- The right next step is to re-evaluate it under the new regime-specific scope rather than add another mechanism family.",
        "",
        "## Minimal Ablation Frame",
        "- Baseline 1: `adaptive_hybrid_hvg` (safe anchor).",
        "- Optional ablations if the route survives gating: remove environment penalty; remove MV residual support.",
        "- No second backup candidate is introduced in this route.",
    ]
    return "\n".join(lines) + "\n"


def render_analysis_gate_definition() -> str:
    lines = [
        "# Analysis Gate Definition",
        "",
        "## Positive Set",
    ]
    for dataset_name in TARGET_MODELING_DATASETS:
        lines.append(f"- `{dataset_name}`")
    lines.extend(
        [
            "",
            "## Protected Controls",
        ]
    )
    for dataset_name in PROTECTED_CONTROL_DATASETS:
        lines.append(f"- `{dataset_name}`")
    lines.extend(
        [
            "",
            "## Gate Metrics",
            "- `targeted_shift_gap`: mean(target top-k shift vs anchor) - mean(control top-k shift vs anchor) >= 0.03",
            "- `winner_overlap_pull_positive`: mean(target delta overlap to best single vs anchor) > 0.00",
            "- `winner_overlap_pull_gap`: mean(target overlap pull) - mean(control overlap pull) >= 0.015",
            "- `winner_corr_pull_gap`: mean(target corr pull) - mean(control corr pull) >= 0.01",
            "- `control_guard`: mean(control overlap pull) >= -0.02",
            "",
            "## Pass Rule",
            "- Pass if at least 4 of the 5 conditions hold.",
            "- If analysis fails, stop immediately and do not run the target+control benchmark.",
            "",
            "## Benchmark Unlock Rule",
            "- Only after analysis passes do we benchmark `{CANDIDATE_METHOD}` against `{ANCHOR_METHOD}` on target_model_dev + protected_control.",
            "- Held-out donor panels stay untouched until after this benchmark stage, and in the current repo they remain reserved because no external dataset is available.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_execution_plan(device_info: dict[str, object]) -> str:
    lines = [
        "# Execution Plan",
        "",
        "## Current Execution Context",
        f"- device={device_info['device']}",
        f"- cuda_available={device_info['cuda_available']}",
        f"- cuda_count={device_info['cuda_count']}",
        "",
        "## Repro Command",
        "```bash",
        "python scripts/run_regime_specific_route.py \\",
        "  --output-dir artifacts_regime_specific_route \\",
        "  --seed 7 \\",
        "  --top-k 200 \\",
        "  --bootstrap-samples 1",
        "```",
        "",
        "## Funnel",
        "1. Build dataset split and headroom diagnosis from existing artifacts.",
        "2. Re-run analysis-only diagnostics for `adaptive_invariant_residual_hvg` on the target_model_dev datasets against protected controls.",
        f"3. If analysis passes, run the minimal target+control benchmark for `{CANDIDATE_METHOD}` vs `{ANCHOR_METHOD}` only.",
        "4. Keep internal donor-heavy holdouts untouched; document the missing truly external validation dataset.",
    ]
    return "\n".join(lines) + "\n"


def load_dataset_resources(real_data_root: Path) -> pkg.DatasetResources:
    manifest_df = pd.read_csv(ROOT / "artifacts_topconf_selector_round2" / "dataset_manifest.csv")
    return pkg.load_dataset_resources(real_data_root=real_data_root, manifest_df=manifest_df)


def build_analysis_rows(
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
    analysis_datasets = (*TARGET_MODELING_DATASETS, *PROTECTED_CONTROL_DATASETS)
    for dataset_name in analysis_datasets:
        dataset = dataset_cache.get(dataset_name, seed)
        current_top_k = min(int(top_k), int(dataset.counts.shape[1]))
        best_single_method = str(headroom_lookup.loc[dataset_name, "best_single_expert"])
        methods_to_compute = tuple(dict.fromkeys([ANCHOR_METHOD, CANDIDATE_METHOD, best_single_method]))
        score_cache: dict[str, np.ndarray] = {}
        topk_cache: dict[str, np.ndarray] = {}
        for method_name in methods_to_compute:
            score_cache[method_name] = np.asarray(
                registry[method_name](dataset.counts, dataset.batches, current_top_k),
                dtype=np.float64,
            )
            topk_cache[method_name] = topk_indices(score_cache[method_name], current_top_k)

        anchor_scores = score_cache[ANCHOR_METHOD]
        anchor_topk = topk_cache[ANCHOR_METHOD]
        best_scores = score_cache[best_single_method]
        best_topk = topk_cache[best_single_method]
        group_name = "positive_headroom" if dataset_name in TARGET_MODELING_DATASETS else "atlas_control"
        anchor_overlap_to_best = jaccard(anchor_topk, best_topk)
        anchor_corr_to_best = spearman_correlation(anchor_scores, best_scores)

        candidate_scores = score_cache[CANDIDATE_METHOD]
        candidate_topk = topk_cache[CANDIDATE_METHOD]
        rows.append(
            {
                "dataset": dataset_name,
                "group_name": group_name,
                "method": CANDIDATE_METHOD,
                "best_single_method": best_single_method,
                "rank_corr_to_anchor": spearman_correlation(candidate_scores, anchor_scores),
                "topk_overlap_to_anchor": jaccard(candidate_topk, anchor_topk),
                "topk_shift_vs_anchor": 1.0 - jaccard(candidate_topk, anchor_topk),
                "topk_overlap_to_best_single": jaccard(candidate_topk, best_topk),
                "delta_overlap_to_best_single_vs_anchor": jaccard(candidate_topk, best_topk) - anchor_overlap_to_best,
                "rank_corr_to_best_single": spearman_correlation(candidate_scores, best_scores),
                "delta_rank_corr_to_best_single_vs_anchor": spearman_correlation(candidate_scores, best_scores) - anchor_corr_to_best,
                "score_dispersion_ratio_vs_anchor": float(
                    np.std(candidate_scores) / max(np.std(anchor_scores), 1e-8)
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_analysis(df: pd.DataFrame) -> pd.DataFrame:
    positive_group = df[df["group_name"] == "positive_headroom"].copy()
    control_group = df[df["group_name"] == "atlas_control"].copy()
    positive_shift = mean_or_nan(positive_group["topk_shift_vs_anchor"])
    control_shift = mean_or_nan(control_group["topk_shift_vs_anchor"])
    positive_overlap_pull = mean_or_nan(positive_group["delta_overlap_to_best_single_vs_anchor"])
    control_overlap_pull = mean_or_nan(control_group["delta_overlap_to_best_single_vs_anchor"])
    positive_corr_pull = mean_or_nan(positive_group["delta_rank_corr_to_best_single_vs_anchor"])
    control_corr_pull = mean_or_nan(control_group["delta_rank_corr_to_best_single_vs_anchor"])
    conditions = {
        "targeted_shift_gap": (positive_shift - control_shift) >= 0.03,
        "winner_overlap_pull_positive": positive_overlap_pull > 0.0,
        "winner_overlap_pull_gap": (positive_overlap_pull - control_overlap_pull) >= 0.015,
        "winner_corr_pull_gap": (positive_corr_pull - control_corr_pull) >= 0.01,
        "control_guard": control_overlap_pull >= -0.02,
    }
    summary_row = {
        "method": CANDIDATE_METHOD,
        "positive_shift_vs_anchor": positive_shift,
        "control_shift_vs_anchor": control_shift,
        "positive_minus_control_shift": positive_shift - control_shift,
        "positive_overlap_pull_vs_anchor": positive_overlap_pull,
        "control_overlap_pull_vs_anchor": control_overlap_pull,
        "overlap_pull_gap": positive_overlap_pull - control_overlap_pull,
        "positive_corr_pull_vs_anchor": positive_corr_pull,
        "control_corr_pull_vs_anchor": control_corr_pull,
        "corr_pull_gap": positive_corr_pull - control_corr_pull,
        "mean_rank_corr_to_anchor": float(df["rank_corr_to_anchor"].mean()),
        "mean_score_dispersion_ratio_vs_anchor": float(df["score_dispersion_ratio_vs_anchor"].mean()),
        "analysis_condition_count": int(sum(bool(value) for value in conditions.values())),
    }
    summary_row["analysis_pass"] = bool(summary_row["analysis_condition_count"] >= 4)
    for condition_name, value in conditions.items():
        summary_row[f"condition_{condition_name}"] = bool(value)
    return pd.DataFrame([summary_row])


def run_minimal_benchmark(
    *,
    dataset_cache: pkg.DatasetCache,
    resources: pkg.DatasetResources,
    top_k: int,
    seed: int,
    refine_epochs: int,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_names = (ANCHOR_METHOD, CANDIDATE_METHOD)
    dataset_names = (*TARGET_MODELING_DATASETS, *PROTECTED_CONTROL_DATASETS)
    rows: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        dataset = dataset_cache.get(dataset_name, seed)
        spec = resources.spec_map[dataset_name]
        rows.extend(
            rr1.run_round1_dataset_benchmark(
                dataset=dataset,
                dataset_id=spec.dataset_id,
                spec=spec,
                method_names=method_names,
                gate_model_path=None,
                refine_epochs=refine_epochs,
                top_k=top_k,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
            )
        )
    raw_df = rr1.add_run_level_scores(pd.DataFrame(rows))
    summary_df = rr1.summarize_by_keys(raw_df, keys=["dataset", "dataset_id", "method"])
    summary_df = rr1.rank_within_group(summary_df, group_cols=["dataset", "dataset_id"], rank_col="dataset_rank")

    candidate_df = summary_df[summary_df["method"] == CANDIDATE_METHOD].copy().set_index("dataset")
    anchor_df = summary_df[summary_df["method"] == ANCHOR_METHOD].copy().set_index("dataset")
    joined = candidate_df.join(
        anchor_df[
            [
                "overall_score",
                "cluster_silhouette",
                "stability",
                "neighbor_preservation",
                "weighted_marker_recall_at_50",
                "runtime_sec",
            ]
        ].add_prefix("anchor_"),
        how="inner",
    )
    joined = joined.reset_index()
    joined["overall_delta_vs_anchor"] = joined["overall_score"] - joined["anchor_overall_score"]
    joined["cluster_delta_vs_anchor"] = joined["cluster_silhouette"] - joined["anchor_cluster_silhouette"]
    joined["stability_delta_vs_anchor"] = joined["stability"] - joined["anchor_stability"]
    joined["neighbor_delta_vs_anchor"] = joined["neighbor_preservation"] - joined["anchor_neighbor_preservation"]
    joined["biology_delta_vs_anchor"] = (
        joined["weighted_marker_recall_at_50"] - joined["anchor_weighted_marker_recall_at_50"]
    )
    joined["runtime_ratio_vs_anchor"] = joined["runtime_sec"] / np.maximum(joined["anchor_runtime_sec"], 1e-8)

    benchmark_row = {
        "method": CANDIDATE_METHOD,
        "positive_headroom_mean_delta": float(
            joined[joined["dataset"].isin(TARGET_MODELING_DATASETS)]["overall_delta_vs_anchor"].mean()
        ),
        "atlas_like_mean_delta": float(
            joined[joined["dataset"].isin(PROTECTED_CONTROL_DATASETS)]["overall_delta_vs_anchor"].mean()
        ),
        "mean_cluster_delta": float(joined["cluster_delta_vs_anchor"].mean()),
        "mean_stability_delta": float(joined["stability_delta_vs_anchor"].mean()),
        "mean_neighbor_delta": float(joined["neighbor_delta_vs_anchor"].mean()),
        "mean_biology_delta": float(joined["biology_delta_vs_anchor"].mean()),
        "mean_runtime_ratio_vs_anchor": float(joined["runtime_ratio_vs_anchor"].mean()),
    }
    benchmark_row["benchmark_pass"] = bool(
        benchmark_row["positive_headroom_mean_delta"] > 0.0
        and benchmark_row["atlas_like_mean_delta"] >= -0.05
        and benchmark_row["mean_biology_delta"] >= -0.02
        and benchmark_row["mean_runtime_ratio_vs_anchor"] <= 1.50
        and (benchmark_row["mean_cluster_delta"] + benchmark_row["mean_stability_delta"]) > 0.0
    )
    return raw_df, joined, pd.DataFrame([benchmark_row])


def render_final_status(
    *,
    split_df: pd.DataFrame,
    analysis_summary_df: pd.DataFrame,
    benchmark_summary_df: pd.DataFrame | None,
    output_dir: Path,
) -> str:
    analysis_row = analysis_summary_df.iloc[0]
    benchmark_ran = benchmark_summary_df is not None and not benchmark_summary_df.empty
    benchmark_pass = bool(benchmark_summary_df.iloc[0]["benchmark_pass"]) if benchmark_ran else False
    if not bool(analysis_row["analysis_pass"]):
        go_no_go = "no-go for now"
        narrative = "analysis gate failed, so the route stops before full benchmark."
    elif benchmark_ran and benchmark_pass:
        go_no_go = "conditional go"
        narrative = "analysis and internal target+control benchmark passed, but external validation is still missing."
    elif benchmark_ran:
        go_no_go = "no-go for now"
        narrative = "analysis passed but the target+control benchmark did not clear the benchmark gate."
    else:
        go_no_go = "analysis-only provisional"
        narrative = "analysis passed but no benchmark result was produced."

    protected_controls = ", ".join(split_df[split_df["role"] == "protected_control"]["dataset"].astype(str).tolist())
    heldout_validation = ", ".join(split_df[split_df["role"] == "heldout_validation_only"]["dataset"].astype(str).tolist())
    target_regime = ", ".join(split_df[split_df["role"] == "target_model_dev"]["dataset"].astype(str).tolist())

    artifact_paths = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    )
    if "final_status.md" not in artifact_paths:
        artifact_paths.append("final_status.md")
        artifact_paths = sorted(artifact_paths)
    lines = [
        "# Final Status",
        "",
        "## Required Summary",
        f"- New problem definition: {QUESTION_ONE_LINER}",
        f"- Target regime summary: modeling datasets={target_regime}",
        f"- Protected controls: {protected_controls}",
        f"- Held-out validation: {heldout_validation}",
        "- Candidate implementation entered: no new scorer was added; the route reused the existing `adaptive_invariant_residual_hvg` implementation.",
        f"- Analysis gates passed: {bool(analysis_row['analysis_pass'])}",
        f"- Entered full benchmark: {benchmark_ran}",
        f"- Current donor-aware route decision: {go_no_go} ({narrative})",
        "",
        "## Gate Readout",
        f"- targeted_shift_gap={fmt_float(analysis_row['positive_minus_control_shift'])} pass={bool(analysis_row['condition_targeted_shift_gap'])}",
        f"- winner_overlap_pull_positive={fmt_float(analysis_row['positive_overlap_pull_vs_anchor'])} pass={bool(analysis_row['condition_winner_overlap_pull_positive'])}",
        f"- winner_overlap_pull_gap={fmt_float(analysis_row['overlap_pull_gap'])} pass={bool(analysis_row['condition_winner_overlap_pull_gap'])}",
        f"- winner_corr_pull_gap={fmt_float(analysis_row['corr_pull_gap'])} pass={bool(analysis_row['condition_winner_corr_pull_gap'])}",
        f"- control_guard={fmt_float(analysis_row['control_overlap_pull_vs_anchor'])} pass={bool(analysis_row['condition_control_guard'])}",
    ]
    if benchmark_ran:
        benchmark_row = benchmark_summary_df.iloc[0]
        lines.extend(
            [
                "",
                "## Benchmark Readout",
                f"- positive_headroom_mean_delta={fmt_float(benchmark_row['positive_headroom_mean_delta'])}",
                f"- atlas_like_mean_delta={fmt_float(benchmark_row['atlas_like_mean_delta'])}",
                f"- mean_biology_delta={fmt_float(benchmark_row['mean_biology_delta'])}",
                f"- mean_runtime_ratio_vs_anchor={fmt_float(benchmark_row['mean_runtime_ratio_vs_anchor'])}",
                f"- benchmark_pass={bool(benchmark_row['benchmark_pass'])}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Benchmark Readout",
                "- Benchmark was not run because the analysis gate did not unlock it.",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifact Paths",
        ]
    )
    for name in artifact_paths:
        lines.append(f"- `{name}`")
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

    headroom_df = load_headroom_table()
    failure_df = load_failure_taxonomy()
    split_df = build_dataset_split(headroom_df=headroom_df, failure_df=failure_df)
    split_df.to_csv(output_dir / "dataset_split.csv", index=False)

    (output_dir / "regime_definition.md").write_text(
        render_regime_definition(split_df),
        encoding="utf-8",
    )
    (output_dir / "dataset_split.md").write_text(
        render_dataset_split_md(split_df),
        encoding="utf-8",
    )

    headroom_summary_df = build_headroom_summary(split_df)
    headroom_summary_df.to_csv(output_dir / "regime_headroom_summary.csv", index=False)
    (output_dir / "regime_headroom_diagnosis.md").write_text(
        render_headroom_diagnosis(split_df),
        encoding="utf-8",
    )
    (output_dir / "candidate_method_spec.md").write_text(
        render_candidate_spec(),
        encoding="utf-8",
    )
    (output_dir / "analysis_gate_definition.md").write_text(
        render_analysis_gate_definition(),
        encoding="utf-8",
    )
    (output_dir / "execution_plan.md").write_text(
        render_execution_plan(device_info),
        encoding="utf-8",
    )

    resources = load_dataset_resources(Path(args.real_data_root))
    dataset_cache = pkg.DatasetCache(resources)
    analysis_df = build_analysis_rows(
        dataset_cache=dataset_cache,
        headroom_df=headroom_df,
        top_k=args.top_k,
        seed=args.seed,
        refine_epochs=args.refine_epochs,
    )
    analysis_summary_df = summarize_analysis(analysis_df)
    analysis_df.to_csv(output_dir / "analysis_dataset_metrics.csv", index=False)
    analysis_summary_df.to_csv(output_dir / "analysis_gate_summary.csv", index=False)

    benchmark_summary_df: pd.DataFrame | None = None
    if bool(analysis_summary_df.iloc[0]["analysis_pass"]):
        benchmark_raw_df, benchmark_delta_df, benchmark_summary_df = run_minimal_benchmark(
            dataset_cache=dataset_cache,
            resources=resources,
            top_k=args.top_k,
            seed=args.seed,
            refine_epochs=args.refine_epochs,
            bootstrap_samples=args.bootstrap_samples,
        )
        benchmark_raw_df.to_csv(output_dir / "benchmark_candidate_vs_anchor_raw.csv", index=False)
        benchmark_delta_df.to_csv(output_dir / "benchmark_candidate_vs_anchor_dataset_deltas.csv", index=False)
        benchmark_summary_df.to_csv(output_dir / "benchmark_candidate_vs_anchor_summary.csv", index=False)

    (output_dir / "final_status.md").write_text(
        render_final_status(
            split_df=split_df,
            analysis_summary_df=analysis_summary_df,
            benchmark_summary_df=benchmark_summary_df,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
