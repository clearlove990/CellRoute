from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spatial_context.cross_sample_differential import build_sample_level_motif_table  # noqa: E402
from spatial_context.motif_embedding import (  # noqa: E402
    SampleAwareMotifFitResult,
    align_sample_aware_motif_model_to_reference,
    assign_sample_aware_motifs,
    build_tissue_motif_feature_bundle,
    fit_sample_aware_tissue_motif_model,
)
from spatial_context.neighborhood import get_runtime_info, load_spatial_h5ad, summarize_neighborhoods  # noqa: E402


WEEK6_DIR = ROOT / "experiments" / "06_cross_sample_differential"
WEEK6B_DIR = ROOT / "experiments" / "06b_sample_aware_cross_sample_differential"
WEEK7A_DIR = ROOT / "experiments" / "07_positive_motif_validation"
WEEK7B_DIR = ROOT / "experiments" / "07_statistical_validity_controls"

AD_DATASET = ROOT / "data" / "spatial_processed" / "gse220442_ad_mtg.h5ad"
BREAST_DATASET = ROOT / "data" / "spatial_processed" / "wu2021_breast_visium.h5ad"

AD_REFERENCE = {
    "dataset_id": "gse220442_ad_mtg",
    "week6_motif_id": "motif_02",
    "week6b_motif_id": "motif_08",
    "week6_label": "Layer 2 + Layer 3 | sparse highE",
    "week6b_label": "Layer 2 + Layer 3 | mixed highE",
    "pathway": "upper_layer_neuron",
    "marker_genes": ["LTK", "NEUROD1", "CARTPT", "MGP", "CALB2", "LAMP5", "C1QL2", "WNT3"],
}

BREAST_DIRECTION_FLIP_MOTIFS = ("motif_03", "motif_01")
DATASET_COLORS = {
    "wu2021_breast_visium": "#C66A3D",
    "gse220442_ad_mtg": "#2E6F95",
}
SUPPORT_MARKERS = {
    "sample_level_supported": "o",
    "heldout_supported": "o",
    "naive_only": "s",
    "not_supported": "^",
}
SHORT_DATASET_NAMES = {
    "wu2021_breast_visium": "breast",
    "gse220442_ad_mtg": "AD",
}


@dataclass(frozen=True)
class SampleAwareArtifact:
    dataset: object
    dataset_id: str
    dataset_name: str
    full_fit: SampleAwareMotifFitResult
    full_sample_level: pd.DataFrame
    loso_spot_table: pd.DataFrame
    loso_sample_level: pd.DataFrame
    fold_summary: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Week 7A and Week 7B experiment outputs.")
    parser.add_argument(
        "--track",
        choices=("week7a", "week7b", "both"),
        default="both",
        help="Which experiment track to materialize.",
    )
    parser.add_argument("--top-variable-genes", type=int, default=256)
    parser.add_argument("--n-expression-programs", type=int, default=6)
    parser.add_argument("--radius-factor", type=float, default=1.6)
    parser.add_argument("--max-train-spots-per-sample", type=int, default=2048)
    parser.add_argument("--random-state", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_info = get_runtime_info()
    week6_diff = pd.read_csv(WEEK6_DIR / "results" / "differential_test_results.csv")
    week6_sample_level = pd.read_csv(WEEK6_DIR / "results" / "sample_level_motif_table.csv")
    week6b_compare = pd.read_csv(WEEK6B_DIR / "results" / "week6_vs_06b_comparison.csv")
    week6b_decision = pd.read_csv(WEEK6B_DIR / "results" / "motif_stability_decision_matrix.csv")
    week6b_consistency = pd.read_csv(WEEK6B_DIR / "results" / "full_vs_loso_differential_consistency.csv")
    week6b_diff = pd.read_csv(WEEK6B_DIR / "results" / "sample_aware_differential_test_results.csv")
    robust_tracking = pd.read_csv(WEEK6B_DIR / "results" / "robust_motif_tracking.csv")

    artifact_cache: dict[str, SampleAwareArtifact] = {}
    if args.track in {"week7b", "both"}:
        artifact_cache["wu2021_breast_visium"] = rebuild_sample_aware_artifact(
            dataset_path=BREAST_DATASET,
            runtime_info=runtime_info,
            args=args,
        )
        build_week7b_outputs(
            runtime_info=runtime_info,
            week6_diff=week6_diff,
            week6b_compare=week6b_compare,
            week6b_decision=week6b_decision,
            week6b_consistency=week6b_consistency,
            week6b_diff=week6b_diff,
            breast_artifact=artifact_cache["wu2021_breast_visium"],
        )
    if args.track in {"week7a", "both"}:
        artifact_cache["gse220442_ad_mtg"] = rebuild_sample_aware_artifact(
            dataset_path=AD_DATASET,
            runtime_info=runtime_info,
            args=args,
        )
        build_week7a_outputs(
            runtime_info=runtime_info,
            week6_sample_level=week6_sample_level,
            robust_tracking=robust_tracking,
            week6b_diff=week6b_diff,
            ad_artifact=artifact_cache["gse220442_ad_mtg"],
        )


def rebuild_sample_aware_artifact(
    *,
    dataset_path: Path,
    runtime_info,
    args: argparse.Namespace,
) -> SampleAwareArtifact:
    dataset = load_spatial_h5ad(dataset_path)
    neighborhood_summary = summarize_neighborhoods(
        dataset,
        runtime_info=runtime_info,
        radius_factor=args.radius_factor,
    )
    feature_bundle = build_tissue_motif_feature_bundle(
        dataset,
        neighborhood_summary,
        runtime_info=runtime_info,
        n_expression_programs=args.n_expression_programs,
        top_variable_genes=args.top_variable_genes,
        random_state=args.random_state,
    )
    full_fit = fit_sample_aware_tissue_motif_model(
        dataset,
        neighborhood_summary,
        runtime_info=runtime_info,
        n_expression_programs=args.n_expression_programs,
        top_variable_genes=args.top_variable_genes,
        feature_bundle=feature_bundle,
        max_train_spots_per_sample=args.max_train_spots_per_sample,
        random_state=args.random_state,
    )
    full_sample_level = build_sample_level_motif_table(
        dataset=dataset,
        spot_table=full_fit.embedding_result.spot_table,
        feature_frame=full_fit.embedding_result.feature_frame,
        expression_program_metadata=full_fit.embedding_result.expression_program_metadata,
        neighborhood_summary=neighborhood_summary,
        analysis_scope="full_model",
    )
    loso_spot_table, fold_summary = build_loso_assignments(
        dataset=dataset,
        neighborhood_summary=neighborhood_summary,
        feature_bundle=feature_bundle,
        reference_fit=full_fit,
        runtime_info=runtime_info,
        args=args,
    )
    loso_sample_level = build_sample_level_motif_table(
        dataset=dataset,
        spot_table=loso_spot_table,
        feature_frame=full_fit.embedding_result.feature_frame,
        expression_program_metadata=full_fit.embedding_result.expression_program_metadata,
        neighborhood_summary=neighborhood_summary,
        analysis_scope="loso",
    )
    return SampleAwareArtifact(
        dataset=dataset,
        dataset_id=str(dataset.dataset_id),
        dataset_name=str(dataset.dataset_name),
        full_fit=full_fit,
        full_sample_level=full_sample_level,
        loso_spot_table=loso_spot_table,
        loso_sample_level=loso_sample_level,
        fold_summary=fold_summary,
    )


def build_loso_assignments(
    *,
    dataset,
    neighborhood_summary,
    feature_bundle,
    reference_fit: SampleAwareMotifFitResult,
    runtime_info,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = sorted(dataset.obs["sample_id"].astype(str).unique().tolist())
    spot_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    reference_spot_table = reference_fit.embedding_result.spot_table
    reference_label_map = reference_fit.frozen_model.motif_label_map

    for fold_index, held_out_sample in enumerate(sample_ids):
        train_samples = [sample_id for sample_id in sample_ids if sample_id != held_out_sample]
        fold_fit = fit_sample_aware_tissue_motif_model(
            dataset,
            neighborhood_summary,
            runtime_info=runtime_info,
            n_expression_programs=args.n_expression_programs,
            top_variable_genes=args.top_variable_genes,
            feature_bundle=feature_bundle,
            train_sample_ids=train_samples,
            max_train_spots_per_sample=args.max_train_spots_per_sample,
            fixed_n_clusters=reference_fit.embedding_result.n_clusters,
            random_state=args.random_state + fold_index + 1,
        )
        motif_id_map, alignment_cost_mean = align_sample_aware_motif_model_to_reference(
            fold_fit.frozen_model,
            reference_fit.frozen_model,
        )
        heldout_table = assign_sample_aware_motifs(
            dataset,
            neighborhood_summary,
            frozen_model=fold_fit.frozen_model,
            runtime_info=runtime_info,
            feature_bundle=feature_bundle,
            selected_sample_ids=[held_out_sample],
            motif_id_map=motif_id_map,
            motif_label_map=reference_label_map,
            n_expression_programs=args.n_expression_programs,
            top_variable_genes=args.top_variable_genes,
            random_state=args.random_state + fold_index + 1,
        )
        reference_subset = reference_spot_table.loc[heldout_table.index]
        agreement = float(
            np.mean(
                heldout_table["motif_id"].astype(str).to_numpy()
                == reference_subset["motif_id"].astype(str).to_numpy()
            )
        )
        fold_rows.append(
            {
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.dataset_name,
                "held_out_sample": held_out_sample,
                "n_train_samples": int(len(train_samples)),
                "n_train_spots": int(fold_fit.frozen_model.training_row_indices.size),
                "n_test_spots": int(heldout_table.shape[0]),
                "training_spots_per_sample": int(fold_fit.frozen_model.training_spots_per_sample),
                "alignment_cost_mean": float(alignment_cost_mean),
                "assignment_agreement_vs_full": agreement,
                "heldout_assignment_distance_mean": float(heldout_table["assignment_distance"].mean()),
                "pca_backend": str(fold_fit.frozen_model.pca_projection.backend),
            }
        )
        spot_frames.append(heldout_table)

    loso_spot_table = pd.concat(spot_frames, axis=0).sort_index()
    fold_summary = pd.DataFrame(fold_rows).sort_values(["dataset_id", "held_out_sample"]).reset_index(drop=True)
    return loso_spot_table, fold_summary


def build_week7b_outputs(
    *,
    runtime_info,
    week6_diff: pd.DataFrame,
    week6b_compare: pd.DataFrame,
    week6b_decision: pd.DataFrame,
    week6b_consistency: pd.DataFrame,
    week6b_diff: pd.DataFrame,
    breast_artifact: SampleAwareArtifact,
) -> None:
    results_dir = WEEK7B_DIR / "results"
    figures_dir = WEEK7B_DIR / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    week6_decision = build_week6_decision_matrix(week6_diff)
    false_positive_summary = build_false_positive_reduction_summary(week6_decision=week6_decision, week6b_decision=week6b_decision)
    stability_summary = build_full_loso_stability_summary(week6b_consistency)
    pvalue_inflation_summary = build_pvalue_inflation_summary(
        week6_vs_06b=week6b_compare,
        week6_diff=week6_diff,
        week6b_diff=week6b_diff,
    )
    direction_flip_details = build_direction_flip_details(
        consistency_table=week6b_consistency,
        differential_table=week6b_diff,
        artifact=breast_artifact,
    )
    final_decision_matrix = build_final_signal_decision_matrix(
        week6_vs_06b=week6b_compare,
        consistency_table=week6b_consistency,
    )

    false_positive_summary.to_csv(results_dir / "false_positive_reduction_summary.csv", index=False)
    stability_summary.to_csv(results_dir / "full_loso_stability_summary.csv", index=False)
    pvalue_inflation_summary.to_csv(results_dir / "pvalue_inflation_summary.csv", index=False)
    direction_flip_details.to_csv(results_dir / "direction_flip_motif_details.csv", index=False)
    final_decision_matrix.to_csv(results_dir / "final_signal_decision_matrix.csv", index=False)

    plot_false_positive_reduction(
        summary=false_positive_summary,
        output_path=figures_dir / "fig7b_false_positive_reduction.png",
    )
    plot_full_vs_loso_stability(
        consistency_table=week6b_consistency,
        stability_summary=stability_summary,
        output_path=figures_dir / "fig7b_full_vs_loso_stability.png",
    )
    plot_pvalue_inflation(
        summary=pvalue_inflation_summary,
        output_path=figures_dir / "fig7b_pvalue_inflation.png",
    )
    shutil.copyfile(
        figures_dir / "fig7b_pvalue_inflation.png",
        figures_dir / "pvalue_inflation_week6_vs_06b.png",
    )
    plot_breast_direction_flips(
        details=direction_flip_details,
        artifact=breast_artifact,
        output_path=figures_dir / "fig7b_breast_direction_flips.png",
    )
    shutil.copyfile(
        figures_dir / "fig7b_breast_direction_flips.png",
        figures_dir / "breast_direction_flip_examples.png",
    )

    write_week7b_protocol(runtime_info=runtime_info)
    write_week7b_analysis(
        false_positive_summary=false_positive_summary,
        stability_summary=stability_summary,
        pvalue_inflation_summary=pvalue_inflation_summary,
        direction_flip_details=direction_flip_details,
        final_decision_matrix=final_decision_matrix,
    )


def build_week7a_outputs(
    *,
    runtime_info,
    week6_sample_level: pd.DataFrame,
    robust_tracking: pd.DataFrame,
    week6b_diff: pd.DataFrame,
    ad_artifact: SampleAwareArtifact,
) -> None:
    results_dir = WEEK7A_DIR / "results"
    figures_dir = WEEK7A_DIR / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tracking_row = select_ad_tracking_row(robust_tracking)
    if tracking_row.empty:
        raise ValueError("Could not locate the AD motif_02 -> motif_08 tracking row.")

    tracking_summary = build_ad_tracking_summary(tracking_row)
    sample_abundance = build_ad_sample_abundance(
        week6_sample_level=week6_sample_level,
        artifact=ad_artifact,
        week6_motif_id=str(tracking_row["week6_motif_id"]),
        week6b_motif_id=str(tracking_row["matched_06b_motif_id"]),
    )
    marker_enrichment = build_ad_marker_enrichment_table(
        artifact=ad_artifact,
        week6b_motif_id=str(tracking_row["matched_06b_motif_id"]),
    )

    tracking_summary.to_csv(results_dir / "ad_motif_tracking_summary.csv", index=False)
    marker_enrichment.to_csv(results_dir / "ad_motif_marker_enrichment.csv", index=False)
    sample_abundance.to_csv(results_dir / "ad_motif_sample_abundance.csv", index=False)

    plot_ad_sample_level_fraction(
        abundance=sample_abundance,
        tracking_row=tracking_row,
        output_path=figures_dir / "ad_motif_sample_level_fraction.png",
    )
    plot_ad_marker_heatmap(
        marker_table=marker_enrichment,
        output_path=figures_dir / "ad_motif_marker_heatmap.png",
    )
    plot_ad_spatial_examples(
        artifact=ad_artifact,
        abundance=sample_abundance,
        motif_id=str(tracking_row["matched_06b_motif_id"]),
        output_path=figures_dir / "ad_motif_spatial_examples.png",
    )

    write_week7a_protocol(runtime_info=runtime_info)
    write_week7a_analysis(
        tracking_row=tracking_row,
        sample_abundance=sample_abundance,
        marker_table=marker_enrichment,
        differential_table=week6b_diff,
    )


def build_week6_decision_matrix(week6_diff: pd.DataFrame) -> pd.DataFrame:
    frame = week6_diff.loc[
        :,
        [
            "dataset_id",
            "dataset_name",
            "comparison_name",
            "motif_id",
            "motif_label",
            "holdout_delta_fraction",
            "holdout_sample_permutation_pvalue_two_sided",
            "holdout_label_max_t_pvalue",
            "holdout_naive_spot_pvalue",
            "holdout_synthetic_null_effect_pvalue",
            "holdout_controlled_support_tier",
        ],
    ].copy()
    frame["decision_label"] = frame.apply(
        lambda row: decision_label_from_metrics(
            support_tier=str(row["holdout_controlled_support_tier"]),
            naive_spot_p=float(row["holdout_naive_spot_pvalue"]),
        ),
        axis=1,
    )
    return frame


def build_false_positive_reduction_summary(
    *,
    week6_decision: pd.DataFrame,
    week6b_decision: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_id in sorted(set(week6_decision["dataset_id"].astype(str)) | set(week6b_decision["dataset_id"].astype(str))):
        left = week6_decision.loc[week6_decision["dataset_id"].astype(str) == str(dataset_id)].copy()
        right = week6b_decision.loc[week6b_decision["dataset_id"].astype(str) == str(dataset_id)].copy()
        week6_naive_only = int(np.sum(left["decision_label"].astype(str) == "naive_only_signal"))
        week6b_naive_only = int(np.sum(right["decision_label"].astype(str) == "naive_only_signal"))
        reduction_abs = int(week6_naive_only - week6b_naive_only)
        reduction_fraction = float(reduction_abs / week6_naive_only) if week6_naive_only > 0 else 0.0
        week6_robust = int(np.sum(left["decision_label"].astype(str) == "robust_biological_signal"))
        week6b_robust = int(np.sum(right["decision_label"].astype(str) == "robust_biological_signal"))
        if reduction_abs > 0:
            interpretation = (
                f"Naive-only motifs shrink from {week6_naive_only} to {week6b_naive_only}; "
                f"sample-aware controls remove {reduction_abs} calls while robust motifs move {week6_robust}->{week6b_robust}."
            )
        elif reduction_abs == 0:
            interpretation = (
                f"Naive-only count stays at {week6_naive_only}; "
                f"robust motifs move {week6_robust}->{week6b_robust}, so biological preservation does not fully solve inflation."
            )
        else:
            interpretation = (
                f"Naive-only count increases from {week6_naive_only} to {week6b_naive_only}; "
                f"inspect motif remapping because the control pipeline created extra unstable calls."
            )
        rows.append(
            {
                "dataset_id": dataset_id,
                "week6_naive_only_count": week6_naive_only,
                "week6b_naive_only_count": week6b_naive_only,
                "reduction_abs": reduction_abs,
                "reduction_fraction": reduction_fraction,
                "week6_robust_count": week6_robust,
                "week6b_robust_count": week6b_robust,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)


def build_full_loso_stability_summary(consistency_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_id, subset in consistency_table.groupby("dataset_id", observed=False):
        direction = float(np.mean(subset["effect_direction_consistent"].astype(bool).astype(float)))
        abundance = float(np.nanmedian(subset["motif_abundance_correlation"].to_numpy(dtype=np.float64)))
        flipped = subset.loc[~subset["effect_direction_consistent"].astype(bool), "motif_id"].astype(str).tolist()
        stable = subset.loc[subset["effect_direction_consistent"].astype(bool), "motif_id"].astype(str).tolist()
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "effect_direction_consistency": direction,
                "median_abundance_correlation": abundance,
                "n_direction_flips": int(len(flipped)),
                "flipped_motifs": ";".join(flipped),
                "stable_motifs": ";".join(stable),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)


def build_pvalue_inflation_summary(
    *,
    week6_vs_06b: pd.DataFrame,
    week6_diff: pd.DataFrame,
    week6b_diff: pd.DataFrame,
) -> pd.DataFrame:
    week6_naive = week6_diff.loc[:, ["dataset_id", "motif_id", "holdout_naive_spot_pvalue"]].rename(
        columns={
            "motif_id": "week6_motif_id",
            "holdout_naive_spot_pvalue": "naive_spot_p",
        }
    )
    week6b_naive = week6b_diff.loc[:, ["dataset_id", "motif_id", "loso_naive_spot_pvalue"]].rename(
        columns={
            "motif_id": "matched_06b_motif_id",
            "loso_naive_spot_pvalue": "sample_aware_naive_spot_p",
        }
    )
    frame = week6_vs_06b.merge(week6_naive, on=["dataset_id", "week6_motif_id"], how="left").merge(
        week6b_naive,
        on=["dataset_id", "matched_06b_motif_id"],
        how="left",
    )
    frame["sample_level_p"] = frame["week6_sample_p_two_sided"].astype(float)
    frame["sample_aware_p"] = frame["week6b_sample_p_two_sided"].astype(float)
    frame["max_t_p"] = frame["week6b_max_t_p"].astype(float)
    frame["null_motif_p"] = frame["week6b_null_effect_p"].astype(float)
    frame["support_tier"] = frame["week6b_controlled_support_tier"].astype(str)
    frame["highlight_label"] = frame.apply(
        lambda row: "AD motif_02 -> motif_08"
        if str(row["dataset_id"]) == AD_REFERENCE["dataset_id"]
        and str(row["week6_motif_id"]) == AD_REFERENCE["week6_motif_id"]
        and str(row["matched_06b_motif_id"]) == AD_REFERENCE["week6b_motif_id"]
        else "",
        axis=1,
    )
    frame["naive_vs_sample_level_log10_gap"] = frame.apply(
        lambda row: safe_neglog10(row["naive_spot_p"], floor=1.0e-300) - safe_neglog10(row["sample_level_p"], floor=1.0e-300),
        axis=1,
    )
    frame["naive_vs_sample_aware_log10_gap"] = frame.apply(
        lambda row: safe_neglog10(row["naive_spot_p"], floor=1.0e-300) - safe_neglog10(row["sample_aware_p"], floor=1.0e-300),
        axis=1,
    )
    frame["naive_vs_max_t_log10_gap"] = frame.apply(
        lambda row: safe_neglog10(row["naive_spot_p"], floor=1.0e-300) - safe_neglog10(row["max_t_p"], floor=1.0e-300),
        axis=1,
    )
    frame["naive_vs_null_log10_gap"] = frame.apply(
        lambda row: safe_neglog10(row["naive_spot_p"], floor=1.0e-300) - safe_neglog10(row["null_motif_p"], floor=1.0e-300),
        axis=1,
    )
    keep_columns = [
        "dataset_id",
        "week6_motif_id",
        "matched_06b_motif_id",
        "support_tier",
        "naive_spot_p",
        "sample_level_p",
        "sample_aware_p",
        "sample_aware_naive_spot_p",
        "max_t_p",
        "null_motif_p",
        "naive_vs_sample_level_log10_gap",
        "naive_vs_sample_aware_log10_gap",
        "naive_vs_max_t_log10_gap",
        "naive_vs_null_log10_gap",
        "highlight_label",
    ]
    return frame.loc[:, keep_columns].sort_values(
        ["dataset_id", "naive_vs_sample_aware_log10_gap", "week6_motif_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_direction_flip_details(
    *,
    consistency_table: pd.DataFrame,
    differential_table: pd.DataFrame,
    artifact: SampleAwareArtifact,
) -> pd.DataFrame:
    full_table = artifact.full_sample_level.loc[:, ["sample_id", "condition", "motif_id", "motif_fraction"]].rename(
        columns={"motif_fraction": "full_fraction"}
    )
    loso_table = artifact.loso_sample_level.loc[:, ["sample_id", "condition", "motif_id", "motif_fraction"]].rename(
        columns={"motif_fraction": "loso_fraction"}
    )
    rows: list[dict[str, object]] = []
    for motif_id in BREAST_DIRECTION_FLIP_MOTIFS:
        diff_row = differential_table.loc[
            (differential_table["dataset_id"].astype(str) == artifact.dataset_id)
            & (differential_table["motif_id"].astype(str) == str(motif_id))
        ]
        consistency_row = consistency_table.loc[
            (consistency_table["dataset_id"].astype(str) == artifact.dataset_id)
            & (consistency_table["motif_id"].astype(str) == str(motif_id))
        ]
        if diff_row.empty or consistency_row.empty:
            continue
        diff_row = diff_row.iloc[0]
        consistency_row = consistency_row.iloc[0]
        merged = full_table.loc[full_table["motif_id"].astype(str) == str(motif_id)].merge(
            loso_table.loc[loso_table["motif_id"].astype(str) == str(motif_id)],
            on=["sample_id", "condition", "motif_id"],
            how="outer",
        ).fillna({"full_fraction": 0.0, "loso_fraction": 0.0})
        merged["abs_fraction_shift"] = np.abs(
            merged["full_fraction"].to_numpy(dtype=np.float64) - merged["loso_fraction"].to_numpy(dtype=np.float64)
        )
        example_sample = str(
            merged.sort_values(["abs_fraction_shift", "sample_id"], ascending=[False, True]).iloc[0]["sample_id"]
        )
        for _, row in merged.sort_values(["condition", "sample_id"]).iterrows():
            rows.append(
                {
                    "dataset_id": artifact.dataset_id,
                    "motif_id": motif_id,
                    "motif_label": str(diff_row["motif_label"]),
                    "sample_id": str(row["sample_id"]),
                    "condition": str(row["condition"]),
                    "full_fraction": float(row["full_fraction"]),
                    "loso_fraction": float(row["loso_fraction"]),
                    "abs_fraction_shift": float(row["abs_fraction_shift"]),
                    "example_sample_for_maps": bool(str(row["sample_id"]) == example_sample),
                    "full_delta_fraction": float(diff_row["full_delta_fraction"]),
                    "loso_delta_fraction": float(diff_row["loso_delta_fraction"]),
                    "effect_direction_consistent": bool(consistency_row["effect_direction_consistent"]),
                    "full_sample_p_two_sided": float(diff_row["full_sample_permutation_pvalue_two_sided"]),
                    "loso_sample_p_two_sided": float(diff_row["loso_sample_permutation_pvalue_two_sided"]),
                    "full_naive_spot_p": float(diff_row["full_naive_spot_pvalue"]),
                    "loso_naive_spot_p": float(diff_row["loso_naive_spot_pvalue"]),
                    "loso_max_t_p": float(diff_row["loso_label_max_t_pvalue"]),
                    "loso_null_effect_p": float(diff_row["loso_synthetic_null_effect_pvalue"]),
                    "full_support_tier": str(diff_row["full_controlled_support_tier"]),
                    "loso_support_tier": str(diff_row["loso_controlled_support_tier"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["motif_id", "condition", "sample_id"]).reset_index(drop=True)


def build_final_signal_decision_matrix(
    *,
    week6_vs_06b: pd.DataFrame,
    consistency_table: pd.DataFrame,
) -> pd.DataFrame:
    merged = week6_vs_06b.merge(
        consistency_table.loc[
            :,
            [
                "dataset_id",
                "motif_id",
                "effect_direction_consistent",
                "motif_abundance_correlation",
                "full_delta_fraction",
                "loso_delta_fraction",
            ],
        ].rename(columns={"motif_id": "matched_06b_motif_id"}),
        on=["dataset_id", "matched_06b_motif_id"],
        how="left",
    )
    labels: list[str] = []
    rationales: list[str] = []
    for _, row in merged.iterrows():
        week6_label = str(row["week6_decision_label"])
        week6b_label = str(row["week6b_decision_label"])
        direction_ok = bool(row.get("effect_direction_consistent", False))
        if not direction_ok and np.isfinite(pd.to_numeric(row.get("full_delta_fraction"), errors="coerce")):
            final_label = "unstable_direction_flip"
            rationale = (
                f"Full vs LOSO effect flips sign ({float(row['full_delta_fraction']):.3f} -> {float(row['loso_delta_fraction']):.3f}); "
                f"abundance correlation={float(row['motif_abundance_correlation']):.2f}."
            )
        elif week6b_label == "robust_biological_signal":
            final_label = "preserved_robust_signal"
            rationale = (
                f"Sample-aware LOSO control retains the signal with p={float(row['week6b_sample_p_two_sided']):.3f} "
                f"and support tier={row['week6b_controlled_support_tier']}."
            )
        elif week6_label == "naive_only_signal" and week6b_label == "naive_only_signal":
            final_label = "persistent_naive_only_signal"
            rationale = (
                f"Naive-only status persists after remapping; naive/control gap stays large "
                f"({float(row['week6b_naive_minus_controlled_log10p']):.2f} log10 units)."
            )
        elif week6_label == "naive_only_signal" and week6b_label != "naive_only_signal":
            final_label = "removed_naive_only_signal"
            rationale = (
                f"Sample-aware control removes the original naive-only call "
                f"({row['week6b_controlled_support_tier']} after remapping)."
            )
        else:
            final_label = "no_signal"
            rationale = "Neither the original pooled test nor the sample-aware remap produces a stable retained signal."
        labels.append(final_label)
        rationales.append(rationale)
    merged["final_signal_label"] = labels
    merged["rationale"] = rationales
    keep_columns = [
        "dataset_id",
        "comparison_name",
        "week6_motif_id",
        "week6_motif_label",
        "matched_06b_motif_id",
        "matched_06b_motif_label",
        "sample_fraction_correlation",
        "alignment_score",
        "week6_decision_label",
        "week6b_decision_label",
        "week6_controlled_support_tier",
        "week6b_controlled_support_tier",
        "effect_direction_consistent",
        "motif_abundance_correlation",
        "week6_sample_p_two_sided",
        "week6b_sample_p_two_sided",
        "week6b_max_t_p",
        "week6b_null_effect_p",
        "final_signal_label",
        "rationale",
    ]
    order = {
        "preserved_robust_signal": 0,
        "removed_naive_only_signal": 1,
        "persistent_naive_only_signal": 2,
        "unstable_direction_flip": 3,
        "no_signal": 4,
    }
    merged["label_order"] = merged["final_signal_label"].map(order).fillna(9)
    return merged.loc[:, keep_columns + ["label_order"]].sort_values(
        ["dataset_id", "label_order", "week6_motif_id"],
        ascending=[True, True, True],
    ).drop(columns="label_order").reset_index(drop=True)


def select_ad_tracking_row(robust_tracking: pd.DataFrame) -> pd.Series:
    matches = robust_tracking.loc[
        (robust_tracking["dataset_id"].astype(str) == AD_REFERENCE["dataset_id"])
        & (robust_tracking["week6_motif_id"].astype(str) == AD_REFERENCE["week6_motif_id"])
        & (robust_tracking["matched_06b_motif_id"].astype(str) == AD_REFERENCE["week6b_motif_id"])
    ]
    if not matches.empty:
        return matches.iloc[0]
    ordered = robust_tracking.loc[
        robust_tracking["dataset_id"].astype(str) == AD_REFERENCE["dataset_id"]
    ].sort_values(
        ["alignment_score", "loso_sample_p_two_sided", "loso_loso_sign_consistency"],
        ascending=[False, True, False],
    )
    return ordered.iloc[0] if not ordered.empty else pd.Series(dtype=object)


def build_ad_tracking_summary(tracking_row: pd.Series) -> pd.DataFrame:
    pathway_overlap = str(tracking_row["top_pathway"]) if bool(tracking_row["pathway_consistent_with_week6"]) else ""
    return pd.DataFrame(
        [
            {
                "week6_motif_id": str(tracking_row["week6_motif_id"]),
                "week6b_motif_id": str(tracking_row["matched_06b_motif_id"]),
                "sample_fraction_correlation": float(tracking_row["sample_fraction_correlation"]),
                "marker_overlap": int(tracking_row["marker_overlap_count"]),
                "pathway_overlap": pathway_overlap,
                "effect_direction_preserved": bool(
                    sign_match(
                        float(tracking_row["week6_delta_fraction"]),
                        float(tracking_row["loso_delta_fraction"]),
                    )
                ),
                "null_p": float(tracking_row["loso_null_effect_p"]),
                "loso_sign_consistency": float(tracking_row["loso_loso_sign_consistency"]),
            }
        ]
    )


def build_ad_sample_abundance(
    *,
    week6_sample_level: pd.DataFrame,
    artifact: SampleAwareArtifact,
    week6_motif_id: str,
    week6b_motif_id: str,
) -> pd.DataFrame:
    week6 = week6_sample_level.loc[
        (week6_sample_level["dataset_id"].astype(str) == artifact.dataset_id)
        & (week6_sample_level["analysis_scope"].astype(str) == "holdout_oof")
        & (week6_sample_level["motif_id"].astype(str) == str(week6_motif_id)),
        ["sample_id", "condition", "motif_fraction"],
    ].rename(columns={"motif_fraction": "week6_fraction"})
    full = artifact.full_sample_level.loc[
        artifact.full_sample_level["motif_id"].astype(str) == str(week6b_motif_id),
        ["sample_id", "condition", "motif_fraction"],
    ].rename(columns={"motif_fraction": "full_fraction"})
    loso = artifact.loso_sample_level.loc[
        artifact.loso_sample_level["motif_id"].astype(str) == str(week6b_motif_id),
        ["sample_id", "condition", "motif_fraction"],
    ].rename(columns={"motif_fraction": "loso_fraction"})
    merged = week6.merge(full, on=["sample_id", "condition"], how="outer").merge(
        loso,
        on=["sample_id", "condition"],
        how="outer",
    )
    merged = merged.fillna(0.0)
    merged["loso_minus_full_shift"] = merged["loso_fraction"].to_numpy(dtype=np.float64) - merged["full_fraction"].to_numpy(dtype=np.float64)
    return merged.sort_values(["condition", "sample_id"]).reset_index(drop=True)


def build_ad_marker_enrichment_table(
    *,
    artifact: SampleAwareArtifact,
    week6b_motif_id: str,
) -> pd.DataFrame:
    marker_table = compute_marker_genes(
        dataset=artifact.dataset,
        spot_table=artifact.loso_spot_table,
        motif_id=week6b_motif_id,
        top_n=12,
    )
    reference_markers = AD_REFERENCE["marker_genes"]
    upper_layer_set = curated_gene_sets(artifact.dataset_id).get("upper_layer_neuron", [])
    marker_lookup = marker_table.set_index("gene").to_dict(orient="index") if not marker_table.empty else {}
    all_genes = sorted(
        set(reference_markers) | set(marker_table["gene"].astype(str).tolist()) | set(upper_layer_set),
        key=lambda gene: (
            gene not in reference_markers,
            gene not in marker_table["gene"].astype(str).tolist(),
            gene not in upper_layer_set,
            -float(marker_lookup.get(gene, {}).get("log2_fc", -1.0)),
            gene,
        ),
    )
    rows: list[dict[str, object]] = []
    for gene in all_genes:
        metric = marker_lookup.get(gene, {})
        rows.append(
            {
                "gene": gene,
                "week6_reference_marker": bool(gene in reference_markers),
                "week6b_top_marker": bool(gene in marker_table["gene"].astype(str).tolist()),
                "upper_layer_neuron_marker": bool(gene in upper_layer_set),
                "log2_fc": float(metric.get("log2_fc", np.nan)),
                "delta_detected_fraction": float(metric.get("delta_detected_fraction", np.nan)),
                "overlap_bucket": overlap_bucket(
                    gene=gene,
                    reference_markers=reference_markers,
                    top_markers=marker_table["gene"].astype(str).tolist(),
                    pathway_markers=upper_layer_set,
                ),
            }
        )
    return pd.DataFrame(rows)


def overlap_bucket(
    *,
    gene: str,
    reference_markers: list[str],
    top_markers: list[str],
    pathway_markers: list[str],
) -> str:
    active = [
        label
        for label, present in (
            ("week6", gene in reference_markers),
            ("week6b", gene in top_markers),
            ("pathway", gene in pathway_markers),
        )
        if present
    ]
    return "+".join(active) if active else "background"


def compute_marker_genes(
    *,
    dataset,
    spot_table: pd.DataFrame,
    motif_id: str,
    top_n: int,
) -> pd.DataFrame:
    motif_mask = spot_table["motif_id"].astype(str).to_numpy() == str(motif_id)
    if not np.any(motif_mask) or np.all(motif_mask):
        return pd.DataFrame(columns=["gene", "log2_fc", "delta_detected_fraction"])
    matrix = dataset.expression.tocsr()
    motif_matrix = matrix[motif_mask]
    other_matrix = matrix[~motif_mask]
    mean_in = np.asarray(motif_matrix.mean(axis=0)).ravel()
    mean_out = np.asarray(other_matrix.mean(axis=0)).ravel()
    frac_in = np.asarray(motif_matrix.getnnz(axis=0)).ravel() / max(motif_matrix.shape[0], 1)
    frac_out = np.asarray(other_matrix.getnnz(axis=0)).ravel() / max(other_matrix.shape[0], 1)
    log2_fc = np.log2((mean_in + 1.0e-4) / (mean_out + 1.0e-4))
    delta_frac = frac_in - frac_out
    score = log2_fc + 0.60 * delta_frac + 0.15 * np.log1p(mean_in)
    gene_names = np.asarray(dataset.var_names, dtype=object).astype(str)
    banned_prefixes = ("AC", "AL", "AP", "LINC", "MIR", "MT-", "RN7", "RNU", "SNOR", "CTD")
    is_interpretable = np.asarray(
        [
            (not any(gene.startswith(prefix) for prefix in banned_prefixes))
            and (not gene.endswith("-AS1"))
            and (not gene.endswith("-AS2"))
            and (len(gene) <= 15)
            for gene in gene_names.tolist()
        ],
        dtype=bool,
    )
    valid = (mean_in >= 0.05) & (frac_in >= 0.10) & (delta_frac >= 0.02) & is_interpretable
    ranked_idx = np.argsort(np.where(valid, score, -np.inf))[::-1]
    rows: list[dict[str, object]] = []
    for gene_idx in ranked_idx[: top_n * 8]:
        if not np.isfinite(score[gene_idx]):
            continue
        if log2_fc[gene_idx] <= 0.0:
            continue
        rows.append(
            {
                "gene": str(gene_names[gene_idx]),
                "log2_fc": float(log2_fc[gene_idx]),
                "delta_detected_fraction": float(delta_frac[gene_idx]),
            }
        )
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def compute_curated_pathway_enrichment(
    marker_genes: list[str],
    universe_genes: list[str],
    *,
    dataset_id: str,
) -> pd.DataFrame:
    gene_sets = curated_gene_sets(dataset_id)
    universe = {gene.upper() for gene in universe_genes}
    query = {gene.upper() for gene in marker_genes if gene}
    rows: list[dict[str, object]] = []
    for pathway_name, genes in gene_sets.items():
        pathway_genes = {gene.upper() for gene in genes if gene.upper() in universe}
        if not pathway_genes:
            continue
        overlap = query & pathway_genes
        if not overlap:
            continue
        contingency = np.asarray(
            [
                [len(overlap), max(len(query) - len(overlap), 0)],
                [max(len(pathway_genes) - len(overlap), 0), max(len(universe - pathway_genes - query), 0)],
            ],
            dtype=np.int64,
        )
        odds_ratio, pvalue = fisher_exact(contingency, alternative="greater")
        rows.append(
            {
                "pathway": pathway_name,
                "overlap_genes": ",".join(sorted(overlap)),
                "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else np.nan,
                "pvalue": float(pvalue),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["pathway", "overlap_genes", "odds_ratio", "pvalue"])
    return pd.DataFrame(rows).sort_values("pvalue", ascending=True).reset_index(drop=True)


def curated_gene_sets(dataset_id: str) -> dict[str, list[str]]:
    common_sets = {
        "hypoxia_stress": ["CA9", "VEGFA", "LDHA", "SLC2A1", "BNIP3", "HILPDA", "PDK1"],
        "immune_lymphocyte": ["CD3D", "CD3E", "IL7R", "LTB", "NKG7", "CCL5", "TRBC1", "TRBC2"],
        "myeloid_inflammation": ["TYROBP", "AIF1", "C1QA", "C1QB", "C1QC", "LST1", "FCER1G", "CTSB"],
    }
    if "gse220442" in dataset_id:
        return {
            **common_sets,
            "upper_layer_neuron": ["RELN", "CUX1", "CUX2", "RORB", "LAMP5", "SATB2", "POU3F2"],
            "deep_layer_neuron": ["BCL11B", "FEZF2", "TLE4", "FOXP2", "TBR1", "CRYM", "PCP4", "ETV1"],
            "white_matter_oligodendrocyte": ["MBP", "MOBP", "PLP1", "MAG", "MOG", "OPALIN", "CLDN11", "CNP"],
            "astrocyte_reactive": ["GFAP", "AQP4", "SLC1A2", "ALDH1L1", "VIM", "CD44", "FABP7"],
            "synaptic_neuronal": ["SNAP25", "SYT1", "CAMK2A", "RBFOX3", "STMN2", "NRGN", "SLC17A7"],
        }
    return {
        **common_sets,
        "epithelial_tumor": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT17", "KRT14", "TACSTD2", "MSLN"],
        "stromal_ecm": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "SPARC", "ACTA2", "TAGLN", "THY1"],
        "adipose_program": ["PLIN1", "FABP4", "ADIPOQ", "LPL", "CFD", "CIDEA"],
        "proliferation": ["MKI67", "TOP2A", "UBE2C", "CENPF", "TYMS", "STMN1"],
    }


def decision_label_from_metrics(*, support_tier: str, naive_spot_p: float, naive_threshold: float = 0.05) -> str:
    if str(support_tier) in {"heldout_supported", "sample_level_supported"}:
        return "robust_biological_signal"
    if np.isfinite(float(naive_spot_p)) and float(naive_spot_p) <= float(naive_threshold):
        return "naive_only_signal"
    return "no_signal"


def sign_match(left: float, right: float, *, tol: float = 1.0e-8) -> bool:
    if not np.isfinite(left) or not np.isfinite(right):
        return False
    if abs(left) <= tol and abs(right) <= tol:
        return True
    if abs(left) <= tol or abs(right) <= tol:
        return False
    return bool(np.sign(left) == np.sign(right))


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.sum(finite) <= 1:
        return float("nan")
    centered_left = left[finite] - float(np.mean(left[finite]))
    centered_right = right[finite] - float(np.mean(right[finite]))
    denominator = float(np.linalg.norm(centered_left) * np.linalg.norm(centered_right))
    if denominator <= 1.0e-12:
        return float("nan")
    return float(np.dot(centered_left, centered_right) / denominator)


def safe_neglog10(value: float | int | np.floating, *, floor: float = 1.0e-12) -> float:
    numeric = float(value)
    return float(-math.log10(max(numeric, floor)))


def _set_publication_style() -> None:
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
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.14,
            "grid.linestyle": "-",
        }
    )


def plot_false_positive_reduction(*, summary: pd.DataFrame, output_path: Path) -> None:
    _set_publication_style()
    x = np.arange(summary.shape[0], dtype=np.float64)
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(
        x - width / 2.0,
        summary["week6_naive_only_count"].to_numpy(dtype=np.float64),
        width=width,
        color="#C66A3D",
        label="Week 6 naive-only",
    )
    ax.bar(
        x + width / 2.0,
        summary["week6b_naive_only_count"].to_numpy(dtype=np.float64),
        width=width,
        color="#2E6F95",
        label="Week 6b naive-only",
    )
    for idx, row in summary.iterrows():
        ax.text(
            idx,
            max(float(row["week6_naive_only_count"]), float(row["week6b_naive_only_count"])) + 0.25,
            f"robust {int(row['week6_robust_count'])}->{int(row['week6b_robust_count'])}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
        ax.text(
            idx + width / 2.0,
            float(row["week6b_naive_only_count"]) + 0.08,
            f"-{int(row['reduction_abs'])}" if int(row["reduction_abs"]) > 0 else "0",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2E6F95",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_DATASET_NAMES.get(value, value) for value in summary["dataset_id"].astype(str)], rotation=0)
    ax.set_ylabel("Naive-only motif count")
    ax.set_title("Week 6 to Week 6b false-positive reduction")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_full_vs_loso_stability(
    *,
    consistency_table: pd.DataFrame,
    stability_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    _set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax = axes[0]
    for dataset_id, subset in consistency_table.groupby("dataset_id", observed=False):
        color = DATASET_COLORS.get(str(dataset_id), "#7A8892")
        ax.scatter(
            subset["full_delta_fraction"].to_numpy(dtype=np.float64),
            subset["loso_delta_fraction"].to_numpy(dtype=np.float64),
            s=88,
            alpha=0.90,
            color=color,
            edgecolors="#FFFFFF",
            linewidths=0.8,
            label=SHORT_DATASET_NAMES.get(str(dataset_id), str(dataset_id)),
        )
    limit = float(
        max(
            np.nanmax(np.abs(consistency_table["full_delta_fraction"].to_numpy(dtype=np.float64))),
            np.nanmax(np.abs(consistency_table["loso_delta_fraction"].to_numpy(dtype=np.float64))),
        )
        * 1.12
    )
    ax.plot([-limit, limit], [-limit, limit], linestyle="--", color="#444444", linewidth=1.0)
    for _, row in consistency_table.loc[
        consistency_table["motif_id"].astype(str).isin(["motif_03", "motif_01", "motif_08"])
    ].iterrows():
        ax.text(
            float(row["full_delta_fraction"]) + 0.01,
            float(row["loso_delta_fraction"]) + 0.01,
            f"{SHORT_DATASET_NAMES.get(str(row['dataset_id']), str(row['dataset_id']))}:{row['motif_id']}",
            fontsize=7.2,
        )
    ax.set_xlabel("Full-model effect")
    ax.set_ylabel("LOSO effect")
    ax.set_title("Motif-level full vs LOSO effects")
    ax.legend(loc="upper left")

    ax = axes[1]
    x = np.arange(stability_summary.shape[0], dtype=np.float64)
    width = 0.32
    ax.bar(
        x - width / 2.0,
        stability_summary["effect_direction_consistency"].to_numpy(dtype=np.float64),
        width=width,
        color="#E9A03B",
        label="Direction consistency",
    )
    ax.bar(
        x + width / 2.0,
        stability_summary["median_abundance_correlation"].to_numpy(dtype=np.float64),
        width=width,
        color="#3E8C73",
        label="Median abundance corr",
    )
    for idx, row in stability_summary.iterrows():
        ax.text(
            idx - width / 2.0,
            float(row["effect_direction_consistency"]) + 0.03,
            f"{float(row['effect_direction_consistency']):.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            idx + width / 2.0,
            float(row["median_abundance_correlation"]) + 0.03,
            f"{float(row['median_abundance_correlation']):.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0.0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_DATASET_NAMES.get(value, value) for value in stability_summary["dataset_id"].astype(str)])
    ax.set_ylabel("Fraction / correlation")
    ax.set_title("Dataset-level stability summary")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_pvalue_inflation(*, summary: pd.DataFrame, output_path: Path) -> None:
    _set_publication_style()
    plot_frame = summary.copy()
    plot_frame["x_plot"] = plot_frame["sample_aware_p"].map(lambda value: safe_neglog10(value, floor=1.0e-16))
    plot_frame["y_plot"] = plot_frame["naive_spot_p"].map(lambda value: safe_neglog10(value, floor=1.0e-16))
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for dataset_id, dataset_df in plot_frame.groupby("dataset_id", observed=False):
        for support_tier, tier_df in dataset_df.groupby("support_tier", observed=False):
            ax.scatter(
                tier_df["x_plot"].to_numpy(dtype=np.float64),
                tier_df["y_plot"].to_numpy(dtype=np.float64),
                s=90,
                alpha=0.92,
                color=DATASET_COLORS.get(str(dataset_id), "#7A8892"),
                marker=SUPPORT_MARKERS.get(str(support_tier), "o"),
                edgecolors="#FFFFFF",
                linewidths=0.8,
            )
    limit = float(
        max(
            plot_frame["x_plot"].to_numpy(dtype=np.float64).max(initial=0.0),
            plot_frame["y_plot"].to_numpy(dtype=np.float64).max(initial=0.0),
        )
        + 0.7
    )
    ax.plot([0.0, limit], [0.0, limit], linestyle="--", color="#444444", linewidth=1.0)
    highlight = plot_frame.loc[plot_frame["highlight_label"].astype(str) != ""]
    if not highlight.empty:
        row = highlight.iloc[0]
        ax.scatter(
            [float(row["x_plot"])],
            [float(row["y_plot"])],
            s=180,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.4,
            marker="*",
            zorder=6,
        )
        ax.annotate(
            str(row["highlight_label"]),
            xy=(float(row["x_plot"]), float(row["y_plot"])),
            xytext=(25, 12),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": 0.8},
        )
    ax.set_xlabel("-log10 sample-aware p-value")
    ax.set_ylabel("-log10 naive spot p-value")
    ax.set_title("P-value inflation from pooled spot analysis")

    color_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="#FFFFFF", markersize=8, label=SHORT_DATASET_NAMES.get(dataset_id, dataset_id))
        for dataset_id, color in DATASET_COLORS.items()
    ]
    marker_handles = [
        Line2D([0], [0], marker=marker, color="#666666", linestyle="None", markersize=8, label=label)
        for label, marker in (
            ("sample-aware preserved", "o"),
            ("naive-only", "s"),
            ("not supported", "^"),
        )
    ]
    legend1 = ax.legend(handles=color_handles, loc="lower left", title="Dataset")
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles, loc="upper right", title="Support tier")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_breast_direction_flips(
    *,
    details: pd.DataFrame,
    artifact: SampleAwareArtifact,
    output_path: Path,
) -> None:
    _set_publication_style()
    motif_order = [motif for motif in BREAST_DIRECTION_FLIP_MOTIFS if motif in details["motif_id"].astype(str).tolist()]
    fig, axes = plt.subplots(len(motif_order), 5, figsize=(16.5, 5.2 * max(len(motif_order), 1)))
    if len(motif_order) == 1:
        axes = np.asarray([axes], dtype=object)
    for row_idx, motif_id in enumerate(motif_order):
        motif_details = details.loc[details["motif_id"].astype(str) == str(motif_id)].copy()
        example_sample = str(motif_details.loc[motif_details["example_sample_for_maps"]].iloc[0]["sample_id"])

        ax = axes[row_idx, 0]
        x = np.arange(motif_details.shape[0], dtype=np.float64)
        ax.plot(x, motif_details["full_fraction"].to_numpy(dtype=np.float64), marker="o", color="#7A8892", linewidth=1.8, label="full")
        ax.plot(x, motif_details["loso_fraction"].to_numpy(dtype=np.float64), marker="s", color="#C44E32", linewidth=1.8, label="LOSO")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{sid}\n{cond}" for sid, cond in zip(motif_details["sample_id"], motif_details["condition"])], rotation=15)
        ax.set_ylabel("Sample motif fraction")
        ax.set_title(f"{motif_id}: sample-level abundance")
        ax.legend(loc="upper right")

        ax = axes[row_idx, 1]
        first = motif_details.iloc[0]
        effect_values = [float(first["full_delta_fraction"]), float(first["loso_delta_fraction"])]
        ax.bar([0, 1], effect_values, color=["#7A8892", "#C44E32"], width=0.55)
        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["full", "LOSO"])
        ax.set_ylabel("Condition effect")
        ax.set_title("Direction flip")
        for idx, value in enumerate(effect_values):
            ax.text(idx, value + np.sign(value) * 0.01, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)

        ax = axes[row_idx, 2]
        pvalues = {
            "naive\nfull": float(first["full_naive_spot_p"]),
            "naive\nLOSO": float(first["loso_naive_spot_p"]),
            "full\nsample": float(first["full_sample_p_two_sided"]),
            "LOSO\nsample": float(first["loso_sample_p_two_sided"]),
            "maxT": float(first["loso_max_t_p"]),
            "null": float(first["loso_null_effect_p"]),
        }
        keys = list(pvalues.keys())
        values = [safe_neglog10(value, floor=1.0e-16) for value in pvalues.values()]
        ax.bar(np.arange(len(keys)), values, color=["#9E9E9E", "#C44E32", "#7A8892", "#C44E32", "#E0A458", "#3E8C73"])
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(keys, rotation=12)
        ax.set_ylabel("-log10 p (floor 1e-16)")
        ax.set_title("Naive vs controlled p-values")

        plot_spatial_map(
            ax=axes[row_idx, 3],
            spot_table=artifact.full_fit.embedding_result.spot_table,
            sample_id=example_sample,
            motif_id=motif_id,
            color="#7A8892",
            title=f"full map\n{example_sample}",
        )
        plot_spatial_map(
            ax=axes[row_idx, 4],
            spot_table=artifact.loso_spot_table,
            sample_id=example_sample,
            motif_id=motif_id,
            color="#C44E32",
            title=f"LOSO map\n{example_sample}",
        )
    fig.suptitle("Breast direction-flip case studies", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_spatial_map(
    *,
    ax,
    spot_table: pd.DataFrame,
    sample_id: str,
    motif_id: str,
    color: str,
    title: str,
) -> None:
    frame = spot_table.loc[spot_table["sample_id"].astype(str) == str(sample_id)].copy()
    mask = frame["motif_id"].astype(str) == str(motif_id)
    ax.scatter(
        frame.loc[~mask, "spatial_x"].to_numpy(dtype=np.float32),
        -frame.loc[~mask, "spatial_y"].to_numpy(dtype=np.float32),
        c="#D9DADB",
        s=6,
        alpha=0.55,
        linewidths=0,
    )
    ax.scatter(
        frame.loc[mask, "spatial_x"].to_numpy(dtype=np.float32),
        -frame.loc[mask, "spatial_y"].to_numpy(dtype=np.float32),
        c=color,
        s=11,
        alpha=0.96,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.grid(False)


def plot_ad_sample_level_fraction(
    *,
    abundance: pd.DataFrame,
    tracking_row: pd.Series,
    output_path: Path,
) -> None:
    _set_publication_style()
    ordered = abundance.copy()
    ordered["condition_order"] = ordered["condition"].astype(str).map({"control": 0, "AD": 1}).fillna(2)
    ordered = ordered.sort_values(["condition_order", "sample_id"]).reset_index(drop=True)
    x = np.arange(ordered.shape[0], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    condition_colors = {"control": "#2E8B8B", "AD": "#C44E32"}
    for idx, row in ordered.iterrows():
        color = condition_colors.get(str(row["condition"]), "#7A8892")
        ax.plot(
            [idx - 0.08, idx + 0.08],
            [float(row["full_fraction"]), float(row["loso_fraction"])],
            color="#B8B8B8",
            linewidth=1.1,
            zorder=1,
        )
        ax.scatter(idx - 0.08, float(row["full_fraction"]), color=color, s=58, marker="o", edgecolors="#FFFFFF", linewidths=0.8, zorder=3)
        ax.scatter(idx + 0.08, float(row["loso_fraction"]), color=color, s=64, marker="s", edgecolors="#FFFFFF", linewidths=0.8, zorder=3)
    boundary = int(np.sum(ordered["condition"].astype(str) == "control")) - 0.5
    ax.axvline(boundary, color="#DDDDDD", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sid}\n{cond}" for sid, cond in zip(ordered["sample_id"], ordered["condition"])], rotation=12)
    ax.set_ylabel("Motif fraction")
    ax.set_title(
        f"AD motif {tracking_row['matched_06b_motif_id']} sample-level abundance "
        f"(full Δ={float(tracking_row['full_delta_fraction']):.3f}, LOSO Δ={float(tracking_row['loso_delta_fraction']):.3f})"
    )
    ax.annotate(
        "control-enriched in both full and LOSO",
        xy=(0.03, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
    )
    legend = [
        Line2D([0], [0], marker="o", color="#555555", linestyle="None", markersize=7, label="full"),
        Line2D([0], [0], marker="s", color="#555555", linestyle="None", markersize=7, label="LOSO"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=condition_colors["control"], markersize=7, label="control"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=condition_colors["AD"], markersize=7, label="AD"),
    ]
    ax.legend(handles=legend, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_ad_marker_heatmap(*, marker_table: pd.DataFrame, output_path: Path) -> None:
    _set_publication_style()
    display = marker_table.copy()
    display["sort_key"] = display.apply(
        lambda row: (
            0 if row["week6_reference_marker"] and row["week6b_top_marker"] else 1,
            0 if row["upper_layer_neuron_marker"] else 1,
            -float(pd.to_numeric(row["log2_fc"], errors="coerce")) if np.isfinite(pd.to_numeric(row["log2_fc"], errors="coerce")) else 999.0,
            str(row["gene"]),
        ),
        axis=1,
    )
    display = display.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)
    matrix = display.loc[:, ["week6_reference_marker", "week6b_top_marker", "upper_layer_neuron_marker"]].astype(int).to_numpy(dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, max(4.8, 0.32 * display.shape[0] + 1.5)), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(["Week 6 ref", "Week 6b top8", "upper-layer"], rotation=20, ha="right")
    ax.set_yticks(np.arange(display.shape[0]))
    ax.set_yticklabels(display["gene"].astype(str).tolist())
    ax.set_title("Marker membership")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="#222222", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    log2_fc = pd.to_numeric(display["log2_fc"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y = np.arange(display.shape[0], dtype=np.float64)
    colors = ["#C44E32" if overlap == "week6+week6b+pathway" or overlap == "week6+week6b" else "#7A8892" for overlap in display["overlap_bucket"].astype(str)]
    ax.barh(y, log2_fc, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("Week 6b LOSO log2 fold-change")
    ax.set_title("Week 6b marker strength")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_ad_spatial_examples(
    *,
    artifact: SampleAwareArtifact,
    abundance: pd.DataFrame,
    motif_id: str,
    output_path: Path,
) -> None:
    _set_publication_style()
    control_row = abundance.loc[abundance["condition"].astype(str) == "control"].sort_values("loso_fraction", ascending=False).iloc[0]
    ad_row = abundance.loc[abundance["condition"].astype(str) == "AD"].sort_values("loso_fraction", ascending=False).iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8))
    for ax, row, title_color in (
        (axes[0], control_row, "#2E8B8B"),
        (axes[1], ad_row, "#C44E32"),
    ):
        plot_spatial_map(
            ax=ax,
            spot_table=artifact.loso_spot_table,
            sample_id=str(row["sample_id"]),
            motif_id=motif_id,
            color=title_color,
            title=f"{row['condition']} sample {row['sample_id']}\nLOSO fraction={float(row['loso_fraction']):.3f}",
        )
    fig.suptitle(f"AD motif {motif_id} spatial examples", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_week7b_protocol(*, runtime_info) -> None:
    lines = [
        "# Protocol",
        "",
        "## Goal",
        "",
        "Systematize the Week 6 and Week 6b validity observations: quantify false-positive shrinkage, measure full-vs-LOSO stability, summarize p-value inflation, isolate breast direction-flip case studies, and produce a final sample-aware decision matrix for paper assembly.",
        "",
        "## Runtime",
        "",
        f"- Device detection: `torch.cuda.is_available()` -> `{runtime_info.cuda_available}`",
        f"- Active device: `{runtime_info.device}`",
        f"- CUDA device count: `{runtime_info.cuda_count}`",
        f"- CUDA device name: `{runtime_info.cuda_name}`",
        f"- Torch version: `{runtime_info.torch_version}`",
        "",
        "## Inputs",
        "",
        "- `experiments/06_cross_sample_differential/results/differential_test_results.csv`",
        "- `experiments/06b_sample_aware_cross_sample_differential/results/week6_vs_06b_comparison.csv`",
        "- `experiments/06b_sample_aware_cross_sample_differential/results/motif_stability_decision_matrix.csv`",
        "- `experiments/06b_sample_aware_cross_sample_differential/results/full_vs_loso_differential_consistency.csv`",
        "- `experiments/06b_sample_aware_cross_sample_differential/results/sample_aware_differential_test_results.csv`",
        "- `data/spatial_processed/wu2021_breast_visium.h5ad`",
        "",
        "## Steps",
        "",
        "1. Rebuild the breast sample-aware motif assignment only to recover full and LOSO spot maps for direction-flip visualization.",
        "2. Count naive-only and robust motifs in Week 6 and Week 6b to quantify false-positive reduction.",
        "3. Aggregate full-vs-LOSO consistency into dataset-level direction-consistency and abundance-correlation summaries.",
        "4. Join Week 6 naive spot p-values with Week 6 sample-level p-values and Week 6b LOSO/maxT/null p-values to quantify p-value inflation.",
        "5. Materialize breast `motif_03` and `motif_01` direction-flip detail tables and multi-panel example figures.",
        "6. Merge Week 6 motif tracking with Week 6b stability metadata to assign final labels: `preserved_robust_signal`, `removed_naive_only_signal`, `persistent_naive_only_signal`, `unstable_direction_flip`, `no_signal`.",
        "",
        "## Outputs",
        "",
        "- `results/false_positive_reduction_summary.csv`",
        "- `results/full_loso_stability_summary.csv`",
        "- `results/pvalue_inflation_summary.csv`",
        "- `results/direction_flip_motif_details.csv`",
        "- `results/final_signal_decision_matrix.csv`",
        "- `figures/fig7b_false_positive_reduction.png`",
        "- `figures/fig7b_full_vs_loso_stability.png`",
        "- `figures/fig7b_pvalue_inflation.png`",
        "- `figures/fig7b_breast_direction_flips.png`",
        "",
        "## Run Command",
        "",
        "- `python scripts/run_week7_parallel_experiments.py --track week7b`",
    ]
    (WEEK7B_DIR / "protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_week7b_analysis(
    *,
    false_positive_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    pvalue_inflation_summary: pd.DataFrame,
    direction_flip_details: pd.DataFrame,
    final_decision_matrix: pd.DataFrame,
) -> None:
    breast_summary = false_positive_summary.loc[false_positive_summary["dataset_id"].astype(str) == "wu2021_breast_visium"].iloc[0]
    ad_summary = false_positive_summary.loc[false_positive_summary["dataset_id"].astype(str) == "gse220442_ad_mtg"].iloc[0]
    breast_stability = stability_summary.loc[stability_summary["dataset_id"].astype(str) == "wu2021_breast_visium"].iloc[0]
    ad_stability = stability_summary.loc[stability_summary["dataset_id"].astype(str) == "gse220442_ad_mtg"].iloc[0]
    biggest_gap = pvalue_inflation_summary.sort_values("naive_vs_sample_aware_log10_gap", ascending=False).iloc[0]
    flip_subset = direction_flip_details.loc[
        direction_flip_details["sample_id"].astype(str)
        == direction_flip_details.groupby("motif_id", observed=False)["sample_id"].transform(
            lambda values: values.iloc[0]
        )
    ].drop_duplicates("motif_id")
    label_counts = (
        final_decision_matrix.groupby(["dataset_id", "final_signal_label"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[
            "preserved_robust_signal",
            "removed_naive_only_signal",
            "persistent_naive_only_signal",
            "unstable_direction_flip",
            "no_signal",
        ], fill_value=0)
    )

    lines = [
        "# Analysis",
        "",
        "## Headline",
        "",
        f"- Breast false-positive burden drops from `{int(breast_summary['week6_naive_only_count'])}` to `{int(breast_summary['week6b_naive_only_count'])}` naive-only motifs (`{float(breast_summary['reduction_fraction']):.3f}` reduction fraction), while AD stays at `{int(ad_summary['week6_naive_only_count'])}` -> `{int(ad_summary['week6b_naive_only_count'])}`.",
        f"- Full-vs-LOSO stability is near-perfect in AD (direction consistency=`{float(ad_stability['effect_direction_consistency']):.2f}`, median abundance corr=`{float(ad_stability['median_abundance_correlation']):.2f}`) but degrades in breast (direction consistency=`{float(breast_stability['effect_direction_consistency']):.2f}`, median abundance corr=`{float(breast_stability['median_abundance_correlation']):.2f}`).",
        "",
        "## P-value inflation",
        "",
        f"- Largest naive-vs-sample-aware inflation row: `{biggest_gap['dataset_id']}:{biggest_gap['week6_motif_id']} -> {biggest_gap['matched_06b_motif_id']}` with naive-to-sample-aware gap=`{float(biggest_gap['naive_vs_sample_aware_log10_gap']):.2f}` log10 units.",
        "- The highlighted AD tracking row (`motif_02 -> motif_08`) keeps a plausible sample-aware signal even though the pooled spot p-value is orders of magnitude smaller than the LOSO-controlled p-value.",
        "",
        "## Direction-flip case study",
        "",
    ]
    for _, row in flip_subset.iterrows():
        lines.append(
            f"- `{row['motif_id']}`: full delta=`{float(row['full_delta_fraction']):.3f}` vs LOSO delta=`{float(row['loso_delta_fraction']):.3f}`, "
            f"full sample p=`{float(row['full_sample_p_two_sided']):.3f}`, LOSO sample p=`{float(row['loso_sample_p_two_sided']):.3f}`, "
            f"naive LOSO p=`{float(row['loso_naive_spot_p']):.3g}`."
        )
    lines.extend(
        [
            "",
            "## Final decision matrix",
            "",
        ]
    )
    for dataset_id, counts in label_counts.iterrows():
        lines.append(
            f"- `{dataset_id}`: preserved=`{int(counts['preserved_robust_signal'])}`, removed naive-only=`{int(counts['removed_naive_only_signal'])}`, "
            f"persistent naive-only=`{int(counts['persistent_naive_only_signal'])}`, unstable flips=`{int(counts['unstable_direction_flip'])}`, no-signal=`{int(counts['no_signal'])}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Ignoring sample structure inflates significance across both datasets, but the damage is qualitatively different: breast shows strong false-positive shrinkage and explicit direction flips, whereas AD mostly preserves the same rank-order story under stricter controls.",
            "- The direction-flip examples are more intuitive than p-value shrinkage alone because they show pooled analysis can reverse the sign of a motif-level condition effect.",
        ]
    )
    (WEEK7B_DIR / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_week7a_protocol(*, runtime_info) -> None:
    lines = [
        "# Protocol",
        "",
        "## Goal",
        "",
        "Validate one AD motif only: determine whether Week 6 `motif_02` tracks to a Week 6b sample-aware preserved motif with interpretable upper-layer neuronal biology.",
        "",
        "## Runtime",
        "",
        f"- Device detection: `torch.cuda.is_available()` -> `{runtime_info.cuda_available}`",
        f"- Active device: `{runtime_info.device}`",
        f"- CUDA device count: `{runtime_info.cuda_count}`",
        f"- CUDA device name: `{runtime_info.cuda_name}`",
        f"- Torch version: `{runtime_info.torch_version}`",
        "",
        "## Inputs",
        "",
        "- `experiments/06_cross_sample_differential/results/sample_level_motif_table.csv`",
        "- `experiments/06b_sample_aware_cross_sample_differential/results/robust_motif_tracking.csv`",
        "- `experiments/06b_sample_aware_cross_sample_differential/results/sample_aware_differential_test_results.csv`",
        "- `data/spatial_processed/gse220442_ad_mtg.h5ad`",
        "",
        "## Steps",
        "",
        "1. Rebuild the AD sample-aware motif assignment to recover full and LOSO per-sample motif fractions plus spatial maps.",
        "2. Extract the tracked `motif_02 -> motif_08` row from the Week 6b robust motif tracking table.",
        "3. Build a one-row tracking summary with sample-fraction correlation, marker overlap, pathway overlap, effect-direction preservation, null p, and LOSO sign consistency.",
        "4. Quantify sample-level abundance per sample across Week 6, Week 6b full, and Week 6b LOSO assignments.",
        "5. Compute Week 6b LOSO marker genes and curated upper-layer pathway overlap for the tracked motif.",
        "6. Write a cautious analysis that states the small-sample limitation explicitly (`n=6`, two-sided p=`0.200`, one-sided p=`0.100`, maxT p=`0.800`).",
        "",
        "## Outputs",
        "",
        "- `results/ad_motif_tracking_summary.csv`",
        "- `results/ad_motif_marker_enrichment.csv`",
        "- `results/ad_motif_sample_abundance.csv`",
        "- `figures/ad_motif_sample_level_fraction.png`",
        "- `figures/ad_motif_marker_heatmap.png`",
        "- `figures/ad_motif_spatial_examples.png`",
        "",
        "## Run Command",
        "",
        "- `python scripts/run_week7_parallel_experiments.py --track week7a`",
    ]
    (WEEK7A_DIR / "protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_week7a_analysis(
    *,
    tracking_row: pd.Series,
    sample_abundance: pd.DataFrame,
    marker_table: pd.DataFrame,
    differential_table: pd.DataFrame,
) -> None:
    sample_n = int(sample_abundance["sample_id"].astype(str).nunique())
    control_mean_full = float(sample_abundance.loc[sample_abundance["condition"].astype(str) == "control", "full_fraction"].mean())
    ad_mean_full = float(sample_abundance.loc[sample_abundance["condition"].astype(str) == "AD", "full_fraction"].mean())
    control_mean_loso = float(sample_abundance.loc[sample_abundance["condition"].astype(str) == "control", "loso_fraction"].mean())
    ad_mean_loso = float(sample_abundance.loc[sample_abundance["condition"].astype(str) == "AD", "loso_fraction"].mean())
    week6b_row = differential_table.loc[
        (differential_table["dataset_id"].astype(str) == AD_REFERENCE["dataset_id"])
        & (differential_table["motif_id"].astype(str) == AD_REFERENCE["week6b_motif_id"])
    ].iloc[0]
    marker_overlap = int(np.sum(marker_table["week6_reference_marker"].astype(bool) & marker_table["week6b_top_marker"].astype(bool)))
    lines = [
        "# Analysis",
        "",
        "## Tracking result",
        "",
        f"- Week 6 `{tracking_row['week6_motif_id']}` tracks to Week 6b `{tracking_row['matched_06b_motif_id']}` with sample-fraction correlation=`{float(tracking_row['sample_fraction_correlation']):.3f}`, marker overlap=`{int(tracking_row['marker_overlap_count'])}`, pathway overlap=`{tracking_row['top_pathway']}`.",
        f"- Effect direction is preserved (`{float(tracking_row['week6_delta_fraction']):.3f}` -> `{float(tracking_row['loso_delta_fraction']):.3f}`), and LOSO sign consistency is `{float(tracking_row['loso_loso_sign_consistency']):.2f}`.",
        "",
        "## Biological interpretation",
        "",
        f"- The Week 6b motif is labeled `{tracking_row['matched_06b_motif_label']}` with dominant cell-type composition `{tracking_row['dominant_cell_types']}` and top pathway `{tracking_row['top_pathway']}` (p=`{float(tracking_row['top_pathway_pvalue']):.3g}`).",
        f"- Marker overlap remains `{marker_overlap}` genes, preserving the Week 6 upper-layer neuronal signature around `Layer 2 / Layer 3` rather than collapsing into a generic null motif.",
        f"- Mean motif fraction stays control-enriched in both full (`control={control_mean_full:.3f}`, `AD={ad_mean_full:.3f}`) and LOSO (`control={control_mean_loso:.3f}`, `AD={ad_mean_loso:.3f}`) assignments.",
        "",
        "## Cautious interpretation",
        "",
        f"- Sample count is only `{sample_n}` (`3 vs 3`), so the exact sample-level test is heavily resolution-limited.",
        f"- Two-sided p=`{float(week6b_row['loso_sample_permutation_pvalue_two_sided']):.3f}`, one-sided p=`{float(week6b_row['loso_sample_permutation_pvalue_one_sided']):.3f}`, maxT p=`{float(week6b_row['loso_label_max_t_pvalue']):.3f}`, null motif p=`{float(week6b_row['loso_synthetic_null_effect_pvalue']):.3f}`.",
        "- This is not a definitive AD biomarker claim.",
        "- The point of the example is narrower: the sample-aware pipeline does not erase plausible biology even when it removes many pooled-analysis false positives elsewhere.",
    ]
    (WEEK7A_DIR / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
