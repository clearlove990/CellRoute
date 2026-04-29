from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import fisher_exact

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spatial_context.cross_sample_differential import (  # noqa: E402
    build_sample_level_motif_table,
    compute_differential_statistics,
)
from spatial_context.motif_embedding import (  # noqa: E402
    SampleAwareMotifFitResult,
    align_sample_aware_motif_model_to_reference,
    assign_sample_aware_motifs,
    build_tissue_motif_feature_bundle,
    fit_sample_aware_tissue_motif_model,
)
from spatial_context.neighborhood import get_runtime_info, load_spatial_h5ad, summarize_neighborhoods  # noqa: E402


DEFAULT_DATASETS = (
    ROOT / "data" / "spatial_processed" / "wu2021_breast_visium.h5ad",
    ROOT / "data" / "spatial_processed" / "gse220442_ad_mtg.h5ad",
)
EXPERIMENT_DIR = ROOT / "experiments" / "06b_sample_aware_cross_sample_differential"
WEEK6_RESULTS_DIR = ROOT / "experiments" / "06_cross_sample_differential" / "results"
WEEK6_ANALYSIS_PATH = ROOT / "experiments" / "06_cross_sample_differential" / "analysis.md"
WEEK6_REFERENCE = {
    "dataset_id": "gse220442_ad_mtg",
    "motif_id": "motif_02",
    "motif_label": "Layer 2 + Layer 3 | sparse highE",
    "pathway": "upper_layer_neuron",
    "marker_genes": ["LTK", "NEUROD1", "CARTPT", "MGP", "CALB2", "LAMP5", "C1QL2", "WNT3"],
}


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset_id: str
    dataset_name: str
    dataset: object
    full_fit: SampleAwareMotifFitResult
    full_sample_level: pd.DataFrame
    loso_spot_table: pd.DataFrame
    loso_sample_level: pd.DataFrame
    differential_table: pd.DataFrame
    fold_summary: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Week 6b sample-aware cross-sample differential analysis from the Week 4b motif recipe."
    )
    parser.add_argument(
        "--dataset-paths",
        nargs="*",
        default=[str(path) for path in DEFAULT_DATASETS],
        help="Processed spatial .h5ad files to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENT_DIR),
        help="Experiment output directory.",
    )
    parser.add_argument("--top-variable-genes", type=int, default=256)
    parser.add_argument("--n-expression-programs", type=int, default=6)
    parser.add_argument("--radius-factor", type=float, default=1.6)
    parser.add_argument("--max-train-spots-per-sample", type=int, default=2048)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--null-iterations", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runtime_info = get_runtime_info()
    dataset_paths = [Path(path).resolve() for path in args.dataset_paths if Path(path).exists()]
    if not dataset_paths:
        raise FileNotFoundError("No processed spatial datasets were found.")

    week6_differential = pd.read_csv(WEEK6_RESULTS_DIR / "differential_test_results.csv")
    week6_sample_level = pd.read_csv(WEEK6_RESULTS_DIR / "sample_level_motif_table.csv")

    dataset_artifacts: dict[str, DatasetArtifacts] = {}
    dataset_summaries: list[dict[str, object]] = []
    differential_frames: list[pd.DataFrame] = []
    null_frames: list[pd.DataFrame] = []
    sample_level_frames: list[pd.DataFrame] = []

    for dataset_path in dataset_paths:
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
        full_sample_level_frame = full_sample_level.copy()
        full_sample_level_frame["assignment_scope"] = "full_model"
        loso_sample_level_frame = loso_sample_level.copy()
        loso_sample_level_frame["assignment_scope"] = "loso"
        sample_level_table = pd.concat([full_sample_level_frame, loso_sample_level_frame], ignore_index=True)
        sample_level_frames.append(sample_level_table)

        core_adjacency = neighborhood_summary.scales[neighborhood_summary.core_scale_name].adjacency.tocsr()
        full_stats = compute_differential_statistics(
            spot_table=full_fit.embedding_result.spot_table.reset_index(drop=True),
            sample_level_table=full_sample_level,
            adjacency=core_adjacency,
            random_state=args.random_state,
            null_iterations=args.null_iterations,
            null_scope_label="full_model",
        )
        loso_stats = compute_differential_statistics(
            spot_table=loso_spot_table.reset_index(drop=True),
            sample_level_table=loso_sample_level,
            adjacency=core_adjacency,
            random_state=args.random_state,
            null_iterations=args.null_iterations,
            null_scope_label="loso",
        )
        merged = merge_scope_summaries(full_stats.summary, loso_stats.summary)
        differential_frames.append(merged)
        null_frames.extend(
            [
                full_stats.null_controls,
                loso_stats.null_controls,
                label_control_rows(full_stats.summary, analysis_scope="full_model"),
                label_control_rows(loso_stats.summary, analysis_scope="loso"),
            ]
        )

        dataset_artifacts[dataset.dataset_id] = DatasetArtifacts(
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.dataset_name,
            dataset=dataset,
            full_fit=full_fit,
            full_sample_level=full_sample_level,
            loso_spot_table=loso_spot_table,
            loso_sample_level=loso_sample_level,
            differential_table=merged,
            fold_summary=fold_summary,
        )
        dataset_summaries.append(
            {
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.dataset_name,
                "n_spots": int(dataset.obs.shape[0]),
                "n_samples": int(dataset.obs["sample_id"].nunique()),
                "condition_counts": dataset.obs.loc[:, ["sample_id", "condition"]].drop_duplicates()["condition"].astype(str).value_counts().to_dict(),
                "n_motifs": int(full_fit.embedding_result.n_clusters),
                "core_scale": neighborhood_summary.core_scale_name,
                "spatial_coherence_zscore": float(full_fit.embedding_result.spatial_coherence_zscore),
                "mean_loso_assignment_agreement": float(fold_summary["assignment_agreement_vs_full"].mean()),
                "mean_alignment_cost": float(fold_summary["alignment_cost_mean"].mean()),
            }
        )

    sample_level_table = pd.concat(sample_level_frames, ignore_index=True) if sample_level_frames else pd.DataFrame()
    differential_table = pd.concat(differential_frames, ignore_index=True) if differential_frames else pd.DataFrame()
    null_control_table = pd.concat([frame for frame in null_frames if not frame.empty], ignore_index=True) if null_frames else pd.DataFrame()
    loso_fold_summary = build_combined_fold_summary(dataset_artifacts=dataset_artifacts)

    consistency_table = build_full_vs_loso_consistency(dataset_artifacts=dataset_artifacts, differential_table=differential_table)
    decision_matrix = build_decision_matrix(differential_table=differential_table)
    week6_vs_06b = build_week6_vs_06b_comparison(
        week6_differential=week6_differential,
        week6_sample_level=week6_sample_level,
        sample_aware_differential=differential_table,
        sample_aware_sample_level=sample_level_table,
        week6_decision=build_week6_decision_matrix(week6_differential),
        sample_aware_decision=decision_matrix,
    )
    robust_tracking = build_robust_motif_tracking(
        artifacts=dataset_artifacts,
        week6_differential=week6_differential,
        week6_sample_level=week6_sample_level,
        sample_aware_differential=differential_table,
        sample_aware_sample_level=sample_level_table,
    )

    sample_level_table.to_csv(results_dir / "sample_level_motif_abundance.csv", index=False)
    loso_fold_summary.to_csv(results_dir / "loso_fold_summary.csv", index=False)
    differential_table.to_csv(results_dir / "sample_aware_differential_test_results.csv", index=False)
    null_control_table.to_csv(results_dir / "sample_aware_null_control_results.csv", index=False)
    week6_vs_06b.to_csv(results_dir / "week6_vs_06b_comparison.csv", index=False)
    consistency_table.to_csv(results_dir / "full_vs_loso_differential_consistency.csv", index=False)
    decision_matrix.to_csv(results_dir / "motif_stability_decision_matrix.csv", index=False)
    robust_tracking.to_csv(results_dir / "robust_motif_tracking.csv", index=False)

    plot_week6_vs_06b_pvalue_shift(week6_vs_06b, figures_dir / "week6_vs_06b_pvalue_shift.png")
    plot_full_vs_loso_effect_scatter(consistency_table, figures_dir / "full_vs_loso_effect_scatter.png")
    plot_naive_only_count_reduction(week6_vs_06b, figures_dir / "naive_only_count_reduction.png")
    plot_ad_motif_tracking(
        robust_tracking=robust_tracking,
        week6_sample_level=week6_sample_level,
        sample_aware_sample_level=sample_level_table,
        output_path=figures_dir / "ad_motif02_tracking.png",
    )
    plot_breast_false_positive_reduction(
        week6_vs_06b=week6_vs_06b,
        output_path=figures_dir / "breast_false_positive_reduction.png",
    )

    write_protocol(
        output_dir=output_dir,
        runtime_info=runtime_info,
        dataset_paths=dataset_paths,
        dataset_summaries=dataset_summaries,
        args=args,
    )
    write_analysis(
        output_dir=output_dir,
        week6_vs_06b=week6_vs_06b,
        decision_matrix=decision_matrix,
        consistency_table=consistency_table,
        robust_tracking=robust_tracking,
    )

    best_tracking = robust_tracking.iloc[0] if not robust_tracking.empty else pd.Series(dtype=object)
    payload = {
        "matched_ad_motif": str(best_tracking.get("matched_06b_motif_id", "")),
        "matched_06b_support_tier": str(best_tracking.get("loso_controlled_support_tier", "")),
        "week6b_naive_only_total": int(np.sum(decision_matrix["decision_label"].astype(str) == "naive_only_signal")) if not decision_matrix.empty else 0,
    }
    print(json.dumps(payload, ensure_ascii=False))


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


def merge_scope_summaries(full_summary: pd.DataFrame, loso_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_id", "dataset_name", "comparison_name", "condition_a", "condition_b", "motif_id", "motif_label"]
    full_prefixed = prefix_frame(full_summary, prefix="full_", key_columns=keys)
    loso_prefixed = prefix_frame(loso_summary, prefix="loso_", key_columns=keys)
    merged = full_prefixed.merge(loso_prefixed, on=keys, how="outer")
    if merged.empty:
        return merged
    merged["primary_rank_score"] = build_rank_score(merged)
    return merged.sort_values(["dataset_id", "primary_rank_score", "motif_id"], ascending=[True, True, True]).reset_index(drop=True)


def prefix_frame(frame: pd.DataFrame, *, prefix: str, key_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rename_map = {column: f"{prefix}{column}" for column in frame.columns if column not in key_columns}
    return frame.rename(columns=rename_map)


def build_rank_score(frame: pd.DataFrame) -> np.ndarray:
    tier_rank = frame["loso_controlled_support_tier"].map(
        {
            "heldout_supported": 0.0,
            "sample_level_supported": 1.0,
            "not_supported": 2.0,
            "naive_only": 3.0,
        }
    ).fillna(4.0)
    loso_p = frame["loso_sample_permutation_pvalue_two_sided"].fillna(1.0).to_numpy(dtype=np.float64)
    loso_max_t = frame["loso_label_max_t_pvalue"].fillna(1.0).to_numpy(dtype=np.float64)
    loso_null = frame["loso_synthetic_null_effect_pvalue"].fillna(1.0).to_numpy(dtype=np.float64)
    loso_consistency = 1.0 - frame["loso_loso_sign_consistency"].fillna(0.0).to_numpy(dtype=np.float64)
    effect_penalty = -np.abs(frame["loso_delta_fraction"].fillna(0.0).to_numpy(dtype=np.float64))
    return tier_rank.to_numpy(dtype=np.float64) + loso_p + loso_max_t + loso_null + loso_consistency + effect_penalty


def label_control_rows(summary: pd.DataFrame, *, analysis_scope: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = summary.loc[:, ["dataset_id", "dataset_name", "motif_id", "motif_label", "delta_fraction", "label_max_t_pvalue"]].copy()
    rows["analysis_scope"] = analysis_scope
    rows["control_name"] = "condition_label_max_t"
    rows["statistic_name"] = "abs_delta_fraction"
    rows["observed_value"] = rows["delta_fraction"].abs()
    rows["null_mean"] = np.nan
    rows["null_std"] = np.nan
    rows["empirical_pvalue"] = rows["label_max_t_pvalue"]
    rows["n_iterations"] = np.nan
    return rows.loc[
        :,
        [
            "dataset_id",
            "dataset_name",
            "analysis_scope",
            "motif_id",
            "motif_label",
            "control_name",
            "statistic_name",
            "observed_value",
            "null_mean",
            "null_std",
            "empirical_pvalue",
            "n_iterations",
        ],
    ]


def build_week6_decision_matrix(week6_differential: pd.DataFrame) -> pd.DataFrame:
    if week6_differential.empty:
        return pd.DataFrame()
    frame = week6_differential.copy()
    frame["decision_label"] = frame.apply(
        lambda row: decision_label_from_metrics(
            support_tier=str(row["holdout_controlled_support_tier"]),
            naive_spot_p=float(row["holdout_naive_spot_pvalue"]),
        ),
        axis=1,
    )
    return frame.loc[
        :,
        [
            "dataset_id",
            "dataset_name",
            "comparison_name",
            "motif_id",
            "motif_label",
            "holdout_delta_fraction",
            "holdout_sample_permutation_pvalue_two_sided",
            "holdout_sample_permutation_pvalue_one_sided",
            "holdout_label_max_t_pvalue",
            "holdout_naive_spot_pvalue",
            "holdout_loso_sign_consistency",
            "holdout_synthetic_null_effect_pvalue",
            "holdout_naive_minus_controlled_log10p",
            "holdout_controlled_support_tier",
            "decision_label",
            "primary_rank_score",
        ],
    ].rename(
        columns={
            "holdout_sample_permutation_pvalue_two_sided": "sample_p_two_sided",
            "holdout_sample_permutation_pvalue_one_sided": "sample_p_one_sided",
            "holdout_label_max_t_pvalue": "max_t_p",
            "holdout_naive_spot_pvalue": "naive_spot_p",
            "holdout_loso_sign_consistency": "loso_sign_consistency",
            "holdout_synthetic_null_effect_pvalue": "synthetic_null_effect_p",
            "holdout_naive_minus_controlled_log10p": "naive_minus_controlled_log10p",
            "holdout_controlled_support_tier": "controlled_support_tier",
        }
    )


def build_decision_matrix(differential_table: pd.DataFrame) -> pd.DataFrame:
    if differential_table.empty:
        return pd.DataFrame()
    frame = differential_table.copy()
    frame["decision_label"] = frame.apply(
        lambda row: decision_label_from_metrics(
            support_tier=str(row["loso_controlled_support_tier"]),
            naive_spot_p=float(row["loso_naive_spot_pvalue"]),
        ),
        axis=1,
    )
    return frame.loc[
        :,
        [
            "dataset_id",
            "dataset_name",
            "comparison_name",
            "motif_id",
            "motif_label",
            "loso_delta_fraction",
            "loso_sample_permutation_pvalue_two_sided",
            "loso_sample_permutation_pvalue_one_sided",
            "loso_label_max_t_pvalue",
            "loso_naive_spot_pvalue",
            "loso_loso_sign_consistency",
            "loso_synthetic_null_effect_pvalue",
            "loso_naive_minus_controlled_log10p",
            "loso_controlled_support_tier",
            "decision_label",
            "primary_rank_score",
        ],
    ].rename(
        columns={
            "loso_sample_permutation_pvalue_two_sided": "sample_p_two_sided",
            "loso_sample_permutation_pvalue_one_sided": "sample_p_one_sided",
            "loso_label_max_t_pvalue": "max_t_p",
            "loso_naive_spot_pvalue": "naive_spot_p",
            "loso_loso_sign_consistency": "loso_sign_consistency",
            "loso_synthetic_null_effect_pvalue": "synthetic_null_effect_p",
            "loso_naive_minus_controlled_log10p": "naive_minus_controlled_log10p",
            "loso_controlled_support_tier": "controlled_support_tier",
        }
    )


def build_full_vs_loso_consistency(
    *,
    dataset_artifacts: dict[str, DatasetArtifacts],
    differential_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_id, artifact in dataset_artifacts.items():
        merged_abundance = artifact.full_sample_level.loc[:, ["sample_id", "motif_id", "motif_fraction"]].merge(
            artifact.loso_sample_level.loc[:, ["sample_id", "motif_id", "motif_fraction"]],
            on=["sample_id", "motif_id"],
            how="outer",
            suffixes=("_full", "_loso"),
        ).fillna({"motif_fraction_full": 0.0, "motif_fraction_loso": 0.0})
        dataset_diff = differential_table.loc[differential_table["dataset_id"].astype(str) == str(dataset_id)].copy()
        for motif_id, motif_df in merged_abundance.groupby("motif_id", observed=False):
            diff_row = dataset_diff.loc[dataset_diff["motif_id"].astype(str) == str(motif_id)]
            if diff_row.empty:
                continue
            row = diff_row.iloc[0]
            full_values = motif_df["motif_fraction_full"].to_numpy(dtype=np.float64)
            loso_values = motif_df["motif_fraction_loso"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": artifact.dataset_name,
                    "comparison_name": str(row["comparison_name"]),
                    "motif_id": str(motif_id),
                    "motif_label": str(row["motif_label"]),
                    "full_delta_fraction": float(row["full_delta_fraction"]),
                    "loso_delta_fraction": float(row["loso_delta_fraction"]),
                    "effect_direction_consistent": bool(sign_match(float(row["full_delta_fraction"]), float(row["loso_delta_fraction"]))),
                    "abs_delta_shift": float(abs(float(row["loso_delta_fraction"]) - float(row["full_delta_fraction"]))),
                    "full_sample_p_two_sided": float(row["full_sample_permutation_pvalue_two_sided"]),
                    "loso_sample_p_two_sided": float(row["loso_sample_permutation_pvalue_two_sided"]),
                    "pvalue_shift_log10": float(safe_neglog10(row["loso_sample_permutation_pvalue_two_sided"]) - safe_neglog10(row["full_sample_permutation_pvalue_two_sided"])),
                    "motif_abundance_correlation": float(pearson_correlation(full_values, loso_values)),
                    "mean_abs_fraction_shift": float(np.mean(np.abs(full_values - loso_values))),
                    "max_abs_fraction_shift": float(np.max(np.abs(full_values - loso_values))),
                    "sample_presence_agreement": float(np.mean(((full_values > 0.0) == (loso_values > 0.0)).astype(np.float64))),
                    "full_controlled_support_tier": str(row["full_controlled_support_tier"]),
                    "loso_controlled_support_tier": str(row["loso_controlled_support_tier"]),
                    "dataset_mean_assignment_agreement": float(artifact.fold_summary["assignment_agreement_vs_full"].mean()),
                    "dataset_mean_alignment_cost": float(artifact.fold_summary["alignment_cost_mean"].mean()),
                    "dataset_mean_assignment_distance": float(artifact.fold_summary["heldout_assignment_distance_mean"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset_id", "loso_sample_p_two_sided", "motif_id"]).reset_index(drop=True)


def build_week6_vs_06b_comparison(
    *,
    week6_differential: pd.DataFrame,
    week6_sample_level: pd.DataFrame,
    sample_aware_differential: pd.DataFrame,
    sample_aware_sample_level: pd.DataFrame,
    week6_decision: pd.DataFrame,
    sample_aware_decision: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    week6_scope = week6_sample_level.loc[week6_sample_level["analysis_scope"].astype(str) == "holdout_oof"].copy()
    sample_aware_scope = sample_aware_sample_level.loc[sample_aware_sample_level["analysis_scope"].astype(str) == "loso"].copy()

    counts = build_naive_only_count_summary(week6_decision=week6_decision, sample_aware_decision=sample_aware_decision)
    count_lookup = counts.set_index("dataset_id").to_dict(orient="index") if not counts.empty else {}

    for dataset_id, week6_dataset in week6_scope.groupby("dataset_id", observed=False):
        sample_aware_dataset = sample_aware_scope.loc[sample_aware_scope["dataset_id"].astype(str) == str(dataset_id)].copy()
        if sample_aware_dataset.empty:
            continue
        week6_matrix = pivot_sample_fraction_table(week6_dataset)
        sample_aware_matrix = pivot_sample_fraction_table(sample_aware_dataset)
        alignment = build_alignment_score_table(
            week6_matrix=week6_matrix,
            sample_aware_matrix=sample_aware_matrix,
            week6_labels=build_motif_label_lookup(week6_dataset),
            sample_aware_labels=build_motif_label_lookup(sample_aware_dataset),
        )
        if alignment.empty:
            continue
        best_alignment = (
            alignment.sort_values(["week6_motif_id", "alignment_score", "sample_fraction_correlation", "label_token_jaccard"], ascending=[True, False, False, False])
            .groupby("week6_motif_id", observed=False)
            .head(1)
            .reset_index(drop=True)
        )
        dataset_counts = count_lookup.get(str(dataset_id), {})
        for _, match_row in best_alignment.iterrows():
            week6_row = week6_differential.loc[
                (week6_differential["dataset_id"].astype(str) == str(dataset_id))
                & (week6_differential["motif_id"].astype(str) == str(match_row["week6_motif_id"]))
            ]
            sample_aware_row = sample_aware_differential.loc[
                (sample_aware_differential["dataset_id"].astype(str) == str(dataset_id))
                & (sample_aware_differential["motif_id"].astype(str) == str(match_row["matched_06b_motif_id"]))
            ]
            if week6_row.empty or sample_aware_row.empty:
                continue
            week6_row = week6_row.iloc[0]
            sample_aware_row = sample_aware_row.iloc[0]
            week6_tier = str(week6_row["holdout_controlled_support_tier"])
            sample_aware_tier = str(sample_aware_row["loso_controlled_support_tier"])
            rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "dataset_name": str(week6_row["dataset_name"]),
                    "comparison_name": str(week6_row["comparison_name"]),
                    "week6_motif_id": str(match_row["week6_motif_id"]),
                    "week6_motif_label": str(week6_row["motif_label"]),
                    "matched_06b_motif_id": str(match_row["matched_06b_motif_id"]),
                    "matched_06b_motif_label": str(sample_aware_row["motif_label"]),
                    "sample_fraction_correlation": float(match_row["sample_fraction_correlation"]),
                    "label_token_jaccard": float(match_row["label_token_jaccard"]),
                    "alignment_score": float(match_row["alignment_score"]),
                    "week6_delta_fraction": float(week6_row["holdout_delta_fraction"]),
                    "week6_sample_p_two_sided": float(week6_row["holdout_sample_permutation_pvalue_two_sided"]),
                    "week6_sample_p_one_sided": float(week6_row["holdout_sample_permutation_pvalue_one_sided"]),
                    "week6_max_t_p": float(week6_row["holdout_label_max_t_pvalue"]),
                    "week6_null_effect_p": float(week6_row["holdout_synthetic_null_effect_pvalue"]),
                    "week6_loso_sign_consistency": float(week6_row["holdout_loso_sign_consistency"]),
                    "week6_controlled_support_tier": week6_tier,
                    "week6_decision_label": decision_label_from_metrics(
                        support_tier=week6_tier,
                        naive_spot_p=float(week6_row["holdout_naive_spot_pvalue"]),
                    ),
                    "week6_naive_minus_controlled_log10p": float(week6_row["holdout_naive_minus_controlled_log10p"]),
                    "week6_primary_rank_score": float(week6_row["primary_rank_score"]),
                    "week6b_full_delta_fraction": float(sample_aware_row["full_delta_fraction"]),
                    "week6b_loso_delta_fraction": float(sample_aware_row["loso_delta_fraction"]),
                    "week6b_sample_p_two_sided": float(sample_aware_row["loso_sample_permutation_pvalue_two_sided"]),
                    "week6b_sample_p_one_sided": float(sample_aware_row["loso_sample_permutation_pvalue_one_sided"]),
                    "week6b_max_t_p": float(sample_aware_row["loso_label_max_t_pvalue"]),
                    "week6b_null_effect_p": float(sample_aware_row["loso_synthetic_null_effect_pvalue"]),
                    "week6b_loso_sign_consistency": float(sample_aware_row["loso_loso_sign_consistency"]),
                    "week6b_controlled_support_tier": sample_aware_tier,
                    "week6b_decision_label": decision_label_from_metrics(
                        support_tier=sample_aware_tier,
                        naive_spot_p=float(sample_aware_row["loso_naive_spot_pvalue"]),
                    ),
                    "week6b_naive_minus_controlled_log10p": float(sample_aware_row["loso_naive_minus_controlled_log10p"]),
                    "week6b_primary_rank_score": float(sample_aware_row["primary_rank_score"]),
                    "controlled_p_shift_log10": float(
                        safe_neglog10(sample_aware_row["loso_sample_permutation_pvalue_two_sided"])
                        - safe_neglog10(week6_row["holdout_sample_permutation_pvalue_two_sided"])
                    ),
                    "naive_only_resolved": bool(week6_tier == "naive_only" and sample_aware_tier != "naive_only"),
                    "naive_only_created": bool(week6_tier != "naive_only" and sample_aware_tier == "naive_only"),
                    "week6_dataset_naive_only_count": int(dataset_counts.get("week6_naive_only_count", 0)),
                    "week6b_dataset_naive_only_count": int(dataset_counts.get("week6b_naive_only_count", 0)),
                    "naive_only_count_delta": int(dataset_counts.get("naive_only_count_delta", 0)),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset_id", "alignment_score", "week6_motif_id"], ascending=[True, False, True]).reset_index(drop=True)


def build_naive_only_count_summary(*, week6_decision: pd.DataFrame, sample_aware_decision: pd.DataFrame) -> pd.DataFrame:
    dataset_ids = sorted(set(week6_decision["dataset_id"].astype(str).tolist()) | set(sample_aware_decision["dataset_id"].astype(str).tolist()))
    rows: list[dict[str, object]] = []
    for dataset_id in dataset_ids:
        week6_subset = week6_decision.loc[week6_decision["dataset_id"].astype(str) == str(dataset_id)]
        sample_aware_subset = sample_aware_decision.loc[sample_aware_decision["dataset_id"].astype(str) == str(dataset_id)]
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "dataset_name": str(week6_subset["dataset_name"].iloc[0] if not week6_subset.empty else sample_aware_subset["dataset_name"].iloc[0]),
                "week6_naive_only_count": int(np.sum(week6_subset["decision_label"].astype(str) == "naive_only_signal")),
                "week6b_naive_only_count": int(np.sum(sample_aware_subset["decision_label"].astype(str) == "naive_only_signal")),
                "week6_robust_count": int(np.sum(week6_subset["decision_label"].astype(str) == "robust_biological_signal")),
                "week6b_robust_count": int(np.sum(sample_aware_subset["decision_label"].astype(str) == "robust_biological_signal")),
                "week6_no_signal_count": int(np.sum(week6_subset["decision_label"].astype(str) == "no_signal")),
                "week6b_no_signal_count": int(np.sum(sample_aware_subset["decision_label"].astype(str) == "no_signal")),
                "naive_only_count_delta": int(np.sum(sample_aware_subset["decision_label"].astype(str) == "naive_only_signal") - np.sum(week6_subset["decision_label"].astype(str) == "naive_only_signal")),
            }
        )
    return pd.DataFrame(rows)


def build_combined_fold_summary(*, dataset_artifacts: dict[str, DatasetArtifacts]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for artifact in dataset_artifacts.values():
        frame = artifact.fold_summary.copy()
        frame["dataset_id"] = artifact.dataset_id
        frame["dataset_name"] = artifact.dataset_name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_robust_motif_tracking(
    *,
    artifacts: dict[str, DatasetArtifacts],
    week6_differential: pd.DataFrame,
    week6_sample_level: pd.DataFrame,
    sample_aware_differential: pd.DataFrame,
    sample_aware_sample_level: pd.DataFrame,
) -> pd.DataFrame:
    dataset_id = str(WEEK6_REFERENCE["dataset_id"])
    if dataset_id not in artifacts:
        return pd.DataFrame()
    artifact = artifacts[dataset_id]
    week6_scope = week6_sample_level.loc[
        (week6_sample_level["dataset_id"].astype(str) == dataset_id)
        & (week6_sample_level["analysis_scope"].astype(str) == "holdout_oof")
        & (week6_sample_level["motif_id"].astype(str) == str(WEEK6_REFERENCE["motif_id"]))
    ].copy()
    sample_aware_scope = sample_aware_sample_level.loc[
        (sample_aware_sample_level["dataset_id"].astype(str) == dataset_id)
        & (sample_aware_sample_level["analysis_scope"].astype(str) == "loso")
    ].copy()
    if week6_scope.empty or sample_aware_scope.empty:
        return pd.DataFrame()

    week6_values = week6_scope.sort_values("sample_id").set_index("sample_id")["motif_fraction"]
    week6_diff_row = week6_differential.loc[
        (week6_differential["dataset_id"].astype(str) == dataset_id)
        & (week6_differential["motif_id"].astype(str) == str(WEEK6_REFERENCE["motif_id"]))
    ]
    if week6_diff_row.empty:
        return pd.DataFrame()
    week6_diff_row = week6_diff_row.iloc[0]

    rows: list[dict[str, object]] = []
    for motif_id, motif_df in sample_aware_scope.groupby("motif_id", observed=False):
        aligned = week6_values.to_frame("week6_fraction").merge(
            motif_df.sort_values("sample_id").set_index("sample_id")["motif_fraction"].rename("week6b_fraction"),
            left_index=True,
            right_index=True,
            how="inner",
        )
        diff_row = sample_aware_differential.loc[
            (sample_aware_differential["dataset_id"].astype(str) == dataset_id)
            & (sample_aware_differential["motif_id"].astype(str) == str(motif_id))
        ]
        if diff_row.empty:
            continue
        diff_row = diff_row.iloc[0]
        label = str(diff_row["motif_label"])
        biology = summarize_sample_aware_motif_biology(
            artifact=artifact,
            motif_id=str(motif_id),
        )
        top_pathway = biology["top_pathway"]
        top_pathway_p = biology["top_pathway_pvalue"]
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": artifact.dataset_name,
                "week6_motif_id": str(WEEK6_REFERENCE["motif_id"]),
                "week6_motif_label": str(WEEK6_REFERENCE["motif_label"]),
                "week6_reference_pathway": str(WEEK6_REFERENCE["pathway"]),
                "week6_reference_markers": ";".join(WEEK6_REFERENCE["marker_genes"]),
                "matched_06b_motif_id": str(motif_id),
                "matched_06b_motif_label": label,
                "sample_fraction_correlation": float(pearson_correlation(aligned["week6_fraction"].to_numpy(dtype=np.float64), aligned["week6b_fraction"].to_numpy(dtype=np.float64))),
                "label_token_jaccard": float(label_token_jaccard(str(WEEK6_REFERENCE["motif_label"]), label)),
                "alignment_score": float(0.80 * max(pearson_correlation(aligned["week6_fraction"].to_numpy(dtype=np.float64), aligned["week6b_fraction"].to_numpy(dtype=np.float64)), 0.0) + 0.20 * label_token_jaccard(str(WEEK6_REFERENCE["motif_label"]), label)),
                "full_delta_fraction": float(diff_row["full_delta_fraction"]),
                "loso_delta_fraction": float(diff_row["loso_delta_fraction"]),
                "loso_sample_p_two_sided": float(diff_row["loso_sample_permutation_pvalue_two_sided"]),
                "loso_sample_p_one_sided": float(diff_row["loso_sample_permutation_pvalue_one_sided"]),
                "loso_max_t_p": float(diff_row["loso_label_max_t_pvalue"]),
                "loso_null_effect_p": float(diff_row["loso_synthetic_null_effect_pvalue"]),
                "loso_loso_sign_consistency": float(diff_row["loso_loso_sign_consistency"]),
                "loso_controlled_support_tier": str(diff_row["loso_controlled_support_tier"]),
                "marker_genes_top8": biology["marker_genes_top8"],
                "dominant_cell_types": biology["dominant_cell_types"],
                "top_pathway": top_pathway,
                "top_pathway_pvalue": float(top_pathway_p),
                "pathway_consistent_with_week6": bool(str(top_pathway) == str(WEEK6_REFERENCE["pathway"])),
                "label_consistent_with_week6": bool(label_token_jaccard(str(WEEK6_REFERENCE["motif_label"]), label) >= 0.34),
                "marker_overlap_count": int(marker_overlap_count(biology["marker_genes_list"], WEEK6_REFERENCE["marker_genes"])),
                "week6_delta_fraction": float(week6_diff_row["holdout_delta_fraction"]),
                "week6_sample_p_two_sided": float(week6_diff_row["holdout_sample_permutation_pvalue_two_sided"]),
                "week6_sample_p_one_sided": float(week6_diff_row["holdout_sample_permutation_pvalue_one_sided"]),
                "week6_max_t_p": float(week6_diff_row["holdout_label_max_t_pvalue"]),
                "week6_null_effect_p": float(week6_diff_row["holdout_synthetic_null_effect_pvalue"]),
                "week6_loso_sign_consistency": float(week6_diff_row["holdout_loso_sign_consistency"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["alignment_score", "loso_sample_p_two_sided", "loso_loso_sign_consistency", "matched_06b_motif_id"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)


def build_alignment_score_table(
    *,
    week6_matrix: pd.DataFrame,
    sample_aware_matrix: pd.DataFrame,
    week6_labels: dict[str, str],
    sample_aware_labels: dict[str, str],
) -> pd.DataFrame:
    if week6_matrix.empty or sample_aware_matrix.empty:
        return pd.DataFrame()
    common_samples = sorted(set(week6_matrix.index.astype(str).tolist()) & set(sample_aware_matrix.index.astype(str).tolist()))
    if not common_samples:
        return pd.DataFrame()
    left = week6_matrix.loc[common_samples]
    right = sample_aware_matrix.loc[common_samples]
    rows: list[dict[str, object]] = []
    for week6_motif_id in left.columns.astype(str).tolist():
        for sample_aware_motif_id in right.columns.astype(str).tolist():
            corr = pearson_correlation(
                left[week6_motif_id].to_numpy(dtype=np.float64),
                right[sample_aware_motif_id].to_numpy(dtype=np.float64),
            )
            label_jaccard = label_token_jaccard(week6_labels.get(week6_motif_id, ""), sample_aware_labels.get(sample_aware_motif_id, ""))
            rows.append(
                {
                    "week6_motif_id": str(week6_motif_id),
                    "matched_06b_motif_id": str(sample_aware_motif_id),
                    "sample_fraction_correlation": float(corr),
                    "label_token_jaccard": float(label_jaccard),
                    "alignment_score": float(0.80 * max(corr, 0.0) + 0.20 * label_jaccard),
                }
            )
    return pd.DataFrame(rows)


def pivot_sample_fraction_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.pivot_table(index="sample_id", columns="motif_id", values="motif_fraction", fill_value=0.0)
        .sort_index()
        .copy()
    )


def build_motif_label_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return frame.loc[:, ["motif_id", "motif_label"]].drop_duplicates().set_index("motif_id")["motif_label"].astype(str).to_dict()


def decision_label_from_metrics(*, support_tier: str, naive_spot_p: float, naive_threshold: float = 0.05) -> str:
    if str(support_tier) in {"heldout_supported", "sample_level_supported"}:
        return "robust_biological_signal"
    if np.isfinite(float(naive_spot_p)) and float(naive_spot_p) <= float(naive_threshold):
        return "naive_only_signal"
    return "no_signal"


def summarize_sample_aware_motif_biology(*, artifact: DatasetArtifacts, motif_id: str) -> dict[str, object]:
    marker_df = compute_marker_genes(dataset=artifact.dataset, spot_table=artifact.loso_spot_table, motif_id=motif_id, top_n=8)
    pathway_df = compute_curated_pathway_enrichment(
        marker_df["gene"].astype(str).tolist(),
        artifact.dataset.var_names.astype(str).tolist(),
        dataset_id=artifact.dataset_id,
    )
    cell_type_summary = (
        artifact.loso_spot_table.loc[artifact.loso_spot_table["motif_id"].astype(str) == str(motif_id), "cell_type"]
        .astype(str)
        .value_counts(normalize=True)
        .head(3)
    )
    top_pathway = ""
    top_pathway_pvalue = float("nan")
    if not pathway_df.empty:
        top_pathway = str(pathway_df.iloc[0]["pathway"])
        top_pathway_pvalue = float(pathway_df.iloc[0]["pvalue"])
    return {
        "marker_genes_list": marker_df["gene"].astype(str).tolist(),
        "marker_genes_top8": ";".join(marker_df["gene"].astype(str).head(8).tolist()),
        "dominant_cell_types": ";".join([f"{idx}={value:.2f}" for idx, value in cell_type_summary.items()]),
        "top_pathway": top_pathway,
        "top_pathway_pvalue": top_pathway_pvalue,
    }


def compute_marker_genes(*, dataset, spot_table: pd.DataFrame, motif_id: str, top_n: int) -> pd.DataFrame:
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


def compute_curated_pathway_enrichment(marker_genes: list[str], universe_genes: list[str], *, dataset_id: str) -> pd.DataFrame:
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
    return pd.DataFrame(rows).sort_values("pvalue", ascending=True).reset_index(drop=True) if rows else pd.DataFrame(columns=["pathway", "overlap_genes", "odds_ratio", "pvalue"])


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


def marker_overlap_count(left: list[str], right: list[str]) -> int:
    return len({value.upper() for value in left} & {value.upper() for value in right})


def label_token_jaccard(left: str, right: str) -> float:
    left_tokens = tokenize_label(left)
    right_tokens = tokenize_label(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return float(len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1))


def tokenize_label(value: str) -> set[str]:
    tokens = {token for token in re.split(r"[^A-Za-z0-9]+", str(value).lower()) if token}
    stop = {"dense", "sparse", "mixed", "high", "low", "mide", "highe", "lowe", "mid"}
    return {token for token in tokens if token not in stop}


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


def safe_neglog10(value: float | int | np.floating) -> float:
    numeric = float(value)
    return float(-math.log10(max(numeric, 1.0e-12)))


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


def plot_week6_vs_06b_pvalue_shift(week6_vs_06b: pd.DataFrame, output_path: Path) -> None:
    _set_publication_style()
    if week6_vs_06b.empty:
        return
    frame = week6_vs_06b.copy()
    frame["x"] = frame["week6_sample_p_two_sided"].map(safe_neglog10)
    frame["y"] = frame["week6b_sample_p_two_sided"].map(safe_neglog10)
    colors = {"wu2021_breast_visium": "#c26b3b", "gse220442_ad_mtg": "#3a78a1"}
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for _, row in frame.iterrows():
        ax.scatter(
            float(row["x"]),
            float(row["y"]),
            s=82,
            color=colors.get(str(row["dataset_id"]), "#66788a"),
            alpha=0.90,
            edgecolors="#ffffff",
            linewidths=0.8,
        )
    limit = float(max(frame["x"].max(), frame["y"].max()) + 0.5)
    ax.plot([0.0, limit], [0.0, limit], linestyle="--", color="#444444", linewidth=1.0)
    highlight = frame.sort_values(["alignment_score", "controlled_p_shift_log10"], ascending=[False, False]).head(4)
    for _, row in highlight.iterrows():
        ax.text(float(row["x"]) + 0.05, float(row["y"]) + 0.03, f"{row['dataset_id']}:{row['week6_motif_id']}", fontsize=7.2)
    ax.set_xlabel("-log10 Week 6 controlled p")
    ax.set_ylabel("-log10 Week 6b controlled p")
    ax.set_title("Week 6 vs Week 6b controlled p-value shift")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_full_vs_loso_effect_scatter(consistency_table: pd.DataFrame, output_path: Path) -> None:
    _set_publication_style()
    if consistency_table.empty:
        return
    frame = consistency_table.copy()
    x = frame["full_delta_fraction"].to_numpy(dtype=np.float64)
    y = frame["loso_delta_fraction"].to_numpy(dtype=np.float64)
    color_value = frame["motif_abundance_correlation"].fillna(0.0).to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    scatter = ax.scatter(
        x,
        y,
        c=color_value,
        cmap="viridis",
        s=88,
        alpha=0.92,
        edgecolors="#ffffff",
        linewidths=0.8,
    )
    limit = float(max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y))) * 1.12) if np.isfinite(x).any() and np.isfinite(y).any() else 1.0
    ax.plot([-limit, limit], [-limit, limit], linestyle="--", color="#444444", linewidth=1.0)
    for _, row in frame.sort_values("loso_sample_p_two_sided", ascending=True).head(5).iterrows():
        ax.text(float(row["full_delta_fraction"]) + 0.01, float(row["loso_delta_fraction"]) + 0.01, f"{row['dataset_id']}:{row['motif_id']}", fontsize=7.1)
    ax.set_xlabel("Full sample-aware effect")
    ax.set_ylabel("LOSO sample-aware effect")
    ax.set_title("Full canonical vs LOSO effect consistency")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Sample-level abundance correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_naive_only_count_reduction(week6_vs_06b: pd.DataFrame, output_path: Path) -> None:
    _set_publication_style()
    if week6_vs_06b.empty:
        return
    summary = week6_vs_06b.loc[:, ["dataset_id", "dataset_name", "week6_dataset_naive_only_count", "week6b_dataset_naive_only_count"]].drop_duplicates()
    if summary.empty:
        return
    x = np.arange(summary.shape[0], dtype=np.float64)
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(x - width / 2.0, summary["week6_dataset_naive_only_count"].to_numpy(dtype=np.float64), width=width, color="#c88b54", label="Week 6")
    ax.bar(x + width / 2.0, summary["week6b_dataset_naive_only_count"].to_numpy(dtype=np.float64), width=width, color="#4f8da6", label="Week 6b")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["dataset_id"].astype(str).tolist(), rotation=10)
    ax.set_ylabel("Naive-only motif count")
    ax.set_title("Naive-only signal shrinkage after sample-aware motifs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_ad_motif_tracking(
    *,
    robust_tracking: pd.DataFrame,
    week6_sample_level: pd.DataFrame,
    sample_aware_sample_level: pd.DataFrame,
    output_path: Path,
) -> None:
    _set_publication_style()
    if robust_tracking.empty:
        return
    tracking_row = robust_tracking.iloc[0]
    dataset_id = str(tracking_row["dataset_id"])
    matched_motif_id = str(tracking_row["matched_06b_motif_id"])
    week6_subset = week6_sample_level.loc[
        (week6_sample_level["dataset_id"].astype(str) == dataset_id)
        & (week6_sample_level["analysis_scope"].astype(str) == "holdout_oof")
        & (week6_sample_level["motif_id"].astype(str) == str(WEEK6_REFERENCE["motif_id"]))
    ].copy()
    week6_subset["source"] = "Week 6"
    sample_aware_subset = sample_aware_sample_level.loc[
        (sample_aware_sample_level["dataset_id"].astype(str) == dataset_id)
        & (sample_aware_sample_level["analysis_scope"].astype(str) == "loso")
        & (sample_aware_sample_level["motif_id"].astype(str) == matched_motif_id)
    ].copy()
    sample_aware_subset["source"] = "Week 6b"
    if week6_subset.empty or sample_aware_subset.empty:
        return

    merged = week6_subset.loc[:, ["sample_id", "condition", "motif_fraction"]].rename(columns={"motif_fraction": "week6_fraction"}).merge(
        sample_aware_subset.loc[:, ["sample_id", "condition", "motif_fraction"]].rename(columns={"motif_fraction": "week6b_fraction"}),
        on=["sample_id", "condition"],
        how="inner",
    )
    merged = merged.sort_values(["condition", "sample_id"]).reset_index(drop=True)
    x = np.arange(merged.shape[0], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    ax = axes[0]
    ax.plot(x, merged["week6_fraction"].to_numpy(dtype=np.float64), marker="o", color="#c26b3b", linewidth=1.8, label="Week 6 motif_02")
    ax.plot(x, merged["week6b_fraction"].to_numpy(dtype=np.float64), marker="o", color="#3a78a1", linewidth=1.8, label=f"Week 6b {matched_motif_id}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['sample_id']}\n{row['condition']}" for _, row in merged.iterrows()], rotation=15)
    ax.set_ylabel("Sample motif fraction")
    ax.set_title("Per-sample motif tracking")
    ax.legend(frameon=False)

    ax = axes[1]
    metrics = ["sample_p_two_sided", "sample_p_one_sided", "max_t_p", "null_effect_p"]
    week6_values = np.asarray(
        [
            tracking_row["week6_sample_p_two_sided"],
            tracking_row["week6_sample_p_one_sided"],
            tracking_row["week6_max_t_p"],
            tracking_row["week6_null_effect_p"],
        ],
        dtype=np.float64,
    )
    week6b_values = np.asarray(
        [
            tracking_row["loso_sample_p_two_sided"],
            tracking_row["loso_sample_p_one_sided"],
            tracking_row["loso_max_t_p"],
            tracking_row["loso_null_effect_p"],
        ],
        dtype=np.float64,
    )
    xi = np.arange(len(metrics), dtype=np.float64)
    width = 0.34
    ax.bar(xi - width / 2.0, [safe_neglog10(value) for value in week6_values], width=width, color="#c26b3b", label="Week 6")
    ax.bar(xi + width / 2.0, [safe_neglog10(value) for value in week6b_values], width=width, color="#3a78a1", label="Week 6b")
    ax.set_xticks(xi)
    ax.set_xticklabels(["two-sided p", "one-sided p", "maxT p", "null p"], rotation=10)
    ax.set_ylabel("-log10 p")
    ax.set_title("Controlled evidence tracking")
    ax.legend(frameon=False)

    fig.suptitle(f"AD MTG Week 6 motif_02 tracking -> {matched_motif_id}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_breast_false_positive_reduction(*, week6_vs_06b: pd.DataFrame, output_path: Path) -> None:
    _set_publication_style()
    if week6_vs_06b.empty:
        return
    breast = week6_vs_06b.loc[week6_vs_06b["dataset_id"].astype(str) == "wu2021_breast_visium"].copy()
    if breast.empty:
        return
    summary = breast.iloc[0]
    values = np.asarray(
        [
            float(summary["week6_dataset_naive_only_count"]),
            float(summary["week6b_dataset_naive_only_count"]),
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.bar([0, 1], values, color=["#c26b3b", "#3a78a1"], width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Week 6", "Week 6b"])
    ax.set_ylabel("Breast naive-only motifs")
    ax.set_title("Breast false-positive reduction")
    for idx, value in enumerate(values.tolist()):
        ax.text(idx, value + 0.1, f"{int(value)}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_protocol(
    *,
    output_dir: Path,
    runtime_info,
    dataset_paths: list[Path],
    dataset_summaries: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Protocol",
        "",
        "## Goal",
        "",
        "Run Week 6b sample-aware cross-sample differential analysis by reusing the Week 4b sample-aware motif recipe, testing whether the Week 6 robust AD motif survives stricter motif construction, whether naive-only calls shrink, and whether full-vs-LOSO conclusions stay aligned.",
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
    ]
    lines.extend([f"- `{path.relative_to(ROOT)}`" for path in dataset_paths])
    lines.extend(
        [
            f"- `experiments\\06_cross_sample_differential\\results\\differential_test_results.csv`",
            f"- `experiments\\06_cross_sample_differential\\results\\sample_level_motif_table.csv`",
            "",
            "## Steps",
            "",
            "1. Refit the Week 4b sample-aware motif baseline on each dataset with per-sample balanced training and per-sample feature normalization.",
            "2. Assign canonical full-model motifs and stricter LOSO motifs aligned back to the canonical motif IDs.",
            "3. Run Week 6-style controlled differential testing on both full and LOSO sample-aware motifs: exact sample permutation, one-sided p, maxT, naive spot Fisher, LOSO sign consistency, and size-matched null controls.",
            "4. Compare Week 6 hold-out motifs against Week 6b LOSO motifs by sample-level abundance correlation and motif-label overlap.",
            "5. Quantify full-vs-LOSO effect direction consistency, p-value shifts, abundance correlation, and fold alignment quality.",
            "6. Track the Week 6 AD MTG robust motif into the best aligned Week 6b motif and check label, marker, and pathway consistency.",
            "",
            "## Outputs",
            "",
            "- `results/sample_aware_differential_test_results.csv`",
            "- `results/sample_aware_null_control_results.csv`",
            "- `results/sample_level_motif_abundance.csv`",
            "- `results/loso_fold_summary.csv`",
            "- `results/week6_vs_06b_comparison.csv`",
            "- `results/full_vs_loso_differential_consistency.csv`",
            "- `results/motif_stability_decision_matrix.csv`",
            "- `results/robust_motif_tracking.csv`",
            "- `figures/week6_vs_06b_pvalue_shift.png`",
            "- `figures/full_vs_loso_effect_scatter.png`",
            "- `figures/naive_only_count_reduction.png`",
            "- `figures/ad_motif02_tracking.png`",
            "- `figures/breast_false_positive_reduction.png`",
            "",
            "## Run Command",
            "",
            f"- `python scripts/run_sample_aware_cross_sample_differential.py --output-dir {output_dir}`",
            "",
            "## Dataset Snapshot",
            "",
        ]
    )
    for row in dataset_summaries:
        condition_counts = ", ".join(f"{key}={value}" for key, value in row["condition_counts"].items())
        lines.append(
            f"- `{row['dataset_id']}`: `{row['n_samples']}` samples ({condition_counts}), `{row['n_spots']}` spots, `{row['n_motifs']}` sample-aware motifs, "
            f"core scale=`{row['core_scale']}`, coherence_z=`{row['spatial_coherence_zscore']:.2f}`, "
            f"mean LOSO agreement=`{row['mean_loso_assignment_agreement']:.3f}`, mean alignment cost=`{row['mean_alignment_cost']:.3f}`."
        )
    lines.extend(
        [
            "",
            "## Statistical Note",
            "",
            f"- Size-matched random null motifs were drawn `{int(args.null_iterations)}` times per motif and per analysis scope (`full_model`, `loso`).",
            "- The exact sample-level p-values remain discrete because the datasets still have six biological samples each.",
        ]
    )
    (output_dir / "protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(
    *,
    output_dir: Path,
    week6_vs_06b: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    consistency_table: pd.DataFrame,
    robust_tracking: pd.DataFrame,
) -> None:
    lines = [
        "# Analysis",
        "",
        "## Q1. Robust AD Motif Retention",
        "",
    ]
    if robust_tracking.empty:
        lines.append("- The Week 6 AD MTG motif could not be aligned to a Week 6b candidate from the current outputs.")
    else:
        top = robust_tracking.iloc[0]
        retained = str(top["loso_controlled_support_tier"]) in {"heldout_supported", "sample_level_supported"}
        lines.append(
            f"- Week 6 reference motif `{top['week6_motif_id']}` aligned best to Week 6b `{top['matched_06b_motif_id']}` (`{top['matched_06b_motif_label']}`), "
            f"sample-fraction correlation `{float(top['sample_fraction_correlation']):.3f}`, LOSO delta `{float(top['loso_delta_fraction']):.3f}`, "
            f"two-sided p `{float(top['loso_sample_p_two_sided']):.3f}`, one-sided p `{float(top['loso_sample_p_one_sided']):.3f}`, "
            f"maxT p `{float(top['loso_max_t_p']):.3f}`, null motif p `{float(top['loso_null_effect_p']):.3f}`, "
            f"LOSO sign consistency `{float(top['loso_loso_sign_consistency']):.2f}`, support tier `{top['loso_controlled_support_tier']}`."
        )
        lines.append(
            f"- Retention verdict: `{'retained' if retained else 'not retained'}` under sample-aware motif construction."
        )
        lines.append(
            f"- Biology consistency: top pathway `{top['top_pathway']}` (p=`{float(top['top_pathway_pvalue']):.3g}`), marker overlap with Week 6 reference `{int(top['marker_overlap_count'])}`, "
            f"label consistency=`{bool(top['label_consistent_with_week6'])}`, pathway consistency=`{bool(top['pathway_consistent_with_week6'])}`."
        )

    lines.extend(["", "## Q2. Naive-Only Reduction", ""])
    if week6_vs_06b.empty:
        lines.append("- Week 6 vs Week 6b comparison table is empty.")
    else:
        count_summary = week6_vs_06b.loc[:, ["dataset_id", "week6_dataset_naive_only_count", "week6b_dataset_naive_only_count"]].drop_duplicates()
        for _, row in count_summary.iterrows():
            lines.append(
                f"- `{row['dataset_id']}`: naive-only count `Week 6={int(row['week6_dataset_naive_only_count'])}` vs `Week 6b={int(row['week6b_dataset_naive_only_count'])}`."
            )
        breast = count_summary.loc[count_summary["dataset_id"].astype(str) == "wu2021_breast_visium"]
        if not breast.empty:
            row = breast.iloc[0]
            change = int(row["week6b_dataset_naive_only_count"]) - int(row["week6_dataset_naive_only_count"])
            lines.append(f"- Breast validity story: naive-only change=`{change}` motifs relative to Week 6.")

    lines.extend(["", "## Q3. Full vs LOSO Consistency", ""])
    if consistency_table.empty:
        lines.append("- Full-vs-LOSO consistency table is empty.")
    else:
        for dataset_id, subset in consistency_table.groupby("dataset_id", observed=False):
            direction = float(np.mean(subset["effect_direction_consistent"].astype(float))) if not subset.empty else float("nan")
            abundance_corr = float(np.nanmedian(subset["motif_abundance_correlation"].to_numpy(dtype=np.float64))) if not subset.empty else float("nan")
            p_shift = float(np.nanmedian(subset["pvalue_shift_log10"].to_numpy(dtype=np.float64))) if not subset.empty else float("nan")
            lines.append(
                f"- `{dataset_id}`: effect-direction consistency=`{direction:.2f}`, median full-vs-LOSO abundance correlation=`{abundance_corr:.2f}`, median p-value shift=`{p_shift:.2f}` log10 units."
            )
        unstable = consistency_table.sort_values(["abs_delta_shift", "motif_abundance_correlation"], ascending=[False, True]).head(3)
        for _, row in unstable.iterrows():
            lines.append(
                f"- Largest full-vs-LOSO drift: `{row['dataset_id']}:{row['motif_id']}` full delta=`{float(row['full_delta_fraction']):.3f}` vs LOSO delta=`{float(row['loso_delta_fraction']):.3f}`, abundance correlation=`{float(row['motif_abundance_correlation']):.2f}`."
            )

    lines.extend(["", "## Decision Matrix", ""])
    if decision_matrix.empty:
        lines.append("- No Week 6b decision matrix rows were generated.")
    else:
        for dataset_id, subset in decision_matrix.groupby("dataset_id", observed=False):
            robust = int(np.sum(subset["decision_label"].astype(str) == "robust_biological_signal"))
            naive_only = int(np.sum(subset["decision_label"].astype(str) == "naive_only_signal"))
            no_signal = int(np.sum(subset["decision_label"].astype(str) == "no_signal"))
            lines.append(
                f"- `{dataset_id}`: robust=`{robust}`, naive-only=`{naive_only}`, no-signal=`{no_signal}`."
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Week 6b should be read as a statistical-validity stress test on the Week 4b sample-aware motif baseline, not as another motif discovery pass.",
            "- The decisive comparisons are the matched Week 6 -> Week 6b motif tracking row, the naive-only count shrinkage, and the full-vs-LOSO consistency table.",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
