from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spatial_context.cross_sample_differential import (  # noqa: E402
    assign_out_of_fold_motifs,
    build_sample_level_motif_table,
    compute_differential_statistics,
)
from spatial_context.motif_embedding import build_tissue_motif_feature_bundle, fit_tissue_motif_model  # noqa: E402
from spatial_context.neighborhood import get_runtime_info, load_spatial_h5ad, summarize_neighborhoods  # noqa: E402


DEFAULT_DATASETS = (
    ROOT / "data" / "spatial_processed" / "wu2021_breast_visium.h5ad",
    ROOT / "data" / "spatial_processed" / "gse220442_ad_mtg.h5ad",
)
EXPERIMENT_DIR = ROOT / "experiments" / "06_cross_sample_differential"


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset_id: str
    dataset_name: str
    dataset: object
    neighborhood_summary: object
    full_result: object
    oof_spot_table: pd.DataFrame
    sample_level_table: pd.DataFrame
    differential_table: pd.DataFrame
    fold_summary: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 6 cross-sample differential analysis with statistical controls.")
    parser.add_argument(
        "--dataset-paths",
        nargs="*",
        default=[str(path) for path in DEFAULT_DATASETS],
        help="Processed .h5ad files to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENT_DIR),
        help="Experiment output directory.",
    )
    parser.add_argument("--top-variable-genes", type=int, default=256)
    parser.add_argument("--n-expression-programs", type=int, default=6)
    parser.add_argument("--radius-factor", type=float, default=1.6)
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

    sample_level_frames: list[pd.DataFrame] = []
    differential_frames: list[pd.DataFrame] = []
    null_frames: list[pd.DataFrame] = []
    dataset_artifacts: dict[str, DatasetArtifacts] = {}
    dataset_summaries: list[dict[str, object]] = []

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
        full_result = fit_tissue_motif_model(
            dataset,
            neighborhood_summary,
            runtime_info=runtime_info,
            n_expression_programs=args.n_expression_programs,
            top_variable_genes=args.top_variable_genes,
            feature_bundle=feature_bundle,
            representation_method="baseline_pca",
            random_state=args.random_state,
        )
        full_spot_table = full_result.spot_table.reset_index(drop=True).copy()
        core_adjacency = neighborhood_summary.scales[neighborhood_summary.core_scale_name].adjacency.tocsr()

        holdout_result = assign_out_of_fold_motifs(
            dataset=dataset,
            full_result=full_result,
            random_state=args.random_state,
        )
        oof_spot_table = holdout_result.spot_table.reset_index(drop=True).copy()

        full_sample_level = build_sample_level_motif_table(
            dataset=dataset,
            spot_table=full_spot_table,
            feature_frame=full_result.feature_frame,
            expression_program_metadata=full_result.expression_program_metadata,
            neighborhood_summary=neighborhood_summary,
            analysis_scope="full_model",
        )
        oof_sample_level = build_sample_level_motif_table(
            dataset=dataset,
            spot_table=oof_spot_table,
            feature_frame=full_result.feature_frame,
            expression_program_metadata=full_result.expression_program_metadata,
            neighborhood_summary=neighborhood_summary,
            analysis_scope="holdout_oof",
        )
        sample_level_table = pd.concat([full_sample_level, oof_sample_level], ignore_index=True)
        sample_level_frames.append(sample_level_table)

        full_stats = compute_differential_statistics(
            spot_table=full_spot_table,
            sample_level_table=full_sample_level,
            adjacency=core_adjacency,
            random_state=args.random_state,
            null_iterations=args.null_iterations,
            null_scope_label="full_model",
        )
        oof_stats = compute_differential_statistics(
            spot_table=oof_spot_table,
            sample_level_table=oof_sample_level,
            adjacency=core_adjacency,
            random_state=args.random_state,
            null_iterations=args.null_iterations,
            null_scope_label="holdout_oof",
        )

        merged_differential = merge_scope_summaries(full_stats.summary, oof_stats.summary)
        differential_frames.append(merged_differential)

        null_frames.extend(
            [
                full_stats.null_controls,
                oof_stats.null_controls,
                label_control_rows(full_stats.summary, analysis_scope="full_model"),
                label_control_rows(oof_stats.summary, analysis_scope="holdout_oof"),
            ]
        )

        dataset_artifacts[dataset.dataset_id] = DatasetArtifacts(
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.dataset_name,
            dataset=dataset,
            neighborhood_summary=neighborhood_summary,
            full_result=full_result,
            oof_spot_table=oof_spot_table,
            sample_level_table=sample_level_table,
            differential_table=merged_differential,
            fold_summary=holdout_result.fold_summary,
        )
        dataset_summaries.append(
            {
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.dataset_name,
                "n_spots": int(dataset.obs.shape[0]),
                "n_samples": int(dataset.obs["sample_id"].nunique()),
                "condition_counts": dataset.obs.loc[:, ["sample_id", "condition"]].drop_duplicates()["condition"].astype(str).value_counts().to_dict(),
                "n_motifs": int(full_result.n_clusters),
                "core_scale": neighborhood_summary.core_scale_name,
                "spatial_coherence_zscore": float(full_result.spatial_coherence_zscore),
                "mean_oof_assignment_agreement": float(holdout_result.fold_summary["assignment_agreement_vs_full"].mean()),
            }
        )

    sample_level_table = pd.concat(sample_level_frames, ignore_index=True) if sample_level_frames else pd.DataFrame()
    differential_table = pd.concat(differential_frames, ignore_index=True) if differential_frames else pd.DataFrame()
    null_control_table = pd.concat([frame for frame in null_frames if not frame.empty], ignore_index=True) if null_frames else pd.DataFrame()

    sample_level_table.to_csv(results_dir / "sample_level_motif_table.csv", index=False)
    differential_table.to_csv(results_dir / "differential_test_results.csv", index=False)
    null_control_table.to_csv(results_dir / "null_control_results.csv", index=False)

    primary_row = choose_primary_motif(differential_table)
    primary_artifact = dataset_artifacts[str(primary_row["dataset_id"])]
    case_summary = build_biological_case_summary(primary_artifact, primary_row)

    plot_sample_level_effect_sizes(
        differential_table=differential_table,
        dataset_id=str(primary_row["dataset_id"]),
        output_path=figures_dir / "sample_level_effect_sizes.png",
    )
    plot_naive_vs_controlled_pvalues(
        differential_table=differential_table,
        output_path=figures_dir / "naive_vs_controlled_pvalues.png",
    )
    plot_leave_one_sample_out_reproducibility(
        differential_table=differential_table,
        output_path=figures_dir / "leave_one_sample_out_reproducibility.png",
    )
    plot_biological_case_motif_map(
        artifact=primary_artifact,
        primary_row=primary_row,
        case_summary=case_summary,
        output_path=figures_dir / "biological_case_motif_map.png",
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
        dataset_summaries=dataset_summaries,
        differential_table=differential_table,
        null_control_table=null_control_table,
        primary_row=primary_row,
        case_summary=case_summary,
    )

    payload = {
        "primary_dataset_id": str(primary_row["dataset_id"]),
        "primary_motif_id": str(primary_row["motif_id"]),
        "primary_support_tier": str(primary_row.get("holdout_controlled_support_tier", "unknown")),
    }
    print(json.dumps(payload, ensure_ascii=False))


def merge_scope_summaries(full_summary: pd.DataFrame, holdout_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_id", "dataset_name", "comparison_name", "condition_a", "condition_b", "motif_id", "motif_label"]
    full_prefixed = prefix_frame(full_summary, prefix="full_", key_columns=keys)
    holdout_prefixed = prefix_frame(holdout_summary, prefix="holdout_", key_columns=keys)
    merged = full_prefixed.merge(holdout_prefixed, on=keys, how="outer")
    if merged.empty:
        return merged
    merged["primary_rank_score"] = build_rank_score(merged)
    merged = merged.sort_values(
        ["dataset_id", "primary_rank_score", "motif_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return merged


def prefix_frame(frame: pd.DataFrame, *, prefix: str, key_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rename_map = {
        column: f"{prefix}{column}"
        for column in frame.columns
        if column not in key_columns
    }
    return frame.rename(columns=rename_map)


def build_rank_score(frame: pd.DataFrame) -> np.ndarray:
    tier_rank = frame["holdout_controlled_support_tier"].map(
        {
            "heldout_supported": 0.0,
            "sample_level_supported": 1.0,
            "not_supported": 2.0,
            "naive_only": 3.0,
        }
    ).fillna(4.0)
    holdout_p = frame["holdout_sample_permutation_pvalue_two_sided"].fillna(1.0).to_numpy(dtype=np.float64)
    holdout_max_t = frame["holdout_label_max_t_pvalue"].fillna(1.0).to_numpy(dtype=np.float64)
    holdout_null = frame["holdout_synthetic_null_effect_pvalue"].fillna(1.0).to_numpy(dtype=np.float64)
    loso = 1.0 - frame["holdout_loso_sign_consistency"].fillna(0.0).to_numpy(dtype=np.float64)
    effect_penalty = -np.abs(frame["holdout_delta_fraction"].fillna(0.0).to_numpy(dtype=np.float64))
    return tier_rank.to_numpy(dtype=np.float64) + holdout_p + holdout_max_t + holdout_null + loso + effect_penalty


def label_control_rows(summary: pd.DataFrame, *, analysis_scope: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = summary.loc[
        :,
        ["dataset_id", "dataset_name", "motif_id", "motif_label", "delta_fraction", "label_max_t_pvalue"],
    ].copy()
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


def choose_primary_motif(differential_table: pd.DataFrame) -> pd.Series:
    if differential_table.empty:
        raise ValueError("Differential table is empty.")
    ranked = differential_table.copy()
    ranked["holdout_sample_permutation_pvalue_two_sided"] = ranked["holdout_sample_permutation_pvalue_two_sided"].fillna(1.0)
    ranked["holdout_label_max_t_pvalue"] = ranked["holdout_label_max_t_pvalue"].fillna(1.0)
    ranked["holdout_synthetic_null_effect_pvalue"] = ranked["holdout_synthetic_null_effect_pvalue"].fillna(1.0)
    ranked["holdout_loso_sign_consistency"] = ranked["holdout_loso_sign_consistency"].fillna(0.0)
    tier_order = {
        "heldout_supported": 0,
        "sample_level_supported": 1,
        "not_supported": 2,
        "naive_only": 3,
    }
    ranked["tier_rank"] = ranked["holdout_controlled_support_tier"].map(tier_order).fillna(4)
    ranked = ranked.sort_values(
        [
            "tier_rank",
            "holdout_sample_permutation_pvalue_two_sided",
            "holdout_label_max_t_pvalue",
            "holdout_synthetic_null_effect_pvalue",
            "holdout_loso_sign_consistency",
            "primary_rank_score",
        ],
        ascending=[True, True, True, True, False, True],
    )
    return ranked.iloc[0]


def build_biological_case_summary(artifact: DatasetArtifacts, primary_row: pd.Series) -> dict[str, object]:
    motif_id = str(primary_row["motif_id"])
    condition_a = str(primary_row["condition_a"])
    condition_b = str(primary_row["condition_b"])
    effect = float(primary_row["holdout_delta_fraction"])
    enriched_condition = condition_b if effect >= 0.0 else condition_a
    depleted_condition = condition_a if effect >= 0.0 else condition_b
    scope_table = artifact.sample_level_table.loc[
        artifact.sample_level_table["analysis_scope"].astype(str) == "holdout_oof"
    ].copy()
    motif_table = scope_table.loc[scope_table["motif_id"].astype(str) == motif_id].copy()
    motif_table = motif_table.sort_values("motif_fraction", ascending=False)
    enriched_sample = motif_table.loc[motif_table["condition"].astype(str) == enriched_condition].iloc[0]
    reference_sample = motif_table.loc[motif_table["condition"].astype(str) == depleted_condition].iloc[0]

    marker_df = compute_marker_genes(
        dataset=artifact.dataset,
        spot_table=artifact.oof_spot_table,
        motif_id=motif_id,
        top_n=12,
    )
    enrichment_df = compute_curated_pathway_enrichment(
        marker_df["gene"].astype(str).tolist(),
        artifact.dataset.var_names.astype(str).tolist(),
        dataset_id=artifact.dataset_id,
    )
    cell_type_summary = (
        artifact.oof_spot_table.loc[artifact.oof_spot_table["motif_id"].astype(str) == motif_id, "cell_type"]
        .astype(str)
        .value_counts(normalize=True)
        .head(5)
    )
    program_summary = summarize_motif_programs(
        scope_table=scope_table,
        motif_id=motif_id,
        program_metadata=artifact.full_result.expression_program_metadata,
    )
    spatial_summary = summarize_spatial_context(
        scope_table=scope_table,
        motif_id=motif_id,
    )
    return {
        "motif_id": motif_id,
        "motif_label": str(primary_row["motif_label"]),
        "dataset_id": artifact.dataset_id,
        "dataset_name": artifact.dataset_name,
        "enriched_condition": enriched_condition,
        "reference_condition": depleted_condition,
        "enriched_sample_id": str(enriched_sample["sample_id"]),
        "reference_sample_id": str(reference_sample["sample_id"]),
        "enriched_sample_fraction": float(enriched_sample["motif_fraction"]),
        "reference_sample_fraction": float(reference_sample["motif_fraction"]),
        "marker_table": marker_df,
        "enrichment_table": enrichment_df,
        "cell_type_summary": cell_type_summary,
        "program_summary": program_summary,
        "spatial_summary": spatial_summary,
    }


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
    valid = (
        (mean_in >= 0.05)
        & (frac_in >= 0.10)
        & (delta_frac >= 0.02)
        & is_interpretable
    )
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


def summarize_motif_programs(
    *,
    scope_table: pd.DataFrame,
    motif_id: str,
    program_metadata: pd.DataFrame,
) -> pd.DataFrame:
    motif_scope = scope_table.loc[
        (scope_table["motif_id"].astype(str) == str(motif_id))
        & (scope_table["motif_spots"].fillna(0).to_numpy(dtype=np.float64) > 0),
    ].copy()
    program_cols = [column for column in motif_scope.columns if column.startswith("program_") and column.endswith("_mean")]
    if motif_scope.empty or not program_cols:
        return pd.DataFrame(columns=["program", "mean_score", "top_genes"])
    program_means = motif_scope.loc[:, program_cols].mean(axis=0).sort_values(ascending=False)
    program_lookup = (
        program_metadata.set_index("component")["top_genes"].astype(str).to_dict()
        if not program_metadata.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    for column, value in program_means.head(3).items():
        component = column.replace("_mean", "")
        rows.append(
            {
                "program": component,
                "mean_score": float(value),
                "top_genes": str(program_lookup.get(component, "")),
            }
        )
    return pd.DataFrame(rows)


def summarize_spatial_context(
    *,
    scope_table: pd.DataFrame,
    motif_id: str,
) -> dict[str, object]:
    motif_scope = scope_table.loc[
        (scope_table["motif_id"].astype(str) == str(motif_id))
        & (scope_table["motif_spots"].fillna(0).to_numpy(dtype=np.float64) > 0),
    ].copy()
    if motif_scope.empty:
        return {}
    weights = motif_scope["motif_spots"].to_numpy(dtype=np.float64)
    same_neighbor = weighted_mean(motif_scope["same_motif_neighbor_fraction"].to_numpy(dtype=np.float64), weights)
    top_neighbor = (
        motif_scope.groupby("top_neighbor_motif", observed=False)["motif_spots"]
        .sum()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    return {
        "same_motif_neighbor_fraction": same_neighbor,
        "top_neighbor_motif": top_neighbor[0] if top_neighbor else "",
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite):
        return float("nan")
    return float(np.average(values[finite], weights=weights[finite]))


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


def plot_sample_level_effect_sizes(
    *,
    differential_table: pd.DataFrame,
    dataset_id: str,
    output_path: Path,
    top_n: int = 6,
) -> None:
    _set_publication_style()
    frame = differential_table.loc[differential_table["dataset_id"].astype(str) == str(dataset_id)].copy()
    if frame.empty:
        return
    frame = frame.sort_values(
        [
            "holdout_sample_permutation_pvalue_two_sided",
            "holdout_label_max_t_pvalue",
            "holdout_synthetic_null_effect_pvalue",
            "holdout_loso_sign_consistency",
        ],
        ascending=[True, True, True, False],
    ).head(top_n)
    labels = [f"{row['motif_id']} ({row['motif_label']})" for _, row in frame.iterrows()]
    y = np.arange(frame.shape[0], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.4, 4.8 + 0.4 * frame.shape[0]))
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax.errorbar(
        frame["full_delta_fraction"].to_numpy(dtype=np.float64),
        y - 0.12,
        xerr=np.vstack(
            [
                np.abs(frame["full_delta_fraction"].to_numpy(dtype=np.float64) - frame["full_bootstrap_ci_low"].to_numpy(dtype=np.float64)),
                np.abs(frame["full_bootstrap_ci_high"].to_numpy(dtype=np.float64) - frame["full_delta_fraction"].to_numpy(dtype=np.float64)),
            ]
        ),
        fmt="o",
        color="#7a8892",
        ecolor="#7a8892",
        capsize=3,
        label="Full model",
    )
    ax.errorbar(
        frame["holdout_delta_fraction"].to_numpy(dtype=np.float64),
        y + 0.12,
        xerr=np.vstack(
            [
                np.abs(frame["holdout_delta_fraction"].to_numpy(dtype=np.float64) - frame["holdout_bootstrap_ci_low"].to_numpy(dtype=np.float64)),
                np.abs(frame["holdout_bootstrap_ci_high"].to_numpy(dtype=np.float64) - frame["holdout_delta_fraction"].to_numpy(dtype=np.float64)),
            ]
        ),
        fmt="o",
        color="#b84a3a",
        ecolor="#b84a3a",
        capsize=3,
        label="Hold-out OOF",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Condition effect on sample-level motif fraction")
    ax.set_title(f"{dataset_id}: full-model vs held-out effect sizes")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_naive_vs_controlled_pvalues(
    *,
    differential_table: pd.DataFrame,
    output_path: Path,
) -> None:
    _set_publication_style()
    if differential_table.empty:
        return
    frame = differential_table.copy()
    frame["x"] = -np.log10(np.clip(frame["full_naive_spot_pvalue"].fillna(1.0).to_numpy(dtype=np.float64), 1.0e-12, 1.0))
    frame["y"] = -np.log10(np.clip(frame["holdout_sample_permutation_pvalue_two_sided"].fillna(1.0).to_numpy(dtype=np.float64), 1.0e-12, 1.0))
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    colors = {"wu2021_breast_visium": "#c26b3b", "gse220442_ad_mtg": "#3a78a1"}
    for _, row in frame.iterrows():
        ax.scatter(
            float(row["x"]),
            float(row["y"]),
            s=72,
            color=colors.get(str(row["dataset_id"]), "#66788a"),
            alpha=0.88,
            edgecolors="#ffffff",
            linewidths=0.8,
        )
    limit = float(max(frame["x"].max(), frame["y"].max()) + 0.5)
    ax.plot([0.0, limit], [0.0, limit], linestyle="--", color="#444444", linewidth=1.0)
    discrepant = frame.sort_values("holdout_naive_minus_controlled_log10p", ascending=False).head(5)
    for _, row in discrepant.iterrows():
        ax.text(float(row["x"]) + 0.05, float(row["y"]) + 0.03, f"{row['dataset_id']}:{row['motif_id']}", fontsize=7.4)
    ax.set_xlabel("-log10 naive spot-level p-value")
    ax.set_ylabel("-log10 controlled hold-out sample-level p-value")
    ax.set_title("Pseudoreplication gap: naive vs controlled")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_leave_one_sample_out_reproducibility(
    *,
    differential_table: pd.DataFrame,
    output_path: Path,
) -> None:
    _set_publication_style()
    if differential_table.empty:
        return
    frame = differential_table.copy()
    x = frame["holdout_delta_fraction"].fillna(frame["full_delta_fraction"]).to_numpy(dtype=np.float64)
    y = frame["holdout_loso_delta_median"].fillna(frame["full_delta_fraction"]).to_numpy(dtype=np.float64)
    color_value = frame["holdout_loso_sign_consistency"].fillna(0.0).to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
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
    limit = float(max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y))) * 1.12)
    ax.plot([-limit, limit], [-limit, limit], linestyle="--", color="#444444", linewidth=1.0)
    for _, row in frame.sort_values("holdout_sample_permutation_pvalue_two_sided", ascending=True).head(5).iterrows():
        ax.text(
            float(row["holdout_delta_fraction"]) + 0.01,
            float(row["holdout_loso_delta_median"]) + 0.01,
            f"{row['dataset_id']}:{row['motif_id']}",
            fontsize=7.2,
        )
    correlation = pearson_correlation(x, y)
    ax.set_xlabel("Hold-out overall effect")
    ax.set_ylabel("LOSO median effect")
    ax.set_title(f"Leave-one-sample-out reproducibility (r={correlation:.2f})")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("LOSO sign consistency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_biological_case_motif_map(
    *,
    artifact: DatasetArtifacts,
    primary_row: pd.Series,
    case_summary: dict[str, object],
    output_path: Path,
) -> None:
    _set_publication_style()
    motif_id = str(case_summary["motif_id"])
    enriched_sample = str(case_summary["enriched_sample_id"])
    reference_sample = str(case_summary["reference_sample_id"])
    sample_order = [enriched_sample, reference_sample]
    titles = [
        f"{case_summary['enriched_condition']} sample {enriched_sample}",
        f"{case_summary['reference_condition']} sample {reference_sample}",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
    for ax, sample_id, title in zip(axes, sample_order, titles, strict=True):
        frame = artifact.oof_spot_table.loc[artifact.oof_spot_table["sample_id"].astype(str) == sample_id].copy()
        mask = frame["motif_id"].astype(str) == motif_id
        ax.scatter(
            frame.loc[~mask, "spatial_x"].to_numpy(dtype=np.float32),
            -frame.loc[~mask, "spatial_y"].to_numpy(dtype=np.float32),
            c="#d8dadc",
            s=6,
            alpha=0.55,
            linewidths=0,
        )
        ax.scatter(
            frame.loc[mask, "spatial_x"].to_numpy(dtype=np.float32),
            -frame.loc[mask, "spatial_y"].to_numpy(dtype=np.float32),
            c="#c44232",
            s=10,
            alpha=0.95,
            linewidths=0,
        )
        fraction = artifact.sample_level_table.loc[
            (artifact.sample_level_table["analysis_scope"].astype(str) == "holdout_oof")
            & (artifact.sample_level_table["sample_id"].astype(str) == sample_id)
            & (artifact.sample_level_table["motif_id"].astype(str) == motif_id),
            "motif_fraction",
        ].iloc[0]
        ax.set_title(f"{title}\n{motif_id} fraction={float(fraction):.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(f"{artifact.dataset_id}: {primary_row['motif_id']} held-out spatial localization", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


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
        "Run Week 6 cross-sample differential analysis on fixed tissue motifs, compare naive spot-level testing against controlled sample-aware testing, and carry the strongest motif into biological interpretation.",
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
            "",
            "## Steps",
            "",
            "1. Rebuild the Week 4 baseline motif embedding from multi-scale neighborhoods and expression programs.",
            "2. Aggregate motif abundance, mean expression, dominant programs, cell-type composition, and spatial co-localization at the sample level.",
            "3. Run exact sample-level permutation, one-sided directional permutation, maxT condition-label permutation, clustered spot mixed-effects, and naive spot-level Fisher testing.",
            "4. Freeze motif labels on training samples only, assign held-out samples out-of-fold, and rerun the differential analysis on those held-out assignments.",
            "5. Stress-test each motif with leave-one-sample-out perturbations and size-matched random null motifs.",
            "6. Interpret the strongest held-out-supported motif with marker genes, curated pathway overlap, cell-type composition, and condition-specific spatial examples.",
            "",
            "## Outputs",
            "",
            "- `results/sample_level_motif_table.csv`",
            "- `results/differential_test_results.csv`",
            "- `results/null_control_results.csv`",
            "- `figures/sample_level_effect_sizes.png`",
            "- `figures/naive_vs_controlled_pvalues.png`",
            "- `figures/leave_one_sample_out_reproducibility.png`",
            "- `figures/biological_case_motif_map.png`",
            "",
            "## Run Command",
            "",
            f"- `python scripts/run_cross_sample_differential.py --output-dir {output_dir}`",
            "",
            "## Dataset snapshot",
            "",
        ]
    )
    for row in dataset_summaries:
        condition_counts = ", ".join(f"{key}={value}" for key, value in row["condition_counts"].items())
        lines.append(
            f"- `{row['dataset_id']}`: `{row['n_samples']}` samples ({condition_counts}), `{row['n_spots']}` spots, "
            f"`{row['n_motifs']}` motifs, core scale=`{row['core_scale']}`, coherence_z=`{row['spatial_coherence_zscore']:.2f}`, "
            f"mean OOF agreement=`{row['mean_oof_assignment_agreement']:.3f}`."
        )
    lines.extend(
        [
            "",
            "## Statistical note",
            "",
            f"- Size-matched random null motifs were drawn `{int(args.null_iterations)}` times per motif and per analysis scope (`full_model`, `holdout_oof`).",
        ]
    )
    (output_dir / "protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(
    *,
    output_dir: Path,
    dataset_summaries: list[dict[str, object]],
    differential_table: pd.DataFrame,
    null_control_table: pd.DataFrame,
    primary_row: pd.Series,
    case_summary: dict[str, object],
) -> None:
    lines = [
        "# Analysis",
        "",
        "## Headline",
        "",
    ]
    lines.append(
        f"- Primary held-out motif: `{primary_row['dataset_id']}:{primary_row['motif_id']}` (`{primary_row['motif_label']}`), "
        f"hold-out delta fraction `{float(primary_row['holdout_delta_fraction']):.3f}`, two-sided sample permutation p `{float(primary_row['holdout_sample_permutation_pvalue_two_sided']):.3f}`, "
        f"one-sided p `{float(primary_row['holdout_sample_permutation_pvalue_one_sided']):.3f}`, maxT p `{float(primary_row['holdout_label_max_t_pvalue']):.3f}`, "
        f"null motif p `{float(primary_row['holdout_synthetic_null_effect_pvalue']):.3f}`, support tier `{primary_row['holdout_controlled_support_tier']}`."
    )
    lines.append(format_sample_permutation_note(primary_row))
    lines.append(
        f"- Naive spot-level testing for the same motif returned p `{float(primary_row['full_naive_spot_pvalue']):.3g}`, showing the expected pseudoreplication gap relative to the controlled hold-out test."
    )

    lines.extend(
        [
            "",
            "## Statistical validity",
            "",
        ]
    )
    for row in dataset_summaries:
        subset = differential_table.loc[differential_table["dataset_id"].astype(str) == str(row["dataset_id"])].copy()
        if subset.empty:
            continue
        near_supported = int(np.sum(subset["holdout_controlled_support_tier"].astype(str).isin(["heldout_supported", "sample_level_supported"])))
        naive_only = int(np.sum(subset["holdout_controlled_support_tier"].astype(str) == "naive_only"))
        median_gap = float(np.nanmedian(subset["holdout_naive_minus_controlled_log10p"].to_numpy(dtype=np.float64)))
        median_loso = float(np.nanmedian(subset["holdout_loso_sign_consistency"].to_numpy(dtype=np.float64)))
        lines.append(
            f"- `{row['dataset_id']}`: `{near_supported}` motifs retained sample-level support after hold-out controls, `{naive_only}` motifs looked significant only under naive spot-level testing, "
            f"median naive-minus-controlled gap=`{median_gap:.2f}` log10 units, median LOSO sign consistency=`{median_loso:.2f}`."
        )

    strongest_gaps = differential_table.sort_values("holdout_naive_minus_controlled_log10p", ascending=False).head(3)
    for _, row in strongest_gaps.iterrows():
        lines.append(
            f"- Largest naive/control discrepancy: `{row['dataset_id']}:{row['motif_id']}` naive p=`{float(row['full_naive_spot_pvalue']):.3g}` vs hold-out sample p=`{float(row['holdout_sample_permutation_pvalue_two_sided']):.3f}`."
        )

    lines.extend(
        [
            "",
            "## Biological case",
            "",
        ]
    )
    lines.append(
        f"- Enriched condition: `{case_summary['enriched_condition']}`. Top example sample: `{case_summary['enriched_sample_id']}` with motif fraction `{float(case_summary['enriched_sample_fraction']):.3f}`; "
        f"reference sample: `{case_summary['reference_sample_id']}` with motif fraction `{float(case_summary['reference_sample_fraction']):.3f}`."
    )
    marker_table = case_summary["marker_table"]
    if not marker_table.empty:
        marker_text = ", ".join(marker_table["gene"].astype(str).head(8).tolist())
        lines.append(f"- Marker genes: `{marker_text}`.")
    enrichment_table = case_summary["enrichment_table"]
    if not enrichment_table.empty:
        top_pathway = enrichment_table.iloc[0]
        lines.append(
            f"- Curated pathway overlap: `{top_pathway['pathway']}` with genes `{top_pathway['overlap_genes']}` and Fisher p `{float(top_pathway['pvalue']):.3g}`."
        )
    cell_type_summary = case_summary["cell_type_summary"]
    if not cell_type_summary.empty:
        composition_text = ", ".join([f"{idx}={value:.2f}" for idx, value in cell_type_summary.head(5).items()])
        lines.append(f"- Cell-type composition inside the motif: `{composition_text}`.")
    program_summary = case_summary["program_summary"]
    if not program_summary.empty:
        program_text = "; ".join(
            [f"{row['program']} ({row['top_genes']})" for _, row in program_summary.head(2).iterrows()]
        )
        lines.append(f"- Dominant local gene programs: `{program_text}`.")
    spatial_summary = case_summary["spatial_summary"]
    if spatial_summary:
        lines.append(
            f"- Spatial localization: same-motif neighbor fraction `{float(spatial_summary['same_motif_neighbor_fraction']):.3f}`, top neighboring motif `{spatial_summary['top_neighbor_motif']}`."
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The datasets are sample-limited, so exact sample-level inference is resolution-limited even when the biological effect size is large.",
            "- Hold-out assignment freezes clustering labels but still reuses the unsupervised feature engineering stage, so this is a double-dipping control rather than a full external validation dataset.",
            "- Pathway interpretation uses small curated gene sets bundled in the script rather than a full MSigDB-scale enrichment database.",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_sample_permutation_note(primary_row: pd.Series) -> str:
    mode = str(primary_row.get("holdout_sample_permutation_mode", ""))
    n_a = int(primary_row.get("holdout_n_samples_a", 0))
    n_b = int(primary_row.get("holdout_n_samples_b", 0))
    total_labelings = pd.to_numeric(pd.Series([primary_row.get("holdout_sample_permutation_total_labelings", np.nan)]), errors="coerce").iloc[0]
    n_permutations = pd.to_numeric(pd.Series([primary_row.get("holdout_sample_permutation_n_permutations", np.nan)]), errors="coerce").iloc[0]
    if mode == "exact":
        if np.isfinite(total_labelings) and float(total_labelings) > 0:
            floor = 1.0 / float(total_labelings)
            return (
                f"- The controlled sample-level permutation operated on `{n_a}` vs `{n_b}` sample labels with exact enumeration "
                f"across `{int(total_labelings)}` labelings, so p-values are discrete and the nominal floor is `{floor:.3g}`."
            )
        return (
            f"- The controlled sample-level permutation operated on `{n_a}` vs `{n_b}` sample labels with exact enumeration, "
            "so p-values are discrete at the sample-label resolution."
        )
    if mode == "approx_monte_carlo":
        if np.isfinite(total_labelings) and np.isfinite(n_permutations):
            return (
                f"- Exact sample-level label enumeration is infeasible for this `{n_a}` vs `{n_b}` split "
                f"(`{int(total_labelings)}` possible labelings), so the controlled test used Monte Carlo permutation with `{int(n_permutations)}` sampled labelings."
            )
        return (
            f"- Exact sample-level label enumeration is infeasible for this `{n_a}` vs `{n_b}` split, "
            "so the controlled test used Monte Carlo permutation."
        )
    return (
        f"- The controlled sample-level test operated on `{n_a}` vs `{n_b}` sample labels rather than treating spots as independent observations."
    )


if __name__ == "__main__":
    main()
