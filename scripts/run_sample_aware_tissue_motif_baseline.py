from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spatial_context.differential_motif import compute_sample_motif_abundance, differential_motif_analysis  # noqa: E402
from spatial_context.motif_embedding import (  # noqa: E402
    SampleAwareMotifFitResult,
    align_sample_aware_motif_model_to_reference,
    assign_sample_aware_motifs,
    build_tissue_motif_feature_bundle,
    fit_sample_aware_tissue_motif_model,
)
from spatial_context.neighborhood import get_runtime_info, load_spatial_h5ad, summarize_neighborhoods  # noqa: E402
from spatial_context.visualization import (  # noqa: E402
    plot_condition_abundance,
    plot_differential_volcano,
    plot_motif_layout,
    plot_motif_spatial_map,
)


DEFAULT_DATASETS = (
    ROOT / "data" / "spatial_processed" / "wu2021_breast_visium.h5ad",
    ROOT / "data" / "spatial_processed" / "gse220442_ad_mtg.h5ad",
)
EXPERIMENT_DIR = ROOT / "experiments" / "04b_sample_aware_tissue_motif_baseline"


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset_id: str
    dataset_name: str
    full_fit: SampleAwareMotifFitResult
    full_abundance: pd.DataFrame
    full_differential: pd.DataFrame
    loso_spot_table: pd.DataFrame
    loso_abundance: pd.DataFrame
    loso_differential: pd.DataFrame
    loso_fold_summary: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 4b sample-aware tissue motif baseline on processed spatial H5AD files.")
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
    parser.add_argument("--max-train-spots-per-sample", type=int, default=2048)
    parser.add_argument("--random-state", type=int, default=7)
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

    full_abundance_frames: list[pd.DataFrame] = []
    full_differential_frames: list[pd.DataFrame] = []
    loso_abundance_frames: list[pd.DataFrame] = []
    loso_differential_frames: list[pd.DataFrame] = []
    fold_summary_frames: list[pd.DataFrame] = []
    normalization_frames: list[pd.DataFrame] = []
    dataset_summaries: list[dict[str, object]] = []
    dataset_artifacts: dict[str, DatasetArtifacts] = {}

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
        full_result = full_fit.embedding_result
        full_abundance = compute_sample_motif_abundance(full_result.spot_table)
        full_differential = differential_motif_analysis(
            full_result.spot_table,
            full_abundance,
            random_state=args.random_state,
        )
        full_abundance_frames.append(full_abundance)
        full_differential_frames.append(full_differential)

        full_norm = full_fit.normalization_summary.copy()
        full_norm["dataset_id"] = dataset.dataset_id
        full_norm["dataset_name"] = dataset.dataset_name
        full_norm["scope"] = "full_model"
        full_norm["held_out_sample"] = ""
        normalization_frames.append(full_norm)

        loso_spot_table, loso_fold_summary, loso_norm = build_loso_assignments(
            dataset=dataset,
            neighborhood_summary=neighborhood_summary,
            feature_bundle=feature_bundle,
            reference_fit=full_fit,
            runtime_info=runtime_info,
            args=args,
        )
        loso_abundance = compute_sample_motif_abundance(loso_spot_table)
        loso_differential = differential_motif_analysis(
            loso_spot_table,
            loso_abundance,
            random_state=args.random_state,
        )
        loso_abundance_frames.append(loso_abundance)
        loso_differential_frames.append(loso_differential)
        fold_summary_frames.append(loso_fold_summary)
        normalization_frames.append(loso_norm)

        dataset_artifacts[dataset.dataset_id] = DatasetArtifacts(
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.dataset_name,
            full_fit=full_fit,
            full_abundance=full_abundance,
            full_differential=full_differential,
            loso_spot_table=loso_spot_table,
            loso_abundance=loso_abundance,
            loso_differential=loso_differential,
            loso_fold_summary=loso_fold_summary,
        )
        dataset_summaries.append(
            {
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.dataset_name,
                "n_spots": int(dataset.obs.shape[0]),
                "n_genes": int(dataset.expression.shape[1]),
                "n_samples": int(dataset.obs["sample_id"].nunique()),
                "conditions": "|".join(sorted(dataset.obs["condition"].astype(str).unique().tolist())),
                "n_motifs": int(full_result.n_clusters),
                "layout_method": full_result.layout_method,
                "spatial_coherence_observed": float(full_result.spatial_coherence_observed),
                "spatial_coherence_perm_mean": float(full_result.spatial_coherence_perm_mean),
                "spatial_coherence_zscore": float(full_result.spatial_coherence_zscore),
                "training_spots_per_sample": int(full_fit.frozen_model.training_spots_per_sample),
                "library_normalization_mode": str(full_fit.frozen_model.library_normalization_mode),
                "normalization_mode": str(full_fit.frozen_model.normalization_mode),
                "pca_backend": str(full_fit.frozen_model.pca_projection.backend),
                "loso_mean_assignment_agreement": float(loso_fold_summary["assignment_agreement_vs_full"].mean()),
                "loso_mean_alignment_cost": float(loso_fold_summary["alignment_cost_mean"].mean()),
                "loso_mean_assignment_distance": float(loso_fold_summary["heldout_assignment_distance_mean"].mean()),
            }
        )

    full_abundance_table = pd.concat(full_abundance_frames, ignore_index=True) if full_abundance_frames else pd.DataFrame()
    full_differential_table = pd.concat(full_differential_frames, ignore_index=True) if full_differential_frames else pd.DataFrame()
    loso_abundance_table = pd.concat(loso_abundance_frames, ignore_index=True) if loso_abundance_frames else pd.DataFrame()
    loso_differential_table = pd.concat(loso_differential_frames, ignore_index=True) if loso_differential_frames else pd.DataFrame()
    fold_summary_table = pd.concat(fold_summary_frames, ignore_index=True) if fold_summary_frames else pd.DataFrame()
    normalization_table = pd.concat(normalization_frames, ignore_index=True) if normalization_frames else pd.DataFrame()

    full_abundance_table.to_csv(results_dir / "motif_abundance_table.csv", index=False)
    full_differential_table.to_csv(results_dir / "differential_motif_results.csv", index=False)
    loso_abundance_table.to_csv(results_dir / "loso_motif_abundance_table.csv", index=False)
    loso_differential_table.to_csv(results_dir / "loso_differential_motif_results.csv", index=False)
    fold_summary_table.to_csv(results_dir / "loso_fold_summary.csv", index=False)
    normalization_table.to_csv(results_dir / "normalization_summary.csv", index=False)
    pd.DataFrame(dataset_summaries).to_csv(results_dir / "dataset_summary.csv", index=False)

    primary_dataset_id, top_row, analysis_scope = choose_primary_visualization_target(
        loso_differential_table=loso_differential_table,
        full_differential_table=full_differential_table,
        dataset_artifacts=dataset_artifacts,
    )
    primary_artifact = dataset_artifacts[primary_dataset_id]
    sample_id = choose_visualization_sample(
        primary_artifact.loso_abundance if analysis_scope == "loso" else primary_artifact.full_abundance,
        dataset_id=primary_dataset_id,
        motif_id=str(top_row["motif_id"]),
    )

    layout_title = f"{primary_artifact.dataset_name} sample-aware motif layout ({primary_artifact.full_fit.embedding_result.layout_method})"
    plot_motif_layout(
        primary_artifact.full_fit.embedding_result.spot_table,
        output_path=figures_dir / "motif_umap.png",
        title=layout_title,
    )
    plot_motif_spatial_map(
        primary_artifact.full_fit.embedding_result.spot_table,
        sample_id=sample_id,
        output_path=figures_dir / "motif_spatial_map_sample1.png",
        title=f"{primary_dataset_id} sample {sample_id} sample-aware motif map",
    )
    abundance_source = primary_artifact.loso_abundance if analysis_scope == "loso" else primary_artifact.full_abundance
    differential_source = primary_artifact.loso_differential if analysis_scope == "loso" else primary_artifact.full_differential
    plot_condition_abundance(
        abundance_source,
        differential_source,
        dataset_id=primary_dataset_id,
        condition_a=str(top_row["condition_a"]),
        condition_b=str(top_row["condition_b"]),
        output_path=figures_dir / "motif_condition_abundance.png",
    )
    plot_differential_volcano(
        differential_source,
        dataset_id=primary_dataset_id,
        output_path=figures_dir / "differential_motif_volcano.png",
        title=f"{primary_dataset_id} {analysis_scope} differential motifs: {top_row['condition_b']} vs {top_row['condition_a']}",
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
        full_differential_table=full_differential_table,
        loso_differential_table=loso_differential_table,
        primary_dataset_id=primary_dataset_id,
        top_row=top_row,
        analysis_scope=analysis_scope,
    )

    payload = {
        "primary_dataset_id": primary_dataset_id,
        "primary_motif": str(top_row["motif_id"]),
        "analysis_scope": analysis_scope,
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_ids = sorted(dataset.obs["sample_id"].astype(str).unique().tolist())
    spot_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    normalization_frames: list[pd.DataFrame] = []
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
        norm = fold_fit.normalization_summary.copy()
        norm["dataset_id"] = dataset.dataset_id
        norm["dataset_name"] = dataset.dataset_name
        norm["scope"] = "loso_train_only"
        norm["held_out_sample"] = held_out_sample
        normalization_frames.append(norm)
        spot_frames.append(heldout_table)

    loso_spot_table = pd.concat(spot_frames, axis=0).sort_index()
    loso_fold_summary = pd.DataFrame(fold_rows).sort_values(["dataset_id", "held_out_sample"]).reset_index(drop=True)
    loso_norm = pd.concat(normalization_frames, ignore_index=True) if normalization_frames else pd.DataFrame()
    return loso_spot_table, loso_fold_summary, loso_norm


def choose_primary_visualization_target(
    *,
    loso_differential_table: pd.DataFrame,
    full_differential_table: pd.DataFrame,
    dataset_artifacts: dict[str, DatasetArtifacts],
) -> tuple[str, pd.Series, str]:
    if not loso_differential_table.empty:
        ranked = loso_differential_table.copy()
        ranked["mixedlm_pvalue"] = ranked["mixedlm_pvalue"].fillna(1.0)
        ranked["evidence_rank"] = ranked["evidence_tier"].map({"strong": 0, "moderate": 1, "weak": 2, "none": 3}).fillna(4)
        top_row = ranked.sort_values(
            ["association_call", "evidence_rank", "mixedlm_pvalue", "delta_fraction"],
            ascending=[False, True, True, False],
        ).iloc[0]
        return str(top_row["dataset_id"]), top_row, "loso"
    if not full_differential_table.empty:
        ranked = full_differential_table.copy()
        ranked["mixedlm_pvalue"] = ranked["mixedlm_pvalue"].fillna(1.0)
        ranked["evidence_rank"] = ranked["evidence_tier"].map({"strong": 0, "moderate": 1, "weak": 2, "none": 3}).fillna(4)
        top_row = ranked.sort_values(
            ["association_call", "evidence_rank", "mixedlm_pvalue", "delta_fraction"],
            ascending=[False, True, True, False],
        ).iloc[0]
        return str(top_row["dataset_id"]), top_row, "full"
    first_id = next(iter(dataset_artifacts))
    artifact = dataset_artifacts[first_id]
    fallback = pd.Series(
        {
            "dataset_id": first_id,
            "motif_id": artifact.full_fit.embedding_result.motif_metadata["motif_id"].iloc[0],
            "condition_a": artifact.full_fit.embedding_result.spot_table["condition"].astype(str).min(),
            "condition_b": artifact.full_fit.embedding_result.spot_table["condition"].astype(str).max(),
        }
    )
    return first_id, fallback, "full"


def choose_visualization_sample(abundance_table: pd.DataFrame, *, dataset_id: str, motif_id: str) -> str:
    subset = abundance_table.loc[
        (abundance_table["dataset_id"] == dataset_id)
        & (abundance_table["motif_id"].astype(str) == str(motif_id))
    ].copy()
    if subset.empty:
        dataset_subset = abundance_table.loc[abundance_table["dataset_id"] == dataset_id]
        if dataset_subset.empty:
            return str(dataset_id)
        return str(dataset_subset["sample_id"].iloc[0])
    top_sample = subset.sort_values("motif_fraction", ascending=False).iloc[0]
    return str(top_sample["sample_id"])


def write_protocol(
    *,
    output_dir: Path,
    runtime_info,
    dataset_paths: list[Path],
    dataset_summaries: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    protocol_lines = [
        "# Protocol",
        "",
        "## Goal",
        "",
        "Rebuild Week 4 as a sample-aware tissue motif baseline that removes pooled-sample dominance from motif construction and evaluates condition effects with leave-one-sample-out motif assignment.",
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
    protocol_lines.extend([f"- `{path.relative_to(ROOT)}`" for path in dataset_paths])
    protocol_lines.extend(
        [
            "",
            "## Steps",
            "",
            "1. Load processed spatial `.h5ad` files and build multi-scale neighborhoods separately within each sample.",
            "2. Reuse the precomputed `lognorm` layer when available for library-normalized program scoring; otherwise apply on-the-fly log1p CP10K normalization before SVD.",
            "3. Construct pooled neighborhood composition, expression-program, density, and entropy features, then z-score every feature independently within each sample.",
            f"4. Draw a balanced training set with the same number of spots from each training sample (cap=`{int(args.max_train_spots_per_sample)}` spots per sample), fit scaler/PCA on those balanced spots, and learn motif centroids without using condition labels.",
            "5. Assign all spots to frozen centroids for the canonical full sample-aware motif map, then rerun motif training in leave-one-sample-out folds and assign each held-out sample to fold-specific frozen centroids aligned back to the canonical motif IDs.",
            "6. Aggregate sample-level motif abundance and run sample permutation, bootstrap, and clustered mixed-effect differential testing for both the canonical full model and the stricter LOSO assignments.",
            "",
            "## Outputs",
            "",
            "- `results/motif_abundance_table.csv`",
            "- `results/differential_motif_results.csv`",
            "- `results/loso_motif_abundance_table.csv`",
            "- `results/loso_differential_motif_results.csv`",
            "- `results/loso_fold_summary.csv`",
            "- `results/normalization_summary.csv`",
            "- `figures/motif_umap.png`",
            "- `figures/motif_spatial_map_sample1.png`",
            "- `figures/motif_condition_abundance.png`",
            "- `figures/differential_motif_volcano.png`",
            "",
            "## Dataset snapshot",
            "",
        ]
    )
    for row in dataset_summaries:
        protocol_lines.append(
            f"- `{row['dataset_id']}`: `{row['n_samples']}` samples, `{row['n_spots']}` spots, `{row['n_motifs']}` motifs, "
            f"balanced_train_spots_per_sample=`{row['training_spots_per_sample']}`, pca_backend=`{row['pca_backend']}`, "
            f"LOSO agreement=`{row['loso_mean_assignment_agreement']:.3f}`, coherence_z=`{row['spatial_coherence_zscore']:.2f}`"
        )
    (output_dir / "protocol.md").write_text("\n".join(protocol_lines) + "\n", encoding="utf-8")


def write_analysis(
    *,
    output_dir: Path,
    dataset_summaries: list[dict[str, object]],
    full_differential_table: pd.DataFrame,
    loso_differential_table: pd.DataFrame,
    primary_dataset_id: str,
    top_row: pd.Series,
    analysis_scope: str,
) -> None:
    summary_df = pd.DataFrame(dataset_summaries)
    lines = [
        "# Analysis",
        "",
        "## Motif quality",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['dataset_id']}` used balanced training with `{row['training_spots_per_sample']}` spots per sample, "
            f"full-model coherence z-score `{row['spatial_coherence_zscore']:.2f}`, and mean LOSO assignment agreement `{row['loso_mean_assignment_agreement']:.3f}`."
        )
    if loso_differential_table.empty and full_differential_table.empty:
        lines.extend(
            [
                "",
                "## Differential readout",
                "",
                "- No two-condition differential motif comparison was available in the current inputs.",
            ]
        )
    else:
        primary_table = loso_differential_table if analysis_scope == "loso" and not loso_differential_table.empty else full_differential_table
        top_dataset = primary_table.loc[primary_table["dataset_id"] == primary_dataset_id].copy()
        top_dataset = top_dataset.sort_values(["association_call", "mixedlm_pvalue", "permutation_pvalue"], ascending=[False, True, True])
        full_calls = 0 if full_differential_table.empty else int(
            full_differential_table.loc[full_differential_table["dataset_id"] == primary_dataset_id, "association_call"].sum()
        )
        loso_calls = 0 if loso_differential_table.empty else int(
            loso_differential_table.loc[loso_differential_table["dataset_id"] == primary_dataset_id, "association_call"].sum()
        )
        lines.extend(
            [
                "",
                "## Differential readout",
                "",
                f"- Primary visualization target: `{primary_dataset_id}` using `{analysis_scope}` differential results for `{top_row['condition_b']} vs {top_row['condition_a']}`.",
                f"- Top motif: `{top_row['motif_id']}` (`{top_row.get('motif_label', '')}`) with delta fraction `{float(top_row.get('delta_fraction', 0.0)):.3f}`, "
                f"log2FC `{float(top_row.get('log2_fold_change', 0.0)):.3f}`, sample permutation p `{float(top_row.get('permutation_pvalue', 1.0)):.3f}`, "
                f"mixed-effect p `{float(top_row.get('mixedlm_pvalue', 1.0)):.3g}`, evidence tier `{top_row.get('evidence_tier', 'none')}`.",
                f"- Condition-associated motifs in `{primary_dataset_id}`: full sample-aware model `{full_calls}`, LOSO out-of-fold model `{loso_calls}`, current figure scope `{analysis_scope}` counts `{int(top_dataset['association_call'].sum())}` / `{int(top_dataset.shape[0])}`.",
                "- The LOSO table is the stricter readout because every sample is assigned by centroids trained without that sample, while canonical motif IDs remain aligned through centroid matching in normalized feature space.",
            ]
        )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
