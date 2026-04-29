from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spatial_context.differential_motif import compute_sample_motif_abundance, differential_motif_analysis
from spatial_context.motif_embedding import MotifEmbeddingResult, fit_tissue_motif_model
from spatial_context.neighborhood import get_runtime_info, load_spatial_h5ad, summarize_neighborhoods
from spatial_context.visualization import (
    plot_condition_abundance,
    plot_differential_volcano,
    plot_motif_layout,
    plot_motif_spatial_map,
)


DEFAULT_DATASETS = (
    ROOT / "data" / "spatial_processed" / "wu2021_breast_visium.h5ad",
    ROOT / "data" / "spatial_processed" / "gse220442_ad_mtg.h5ad",
)
EXPERIMENT_DIR = ROOT / "experiments" / "04_tissue_motif_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 4 tissue motif baseline on processed spatial H5AD files.")
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

    abundance_frames: list[pd.DataFrame] = []
    differential_frames: list[pd.DataFrame] = []
    motif_results: dict[str, MotifEmbeddingResult] = {}
    dataset_summaries: list[dict[str, object]] = []

    for dataset_path in dataset_paths:
        dataset = load_spatial_h5ad(dataset_path)
        neighborhood_summary = summarize_neighborhoods(
            dataset,
            runtime_info=runtime_info,
            radius_factor=args.radius_factor,
        )
        motif_result = fit_tissue_motif_model(
            dataset,
            neighborhood_summary,
            runtime_info=runtime_info,
            n_expression_programs=args.n_expression_programs,
            top_variable_genes=args.top_variable_genes,
            random_state=args.random_state,
        )
        motif_results[dataset.dataset_id] = motif_result
        abundance_df = compute_sample_motif_abundance(motif_result.spot_table)
        differential_df = differential_motif_analysis(
            motif_result.spot_table,
            abundance_df,
            random_state=args.random_state,
        )
        abundance_frames.append(abundance_df)
        differential_frames.append(differential_df)
        dataset_summaries.append(
            {
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.dataset_name,
                "n_spots": int(dataset.obs.shape[0]),
                "n_genes": int(dataset.expression.shape[1]),
                "n_samples": int(dataset.obs["sample_id"].nunique()),
                "conditions": "|".join(sorted(dataset.obs["condition"].astype(str).unique().tolist())),
                "n_motifs": int(motif_result.n_clusters),
                "layout_method": motif_result.layout_method,
                "spatial_coherence_observed": float(motif_result.spatial_coherence_observed),
                "spatial_coherence_perm_mean": float(motif_result.spatial_coherence_perm_mean),
                "spatial_coherence_zscore": float(motif_result.spatial_coherence_zscore),
            }
        )

    abundance_table = pd.concat(abundance_frames, ignore_index=True) if abundance_frames else pd.DataFrame()
    differential_table = pd.concat(differential_frames, ignore_index=True) if differential_frames else pd.DataFrame()
    abundance_table.to_csv(results_dir / "motif_abundance_table.csv", index=False)
    differential_table.to_csv(results_dir / "differential_motif_results.csv", index=False)
    pd.DataFrame(dataset_summaries).to_csv(results_dir / "dataset_summary.csv", index=False)

    primary_dataset_id, top_row = choose_primary_visualization_target(differential_table, motif_results)
    primary_result = motif_results[primary_dataset_id]
    sample_id = choose_visualization_sample(abundance_table, dataset_id=primary_dataset_id, motif_id=str(top_row["motif_id"]))

    layout_title = f"{primary_result.dataset_name} motif layout ({primary_result.layout_method})"
    plot_motif_layout(
        primary_result.spot_table,
        output_path=figures_dir / "motif_umap.png",
        title=layout_title,
    )
    plot_motif_spatial_map(
        primary_result.spot_table,
        sample_id=sample_id,
        output_path=figures_dir / "motif_spatial_map_sample1.png",
        title=f"{primary_dataset_id} sample {sample_id} motif map",
    )
    plot_condition_abundance(
        abundance_table,
        differential_table,
        dataset_id=primary_dataset_id,
        condition_a=str(top_row["condition_a"]),
        condition_b=str(top_row["condition_b"]),
        output_path=figures_dir / "motif_condition_abundance.png",
    )
    plot_differential_volcano(
        differential_table,
        dataset_id=primary_dataset_id,
        output_path=figures_dir / "differential_motif_volcano.png",
        title=f"{primary_dataset_id} differential motifs: {top_row['condition_b']} vs {top_row['condition_a']}",
    )

    write_protocol(
        output_dir=output_dir,
        runtime_info=runtime_info,
        dataset_paths=dataset_paths,
        dataset_summaries=dataset_summaries,
    )
    write_analysis(
        output_dir=output_dir,
        dataset_summaries=dataset_summaries,
        differential_table=differential_table,
        primary_dataset_id=primary_dataset_id,
        top_row=top_row,
    )
    payload = {"primary_dataset_id": primary_dataset_id, "top_motif": json.loads(top_row.to_json(force_ascii=False))}
    print(json.dumps(payload, ensure_ascii=False))


def choose_primary_visualization_target(
    differential_table: pd.DataFrame,
    motif_results: dict[str, MotifEmbeddingResult],
) -> tuple[str, pd.Series]:
    if differential_table.empty:
        first_id = next(iter(motif_results))
        fallback = pd.Series(
            {
                "dataset_id": first_id,
                "motif_id": motif_results[first_id].motif_metadata["motif_id"].iloc[0],
                "condition_a": motif_results[first_id].spot_table["condition"].astype(str).min(),
                "condition_b": motif_results[first_id].spot_table["condition"].astype(str).max(),
            }
        )
        return first_id, fallback
    ranked = differential_table.copy()
    ranked["mixedlm_pvalue"] = ranked["mixedlm_pvalue"].fillna(1.0)
    ranked["evidence_rank"] = ranked["evidence_tier"].map({"strong": 0, "moderate": 1, "weak": 2, "none": 3}).fillna(4)
    top_row = ranked.sort_values(
        ["association_call", "evidence_rank", "mixedlm_pvalue", "delta_fraction"],
        ascending=[False, True, True, False],
    ).iloc[0]
    return str(top_row["dataset_id"]), top_row


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
) -> None:
    protocol_lines = [
        "# Protocol",
        "",
        "## Goal",
        "",
        "Build a Week 4 tissue motif baseline from multi-scale spatial neighborhoods, cluster motif embeddings, and test differential motif abundance with sample-aware statistics.",
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
            "1. Load processed spatial `.h5ad` files without requiring `anndata`, using `h5py` to read `obs`, `var`, `layers['lognorm']`, and spatial coordinates.",
            "2. Build three neighborhood scales per sample: `kNN-6`, adaptive radius (`median 6-NN distance * 1.6`), and `kNN-18`.",
            "3. Aggregate neighbor cell-type composition, local expression program scores, density features, and neighborhood entropy into a motif embedding.",
            "4. Standardize the motif embedding, reduce with PCA, cluster with KMeans, and label motifs from the dominant neighborhood composition.",
            "5. Summarize sample-level motif abundance and test condition effects with exact sample permutation, sample bootstrap, and a spot-level mixed-effect approximation with sample random intercept.",
            "",
            "## Outputs",
            "",
            "- `results/motif_abundance_table.csv`",
            "- `results/differential_motif_results.csv`",
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
            f"layout=`{row['layout_method']}`, coherence_z=`{row['spatial_coherence_zscore']:.2f}`"
        )
    (output_dir / "protocol.md").write_text("\n".join(protocol_lines) + "\n", encoding="utf-8")


def write_analysis(
    *,
    output_dir: Path,
    dataset_summaries: list[dict[str, object]],
    differential_table: pd.DataFrame,
    primary_dataset_id: str,
    top_row: pd.Series,
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
            f"- `{row['dataset_id']}` produced `{row['n_motifs']}` motifs with spatial coherence z-score `{row['spatial_coherence_zscore']:.2f}` "
            f"(observed `{row['spatial_coherence_observed']:.3f}` vs permutation mean `{row['spatial_coherence_perm_mean']:.3f}`)."
        )
    if differential_table.empty:
        lines.extend(
            [
                "",
                "## Differential readout",
                "",
                "- No two-condition differential motif comparison was available in the current inputs.",
            ]
        )
    else:
        top_dataset = differential_table.loc[differential_table["dataset_id"] == primary_dataset_id].copy()
        top_dataset = top_dataset.sort_values(["association_call", "mixedlm_pvalue", "permutation_pvalue"], ascending=[False, True, True])
        lines.extend(
            [
                "",
                "## Differential readout",
                "",
                f"- Primary visualization target: `{primary_dataset_id}` with comparison `{top_row['condition_b']} vs {top_row['condition_a']}`.",
                f"- Top motif: `{top_row['motif_id']}` (`{top_row.get('motif_label', '')}`) with delta fraction `{float(top_row.get('delta_fraction', 0.0)):.3f}`, "
                f"log2FC `{float(top_row.get('log2_fold_change', 0.0)):.3f}`, sample-level permutation p `{float(top_row.get('permutation_pvalue', 1.0)):.3f}`, "
                f"mixed-effect p `{float(top_row.get('mixedlm_pvalue', 1.0)):.3g}`, evidence tier `{top_row.get('evidence_tier', 'none')}`.",
                f"- Condition-associated motifs called by the baseline: `{int(top_dataset['association_call'].sum())}` / `{int(top_dataset.shape[0])}` in `{primary_dataset_id}`.",
                "- In this small-sample setting the mixed-effect approximation was conservative; the primary baseline call is driven by sample-level permutation plus bootstrap support, not by spot-level pseudoreplication.",
            ]
        )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
