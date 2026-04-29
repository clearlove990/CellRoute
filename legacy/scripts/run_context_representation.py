from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spatial_context.differential_motif import compute_sample_motif_abundance, differential_motif_analysis
from spatial_context.graph_ssl import GraphSSLConfig
from spatial_context.motif_embedding import (
    MotifEmbeddingResult,
    build_tissue_motif_feature_bundle,
    fit_tissue_motif_model,
)
from spatial_context.neighborhood import get_runtime_info, load_spatial_h5ad, summarize_neighborhoods
from spatial_context.visualization import (
    plot_metric_boxplot,
    plot_pareto_frontier,
    plot_representation_comparison,
    plot_signal_vs_leakage,
)


DEFAULT_DATASETS = (
    ROOT / "data" / "spatial_processed" / "wu2021_breast_visium.h5ad",
    ROOT / "data" / "spatial_processed" / "gse220442_ad_mtg.h5ad",
)
EXPERIMENT_DIR = ROOT / "experiments" / "05_context_representation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 5 context-aware representation learning on processed spatial H5AD files.")
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
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["baseline_pca", "graph_ssl_main", "graph_ssl_no_sample_reg", "graph_ssl_no_neighbor"],
        help="Method names to run.",
    )
    return parser.parse_args()


def build_method_specs(random_state: int) -> dict[str, dict[str, object]]:
    return {
        "baseline_pca": {
            "label": "Week 4 baseline PCA",
            "representation_method": "baseline_pca",
            "representation_config": None,
        },
        "graph_ssl_main": {
            "label": "Graph SSL main",
            "representation_method": "graph_ssl",
            "representation_config": GraphSSLConfig(
                random_state=random_state,
                neighbor_weight=0.35,
                sample_balance_weight=0.18,
                condition_spread_weight=0.05,
            ),
        },
        "graph_ssl_no_sample_reg": {
            "label": "Graph SSL no sample regularizer",
            "representation_method": "graph_ssl",
            "representation_config": GraphSSLConfig(
                random_state=random_state,
                neighbor_weight=0.35,
                sample_balance_weight=0.0,
                condition_spread_weight=0.05,
            ),
        },
        "graph_ssl_no_neighbor": {
            "label": "Graph SSL no neighbor loss",
            "representation_method": "graph_ssl",
            "representation_config": GraphSSLConfig(
                random_state=random_state,
                neighbor_weight=0.0,
                sample_balance_weight=0.18,
                condition_spread_weight=0.05,
            ),
        },
    }


def main() -> None:
    args = parse_args()
    method_specs = build_method_specs(args.random_state)
    selected_methods = [method_name for method_name in args.methods if method_name in method_specs]
    if not selected_methods:
        raise ValueError("No valid methods were selected.")

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runtime_info = get_runtime_info()
    dataset_paths = [Path(path).resolve() for path in args.dataset_paths if Path(path).exists()]
    if not dataset_paths:
        raise FileNotFoundError("No processed spatial datasets were found.")

    metric_rows: list[dict[str, object]] = []
    motif_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    method_results: dict[str, dict[str, MotifEmbeddingResult]] = {}
    abundance_tables: dict[tuple[str, str], pd.DataFrame] = {}
    differential_tables: dict[tuple[str, str], pd.DataFrame] = {}
    dataset_labels: dict[str, str] = {}

    for dataset_path in dataset_paths:
        dataset = load_spatial_h5ad(dataset_path)
        dataset_labels[dataset.dataset_id] = dataset.dataset_name
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
        method_results[dataset.dataset_id] = {}

        for method_name in selected_methods:
            method_spec = method_specs[method_name]
            result = fit_tissue_motif_model(
                dataset,
                neighborhood_summary,
                runtime_info=runtime_info,
                n_expression_programs=args.n_expression_programs,
                top_variable_genes=args.top_variable_genes,
                representation_method=str(method_spec["representation_method"]),
                representation_config=method_spec["representation_config"],
                feature_bundle=feature_bundle,
                random_state=args.random_state,
            )
            abundance_table = compute_sample_motif_abundance(result.spot_table)
            differential_table = differential_motif_analysis(
                result.spot_table,
                abundance_table,
                random_state=args.random_state,
            )
            method_results[dataset.dataset_id][method_name] = result
            abundance_tables[(dataset.dataset_id, method_name)] = abundance_table
            differential_tables[(dataset.dataset_id, method_name)] = differential_table

            stability_df = compute_motif_stability_by_sample(result)
            stability_df["method"] = method_name
            stability_rows.extend(stability_df.to_dict(orient="records"))

            metric_rows.append(
                summarize_representation_metrics(
                    dataset=dataset,
                    neighborhood_adjacency=neighborhood_summary.scales[neighborhood_summary.core_scale_name].adjacency,
                    result=result,
                    method_name=method_name,
                    method_label=str(method_spec["label"]),
                    stability_df=stability_df,
                    abundance_table=abundance_table,
                    differential_table=differential_table,
                    random_state=args.random_state,
                )
            )
            motif_rows.extend(
                build_motif_comparison_rows(
                    result=result,
                    method_name=method_name,
                    method_label=str(method_spec["label"]),
                    stability_df=stability_df,
                    abundance_table=abundance_table,
                    differential_table=differential_table,
                )
            )

    metric_frame = pd.DataFrame(metric_rows)
    stability_frame = pd.DataFrame(stability_rows)
    motif_frame = pd.DataFrame(motif_rows)
    metric_frame = add_baseline_deltas(metric_frame)
    metric_frame["pareto_optimal"] = compute_pareto_optimal(
        x_values=metric_frame["batch_sample_leakage"].to_numpy(dtype=np.float64),
        y_values=metric_frame["overall_biological_score"].to_numpy(dtype=np.float64),
    )

    metric_frame.to_csv(results_dir / "embedding_metrics.csv", index=False)
    motif_frame.to_csv(results_dir / "motif_comparison.csv", index=False)

    primary_dataset_id = choose_primary_dataset(metric_frame)
    primary_baseline = method_results[primary_dataset_id]["baseline_pca"]
    primary_ssl = method_results[primary_dataset_id].get("graph_ssl_main", primary_baseline)
    plot_representation_comparison(
        primary_baseline.spot_table,
        primary_ssl.spot_table,
        output_path=figures_dir / "baseline_vs_ssl_umap.png",
        baseline_title=f"{primary_dataset_id}: baseline_pca",
        ssl_title=f"{primary_dataset_id}: graph_ssl_main",
    )
    plot_metric_boxplot(
        stability_frame,
        metric_col="motif_stability",
        output_path=figures_dir / "motif_stability_boxplot.png",
        title="Motif stability across samples",
        ylabel="Cosine similarity to global motif centroid",
    )
    plot_signal_vs_leakage(
        metric_frame,
        output_path=figures_dir / "sample_mixing_vs_condition_signal.png",
    )
    plot_pareto_frontier(
        metric_frame,
        output_path=figures_dir / "representation_pareto_frontier.png",
    )

    write_protocol(
        output_dir=output_dir,
        runtime_info=runtime_info,
        dataset_paths=dataset_paths,
        method_specs={name: method_specs[name] for name in selected_methods},
        metric_frame=metric_frame,
    )
    write_analysis(
        output_dir=output_dir,
        metric_frame=metric_frame,
        motif_frame=motif_frame,
        primary_dataset_id=primary_dataset_id,
    )
    summary_payload = {
        "primary_dataset_id": primary_dataset_id,
        "graph_ssl_success_datasets": metric_frame.loc[
            (metric_frame["method"] == "graph_ssl_main") & metric_frame["success_candidate"].astype(bool),
            "dataset_id",
        ].astype(str).tolist(),
    }
    print(json.dumps(summary_payload, ensure_ascii=False))


def compute_motif_stability_by_sample(
    result: MotifEmbeddingResult,
    *,
    min_spots_per_group: int = 25,
) -> pd.DataFrame:
    feature_matrix = result.feature_frame.to_numpy(dtype=np.float32, copy=False)
    spot_table = result.spot_table.loc[:, ["dataset_id", "dataset_name", "sample_id", "condition", "motif_id", "motif_label"]].copy()
    spot_table["row_idx"] = np.arange(spot_table.shape[0], dtype=np.int64)
    global_centroids: dict[str, np.ndarray] = {}
    for motif_id, motif_df in spot_table.groupby("motif_id", observed=False):
        motif_idx = motif_df["row_idx"].to_numpy(dtype=np.int64, copy=False)
        global_centroids[str(motif_id)] = feature_matrix[motif_idx].mean(axis=0)

    rows: list[dict[str, object]] = []
    for (sample_id, motif_id), motif_df in spot_table.groupby(["sample_id", "motif_id"], observed=False):
        n_spots = int(motif_df.shape[0])
        if n_spots < min_spots_per_group:
            continue
        motif_idx = motif_df["row_idx"].to_numpy(dtype=np.int64, copy=False)
        local_centroid = feature_matrix[motif_idx].mean(axis=0)
        global_centroid = global_centroids[str(motif_id)]
        rows.append(
            {
                "dataset_id": str(motif_df["dataset_id"].iloc[0]),
                "dataset_name": str(motif_df["dataset_name"].iloc[0]),
                "sample_id": str(sample_id),
                "condition": str(motif_df["condition"].iloc[0]),
                "motif_id": str(motif_id),
                "motif_label": str(motif_df["motif_label"].iloc[0]),
                "n_spots": n_spots,
                "motif_stability": cosine_similarity_np(local_centroid, global_centroid),
            }
        )
    return pd.DataFrame(rows)


def summarize_representation_metrics(
    *,
    dataset,
    neighborhood_adjacency: sparse.csr_matrix,
    result: MotifEmbeddingResult,
    method_name: str,
    method_label: str,
    stability_df: pd.DataFrame,
    abundance_table: pd.DataFrame,
    differential_table: pd.DataFrame,
    random_state: int,
) -> dict[str, object]:
    condition_separability = compute_condition_separability(result.representation_embedding, result.spot_table)
    sample_leakage = compute_batch_sample_leakage(result.representation_embedding, result.spot_table)
    spatial_smoothness = compute_spatial_smoothness(result.representation_embedding, neighborhood_adjacency)
    cell_type_conservation = compute_cell_type_conservation(
        result.representation_embedding,
        result.spot_table["cell_type"].astype(str).to_numpy(),
        random_state=random_state,
    )
    differential_reproducibility = compute_differential_motif_reproducibility(
        abundance_table,
        random_state=random_state,
    )
    motif_stability = float(stability_df["motif_stability"].mean()) if not stability_df.empty else float("nan")
    biological_metrics = [
        motif_stability,
        condition_separability,
        spatial_smoothness,
        cell_type_conservation,
        differential_reproducibility,
    ]
    overall_biological_score = float(np.nanmean(biological_metrics))
    sample_mixing_score = float(1.0 / (1.0 + max(sample_leakage, 0.0)))
    association_count = int(differential_table["association_call"].sum()) if not differential_table.empty else 0
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_name": dataset.dataset_name,
        "method": method_name,
        "method_label": method_label,
        "representation_name": result.representation_name,
        "n_spots": int(dataset.obs.shape[0]),
        "n_samples": int(dataset.obs["sample_id"].nunique()),
        "n_motifs": int(result.n_clusters),
        "motif_stability": motif_stability,
        "condition_separability": condition_separability,
        "batch_sample_leakage": sample_leakage,
        "sample_mixing_score": sample_mixing_score,
        "spatial_smoothness": spatial_smoothness,
        "cell_type_conservation": cell_type_conservation,
        "differential_motif_reproducibility": differential_reproducibility,
        "overall_biological_score": overall_biological_score,
        "spatial_coherence_zscore": float(result.spatial_coherence_zscore),
        "associated_motif_count": association_count,
    }


def compute_condition_separability(embedding: np.ndarray, spot_table: pd.DataFrame) -> float:
    sample_summary = aggregate_sample_embeddings(embedding, spot_table)
    if sample_summary.empty or sample_summary["condition"].nunique() <= 1:
        return float("nan")
    condition_centroids = {
        str(condition): group.loc[:, [column for column in sample_summary.columns if column.startswith("emb_")]].to_numpy(dtype=np.float64).mean(axis=0)
        for condition, group in sample_summary.groupby("condition", observed=False)
    }
    between_distances: list[float] = []
    condition_keys = sorted(condition_centroids)
    for left_index in range(len(condition_keys)):
        for right_index in range(left_index + 1, len(condition_keys)):
            left = condition_centroids[condition_keys[left_index]]
            right = condition_centroids[condition_keys[right_index]]
            between_distances.append(float(np.linalg.norm(left - right)))
    if not between_distances:
        return float("nan")
    within = 0.0
    embedding_cols = [column for column in sample_summary.columns if column.startswith("emb_")]
    for _, row in sample_summary.iterrows():
        centroid = condition_centroids[str(row["condition"])]
        within += float(np.linalg.norm(row[embedding_cols].to_numpy(dtype=np.float64) - centroid))
    within /= max(sample_summary.shape[0], 1)
    between = float(np.mean(between_distances))
    return float(between / (between + within + 1.0e-6))


def compute_batch_sample_leakage(embedding: np.ndarray, spot_table: pd.DataFrame) -> float:
    sample_summary = aggregate_sample_embeddings(embedding, spot_table)
    if sample_summary.empty or sample_summary.shape[0] <= 1:
        return float("nan")
    embedding_cols = [column for column in sample_summary.columns if column.startswith("emb_")]
    global_centroid = sample_summary.loc[:, embedding_cols].to_numpy(dtype=np.float64).mean(axis=0)
    condition_centroids = {
        str(condition): group.loc[:, embedding_cols].to_numpy(dtype=np.float64).mean(axis=0)
        for condition, group in sample_summary.groupby("condition", observed=False)
    }
    numerator = 0.0
    denominator = 0.0
    for _, row in sample_summary.iterrows():
        sample_embedding = row[embedding_cols].to_numpy(dtype=np.float64)
        numerator += float(np.linalg.norm(sample_embedding - condition_centroids[str(row["condition"])]))
        denominator += float(np.linalg.norm(sample_embedding - global_centroid))
    numerator /= max(sample_summary.shape[0], 1)
    denominator /= max(sample_summary.shape[0], 1)
    return float(numerator / (denominator + 1.0e-6))


def compute_spatial_smoothness(embedding: np.ndarray, adjacency: sparse.csr_matrix) -> float:
    normalized = row_normalize_numpy(embedding.astype(np.float32, copy=False))
    neighbor_embedding = np.asarray(adjacency.dot(normalized), dtype=np.float32)
    neighbor_embedding = row_normalize_numpy(neighbor_embedding)
    return float(np.mean(np.sum(normalized * neighbor_embedding, axis=1)))


def compute_cell_type_conservation(
    embedding: np.ndarray,
    cell_types: np.ndarray,
    *,
    random_state: int,
    max_obs: int = 5000,
    n_neighbors: int = 15,
) -> float:
    n_obs = embedding.shape[0]
    if n_obs <= 2:
        return float("nan")
    rng = np.random.default_rng(random_state)
    if n_obs > max_obs:
        subset_idx = np.sort(rng.choice(n_obs, size=max_obs, replace=False))
    else:
        subset_idx = np.arange(n_obs, dtype=np.int64)
    subset_embedding = embedding[subset_idx]
    subset_labels = cell_types[subset_idx].astype(str)
    effective_k = int(min(n_neighbors + 1, subset_embedding.shape[0]))
    if effective_k <= 1:
        return float("nan")
    neighbors = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
    neighbors.fit(subset_embedding)
    knn_idx = neighbors.kneighbors(return_distance=False)[:, 1:]
    if knn_idx.size == 0:
        return float("nan")
    same_label_fraction = []
    for row_index in range(knn_idx.shape[0]):
        same_label_fraction.append(float(np.mean(subset_labels[knn_idx[row_index]] == subset_labels[row_index])))
    return float(np.mean(same_label_fraction))


def compute_differential_motif_reproducibility(
    abundance_table: pd.DataFrame,
    *,
    random_state: int,
    n_bootstraps: int = 128,
) -> float:
    if abundance_table.empty:
        return float("nan")
    conditions = sorted(abundance_table["condition"].astype(str).unique().tolist())
    if len(conditions) != 2:
        return float("nan")
    condition_a, condition_b = conditions
    top_k = int(max(2, min(4, abundance_table["motif_id"].nunique() // 2)))
    full_rank = rank_motifs_from_abundance(abundance_table, condition_a=condition_a, condition_b=condition_b)
    if full_rank.empty:
        return float("nan")
    reference_set = set(full_rank.head(top_k)["motif_id"].astype(str).tolist())
    if not reference_set:
        return float("nan")
    sample_meta = abundance_table.loc[:, ["sample_id", "condition"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    scores: list[float] = []
    for _ in range(n_bootstraps):
        sampled_ids: list[str] = []
        for condition_name in [condition_a, condition_b]:
            condition_samples = sample_meta.loc[sample_meta["condition"].astype(str) == condition_name, "sample_id"].astype(str).tolist()
            if not condition_samples:
                continue
            sampled_ids.extend(rng.choice(condition_samples, size=len(condition_samples), replace=True).tolist())
        weights = Counter(sampled_ids)
        if not weights:
            continue
        weight_frame = pd.DataFrame(
            {
                "sample_id": list(weights.keys()),
                "bootstrap_weight": list(weights.values()),
            }
        )
        weighted = abundance_table.merge(weight_frame, on="sample_id", how="inner")
        if weighted.empty:
            continue
        boot_rank = rank_motifs_from_abundance(
            weighted,
            condition_a=condition_a,
            condition_b=condition_b,
            weight_col="bootstrap_weight",
        )
        if boot_rank.empty:
            continue
        boot_set = set(boot_rank.head(top_k)["motif_id"].astype(str).tolist())
        union = reference_set | boot_set
        score = 1.0 if not union else float(len(reference_set & boot_set) / len(union))
        scores.append(score)
    return float(np.mean(scores)) if scores else float("nan")


def rank_motifs_from_abundance(
    abundance_table: pd.DataFrame,
    *,
    condition_a: str,
    condition_b: str,
    weight_col: str | None = None,
) -> pd.DataFrame:
    frame = abundance_table.copy()
    if weight_col is None:
        frame["__weight"] = 1.0
        weight_col = "__weight"
    grouped = (
        frame.groupby(["motif_id", "condition"], observed=False)
        .apply(
            lambda group: np.average(
                group["motif_fraction"].to_numpy(dtype=np.float64),
                weights=group[weight_col].to_numpy(dtype=np.float64),
            ),
            include_groups=False,
        )
        .rename("weighted_fraction")
        .reset_index()
    )
    pivot = grouped.pivot(index="motif_id", columns="condition", values="weighted_fraction").fillna(0.0)
    if condition_a not in pivot.columns or condition_b not in pivot.columns:
        return pd.DataFrame(columns=["motif_id", "delta"])
    rank = pivot.assign(delta=lambda df: df[condition_b] - df[condition_a]).reset_index()
    rank["abs_delta"] = rank["delta"].abs()
    return rank.sort_values(["abs_delta", "delta"], ascending=[False, False]).reset_index(drop=True)


def build_motif_comparison_rows(
    *,
    result: MotifEmbeddingResult,
    method_name: str,
    method_label: str,
    stability_df: pd.DataFrame,
    abundance_table: pd.DataFrame,
    differential_table: pd.DataFrame,
) -> list[dict[str, object]]:
    stability_summary = (
        stability_df.groupby("motif_id", observed=False)["motif_stability"].mean().rename("motif_stability").reset_index()
        if not stability_df.empty
        else pd.DataFrame(columns=["motif_id", "motif_stability"])
    )
    abundance_summary = (
        abundance_table.groupby(["motif_id", "motif_label"], observed=False)
        .agg(
            mean_sample_fraction=("motif_fraction", "mean"),
            samples_with_support=("motif_spots", lambda values: int(np.sum(np.asarray(values, dtype=np.int64) > 0))),
        )
        .reset_index()
    )
    comparison = result.motif_metadata.copy()
    comparison["method"] = method_name
    comparison["method_label"] = method_label
    comparison["dataset_id"] = result.dataset_id
    comparison["dataset_name"] = result.dataset_name
    comparison = comparison.merge(abundance_summary, on=["motif_id", "motif_label"], how="left")
    comparison = comparison.merge(stability_summary, on="motif_id", how="left")
    if not differential_table.empty:
        comparison = comparison.merge(
            differential_table.loc[
                :,
                [
                    "motif_id",
                    "delta_fraction",
                    "log2_fold_change",
                    "permutation_pvalue",
                    "mixedlm_pvalue",
                    "evidence_tier",
                    "association_call",
                ],
            ],
            on="motif_id",
            how="left",
        )
    comparison["representation_name"] = result.representation_name
    comparison["spatial_coherence_zscore"] = float(result.spatial_coherence_zscore)
    return comparison.to_dict(orient="records")


def aggregate_sample_embeddings(embedding: np.ndarray, spot_table: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(embedding.astype(np.float32, copy=False), index=spot_table.index)
    frame.columns = [f"emb_{index:02d}" for index in range(frame.shape[1])]
    meta = spot_table.loc[:, ["sample_id", "condition"]].copy()
    merged = pd.concat([meta, frame], axis=1)
    return merged.groupby(["sample_id", "condition"], observed=False).mean(numeric_only=True).reset_index()


def cosine_similarity_np(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def row_normalize_numpy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms > 1.0e-6, norms, 1.0)
    return values / norms


def add_baseline_deltas(metric_frame: pd.DataFrame) -> pd.DataFrame:
    if metric_frame.empty:
        return metric_frame
    frame = metric_frame.copy()
    higher_better_metrics = [
        "motif_stability",
        "condition_separability",
        "spatial_smoothness",
        "cell_type_conservation",
        "differential_motif_reproducibility",
        "overall_biological_score",
    ]
    for metric_name in higher_better_metrics:
        frame[f"delta_{metric_name}_vs_baseline"] = np.nan
    frame["delta_batch_sample_leakage_vs_baseline"] = np.nan
    frame["success_candidate"] = False
    for dataset_id, group in frame.groupby("dataset_id", observed=False):
        baseline = group.loc[group["method"] == "baseline_pca"]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        for row_index in group.index:
            for metric_name in higher_better_metrics:
                frame.at[row_index, f"delta_{metric_name}_vs_baseline"] = float(frame.at[row_index, metric_name]) - float(baseline_row[metric_name])
            frame.at[row_index, "delta_batch_sample_leakage_vs_baseline"] = float(frame.at[row_index, "batch_sample_leakage"]) - float(baseline_row["batch_sample_leakage"])
            if frame.at[row_index, "method"] == "baseline_pca":
                continue
            gain_metrics = [
                float(frame.at[row_index, "delta_motif_stability_vs_baseline"]),
                float(frame.at[row_index, "delta_condition_separability_vs_baseline"]),
                float(frame.at[row_index, "delta_spatial_smoothness_vs_baseline"]),
                float(frame.at[row_index, "delta_cell_type_conservation_vs_baseline"]),
                float(frame.at[row_index, "delta_differential_motif_reproducibility_vs_baseline"]),
            ]
            leakage_delta = float(frame.at[row_index, "delta_batch_sample_leakage_vs_baseline"])
            frame.at[row_index, "success_candidate"] = bool(
                np.nanmax(gain_metrics) > 0.01
                and leakage_delta <= 0.05
                and float(frame.at[row_index, "delta_overall_biological_score_vs_baseline"]) >= -0.02
            )
    return frame


def compute_pareto_optimal(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    keep = finite.copy()
    for index in range(x_values.shape[0]):
        if not finite[index]:
            continue
        for other_index in range(x_values.shape[0]):
            if index == other_index:
                continue
            if not finite[other_index]:
                continue
            dominates = (
                x_values[other_index] <= x_values[index]
                and y_values[other_index] >= y_values[index]
                and (x_values[other_index] < x_values[index] or y_values[other_index] > y_values[index])
            )
            if dominates:
                keep[index] = False
                break
    return keep


def choose_primary_dataset(metric_frame: pd.DataFrame) -> str:
    main_rows = metric_frame.loc[metric_frame["method"] == "graph_ssl_main"].copy()
    if main_rows.empty:
        return str(metric_frame["dataset_id"].iloc[0])
    ranked = main_rows.sort_values(
        ["success_candidate", "delta_overall_biological_score_vs_baseline", "delta_condition_separability_vs_baseline"],
        ascending=[False, False, False],
    )
    return str(ranked.iloc[0]["dataset_id"])


def write_protocol(
    *,
    output_dir: Path,
    runtime_info,
    dataset_paths: list[Path],
    method_specs: dict[str, dict[str, object]],
    metric_frame: pd.DataFrame,
) -> None:
    lines = [
        "# Protocol",
        "",
        "## Goal",
        "",
        "Add a lightweight context-aware representation branch on top of the Week 4 tissue motif baseline.",
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
            "## Methods",
            "",
        ]
    )
    for method_name, method_spec in method_specs.items():
        lines.append(
            f"- `{method_name}`: `{method_spec['label']}` with representation=`{method_spec['representation_method']}`."
        )
    lines.extend(
        [
            "",
            "## Steps",
            "",
            "1. Reuse the Week 4 neighborhood summary and motif feature frame built from cell-type composition, density, entropy, and expression programs.",
            "2. Train a lightweight graph contrastive encoder on the core spatial graph with feature dropout, edge dropout, and neighborhood masking.",
            "3. Add sample-aware regularization to shrink within-condition sample centroids while preserving condition-level separation.",
            "4. Cluster motifs from either baseline PCA or graph SSL representations and keep motif labels anchored to interpretable neighborhood features.",
            "5. Score representation quality using motif stability, condition separability, batch/sample leakage, spatial smoothness, cell-type conservation, and differential motif reproducibility.",
            "6. Compare baseline and SSL variants on a Pareto plot rather than chasing a single metric.",
            "",
            "## Output Tables",
            "",
            "- `results/embedding_metrics.csv`",
            "- `results/motif_comparison.csv`",
            "",
            "## Output Figures",
            "",
            "- `figures/baseline_vs_ssl_umap.png`",
            "- `figures/motif_stability_boxplot.png`",
            "- `figures/sample_mixing_vs_condition_signal.png`",
            "- `figures/representation_pareto_frontier.png`",
            "",
            "## Dataset snapshot",
            "",
        ]
    )
    for _, row in metric_frame.loc[metric_frame["method"] == "baseline_pca"].iterrows():
        lines.append(
            f"- `{row['dataset_id']}`: `{row['n_samples']}` samples, `{row['n_spots']}` spots, baseline motifs=`{row['n_motifs']}`, "
            f"baseline stability=`{float(row['motif_stability']):.3f}`, baseline coherence_z=`{float(row['spatial_coherence_zscore']):.2f}`"
        )
    (output_dir / "protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(
    *,
    output_dir: Path,
    metric_frame: pd.DataFrame,
    motif_frame: pd.DataFrame,
    primary_dataset_id: str,
) -> None:
    main_rows = metric_frame.loc[metric_frame["method"] == "graph_ssl_main"].copy()
    success_rows = main_rows.loc[main_rows["success_candidate"].astype(bool)].copy()
    lines = [
        "# Analysis",
        "",
        "## Headline",
        "",
    ]
    if success_rows.empty:
        lines.append("- The current graph SSL branch did not clear the success criterion on the evaluated datasets; the representation improved some axes but still looks like a trade-off rather than a clean Pareto gain.")
    else:
        success_dataset_ids = success_rows["dataset_id"].astype(str).tolist()
        lines.append(
            f"- `graph_ssl_main` cleared the Week 5 success criterion on `{', '.join(success_dataset_ids)}` by improving at least one biological axis without a material sample-leakage penalty."
        )

    lines.extend(
        [
            "",
            "## Per-dataset summary",
            "",
        ]
    )
    for dataset_id, group in metric_frame.groupby("dataset_id", observed=False):
        baseline = group.loc[group["method"] == "baseline_pca"].iloc[0]
        main = group.loc[group["method"] == "graph_ssl_main"].iloc[0] if not group.loc[group["method"] == "graph_ssl_main"].empty else baseline
        lines.append(
            f"- `{dataset_id}`: overall biological score `{float(baseline['overall_biological_score']):.3f}` -> `{float(main['overall_biological_score']):.3f}`, "
            f"motif stability `{float(baseline['motif_stability']):.3f}` -> `{float(main['motif_stability']):.3f}`, "
            f"condition separability `{float(baseline['condition_separability']):.3f}` -> `{float(main['condition_separability']):.3f}`, "
            f"sample leakage `{float(baseline['batch_sample_leakage']):.3f}` -> `{float(main['batch_sample_leakage']):.3f}`."
        )

    lines.extend(
        [
            "",
            "## Pareto diagnostic",
            "",
        ]
    )
    frontier_rows = metric_frame.loc[metric_frame["pareto_optimal"].astype(bool)].copy()
    frontier_labels = [f"{row['dataset_id']}:{row['method']}" for _, row in frontier_rows.iterrows()]
    lines.append(f"- Global frontier members: `{', '.join(frontier_labels)}`.")
    primary_main = metric_frame.loc[
        (metric_frame["dataset_id"] == primary_dataset_id) & (metric_frame["method"] == "graph_ssl_main")
    ]
    if not primary_main.empty:
        row = primary_main.iloc[0]
        lines.append(
            f"- Primary dataset `{primary_dataset_id}`: graph SSL delta overall=`{float(row['delta_overall_biological_score_vs_baseline']):.3f}`, "
            f"delta leakage=`{float(row['delta_batch_sample_leakage_vs_baseline']):.3f}`."
        )

    lines.extend(
        [
            "",
            "## Motif-level readout",
            "",
        ]
    )
    primary_motifs = motif_frame.loc[
        (motif_frame["dataset_id"] == primary_dataset_id)
        & (motif_frame["method"] == "graph_ssl_main")
    ].copy()
    if primary_motifs.empty:
        lines.append("- No motif-level comparison was available for the primary dataset.")
    else:
        ranked = primary_motifs.sort_values(
            ["association_call", "motif_stability", "mean_sample_fraction"],
            ascending=[False, False, False],
        ).head(3)
        for _, row in ranked.iterrows():
            lines.append(
                f"- `{row['motif_id']}` (`{row['motif_label']}`): stability=`{float(row.get('motif_stability', np.nan)):.3f}`, "
                f"mean fraction=`{float(row.get('mean_sample_fraction', np.nan)):.3f}`, association=`{row.get('evidence_tier', 'na')}`."
            )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
