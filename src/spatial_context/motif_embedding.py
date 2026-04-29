from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

from .graph_ssl import GraphSSLConfig, train_graph_context_embedding
from .neighborhood import (
    NeighborhoodSummary,
    RuntimeInfo,
    SpatialDataset,
    aggregate_feature_matrix,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is optional for import but expected in this repo
    torch = None


@dataclass(frozen=True)
class MotifFeatureBundle:
    feature_frame: pd.DataFrame
    expression_program_metadata: pd.DataFrame
    library_normalization_mode: str = "input_expression"


@dataclass(frozen=True)
class PCAProjection:
    components: np.ndarray
    mean: np.ndarray
    explained_variance_ratio: np.ndarray
    backend: str


@dataclass(frozen=True)
class FrozenSampleAwareMotifModel:
    feature_columns: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_projection: PCAProjection
    embedding_centroids: np.ndarray
    normalized_feature_centroids: np.ndarray
    cluster_dim: int
    motif_ids: tuple[str, ...]
    motif_label_map: dict[str, str]
    training_row_indices: np.ndarray
    training_sample_ids: tuple[str, ...]
    training_spots_per_sample: int
    representation_name: str
    normalization_mode: str
    library_normalization_mode: str
    model_metadata: dict[str, Any]


@dataclass(frozen=True)
class SampleAwareMotifFitResult:
    embedding_result: "MotifEmbeddingResult"
    frozen_model: FrozenSampleAwareMotifModel
    normalized_feature_frame: pd.DataFrame
    normalization_summary: pd.DataFrame


@dataclass(frozen=True)
class MotifEmbeddingResult:
    dataset_id: str
    dataset_name: str
    feature_frame: pd.DataFrame
    pca_embedding: np.ndarray
    representation_name: str
    representation_embedding: np.ndarray
    layout_2d: np.ndarray
    layout_method: str
    spot_table: pd.DataFrame
    motif_metadata: pd.DataFrame
    expression_program_metadata: pd.DataFrame
    representation_training_history: pd.DataFrame | None
    n_clusters: int
    spatial_coherence_observed: float
    spatial_coherence_perm_mean: float
    spatial_coherence_zscore: float
    model_metadata: dict[str, Any]


def fit_tissue_motif_model(
    dataset: SpatialDataset,
    neighborhood_summary: NeighborhoodSummary,
    *,
    runtime_info: RuntimeInfo,
    n_expression_programs: int = 6,
    top_variable_genes: int = 256,
    representation_method: str = "baseline_pca",
    representation_config: GraphSSLConfig | dict[str, Any] | None = None,
    feature_bundle: MotifFeatureBundle | None = None,
    random_state: int = 7,
) -> MotifEmbeddingResult:
    feature_bundle = feature_bundle or build_tissue_motif_feature_bundle(
        dataset,
        neighborhood_summary,
        runtime_info=runtime_info,
        n_expression_programs=n_expression_programs,
        top_variable_genes=top_variable_genes,
        random_state=random_state,
    )
    feature_frame = feature_bundle.feature_frame
    program_metadata = feature_bundle.expression_program_metadata

    standardized = StandardScaler().fit_transform(feature_frame.to_numpy(dtype=np.float32, copy=False))
    n_pca_components = int(max(2, min(12, standardized.shape[0] - 1, standardized.shape[1])))
    pca_model = PCA(n_components=n_pca_components, random_state=random_state)
    pca_embedding = pca_model.fit_transform(standardized).astype(np.float32, copy=False)
    representation_name = str(representation_method).strip().lower()
    representation_embedding, representation_training_history, representation_metadata = choose_representation(
        representation_method=representation_name,
        feature_frame=feature_frame,
        pca_embedding=pca_embedding,
        adjacency=neighborhood_summary.scales[neighborhood_summary.core_scale_name].adjacency,
        runtime_info=runtime_info,
        sample_ids=dataset.obs["sample_id"].astype(str).to_numpy(),
        condition_ids=dataset.obs["condition"].astype(str).to_numpy(),
        representation_config=representation_config,
        random_state=random_state,
    )

    n_clusters = choose_cluster_count(representation_embedding, random_state=random_state)
    cluster_embedding = representation_embedding[:, : min(8, representation_embedding.shape[1])]
    cluster_model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    cluster_ids = cluster_model.fit_predict(cluster_embedding)
    motif_ids = np.asarray([f"motif_{cluster_id:02d}" for cluster_id in cluster_ids], dtype=object)

    layout_2d, layout_method = compute_layout_2d(representation_embedding, random_state=random_state)
    motif_label_map = label_motifs(
        motif_ids=motif_ids,
        feature_frame=feature_frame,
        neighborhood_summary=neighborhood_summary,
    )

    spot_table = dataset.obs.copy()
    spot_table["motif_cluster"] = cluster_ids.astype(np.int64, copy=False)
    spot_table["motif_id"] = motif_ids
    spot_table["motif_label"] = pd.Series(motif_ids, index=spot_table.index).map(motif_label_map).astype(str)
    spot_table["layout_1"] = layout_2d[:, 0].astype(np.float32, copy=False)
    spot_table["layout_2"] = layout_2d[:, 1].astype(np.float32, copy=False)
    spot_table["layout_method"] = layout_method
    spot_table["representation_method"] = representation_name

    motif_metadata = (
        spot_table.groupby(["motif_id", "motif_label"], observed=False)
        .size()
        .rename("n_spots")
        .reset_index()
        .sort_values("motif_id")
        .reset_index(drop=True)
    )
    coherence_observed, coherence_perm_mean, coherence_zscore = compute_spatial_coherence(
        motif_ids=motif_ids,
        adjacency=neighborhood_summary.scales[neighborhood_summary.core_scale_name].adjacency,
        sample_ids=spot_table["sample_id"].astype(str).to_numpy(),
        random_state=random_state,
    )
    program_count = int(program_metadata.shape[0]) if not program_metadata.empty else 0
    model_metadata = {
        "n_expression_programs": program_count,
        "top_variable_genes": int(min(top_variable_genes, dataset.expression.shape[1])),
        "n_clusters": int(n_clusters),
        "core_scale": neighborhood_summary.core_scale_name,
        "pca_explained_variance_ratio": pca_model.explained_variance_ratio_.astype(float).tolist(),
        "representation_method": representation_name,
        "representation_dim": int(representation_embedding.shape[1]),
        "representation_metadata": representation_metadata,
    }
    return MotifEmbeddingResult(
        dataset_id=dataset.dataset_id,
        dataset_name=dataset.dataset_name,
        feature_frame=feature_frame,
        pca_embedding=pca_embedding,
        representation_name=representation_name,
        representation_embedding=representation_embedding,
        layout_2d=layout_2d,
        layout_method=layout_method,
        spot_table=spot_table,
        motif_metadata=motif_metadata,
        expression_program_metadata=program_metadata,
        representation_training_history=representation_training_history,
        n_clusters=n_clusters,
        spatial_coherence_observed=float(coherence_observed),
        spatial_coherence_perm_mean=float(coherence_perm_mean),
        spatial_coherence_zscore=float(coherence_zscore),
        model_metadata=model_metadata,
    )


def build_tissue_motif_feature_bundle(
    dataset: SpatialDataset,
    neighborhood_summary: NeighborhoodSummary,
    *,
    runtime_info: RuntimeInfo,
    n_expression_programs: int = 6,
    top_variable_genes: int = 256,
    random_state: int = 7,
) -> MotifFeatureBundle:
    program_expression, library_normalization_mode = prepare_expression_matrix_for_program_scoring(dataset)
    program_scores, program_metadata = compute_expression_program_scores(
        program_expression,
        dataset.var_names,
        n_components=n_expression_programs,
        top_variable_genes=top_variable_genes,
        random_state=random_state,
    )
    program_frame = aggregate_expression_programs(
        neighborhood_summary=neighborhood_summary,
        program_scores=program_scores,
        runtime_info=runtime_info,
    )
    feature_frame = pd.concat(
        [
            neighborhood_summary.composition,
            program_frame,
            neighborhood_summary.density,
            neighborhood_summary.entropy,
        ],
        axis=1,
    )
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return MotifFeatureBundle(
        feature_frame=feature_frame,
        expression_program_metadata=program_metadata,
        library_normalization_mode=library_normalization_mode,
    )


def fit_sample_aware_tissue_motif_model(
    dataset: SpatialDataset,
    neighborhood_summary: NeighborhoodSummary,
    *,
    runtime_info: RuntimeInfo,
    n_expression_programs: int = 6,
    top_variable_genes: int = 256,
    feature_bundle: MotifFeatureBundle | None = None,
    train_sample_ids: list[str] | tuple[str, ...] | None = None,
    max_train_spots_per_sample: int = 2048,
    fixed_n_clusters: int | None = None,
    random_state: int = 7,
) -> SampleAwareMotifFitResult:
    feature_bundle = feature_bundle or build_tissue_motif_feature_bundle(
        dataset,
        neighborhood_summary,
        runtime_info=runtime_info,
        n_expression_programs=n_expression_programs,
        top_variable_genes=top_variable_genes,
        random_state=random_state,
    )
    raw_feature_frame = feature_bundle.feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sample_ids = dataset.obs["sample_id"].astype(str).to_numpy()
    normalized_feature_frame, normalization_summary = normalize_feature_frame_by_sample(
        raw_feature_frame,
        sample_ids=sample_ids,
    )
    train_idx, training_spots_per_sample = select_sample_balanced_training_indices(
        sample_ids=sample_ids,
        allowed_sample_ids=train_sample_ids,
        max_spots_per_sample=max_train_spots_per_sample,
        random_state=random_state,
    )
    normalized_matrix = normalized_feature_frame.to_numpy(dtype=np.float32, copy=False)
    train_matrix = normalized_matrix[train_idx]
    scaler_mean, scaler_scale = fit_standardization_stats(train_matrix)
    standardized_matrix = standardize_dense_matrix(
        normalized_matrix,
        mean=scaler_mean,
        scale=scaler_scale,
    )
    train_standardized = standardized_matrix[train_idx]
    n_pca_components = int(max(2, min(12, train_standardized.shape[0] - 1, train_standardized.shape[1])))
    train_embedding, pca_projection = fit_linear_pca_projection(
        train_standardized,
        n_components=n_pca_components,
        runtime_info=runtime_info,
        random_state=random_state,
    )
    representation_embedding = transform_linear_pca_projection(
        standardized_matrix,
        projection=pca_projection,
    )
    cluster_dim = int(min(8, max(2, representation_embedding.shape[1])))
    train_cluster_embedding = train_embedding[:, :cluster_dim]
    n_clusters = int(fixed_n_clusters) if fixed_n_clusters is not None else int(
        choose_cluster_count(train_cluster_embedding, random_state=random_state)
    )
    cluster_model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    train_cluster_ids = cluster_model.fit_predict(train_cluster_embedding)
    embedding_centroids = cluster_model.cluster_centers_.astype(np.float32, copy=False)
    cluster_ids, assignment_distance = assign_nearest_centroids(
        representation_embedding[:, :cluster_dim],
        embedding_centroids,
        runtime_info=runtime_info,
    )
    motif_catalog = tuple(f"motif_{cluster_id:02d}" for cluster_id in range(n_clusters))
    motif_ids = np.asarray([motif_catalog[cluster_id] for cluster_id in cluster_ids.tolist()], dtype=object)
    layout_2d, layout_method = compute_layout_2d(representation_embedding, random_state=random_state)
    motif_label_map = label_motifs(
        motif_ids=motif_ids,
        feature_frame=raw_feature_frame,
        neighborhood_summary=neighborhood_summary,
    )

    spot_table = dataset.obs.copy()
    spot_table["motif_cluster"] = cluster_ids.astype(np.int64, copy=False)
    spot_table["motif_id"] = motif_ids
    spot_table["motif_label"] = pd.Series(motif_ids, index=spot_table.index).map(motif_label_map).astype(str)
    spot_table["layout_1"] = layout_2d[:, 0].astype(np.float32, copy=False)
    spot_table["layout_2"] = layout_2d[:, 1].astype(np.float32, copy=False)
    spot_table["layout_method"] = layout_method
    spot_table["representation_method"] = "sample_aware_pca"
    spot_table["assignment_distance"] = assignment_distance.astype(np.float32, copy=False)

    motif_metadata = (
        spot_table.groupby(["motif_id", "motif_label"], observed=False)
        .size()
        .rename("n_spots")
        .reset_index()
        .sort_values("motif_id")
        .reset_index(drop=True)
    )
    coherence_observed, coherence_perm_mean, coherence_zscore = compute_spatial_coherence(
        motif_ids=motif_ids,
        adjacency=neighborhood_summary.scales[neighborhood_summary.core_scale_name].adjacency,
        sample_ids=spot_table["sample_id"].astype(str).to_numpy(),
        random_state=random_state,
    )
    normalized_feature_centroids = (
        pd.DataFrame(normalized_matrix[train_idx], columns=normalized_feature_frame.columns)
        .assign(cluster_id=train_cluster_ids.astype(np.int64, copy=False))
        .groupby("cluster_id", observed=False)
        .mean(numeric_only=True)
        .reindex(range(n_clusters))
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    model_metadata = {
        "training_mode": "sample_balanced_per_sample",
        "normalization_mode": "per_sample_zscore_all_features",
        "library_normalization_mode": feature_bundle.library_normalization_mode,
        "n_expression_programs": int(feature_bundle.expression_program_metadata.shape[0]),
        "top_variable_genes": int(min(top_variable_genes, dataset.expression.shape[1])),
        "n_clusters": int(n_clusters),
        "core_scale": neighborhood_summary.core_scale_name,
        "train_sample_ids": sorted(set(dataset.obs.iloc[train_idx]["sample_id"].astype(str).tolist())),
        "training_spots_per_sample": int(training_spots_per_sample),
        "n_training_spots": int(train_idx.size),
        "pca_backend": pca_projection.backend,
        "pca_explained_variance_ratio": pca_projection.explained_variance_ratio.astype(float).tolist(),
        "representation_method": "sample_aware_pca",
        "representation_dim": int(representation_embedding.shape[1]),
        "cluster_dim": int(cluster_dim),
        "assignment_distance_mean": float(np.mean(assignment_distance)),
    }
    embedding_result = MotifEmbeddingResult(
        dataset_id=dataset.dataset_id,
        dataset_name=dataset.dataset_name,
        feature_frame=raw_feature_frame,
        pca_embedding=representation_embedding,
        representation_name="sample_aware_pca",
        representation_embedding=representation_embedding,
        layout_2d=layout_2d,
        layout_method=layout_method,
        spot_table=spot_table,
        motif_metadata=motif_metadata,
        expression_program_metadata=feature_bundle.expression_program_metadata,
        representation_training_history=None,
        n_clusters=n_clusters,
        spatial_coherence_observed=float(coherence_observed),
        spatial_coherence_perm_mean=float(coherence_perm_mean),
        spatial_coherence_zscore=float(coherence_zscore),
        model_metadata=model_metadata,
    )
    frozen_model = FrozenSampleAwareMotifModel(
        feature_columns=tuple(raw_feature_frame.columns.astype(str).tolist()),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        pca_projection=pca_projection,
        embedding_centroids=embedding_centroids,
        normalized_feature_centroids=normalized_feature_centroids,
        cluster_dim=cluster_dim,
        motif_ids=motif_catalog,
        motif_label_map=motif_label_map,
        training_row_indices=train_idx.astype(np.int64, copy=False),
        training_sample_ids=tuple(sorted(set(dataset.obs.iloc[train_idx]["sample_id"].astype(str).tolist()))),
        training_spots_per_sample=int(training_spots_per_sample),
        representation_name="sample_aware_pca",
        normalization_mode="per_sample_zscore_all_features",
        library_normalization_mode=feature_bundle.library_normalization_mode,
        model_metadata=model_metadata,
    )
    return SampleAwareMotifFitResult(
        embedding_result=embedding_result,
        frozen_model=frozen_model,
        normalized_feature_frame=normalized_feature_frame,
        normalization_summary=normalization_summary,
    )


def assign_sample_aware_motifs(
    dataset: SpatialDataset,
    neighborhood_summary: NeighborhoodSummary,
    *,
    frozen_model: FrozenSampleAwareMotifModel,
    runtime_info: RuntimeInfo,
    feature_bundle: MotifFeatureBundle | None = None,
    selected_sample_ids: list[str] | tuple[str, ...] | None = None,
    motif_id_map: dict[str, str] | None = None,
    motif_label_map: dict[str, str] | None = None,
    n_expression_programs: int = 6,
    top_variable_genes: int = 256,
    random_state: int = 7,
) -> pd.DataFrame:
    feature_bundle = feature_bundle or build_tissue_motif_feature_bundle(
        dataset,
        neighborhood_summary,
        runtime_info=runtime_info,
        n_expression_programs=n_expression_programs,
        top_variable_genes=top_variable_genes,
        random_state=random_state,
    )
    raw_feature_frame = feature_bundle.feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sample_ids = dataset.obs["sample_id"].astype(str).to_numpy()
    normalized_feature_frame, _ = normalize_feature_frame_by_sample(
        raw_feature_frame,
        sample_ids=sample_ids,
    )
    if selected_sample_ids is None:
        mask = np.ones(sample_ids.shape[0], dtype=bool)
    else:
        selected_set = {str(sample_id) for sample_id in selected_sample_ids}
        mask = np.asarray([sample_id in selected_set for sample_id in sample_ids], dtype=bool)
    if not np.any(mask):
        return dataset.obs.iloc[0:0].copy()
    selected_matrix = normalized_feature_frame.loc[mask, list(frozen_model.feature_columns)].to_numpy(dtype=np.float32, copy=False)
    standardized_matrix = standardize_dense_matrix(
        selected_matrix,
        mean=frozen_model.scaler_mean,
        scale=frozen_model.scaler_scale,
    )
    representation_embedding = transform_linear_pca_projection(
        standardized_matrix,
        projection=frozen_model.pca_projection,
    )
    cluster_ids, assignment_distance = assign_nearest_centroids(
        representation_embedding[:, : frozen_model.cluster_dim],
        frozen_model.embedding_centroids,
        runtime_info=runtime_info,
    )
    source_motif_ids = np.asarray([frozen_model.motif_ids[cluster_id] for cluster_id in cluster_ids.tolist()], dtype=object)
    if motif_id_map:
        final_motif_ids = np.asarray([motif_id_map.get(str(motif_id), str(motif_id)) for motif_id in source_motif_ids.tolist()], dtype=object)
    else:
        final_motif_ids = source_motif_ids
    label_lookup = dict(frozen_model.motif_label_map)
    if motif_label_map:
        label_lookup.update({str(key): str(value) for key, value in motif_label_map.items()})
    spot_table = dataset.obs.loc[mask].copy()
    spot_table["motif_cluster"] = cluster_ids.astype(np.int64, copy=False)
    spot_table["motif_id"] = final_motif_ids
    spot_table["motif_label"] = pd.Series(final_motif_ids, index=spot_table.index).map(label_lookup).fillna(
        pd.Series(final_motif_ids, index=spot_table.index)
    ).astype(str)
    spot_table["layout_1"] = representation_embedding[:, 0].astype(np.float32, copy=False)
    spot_table["layout_2"] = representation_embedding[:, 1].astype(np.float32, copy=False)
    spot_table["layout_method"] = "projection_only"
    spot_table["representation_method"] = frozen_model.representation_name
    spot_table["assignment_distance"] = assignment_distance.astype(np.float32, copy=False)
    return spot_table.copy()


def align_sample_aware_motif_model_to_reference(
    frozen_model: FrozenSampleAwareMotifModel,
    reference_model: FrozenSampleAwareMotifModel,
) -> tuple[dict[str, str], float]:
    if len(frozen_model.motif_ids) != len(reference_model.motif_ids):
        raise ValueError("Frozen sample-aware models must expose the same number of motifs for alignment.")
    cost_matrix = pairwise_squared_distance(
        frozen_model.normalized_feature_centroids,
        reference_model.normalized_feature_centroids,
    )
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    motif_map = {
        str(frozen_model.motif_ids[row]): str(reference_model.motif_ids[col])
        for row, col in zip(row_idx.tolist(), col_idx.tolist(), strict=False)
    }
    mean_cost = float(np.mean(cost_matrix[row_idx, col_idx])) if row_idx.size else float("nan")
    return motif_map, mean_cost


def choose_representation(
    *,
    representation_method: str,
    feature_frame: pd.DataFrame,
    pca_embedding: np.ndarray,
    adjacency: sparse.csr_matrix,
    runtime_info: RuntimeInfo,
    sample_ids: np.ndarray,
    condition_ids: np.ndarray,
    representation_config: GraphSSLConfig | dict[str, Any] | None,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame | None, dict[str, Any]]:
    if representation_method == "baseline_pca":
        return pca_embedding, None, {"kind": "baseline_pca"}
    if representation_method == "graph_ssl":
        if isinstance(representation_config, GraphSSLConfig):
            ssl_config = representation_config
        else:
            config_payload = dict(representation_config or {})
            config_payload.setdefault("random_state", random_state)
            ssl_config = GraphSSLConfig.from_dict(config_payload)
        ssl_result = train_graph_context_embedding(
            feature_frame,
            adjacency,
            runtime_info=runtime_info,
            sample_ids=sample_ids,
            condition_ids=condition_ids,
            config=ssl_config,
        )
        return ssl_result.embedding, ssl_result.training_history, ssl_result.model_metadata
    raise ValueError(f"Unsupported representation_method: {representation_method!r}")


def prepare_expression_matrix_for_program_scoring(dataset: SpatialDataset) -> tuple[sparse.csr_matrix, str]:
    matrix = sparse.csr_matrix(dataset.expression, dtype=np.float32)
    if str(dataset.expression_layer).strip().lower() == "lognorm":
        return matrix, "precomputed_lognorm_input"
    library_size = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32, copy=False)
    scale = np.divide(
        np.full(matrix.shape[0], 1.0e4, dtype=np.float32),
        np.maximum(library_size, 1.0),
        out=np.ones(matrix.shape[0], dtype=np.float32),
        where=library_size > 0,
    )
    normalized = sparse.diags(scale).dot(matrix).tocsr()
    normalized.data = np.log1p(normalized.data.astype(np.float32, copy=False))
    return normalized, "on_the_fly_log1p_cp10k"


def normalize_feature_frame_by_sample(
    feature_frame: pd.DataFrame,
    *,
    sample_ids: np.ndarray,
    eps: float = 1.0e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = np.asarray(sample_ids, dtype=object)
    matrix = feature_frame.to_numpy(dtype=np.float32, copy=True)
    normalized = np.zeros_like(matrix, dtype=np.float32)
    summary_rows: list[dict[str, object]] = []
    for sample_id in sorted(set(sample_ids.tolist())):
        sample_mask = sample_ids == sample_id
        sample_matrix = matrix[sample_mask]
        if sample_matrix.size == 0:
            continue
        sample_mean = sample_matrix.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        sample_std = sample_matrix.std(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        zero_var_mask = sample_std <= eps
        safe_std = sample_std.copy()
        safe_std[zero_var_mask] = 1.0
        normalized[sample_mask] = (sample_matrix - sample_mean) / safe_std
        summary_rows.append(
            {
                "sample_id": str(sample_id),
                "n_spots": int(sample_matrix.shape[0]),
                "mean_abs_feature_mean_after_centering": float(np.mean(np.abs(normalized[sample_mask].mean(axis=0)))),
                "mean_feature_std_after_scaling": float(np.mean(normalized[sample_mask].std(axis=0))),
                "n_zero_variance_features": int(np.sum(zero_var_mask)),
            }
        )
    normalized_frame = pd.DataFrame(
        normalized,
        index=feature_frame.index,
        columns=feature_frame.columns,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return normalized_frame, pd.DataFrame(summary_rows).sort_values("sample_id").reset_index(drop=True)


def select_sample_balanced_training_indices(
    *,
    sample_ids: np.ndarray,
    allowed_sample_ids: list[str] | tuple[str, ...] | None,
    max_spots_per_sample: int,
    random_state: int,
) -> tuple[np.ndarray, int]:
    sample_ids = np.asarray(sample_ids, dtype=object)
    if allowed_sample_ids is None:
        selected_samples = sorted(set(sample_ids.tolist()))
    else:
        selected_set = {str(sample_id) for sample_id in allowed_sample_ids}
        selected_samples = [sample_id for sample_id in sorted(set(sample_ids.tolist())) if str(sample_id) in selected_set]
    if not selected_samples:
        raise ValueError("No samples are available for sample-balanced motif training.")
    sample_counts = {
        str(sample_id): int(np.sum(sample_ids == sample_id))
        for sample_id in selected_samples
    }
    training_spots_per_sample = int(min(min(sample_counts.values()), max(1, int(max_spots_per_sample))))
    if training_spots_per_sample <= 0:
        raise ValueError("Sample-balanced motif training requires at least one spot per sample.")
    rng = np.random.default_rng(random_state)
    selected_idx: list[np.ndarray] = []
    for sample_id in selected_samples:
        sample_idx = np.flatnonzero(sample_ids == sample_id)
        if sample_idx.size <= training_spots_per_sample:
            selected_idx.append(sample_idx.astype(np.int64, copy=False))
            continue
        draw = np.sort(rng.choice(sample_idx, size=training_spots_per_sample, replace=False))
        selected_idx.append(draw.astype(np.int64, copy=False))
    return np.concatenate(selected_idx).astype(np.int64, copy=False), training_spots_per_sample


def fit_standardization_stats(matrix: np.ndarray, *, eps: float = 1.0e-6) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(matrix, dtype=np.float32)
    mean = array.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    scale = array.std(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    scale = np.where(scale > eps, scale, 1.0).astype(np.float32, copy=False)
    return mean, scale


def standardize_dense_matrix(matrix: np.ndarray, *, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    centered = array - np.asarray(mean, dtype=np.float32)
    return np.divide(
        centered,
        np.asarray(scale, dtype=np.float32),
        out=np.zeros_like(centered, dtype=np.float32),
        where=np.asarray(scale, dtype=np.float32) > 0,
    ).astype(np.float32, copy=False)


def fit_linear_pca_projection(
    matrix: np.ndarray,
    *,
    n_components: int,
    runtime_info: RuntimeInfo,
    random_state: int,
) -> tuple[np.ndarray, PCAProjection]:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("PCA projection requires a non-empty 2D matrix.")
    effective_components = int(max(2, min(n_components, array.shape[0] - 1, array.shape[1])))
    if torch is not None and runtime_info.cuda_available:
        try:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)
            device = torch.device("cuda")
            x_tensor = torch.from_numpy(array).to(device, non_blocking=True)
            mean_tensor = x_tensor.mean(dim=0, keepdim=True)
            centered = x_tensor - mean_tensor
            u_mat, singular_vals, v_mat = torch.pca_lowrank(centered, q=effective_components, center=False)
            embedding = (u_mat[:, :effective_components] * singular_vals[:effective_components]).detach().cpu().numpy().astype(np.float32, copy=False)
            components = v_mat[:, :effective_components].T.detach().cpu().numpy().astype(np.float32, copy=False)
            mean = mean_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            total_variance = float(np.var(array, axis=0, ddof=1).sum()) if array.shape[0] > 1 else 0.0
            explained_variance = np.square(singular_vals[:effective_components].detach().cpu().numpy().astype(np.float64)) / max(array.shape[0] - 1, 1)
            if total_variance > 0.0:
                explained_ratio = (explained_variance / total_variance).astype(np.float32, copy=False)
            else:
                explained_ratio = np.zeros(effective_components, dtype=np.float32)
            return embedding, PCAProjection(
                components=components,
                mean=mean,
                explained_variance_ratio=explained_ratio,
                backend="torch_pca_cuda",
            )
        except RuntimeError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    pca_model = PCA(n_components=effective_components, random_state=random_state)
    embedding = pca_model.fit_transform(array).astype(np.float32, copy=False)
    return embedding, PCAProjection(
        components=pca_model.components_.astype(np.float32, copy=False),
        mean=pca_model.mean_.astype(np.float32, copy=False),
        explained_variance_ratio=pca_model.explained_variance_ratio_.astype(np.float32, copy=False),
        backend="sklearn_pca_cpu",
    )


def transform_linear_pca_projection(
    matrix: np.ndarray,
    *,
    projection: PCAProjection,
) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    centered = array - np.asarray(projection.mean, dtype=np.float32)
    return centered.dot(np.asarray(projection.components, dtype=np.float32).T).astype(np.float32, copy=False)


def assign_nearest_centroids(
    embedding: np.ndarray,
    centroids: np.ndarray,
    *,
    runtime_info: RuntimeInfo,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(embedding, dtype=np.float32)
    right = np.asarray(centroids, dtype=np.float32)
    if torch is not None and runtime_info.cuda_available:
        try:
            device = torch.device("cuda")
            left_tensor = torch.from_numpy(left).to(device, non_blocking=True)
            right_tensor = torch.from_numpy(right).to(device, non_blocking=True)
            distance = torch.cdist(left_tensor, right_tensor, p=2.0)
            cluster_ids = torch.argmin(distance, dim=1)
            min_distance = distance.gather(1, cluster_ids[:, None]).squeeze(1).pow(2)
            return (
                cluster_ids.detach().cpu().numpy().astype(np.int64, copy=False),
                min_distance.detach().cpu().numpy().astype(np.float32, copy=False),
            )
        except RuntimeError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    distance = pairwise_squared_distance(left, right)
    cluster_ids = np.argmin(distance, axis=1).astype(np.int64, copy=False)
    min_distance = distance[np.arange(distance.shape[0]), cluster_ids].astype(np.float32, copy=False)
    return cluster_ids, min_distance


def pairwise_squared_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_sq = np.sum(np.square(left), axis=1, keepdims=True)
    right_sq = np.sum(np.square(right), axis=1, keepdims=True).T
    return np.maximum(left_sq + right_sq - 2.0 * left.dot(right.T), 0.0)


def compute_expression_program_scores(
    expression: sparse.csr_matrix,
    var_names: np.ndarray,
    *,
    n_components: int = 6,
    top_variable_genes: int = 256,
    random_state: int = 7,
) -> tuple[np.ndarray, pd.DataFrame]:
    matrix = sparse.csr_matrix(expression, dtype=np.float32)
    n_obs, n_vars = matrix.shape
    top_k = int(max(8, min(top_variable_genes, n_vars)))
    mean_per_gene = np.asarray(matrix.mean(axis=0)).ravel()
    sq_mean_per_gene = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variance_per_gene = np.clip(sq_mean_per_gene - np.square(mean_per_gene), a_min=0.0, a_max=None)
    gene_idx = np.argsort(variance_per_gene)[-top_k:]
    gene_idx = np.sort(gene_idx.astype(np.int64, copy=False))
    subset = matrix[:, gene_idx]
    effective_components = int(max(2, min(n_components, subset.shape[0] - 1, subset.shape[1] - 1)))
    svd = TruncatedSVD(n_components=effective_components, random_state=random_state)
    scores = svd.fit_transform(subset).astype(np.float32, copy=False)
    metadata_rows: list[dict[str, object]] = []
    components = svd.components_.astype(np.float32, copy=False)
    for component_idx in range(components.shape[0]):
        top_loading_idx = np.argsort(np.abs(components[component_idx]))[-12:]
        top_genes = [str(var_names[gene_idx[index]]) for index in top_loading_idx[::-1]]
        metadata_rows.append(
            {
                "component": f"program_{component_idx:02d}",
                "explained_variance_ratio": float(svd.explained_variance_ratio_[component_idx]),
                "top_genes": ";".join(top_genes),
            }
        )
    return scores, pd.DataFrame(metadata_rows)


def aggregate_expression_programs(
    *,
    neighborhood_summary: NeighborhoodSummary,
    program_scores: np.ndarray,
    runtime_info: RuntimeInfo,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scale_name, scale in neighborhood_summary.scales.items():
        aggregated = aggregate_feature_matrix(scale.adjacency, program_scores, runtime_info=runtime_info)
        columns = [f"{scale_name}__program_{index:02d}" for index in range(aggregated.shape[1])]
        frames.append(pd.DataFrame(aggregated, index=neighborhood_summary.obs_index, columns=columns))
    return pd.concat(frames, axis=1)


def choose_cluster_count(pca_embedding: np.ndarray, *, random_state: int = 7) -> int:
    n_obs = pca_embedding.shape[0]
    upper = int(min(10, max(4, round(np.sqrt(max(n_obs, 1) / 600.0)) + 4)))
    candidates = [value for value in range(4, upper + 1) if value < n_obs]
    if not candidates:
        return max(2, min(4, n_obs))
    rng = np.random.default_rng(random_state)
    sample_size = int(min(2000, n_obs))
    if sample_size < n_obs:
        sample_idx = np.sort(rng.choice(n_obs, size=sample_size, replace=False))
        sample = pca_embedding[sample_idx, : min(8, pca_embedding.shape[1])]
    else:
        sample = pca_embedding[:, : min(8, pca_embedding.shape[1])]
    best_k = candidates[0]
    best_score = -np.inf
    for n_clusters in candidates:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=max(6144, sample.shape[0]),
            n_init=12,
            random_state=random_state,
        )
        labels = model.fit_predict(sample)
        if len(np.unique(labels)) < 2:
            continue
        score = calinski_harabasz_score(sample, labels) - (0.05 * n_clusters)
        if score > best_score:
            best_score = score
            best_k = n_clusters
    return int(best_k)


def compute_layout_2d(pca_embedding: np.ndarray, *, random_state: int = 7) -> tuple[np.ndarray, str]:
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.25,
            metric="euclidean",
            random_state=random_state,
        )
        return reducer.fit_transform(pca_embedding[:, : min(10, pca_embedding.shape[1])]).astype(np.float32, copy=False), "umap"
    except Exception:
        return pca_embedding[:, :2].astype(np.float32, copy=False), "pca_fallback"


def label_motifs(
    *,
    motif_ids: np.ndarray,
    feature_frame: pd.DataFrame,
    neighborhood_summary: NeighborhoodSummary,
) -> dict[str, str]:
    profile = feature_frame.copy()
    profile["motif_id"] = motif_ids
    motif_means = profile.groupby("motif_id", observed=False).mean(numeric_only=True)
    core_scale = neighborhood_summary.core_scale_name
    composition_cols = neighborhood_summary.composition_columns[core_scale]
    entropy_col = f"{core_scale}__entropy"
    density_col = f"{core_scale}__area_density"
    entropy_quantiles = np.quantile(
        motif_means[entropy_col].to_numpy(dtype=np.float32, copy=False),
        [0.33, 0.66],
    )
    density_quantiles = np.quantile(
        motif_means[density_col].to_numpy(dtype=np.float32, copy=False),
        [0.33, 0.66],
    )
    label_map: dict[str, str] = {}
    for motif_id, row in motif_means.iterrows():
        top_columns = row[composition_cols].sort_values(ascending=False).head(2).index.tolist()
        top_names = [neighborhood_summary.cell_type_lookup[column] for column in top_columns]
        if len(top_names) == 1:
            top_names = top_names + [top_names[0]]
        entropy_state = _quantile_state(float(row[entropy_col]), entropy_quantiles, labels=("lowE", "midE", "highE"))
        density_state = _quantile_state(float(row[density_col]), density_quantiles, labels=("sparse", "mixed", "dense"))
        label_map[str(motif_id)] = f"{top_names[0]} + {top_names[1]} | {density_state} {entropy_state}"
    return label_map


def compute_spatial_coherence(
    *,
    motif_ids: np.ndarray,
    adjacency: sparse.csr_matrix,
    sample_ids: np.ndarray,
    n_permutations: int = 100,
    random_state: int = 7,
) -> tuple[float, float, float]:
    categories = pd.Categorical(motif_ids)
    codes = np.asarray(categories.codes, dtype=np.int64)
    one_hot = np.zeros((codes.shape[0], len(categories.categories)), dtype=np.float32)
    one_hot[np.arange(codes.shape[0]), codes] = 1.0
    observed_matrix = np.asarray(adjacency.dot(one_hot), dtype=np.float32)
    observed = float(np.mean(observed_matrix[np.arange(codes.shape[0]), codes]))

    rng = np.random.default_rng(random_state)
    permutation_scores = np.zeros(n_permutations, dtype=np.float32)
    unique_samples = np.unique(sample_ids.astype(str))
    for permutation_idx in range(n_permutations):
        perm_codes = codes.copy()
        for sample_id in unique_samples:
            sample_mask = sample_ids == sample_id
            perm_codes[sample_mask] = rng.permutation(perm_codes[sample_mask])
        perm_one_hot = np.zeros_like(one_hot)
        perm_one_hot[np.arange(perm_codes.shape[0]), perm_codes] = 1.0
        perm_matrix = np.asarray(adjacency.dot(perm_one_hot), dtype=np.float32)
        permutation_scores[permutation_idx] = float(np.mean(perm_matrix[np.arange(perm_codes.shape[0]), perm_codes]))
    perm_mean = float(permutation_scores.mean())
    perm_std = float(permutation_scores.std(ddof=0))
    z_score = 0.0 if perm_std <= 1.0e-8 else float((observed - perm_mean) / perm_std)
    return observed, perm_mean, z_score


def _quantile_state(value: float, quantiles: np.ndarray, *, labels: tuple[str, str, str]) -> str:
    lower, upper = float(quantiles[0]), float(quantiles[1])
    if value <= lower:
        return labels[0]
    if value <= upper:
        return labels[1]
    return labels[2]
