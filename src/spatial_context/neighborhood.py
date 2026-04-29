from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is optional for import but expected in this repo
    torch = None


@dataclass(frozen=True)
class RuntimeInfo:
    device: str
    cuda_available: bool
    cuda_count: int
    cuda_name: str
    torch_version: str


@dataclass(frozen=True)
class SpatialDataset:
    path: Path
    dataset_id: str
    dataset_name: str
    obs: pd.DataFrame
    var_names: np.ndarray
    expression: sparse.csr_matrix
    spatial: np.ndarray
    expression_layer: str


@dataclass(frozen=True)
class NeighborhoodScale:
    name: str
    adjacency: sparse.csr_matrix
    neighbor_count: np.ndarray
    mean_distance: np.ndarray
    support_radius: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class NeighborhoodSummary:
    dataset_id: str
    dataset_name: str
    obs_index: pd.Index
    scales: dict[str, NeighborhoodScale]
    core_scale_name: str
    composition: pd.DataFrame
    entropy: pd.DataFrame
    density: pd.DataFrame
    cell_type_categories: tuple[str, ...]
    composition_columns: dict[str, list[str]]
    cell_type_lookup: dict[str, str]


def get_runtime_info() -> RuntimeInfo:
    if torch is None:
        return RuntimeInfo(
            device="cpu",
            cuda_available=False,
            cuda_count=0,
            cuda_name="cpu",
            torch_version="unavailable",
        )
    cuda_available = bool(torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    return RuntimeInfo(
        device="cuda" if cuda_available else "cpu",
        cuda_available=cuda_available,
        cuda_count=cuda_count,
        cuda_name=str(cuda_name),
        torch_version=str(torch.__version__),
    )


def load_spatial_h5ad(
    path: str | Path,
    *,
    expression_layer: str = "lognorm",
    obs_columns: tuple[str, ...] = (
        "sample_id",
        "condition",
        "cell_type",
        "cluster",
        "dataset_id",
        "dataset_name",
        "patient_id",
        "spot_barcode",
        "spatial_x",
        "spatial_y",
        "n_counts",
        "n_genes_by_counts",
    ),
) -> SpatialDataset:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        obs = _read_dataframe(handle["obs"], columns=obs_columns)
        var_names = _read_string_array(handle["var"]["feature_name"][...])
        if "layers" in handle and expression_layer in handle["layers"]:
            expression = _read_csr_matrix(handle["layers"][expression_layer])
            layer_name = expression_layer
        else:
            expression = _read_csr_matrix(handle["X"])
            layer_name = "X"

        if {"spatial_x", "spatial_y"}.issubset(obs.columns):
            spatial = obs.loc[:, ["spatial_x", "spatial_y"]].to_numpy(dtype=np.float32, copy=True)
        else:
            spatial = np.asarray(handle["obsm"]["spatial"][...], dtype=np.float32)
            obs["spatial_x"] = spatial[:, 0]
            obs["spatial_y"] = spatial[:, 1]

    obs["sample_id"] = _string_series_or_default(obs, "sample_id", "sample_0")
    obs["condition"] = _string_series_or_default(obs, "condition", "unknown")
    obs["cell_type"] = _string_series_or_default(obs, "cell_type", "unassigned")
    obs["cluster"] = _string_series_or_default(obs, "cluster", "")
    obs["dataset_id"] = _string_series_or_default(obs, "dataset_id", path.stem)
    obs["dataset_name"] = _string_series_or_default(obs, "dataset_name", path.stem)
    if "spot_barcode" in obs.columns:
        obs["spot_barcode"] = _string_series_or_default(obs, "spot_barcode", "")
    else:
        obs["spot_barcode"] = obs.index.astype(str)
    if "n_counts" in obs.columns:
        obs["n_counts"] = pd.to_numeric(obs["n_counts"], errors="coerce").fillna(0.0).astype(np.float32)
    if "n_genes_by_counts" in obs.columns:
        obs["n_genes_by_counts"] = pd.to_numeric(obs["n_genes_by_counts"], errors="coerce").fillna(0.0).astype(np.float32)
    obs["spatial_x"] = pd.to_numeric(obs["spatial_x"], errors="coerce").astype(np.float32)
    obs["spatial_y"] = pd.to_numeric(obs["spatial_y"], errors="coerce").astype(np.float32)
    dataset_id = str(obs["dataset_id"].iloc[0])
    dataset_name = str(obs["dataset_name"].iloc[0])
    return SpatialDataset(
        path=path,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        obs=obs,
        var_names=np.asarray(var_names, dtype=object),
        expression=sparse.csr_matrix(expression, dtype=np.float32),
        spatial=np.asarray(spatial, dtype=np.float32),
        expression_layer=layer_name,
    )


def load_spatial_h5ad_metadata_only(
    path: str | Path,
    *,
    obs_columns: tuple[str, ...] = (
        "sample_id",
        "condition",
        "cell_type",
        "cluster",
        "dataset_id",
        "dataset_name",
        "patient_id",
        "spot_barcode",
        "spatial_x",
        "spatial_y",
        "n_counts",
        "n_genes_by_counts",
    ),
) -> SpatialDataset:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        obs = _read_dataframe(handle["obs"], columns=obs_columns)
        if {"spatial_x", "spatial_y"}.issubset(obs.columns):
            spatial = obs.loc[:, ["spatial_x", "spatial_y"]].to_numpy(dtype=np.float32, copy=True)
        else:
            spatial = np.asarray(handle["obsm"]["spatial"][...], dtype=np.float32)
            obs["spatial_x"] = spatial[:, 0]
            obs["spatial_y"] = spatial[:, 1]

    obs["sample_id"] = _string_series_or_default(obs, "sample_id", "sample_0")
    obs["condition"] = _string_series_or_default(obs, "condition", "unknown")
    obs["cell_type"] = _string_series_or_default(obs, "cell_type", "unassigned")
    obs["cluster"] = _string_series_or_default(obs, "cluster", "")
    obs["dataset_id"] = _string_series_or_default(obs, "dataset_id", path.stem)
    obs["dataset_name"] = _string_series_or_default(obs, "dataset_name", path.stem)
    if "spot_barcode" in obs.columns:
        obs["spot_barcode"] = _string_series_or_default(obs, "spot_barcode", "")
    else:
        obs["spot_barcode"] = obs.index.astype(str)
    if "n_counts" in obs.columns:
        obs["n_counts"] = pd.to_numeric(obs["n_counts"], errors="coerce").fillna(0.0).astype(np.float32)
    if "n_genes_by_counts" in obs.columns:
        obs["n_genes_by_counts"] = pd.to_numeric(obs["n_genes_by_counts"], errors="coerce").fillna(0.0).astype(np.float32)
    obs["spatial_x"] = pd.to_numeric(obs["spatial_x"], errors="coerce").astype(np.float32)
    obs["spatial_y"] = pd.to_numeric(obs["spatial_y"], errors="coerce").astype(np.float32)
    dataset_id = str(obs["dataset_id"].iloc[0])
    dataset_name = str(obs["dataset_name"].iloc[0])
    empty_expression = sparse.csr_matrix((obs.shape[0], 0), dtype=np.float32)
    return SpatialDataset(
        path=path,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        obs=obs,
        var_names=np.asarray([], dtype=object),
        expression=empty_expression,
        spatial=np.asarray(spatial, dtype=np.float32),
        expression_layer="metadata_only",
    )


def summarize_neighborhoods(
    dataset: SpatialDataset,
    *,
    runtime_info: RuntimeInfo,
    sample_key: str = "sample_id",
    cell_type_key: str = "cell_type",
    k_neighbors: tuple[int, ...] = (6, 18),
    radius_factor: float = 1.6,
    radius_reference_k: int = 6,
) -> NeighborhoodSummary:
    scales = build_multiscale_neighborhoods(
        dataset,
        sample_key=sample_key,
        k_neighbors=k_neighbors,
        radius_factor=radius_factor,
        radius_reference_k=radius_reference_k,
    )
    categories = tuple(sorted(dataset.obs[cell_type_key].astype(str).fillna("unassigned").unique().tolist()))
    codes = pd.Categorical(dataset.obs[cell_type_key].astype(str), categories=categories)
    code_values = np.asarray(codes.codes, dtype=np.int64)
    one_hot = np.zeros((dataset.obs.shape[0], len(categories)), dtype=np.float32)
    valid = code_values >= 0
    one_hot[np.flatnonzero(valid), code_values[valid]] = 1.0

    composition_frames: list[pd.DataFrame] = []
    composition_columns: dict[str, list[str]] = {}
    cell_type_lookup: dict[str, str] = {}
    entropy_data: dict[str, np.ndarray] = {}
    density_data: dict[str, np.ndarray] = {}

    for scale_name, scale in scales.items():
        composition = aggregate_feature_matrix(scale.adjacency, one_hot, runtime_info=runtime_info)
        columns: list[str] = []
        for category in categories:
            feature_name = f"{scale_name}__ct__{safe_feature_name(category)}"
            columns.append(feature_name)
            cell_type_lookup[feature_name] = category
        composition_columns[scale_name] = columns
        composition_frames.append(pd.DataFrame(composition, index=dataset.obs.index, columns=columns))
        entropy_data[f"{scale_name}__entropy"] = compute_normalized_entropy(composition)
        density_data[f"{scale_name}__neighbor_count"] = scale.neighbor_count.astype(np.float32, copy=False)
        density_data[f"{scale_name}__mean_distance"] = scale.mean_distance.astype(np.float32, copy=False)
        density_data[f"{scale_name}__area_density"] = np.divide(
            scale.neighbor_count,
            np.maximum(np.pi * np.square(scale.support_radius), 1.0e-6),
            out=np.zeros_like(scale.neighbor_count, dtype=np.float32),
            where=scale.support_radius > 0,
        ).astype(np.float32, copy=False)

    core_scale_name = min(
        (name for name in scales if name.startswith("knn_")),
        key=lambda value: int(value.split("_", maxsplit=1)[1]),
        default=next(iter(scales)),
    )
    composition_df = pd.concat(composition_frames, axis=1)
    entropy_df = pd.DataFrame(entropy_data, index=dataset.obs.index)
    density_df = pd.DataFrame(density_data, index=dataset.obs.index)
    return NeighborhoodSummary(
        dataset_id=dataset.dataset_id,
        dataset_name=dataset.dataset_name,
        obs_index=dataset.obs.index,
        scales=scales,
        core_scale_name=core_scale_name,
        composition=composition_df,
        entropy=entropy_df,
        density=density_df,
        cell_type_categories=categories,
        composition_columns=composition_columns,
        cell_type_lookup=cell_type_lookup,
    )


def build_multiscale_neighborhoods(
    dataset: SpatialDataset,
    *,
    sample_key: str = "sample_id",
    k_neighbors: tuple[int, ...] = (6, 18),
    radius_factor: float = 1.6,
    radius_reference_k: int = 6,
) -> dict[str, NeighborhoodScale]:
    n_obs = dataset.obs.shape[0]
    scale_payloads: dict[str, dict[str, list[np.ndarray]]] = {}
    for k in sorted(set(int(value) for value in k_neighbors)):
        scale_payloads[f"knn_{k:02d}"] = {
            "rows": [],
            "cols": [],
            "data": [],
            "neighbor_count": [],
            "mean_distance": [],
            "support_radius": [],
        }
    scale_payloads["radius"] = {
        "rows": [],
        "cols": [],
        "data": [],
        "neighbor_count": [],
        "mean_distance": [],
        "support_radius": [],
    }
    radius_per_sample: dict[str, float] = {}

    sample_values = dataset.obs[sample_key].astype(str).to_numpy()
    for sample_id in sorted(set(sample_values.tolist())):
        sample_idx = np.flatnonzero(sample_values == sample_id)
        coords = dataset.spatial[sample_idx]
        radius = infer_radius_from_knn(coords, k=radius_reference_k, scale_factor=radius_factor)
        radius_per_sample[sample_id] = radius
        for k in sorted(set(int(value) for value in k_neighbors)):
            local_adj, local_count, local_mean_distance, local_support_radius = _build_knn_local(coords, k=k)
            _append_scale_payload(
                payload=scale_payloads[f"knn_{k:02d}"],
                local_adjacency=local_adj,
                sample_idx=sample_idx,
                neighbor_count=local_count,
                mean_distance=local_mean_distance,
                support_radius=local_support_radius,
            )
        local_adj, local_count, local_mean_distance, local_support_radius = _build_radius_local(coords, radius=radius)
        _append_scale_payload(
            payload=scale_payloads["radius"],
            local_adjacency=local_adj,
            sample_idx=sample_idx,
            neighbor_count=local_count,
            mean_distance=local_mean_distance,
            support_radius=local_support_radius,
        )

    scales: dict[str, NeighborhoodScale] = {}
    for scale_name, payload in scale_payloads.items():
        rows = np.concatenate(payload["rows"]) if payload["rows"] else np.empty(0, dtype=np.int64)
        cols = np.concatenate(payload["cols"]) if payload["cols"] else np.empty(0, dtype=np.int64)
        data = np.concatenate(payload["data"]) if payload["data"] else np.empty(0, dtype=np.float32)
        adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(n_obs, n_obs), dtype=np.float32)
        scales[scale_name] = NeighborhoodScale(
            name=scale_name,
            adjacency=adjacency,
            neighbor_count=np.concatenate(payload["neighbor_count"]).astype(np.float32, copy=False),
            mean_distance=np.concatenate(payload["mean_distance"]).astype(np.float32, copy=False),
            support_radius=np.concatenate(payload["support_radius"]).astype(np.float32, copy=False),
            metadata={
                "kind": "radius" if scale_name == "radius" else "knn",
                "radius_factor": float(radius_factor),
                "radius_reference_k": int(radius_reference_k),
                "sample_radius_mean": float(np.mean(list(radius_per_sample.values()))) if radius_per_sample else 0.0,
            },
        )
    return scales


def aggregate_feature_matrix(
    adjacency: sparse.csr_matrix,
    features: np.ndarray,
    *,
    runtime_info: RuntimeInfo,
) -> np.ndarray:
    feature_matrix = np.asarray(features, dtype=np.float32)
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix[:, None]
    if torch is not None and runtime_info.cuda_available:
        try:
            coo = adjacency.tocoo()
            indices = np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
            values = coo.data.astype(np.float32, copy=False)
            sparse_tensor = torch.sparse_coo_tensor(
                torch.from_numpy(indices),
                torch.from_numpy(values),
                size=adjacency.shape,
                device="cuda",
            ).coalesce()
            feature_tensor = torch.as_tensor(feature_matrix, dtype=torch.float32, device="cuda")
            aggregated = torch.sparse.mm(sparse_tensor, feature_tensor)
            return aggregated.detach().cpu().numpy()
        except RuntimeError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.asarray(adjacency.dot(feature_matrix), dtype=np.float32)


def compute_normalized_entropy(composition: np.ndarray) -> np.ndarray:
    probs = np.asarray(composition, dtype=np.float32)
    probs = np.clip(probs, 1.0e-8, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = math.log(max(probs.shape[1], 2))
    return np.asarray(entropy / max_entropy, dtype=np.float32)


def safe_feature_name(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_").lower()
    return normalized or "unknown"


def infer_radius_from_knn(coords: np.ndarray, *, k: int, scale_factor: float) -> float:
    if coords.shape[0] <= 1:
        return 1.0
    effective_k = int(max(1, min(k, coords.shape[0] - 1)))
    model = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    model.fit(coords)
    distances, _ = model.kneighbors(coords)
    kth_distance = distances[:, effective_k]
    positive = kth_distance[kth_distance > 0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive) * scale_factor)


def _append_scale_payload(
    *,
    payload: dict[str, list[np.ndarray]],
    local_adjacency: sparse.csr_matrix,
    sample_idx: np.ndarray,
    neighbor_count: np.ndarray,
    mean_distance: np.ndarray,
    support_radius: np.ndarray,
) -> None:
    coo = local_adjacency.tocoo()
    payload["rows"].append(sample_idx[coo.row].astype(np.int64, copy=False))
    payload["cols"].append(sample_idx[coo.col].astype(np.int64, copy=False))
    payload["data"].append(coo.data.astype(np.float32, copy=False))
    payload["neighbor_count"].append(neighbor_count.astype(np.float32, copy=False))
    payload["mean_distance"].append(mean_distance.astype(np.float32, copy=False))
    payload["support_radius"].append(support_radius.astype(np.float32, copy=False))


def _build_knn_local(coords: np.ndarray, *, k: int) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    n_obs = coords.shape[0]
    if n_obs <= 1:
        adjacency = sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 1), dtype=np.float32)
        return adjacency, np.asarray([1.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32)
    effective_k = int(max(1, min(k, n_obs - 1)))
    model = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    model.fit(coords)
    distances, indices = model.kneighbors(coords)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    neighbor_count = np.zeros(n_obs, dtype=np.float32)
    mean_distance = np.zeros(n_obs, dtype=np.float32)
    support_radius = np.zeros(n_obs, dtype=np.float32)
    for row_idx in range(n_obs):
        neighbors = indices[row_idx, 1:]
        neighbor_distances = distances[row_idx, 1:]
        weights = np.divide(
            np.ones_like(neighbor_distances, dtype=np.float32),
            np.maximum(neighbor_distances.astype(np.float32, copy=False), 1.0e-3),
            out=np.ones_like(neighbor_distances, dtype=np.float32),
        )
        rows.append(np.full(neighbors.shape[0], row_idx, dtype=np.int64))
        cols.append(neighbors.astype(np.int64, copy=False))
        values.append(weights.astype(np.float32, copy=False))
        neighbor_count[row_idx] = float(neighbors.shape[0])
        mean_distance[row_idx] = float(neighbor_distances.mean()) if neighbor_distances.size else 0.0
        support_radius[row_idx] = float(neighbor_distances.max()) if neighbor_distances.size else 1.0
    adjacency = sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_obs, n_obs),
        dtype=np.float32,
    )
    return row_normalize_csr(adjacency), neighbor_count, mean_distance, support_radius


def _build_radius_local(coords: np.ndarray, *, radius: float) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    n_obs = coords.shape[0]
    if n_obs <= 1:
        adjacency = sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 1), dtype=np.float32)
        return adjacency, np.asarray([1.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32)

    fallback_model = NearestNeighbors(n_neighbors=2, metric="euclidean")
    fallback_model.fit(coords)
    fallback_distances, fallback_indices = fallback_model.kneighbors(coords)

    model = NearestNeighbors(radius=radius, metric="euclidean")
    model.fit(coords)
    all_distances, all_indices = model.radius_neighbors(coords, return_distance=True, sort_results=True)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    neighbor_count = np.zeros(n_obs, dtype=np.float32)
    mean_distance = np.zeros(n_obs, dtype=np.float32)
    support_radius = np.zeros(n_obs, dtype=np.float32)
    for row_idx in range(n_obs):
        indices = np.asarray(all_indices[row_idx], dtype=np.int64)
        distances = np.asarray(all_distances[row_idx], dtype=np.float32)
        keep = indices != row_idx
        indices = indices[keep]
        distances = distances[keep]
        if indices.size == 0:
            indices = np.asarray([fallback_indices[row_idx, 1]], dtype=np.int64)
            distances = np.asarray([fallback_distances[row_idx, 1]], dtype=np.float32)
        weights = np.divide(
            np.ones_like(distances, dtype=np.float32),
            np.maximum(distances, 1.0e-3),
            out=np.ones_like(distances, dtype=np.float32),
        )
        rows.append(np.full(indices.shape[0], row_idx, dtype=np.int64))
        cols.append(indices.astype(np.int64, copy=False))
        values.append(weights.astype(np.float32, copy=False))
        neighbor_count[row_idx] = float(indices.shape[0])
        mean_distance[row_idx] = float(distances.mean()) if distances.size else 0.0
        support_radius[row_idx] = float(max(radius, distances.max())) if distances.size else float(radius)
    adjacency = sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_obs, n_obs),
        dtype=np.float32,
    )
    return row_normalize_csr(adjacency), neighbor_count, mean_distance, support_radius


def row_normalize_csr(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32, copy=False)
    row_sum = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    row_sum = np.where(row_sum > 0, row_sum, 1.0).astype(np.float32, copy=False)
    inv = sparse.diags(1.0 / row_sum)
    return inv.dot(matrix).tocsr()


def _read_dataframe(group: h5py.Group, *, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    index_name = _decode_scalar(group.attrs.get("_index", "index"))
    index_values = _as_index_values(_read_node(group[index_name]))
    frame = pd.DataFrame(index=pd.Index(index_values, name=index_name))
    column_order = [_decode_scalar(value) for value in group.attrs.get("column-order", list(group.keys()))]
    for column_name in column_order:
        if column_name == index_name:
            continue
        if columns is not None and column_name not in columns:
            continue
        frame[column_name] = _read_node(group[column_name])
    if columns is not None:
        for column_name in columns:
            if column_name not in frame.columns and column_name in group:
                frame[column_name] = _read_node(group[column_name])
    return frame


def _read_csr_matrix(group: h5py.Group) -> sparse.csr_matrix:
    shape = tuple(int(value) for value in group.attrs["shape"])
    data = np.asarray(group["data"][...], dtype=np.float32)
    indices = np.asarray(group["indices"][...], dtype=np.int32)
    indptr = np.asarray(group["indptr"][...], dtype=np.int64)
    return sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32)


def _read_node(node: h5py.Dataset | h5py.Group) -> Any:
    if isinstance(node, h5py.Dataset):
        raw = node[...]
        if getattr(raw, "dtype", None) is not None and raw.dtype.kind in {"O", "S", "U"}:
            return _read_string_array(raw)
        return raw
    encoding_type = _decode_scalar(node.attrs.get("encoding-type", ""))
    if encoding_type == "categorical":
        categories = _read_string_array(node["categories"][...])
        codes = np.asarray(node["codes"][...], dtype=np.int64)
        values = np.empty(codes.shape[0], dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        values[~valid] = None
        return pd.Categorical(values, categories=list(categories), ordered=bool(node.attrs.get("ordered", False)))
    raise TypeError(f"Unsupported H5AD node type: {encoding_type!r}")


def _read_string_array(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values)
    decoded = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in flat.reshape(-1).tolist()
    ]
    return np.asarray(decoded, dtype=object).reshape(flat.shape)


def _as_index_values(values: Any) -> np.ndarray:
    if isinstance(values, pd.Categorical):
        return np.asarray(values.astype(str), dtype=object)
    if isinstance(values, np.ndarray):
        if values.dtype.kind in {"O", "S", "U"}:
            return np.asarray([_decode_scalar(item) for item in values.tolist()], dtype=object)
        return values
    return np.asarray(values, dtype=object)


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _string_series_or_default(frame: pd.DataFrame, column_name: str, default_value: str) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series(default_value, index=frame.index, dtype=object)
    values = frame[column_name]
    values = pd.Series(values, index=frame.index, copy=False)
    values = values.astype(object)
    values = values.where(values.notna(), default_value)
    return values.astype(str)
