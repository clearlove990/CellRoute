from __future__ import annotations

import csv
import gzip
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from .spatial_screening import (
    EXPERIMENT_DIR,
    ROOT,
    SPATIAL_PROCESSED_DIR,
    SPATIAL_RAW_DIR,
    CandidateDatasetSpec,
    candidate_dataset_specs,
    ensure_output_dirs,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


try:
    import torch
except ModuleNotFoundError:
    torch = None

import anndata as ad


ANACONDA_PYTHON = Path(r"D:\anaconda\python.exe")
GPU_CLUSTER_HELPER = ROOT / "scripts" / "gpu_cluster_from_array.py"


@dataclass(frozen=True)
class RuntimeInfo:
    device_type: str
    cuda_available: bool
    cuda_count: int
    cuda_name: str


@dataclass(frozen=True)
class ProcessedSpatialDataset:
    dataset_id: str
    dataset_name: str
    selection_role: str
    processed_path: Path
    n_samples: int
    n_spots: int
    n_genes: int


def get_runtime_info() -> RuntimeInfo:
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
        cuda_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
        if cuda_available:
            return RuntimeInfo(
                device_type="cuda",
                cuda_available=True,
                cuda_count=cuda_count,
                cuda_name=cuda_name,
            )
    external_runtime = query_external_torch_runtime()
    if external_runtime is not None:
        return external_runtime
    cuda_available = False
    cuda_count = 0
    cuda_name = "cpu"
    return RuntimeInfo(
        device_type="cuda" if cuda_available else "cpu",
        cuda_available=cuda_available,
        cuda_count=cuda_count,
        cuda_name=cuda_name,
    )


def query_external_torch_runtime() -> RuntimeInfo | None:
    if not ANACONDA_PYTHON.exists():
        return None
    probe_command = [
        str(ANACONDA_PYTHON),
        "-c",
        (
            "import json, torch; "
            "print(json.dumps({"
            "'cuda_available': bool(torch.cuda.is_available()), "
            "'cuda_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "
            "'cuda_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}))"
        ),
    ]
    try:
        result = subprocess.run(probe_command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout.strip())
    except (subprocess.CalledProcessError, OSError, json.JSONDecodeError):
        return None
    return RuntimeInfo(
        device_type="cuda" if payload["cuda_available"] else "cpu",
        cuda_available=bool(payload["cuda_available"]),
        cuda_count=int(payload["cuda_count"]),
        cuda_name=str(payload["cuda_name"]),
    )


def run_spatial_dataset_screening() -> tuple[list[ProcessedSpatialDataset], pd.DataFrame]:
    ensure_output_dirs()
    runtime_info = get_runtime_info()
    processed: list[ProcessedSpatialDataset] = []
    summary_frames: list[pd.DataFrame] = []

    selected_specs = [spec for spec in candidate_dataset_specs() if spec.selected == "yes"]
    for spec in selected_specs:
        adata = load_selected_dataset(spec=spec, runtime_info=runtime_info)
        processed_path = SPATIAL_PROCESSED_DIR / f"{spec.dataset_id}.h5ad"
        adata.write_h5ad(processed_path)
        processed.append(
            ProcessedSpatialDataset(
                dataset_id=spec.dataset_id,
                dataset_name=spec.dataset_name,
                selection_role=spec.selection_role,
                processed_path=processed_path,
                n_samples=int(adata.obs["sample_id"].nunique()),
                n_spots=int(adata.n_obs),
                n_genes=int(adata.n_vars),
            )
        )
        summary_frames.append(build_sample_qc_summary(adata=adata, spec=spec))

    summary_df = pd.concat(summary_frames, ignore_index=True)
    results_dir = EXPERIMENT_DIR / "results"
    figures_dir = EXPERIMENT_DIR / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "dataset_qc_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_spots_per_sample(summary_df=summary_df, output_path=figures_dir / "qc_spots_per_sample.png")
    plot_genes_per_sample(summary_df=summary_df, output_path=figures_dir / "qc_genes_per_sample.png")
    plot_spatial_preview(processed=processed, output_path=figures_dir / "spatial_layout_preview.png")

    write_protocol(processed=processed, runtime_info=runtime_info)
    write_analysis(processed=processed, summary_df=summary_df, runtime_info=runtime_info)
    return processed, summary_df


def load_selected_dataset(*, spec: CandidateDatasetSpec, runtime_info: RuntimeInfo) -> ad.AnnData:
    if spec.dataset_id == "wu2021_breast_visium":
        return load_wu_breast_visium(spec=spec, runtime_info=runtime_info)
    if spec.dataset_id == "gse220442_ad_mtg":
        return load_gse220442_ad_mtg(spec=spec, runtime_info=runtime_info)
    raise KeyError(f"Unsupported selected dataset: {spec.dataset_id}")


def load_wu_breast_visium(*, spec: CandidateDatasetSpec, runtime_info: RuntimeInfo) -> ad.AnnData:
    base_dir = SPATIAL_RAW_DIR / spec.dataset_id / "extracted"
    counts_root = base_dir / "counts" / "filtered_count_matrices"
    spatial_root = base_dir / "spatial" / "spatial"
    metadata_root = base_dir / "metadata" / "metadata"

    sample_adatas: list[ad.AnnData] = []
    sample_assets: dict[str, dict[str, str]] = {}
    for count_dir in sorted(path for path in counts_root.iterdir() if path.is_dir()):
        sample_id = count_dir.name.removesuffix("_filtered_count_matrix")
        spatial_dir = spatial_root / f"{sample_id}_spatial"
        metadata_path = metadata_root / f"{sample_id}_metadata.csv"

        counts, barcodes, var_df = load_10x_count_dir(count_dir)
        positions = read_spatial_positions(spatial_dir / "tissue_positions_list.csv")
        metadata = pd.read_csv(metadata_path).rename(
            columns={
                "Unnamed: 0": "barcode",
                "Classification": "cell_type",
                "subtype": "condition",
                "patientid": "patient_id",
            }
        )
        metadata["barcode"] = metadata["barcode"].astype(str)
        aligned_obs = metadata.set_index("barcode").reindex(barcodes)
        aligned_obs.index = pd.Index(barcodes, name="barcode")
        aligned_obs["cluster"] = pd.NA

        sample_assets[sample_id] = collect_spatial_assets(spatial_dir)
        sample_adatas.append(
            build_sample_adata(
                counts=counts,
                barcodes=barcodes,
                var_df=var_df,
                positions=positions,
                aligned_obs=aligned_obs,
                sample_id=sample_id,
                dataset_id=spec.dataset_id,
                dataset_name=spec.dataset_name,
            )
        )

    adata = concatenate_samples(sample_adatas)
    assign_clusters_if_missing(adata=adata, runtime_info=runtime_info)
    attach_runtime_metadata(adata=adata, spec=spec, runtime_info=runtime_info, sample_assets=sample_assets)
    return adata


def load_gse220442_ad_mtg(*, spec: CandidateDatasetSpec, runtime_info: RuntimeInfo) -> ad.AnnData:
    dataset_dir = SPATIAL_RAW_DIR / spec.dataset_id
    base_dir = dataset_dir / "extracted" / "counts_and_images" / "counts_and_images"
    metadata_path = dataset_dir / "downloads" / "GSE220442_metadata.csv.gz"
    metadata = pd.read_csv(metadata_path).rename(
        columns={
            "Unnamed: 0": "barcode_with_sample",
            "category": "condition",
            "Layer": "cell_type",
            "patientID": "patient_id",
        }
    )
    metadata["barcode_stub"] = metadata["barcode_with_sample"].astype(str).str.rsplit("-", n=1).str[0]
    metadata["sample_id"] = metadata["patient_id"].astype(str).str.replace("_", "-", regex=False)

    sample_adatas: list[ad.AnnData] = []
    sample_assets: dict[str, dict[str, str]] = {}
    for sample_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        sample_id = sample_dir.name
        counts, barcodes, var_df = load_10x_count_dir(sample_dir / "filtered_feature_bc_matrix")
        positions = read_spatial_positions(resolve_existing_path(sample_dir / "spatial", "tissue_positions_list.csv"))
        sample_meta = metadata.loc[metadata["sample_id"] == sample_id].copy()
        sample_meta = sample_meta.set_index("barcode_stub")
        aligned_obs = sample_meta.reindex([barcode.rsplit("-", maxsplit=1)[0] for barcode in barcodes]).copy()
        aligned_obs.index = pd.Index(barcodes, name="barcode")
        aligned_obs["cluster"] = aligned_obs["seurat_clusters"].astype("Int64").astype(str)
        sample_assets[sample_id] = collect_spatial_assets(sample_dir / "spatial")
        sample_adatas.append(
            build_sample_adata(
                counts=counts,
                barcodes=barcodes,
                var_df=var_df,
                positions=positions,
                aligned_obs=aligned_obs,
                sample_id=sample_id,
                dataset_id=spec.dataset_id,
                dataset_name=spec.dataset_name,
            )
        )

    adata = concatenate_samples(sample_adatas)
    assign_clusters_if_missing(adata=adata, runtime_info=runtime_info)
    attach_runtime_metadata(adata=adata, spec=spec, runtime_info=runtime_info, sample_assets=sample_assets)
    return adata


def concatenate_samples(sample_adatas: list[ad.AnnData]) -> ad.AnnData:
    combined = ad.concat(sample_adatas, axis=0, join="outer", merge="same")
    combined.var_names_make_unique()
    combined.obs["condition"] = combined.obs["condition"].fillna("unknown").astype(str)
    combined.obs["cell_type"] = combined.obs["cell_type"].fillna("unassigned").astype(str)
    combined.obs["cluster"] = combined.obs["cluster"].fillna("").astype(str)
    return combined


def attach_runtime_metadata(
    *,
    adata: ad.AnnData,
    spec: CandidateDatasetSpec,
    runtime_info: RuntimeInfo,
    sample_assets: dict[str, dict[str, str]],
) -> None:
    metadata = dict(adata.uns.get("dataset_screening", {}))
    metadata.update(
        {
        "dataset_id": spec.dataset_id,
        "dataset_name": spec.dataset_name,
        "selection_role": spec.selection_role,
        "runtime": {
            "device_type": runtime_info.device_type,
            "cuda_available": runtime_info.cuda_available,
            "cuda_count": runtime_info.cuda_count,
            "cuda_name": runtime_info.cuda_name,
        },
        "sample_assets": sample_assets,
        "x_representation": "raw_counts",
        "normalized_layer": "lognorm",
        }
    )
    metadata.setdefault("cluster_method", "provided_metadata")
    adata.uns["dataset_screening"] = metadata


def assign_clusters_if_missing(*, adata: ad.AnnData, runtime_info: RuntimeInfo) -> None:
    existing = adata.obs["cluster"].astype(str).str.len() > 0
    if bool(existing.all()):
        return
    labels, cluster_method = compute_basic_clusters(matrix=adata.layers["lognorm"], runtime_info=runtime_info)
    adata.obs.loc[:, "cluster"] = np.asarray([f"cluster_{label}" for label in labels], dtype=object)
    adata.uns.setdefault("dataset_screening", {})
    adata.uns["dataset_screening"]["cluster_method"] = cluster_method


def compute_basic_clusters(*, matrix: sparse.spmatrix, runtime_info: RuntimeInfo) -> tuple[np.ndarray, str]:
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    n_obs, n_vars = matrix.shape
    n_top_genes = int(min(1024, n_vars))
    n_components = int(max(2, min(20, n_obs - 1, n_top_genes - 1)))
    if n_components < 2:
        return np.zeros(n_obs, dtype=np.int64), "degenerate"

    gene_idx = select_variable_genes(matrix=matrix, top_k=n_top_genes)
    subset = matrix[:, gene_idx].tocsr()
    n_clusters = int(max(2, min(8, np.sqrt(max(n_obs, 1) / 350.0) + 2)))

    if torch is not None and runtime_info.cuda_available:
        try:
            dense = subset.toarray().astype(np.float32, copy=False)
            x_tensor = torch.from_numpy(dense).to("cuda", non_blocking=True)
            x_tensor = x_tensor - x_tensor.mean(dim=0, keepdim=True)
            u_mat, singular_vals, _ = torch.pca_lowrank(x_tensor, q=n_components, center=False)
            embedding = (u_mat[:, :n_components] * singular_vals[:n_components]).cpu().numpy()
            labels = KMeans(n_clusters=n_clusters, random_state=7, n_init=10).fit_predict(embedding)
            return labels.astype(np.int64), "torch_pca_cuda_kmeans"
        except RuntimeError:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    external_labels = run_external_gpu_cluster(
        subset=subset,
        n_components=n_components,
        n_clusters=n_clusters,
    )
    if external_labels is not None:
        return external_labels.astype(np.int64), "torch_pca_external_cuda_kmeans"

    embedding = TruncatedSVD(n_components=n_components, random_state=7).fit_transform(subset)
    labels = KMeans(n_clusters=n_clusters, random_state=7, n_init=10).fit_predict(embedding)
    return labels.astype(np.int64), "truncated_svd_cpu_kmeans"


def select_variable_genes(*, matrix: sparse.csr_matrix, top_k: int) -> np.ndarray:
    mean_per_gene = np.asarray(matrix.mean(axis=0)).ravel()
    sq_mean_per_gene = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    var_per_gene = np.clip(sq_mean_per_gene - np.square(mean_per_gene), a_min=0.0, a_max=None)
    selected = np.argsort(var_per_gene)[-top_k:]
    return np.sort(selected.astype(np.int64))


def run_external_gpu_cluster(
    *,
    subset: sparse.csr_matrix,
    n_components: int,
    n_clusters: int,
) -> np.ndarray | None:
    if not ANACONDA_PYTHON.exists() or not GPU_CLUSTER_HELPER.exists():
        return None
    tmp_dir = EXPERIMENT_DIR / "tmp_gpu_cluster"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_path = tmp_dir / "cluster_input.npy"
    output_path = tmp_dir / "cluster_labels.npy"
    np.save(input_path, subset.toarray().astype(np.float32, copy=False))
    command = [
        str(ANACONDA_PYTHON),
        str(GPU_CLUSTER_HELPER),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--n-components",
        str(n_components),
        "--n-clusters",
        str(n_clusters),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError):
        return None
    if not output_path.exists():
        return None
    return np.load(output_path)


def build_sample_adata(
    *,
    counts: sparse.csr_matrix,
    barcodes: list[str],
    var_df: pd.DataFrame,
    positions: pd.DataFrame,
    aligned_obs: pd.DataFrame,
    sample_id: str,
    dataset_id: str,
    dataset_name: str,
) -> ad.AnnData:
    obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
    obs["spot_barcode"] = obs.index.astype(str)
    obs["sample_id"] = sample_id
    obs["dataset_id"] = dataset_id
    obs["dataset_name"] = dataset_name

    positions = positions.reindex(obs.index)
    if positions.isna().any().any():
        missing = positions.index[positions.isna().any(axis=1)].tolist()[:5]
        raise ValueError(f"Missing spatial coordinates for sample={sample_id} barcodes={missing}")
    obs = pd.concat([obs, positions], axis=1)

    aligned_obs = aligned_obs.reindex(obs.index)
    obs = pd.concat([obs, aligned_obs], axis=1)
    obs = obs.loc[:, ~obs.columns.duplicated()]
    obs["condition"] = obs.get("condition", "unknown")
    obs["cell_type"] = obs.get("cell_type", "unassigned")
    obs["cluster"] = obs.get("cluster", "")
    obs["condition"] = obs["condition"].fillna("unknown").astype(str)
    obs["cell_type"] = obs["cell_type"].fillna("unassigned").astype(str)
    obs["cluster"] = obs["cluster"].fillna("").astype(str)

    counts = counts.tocsr().astype(np.float32)
    obs["n_counts"] = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    obs["n_genes_by_counts"] = np.asarray((counts > 0).sum(axis=1)).ravel().astype(np.int32)
    obs["spatial_x"] = obs["pxl_col_in_fullres"].astype(np.float32)
    obs["spatial_y"] = obs["pxl_row_in_fullres"].astype(np.float32)

    adata = ad.AnnData(X=counts, obs=obs, var=var_df.copy())
    adata.var_names = pd.Index(var_df["gene_symbol"].astype(str), name="feature_name")
    adata.var_names_make_unique()
    adata.layers["lognorm"] = normalize_sparse_log1p(counts)
    adata.obsm["spatial"] = obs[["spatial_x", "spatial_y"]].to_numpy(dtype=np.float32)
    adata.obs_names = pd.Index([f"{sample_id}:{barcode}" for barcode in barcodes], name="obs_name")
    return adata


def normalize_sparse_log1p(counts: sparse.csr_matrix, target_sum: float = 1.0e4) -> sparse.csr_matrix:
    counts = counts.astype(np.float32)
    total_counts = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    scale = np.divide(
        np.full_like(total_counts, fill_value=target_sum, dtype=np.float32),
        np.maximum(total_counts, 1.0),
        out=np.ones_like(total_counts, dtype=np.float32),
        where=total_counts > 0,
    )
    normalized = counts.multiply(scale[:, None]).tocsr().astype(np.float32)
    normalized.data = np.log1p(normalized.data)
    return normalized


def load_10x_count_dir(count_dir: Path) -> tuple[sparse.csr_matrix, list[str], pd.DataFrame]:
    matrix_path = resolve_existing_path(count_dir, "matrix.mtx")
    barcodes_path = resolve_existing_path(count_dir, "barcodes.tsv")
    features_path = resolve_existing_path(count_dir, "features.tsv", "genes.tsv")

    with open_binary_auto(matrix_path) as handle:
        matrix_raw = mmread(handle)
    matrix = sparse.csr_matrix(matrix_raw) if sparse.issparse(matrix_raw) else sparse.csr_matrix(np.asarray(matrix_raw))

    barcodes = read_single_column_text(barcodes_path)
    feature_table = read_feature_table(features_path)
    if matrix.shape == (len(feature_table), len(barcodes)):
        matrix = matrix.transpose().tocsr()
    elif matrix.shape != (len(barcodes), len(feature_table)):
        raise ValueError(
            f"Matrix shape mismatch in {count_dir}: matrix={matrix.shape}, barcodes={len(barcodes)}, genes={len(feature_table)}"
        )
    return matrix.astype(np.float32), barcodes, feature_table


def read_feature_table(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    with open_text_auto(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"Feature table is empty: {path}")
    if len(rows[0]) == 1:
        gene_ids = [row[0] for row in rows]
        gene_symbols = [row[0] for row in rows]
        feature_types = ["Gene Expression"] * len(rows)
    else:
        gene_ids = [row[0] for row in rows]
        gene_symbols = [row[1] if len(row) > 1 and row[1] else row[0] for row in rows]
        feature_types = [row[2] if len(row) > 2 else "Gene Expression" for row in rows]
    return pd.DataFrame({"gene_id": gene_ids, "gene_symbol": gene_symbols, "feature_type": feature_types})


def read_single_column_text(path: Path) -> list[str]:
    values: list[str] = []
    with open_text_auto(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                values.append(stripped.split("\t")[0])
    return values


def read_spatial_positions(path: Path) -> pd.DataFrame:
    with open_text_auto(path) as handle:
        sample = handle.readline()
    has_header = sample.lower().startswith("barcode")
    dataframe = pd.read_csv(
        path,
        compression="gzip" if is_gzip_file(path) else "infer",
        header=0 if has_header else None,
    )
    if has_header:
        rename_map = {
            "barcode": "barcode",
            "in_tissue": "in_tissue",
            "array_row": "array_row",
            "array_col": "array_col",
            "pxl_row_in_fullres": "pxl_row_in_fullres",
            "pxl_col_in_fullres": "pxl_col_in_fullres",
        }
        dataframe = dataframe.rename(columns=rename_map)
        expected_cols = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
        dataframe = dataframe[expected_cols]
    else:
        dataframe = dataframe.iloc[:, :6]
        dataframe.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
    dataframe["barcode"] = dataframe["barcode"].astype(str)
    return dataframe.set_index("barcode")


def collect_spatial_assets(spatial_dir: Path) -> dict[str, str]:
    asset_names = (
        "tissue_hires_image.png",
        "tissue_lowres_image.png",
        "detected_tissue_image.jpg",
        "aligned_fiducials.jpg",
        "scalefactors_json.json",
    )
    assets: dict[str, str] = {}
    for asset_name in asset_names:
        asset_path = resolve_existing_path_optional(spatial_dir, asset_name)
        if asset_path is not None:
            assets[asset_name] = str(asset_path)
    return assets


def resolve_existing_path(directory: Path, *base_names: str) -> Path:
    resolved = resolve_existing_path_optional(directory, *base_names)
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve any of {base_names} under {directory}")
    return resolved


def resolve_existing_path_optional(directory: Path, *base_names: str) -> Path | None:
    suffixes = ("", ".gz")
    for base_name in base_names:
        for suffix in suffixes:
            candidate = directory / f"{base_name}{suffix}"
            if candidate.exists():
                return candidate
    return None


def is_gzip_file(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


@contextmanager
def open_text_auto(path: Path) -> Iterator[object]:
    if is_gzip_file(path):
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            yield handle
    else:
        with path.open("rt", encoding="utf-8") as handle:
            yield handle


@contextmanager
def open_binary_auto(path: Path) -> Iterator[object]:
    if is_gzip_file(path):
        with gzip.open(path, mode="rb") as handle:
            yield handle
    else:
        with path.open("rb") as handle:
            yield handle


def build_sample_qc_summary(*, adata: ad.AnnData, spec: CandidateDatasetSpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, frame in adata.obs.groupby("sample_id", sort=True, observed=False):
        rows.append(
            {
                "dataset_id": spec.dataset_id,
                "dataset_name": spec.dataset_name,
                "selection_role": spec.selection_role,
                "sample_id": sample_id,
                "condition": "|".join(sorted(frame["condition"].dropna().astype(str).unique().tolist())),
                "n_spots": int(frame.shape[0]),
                "median_counts": float(np.median(frame["n_counts"].to_numpy(dtype=np.float32))),
                "median_genes": float(np.median(frame["n_genes_by_counts"].to_numpy(dtype=np.float32))),
                "n_cell_type_labels": int(frame["cell_type"].astype(str).nunique()),
                "n_clusters": int(frame["cluster"].astype(str).nunique()),
                "pct_in_tissue": float(frame["in_tissue"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_spots_per_sample(*, summary_df: pd.DataFrame, output_path: Path) -> None:
    ordered = summary_df.sort_values(["selection_role", "dataset_name", "sample_id"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#b55439" if role == "primary" else "#3b6fb6" for role in ordered["selection_role"]]
    ax.bar(ordered["sample_id"], ordered["n_spots"], color=colors)
    ax.set_title("Spots Per Sample")
    ax.set_ylabel("Spots")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_genes_per_sample(*, summary_df: pd.DataFrame, output_path: Path) -> None:
    ordered = summary_df.sort_values(["selection_role", "dataset_name", "sample_id"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#b55439" if role == "primary" else "#3b6fb6" for role in ordered["selection_role"]]
    ax.bar(ordered["sample_id"], ordered["median_genes"], color=colors)
    ax.set_title("Median Genes Per Sample")
    ax.set_ylabel("Median detected genes")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_spatial_preview(*, processed: list[ProcessedSpatialDataset], output_path: Path) -> None:
    adatas = [(item.dataset_name, ad.read_h5ad(item.processed_path)) for item in processed]
    sample_panels: list[tuple[str, str, pd.DataFrame]] = []
    for dataset_name, adata in adatas:
        obs = adata.obs.copy()
        for sample_id, frame in obs.groupby("sample_id", sort=True, observed=False):
            sample_panels.append((dataset_name, sample_id, frame))

    n_panels = len(sample_panels)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, panel in zip(axes, sample_panels, strict=False):
        dataset_name, sample_id, frame = panel
        codes, _ = pd.factorize(frame["cluster"].astype(str))
        ax.scatter(
            frame["spatial_x"].to_numpy(dtype=np.float32),
            -frame["spatial_y"].to_numpy(dtype=np.float32),
            c=codes,
            s=5,
            cmap="tab20",
            linewidths=0,
            alpha=0.85,
        )
        condition = "|".join(sorted(frame["condition"].astype(str).unique().tolist()))
        ax.set_title(f"{sample_id} | {condition}")
        ax.set_xlabel(dataset_name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    for ax in axes[n_panels:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_protocol(*, processed: list[ProcessedSpatialDataset], runtime_info: RuntimeInfo) -> None:
    protocol_path = EXPERIMENT_DIR / "protocol.md"
    dataset_lines = [f"- `{item.dataset_id}` -> `{item.processed_path.relative_to(ROOT)}`" for item in processed]
    protocol_text = "\n".join(
        [
            "# Protocol",
            "",
            "## Goal",
            "",
            "Screen 2 selected spatial transcriptomics datasets, standardize them into spot-level AnnData, and generate QC figures for sample-level screening.",
            "",
            "## Runtime",
            "",
            f"- Device detection: `torch.cuda.is_available()` -> `{runtime_info.cuda_available}`",
            f"- Active device: `{runtime_info.device_type}`",
            f"- CUDA device: `{runtime_info.cuda_name}`",
            "",
            "## Steps",
            "",
            "1. Build `data_catalog/download_manifest.csv` and shortlist notes from curated public dataset sources.",
            "2. Download and extract the selected raw archives into `data/spatial_raw/`.",
            "3. Parse 10x-style count matrices and spatial coordinates.",
            "4. Standardize spot metadata to `sample_id`, `condition`, `cell_type`, `cluster`, counts, log-normalized expression, and `obsm['spatial']`.",
            "5. Write one merged `.h5ad` per dataset under `data/spatial_processed/`.",
            "6. Summarize per-sample QC and render screening figures.",
            "",
            "## Outputs",
            "",
            *dataset_lines,
            "",
        ]
    )
    protocol_path.write_text(protocol_text + "\n", encoding="utf-8")


def write_analysis(*, processed: list[ProcessedSpatialDataset], summary_df: pd.DataFrame, runtime_info: RuntimeInfo) -> None:
    analysis_path = EXPERIMENT_DIR / "analysis.md"
    primary = next(item for item in processed if item.selection_role == "primary")
    validation = next(item for item in processed if item.selection_role == "validation")
    total_spots = int(summary_df["n_spots"].sum())
    median_genes = float(summary_df["median_genes"].median())
    analysis_lines = [
        "# Analysis",
        "",
        "## Selection decision",
        "",
        f"- Primary dataset: `{primary.dataset_name}` ({primary.n_samples} samples, {primary.n_spots} spots, tumor microenvironment narrative).",
        f"- Validation dataset: `{validation.dataset_name}` ({validation.n_samples} samples, {validation.n_spots} spots, disease/control validation).",
        "",
        "## QC snapshot",
        "",
        f"- Total screened spots across active datasets: `{total_spots}`",
        f"- Median genes per sample across the active screen: `{median_genes:.1f}`",
        f"- GPU-assisted clustering path available: `{runtime_info.cuda_available}` on `{runtime_info.cuda_name}`",
        "",
        "## Immediate readout",
        "",
        "- The breast cohort is the better main-story anchor because histology-aware pathology labels already separate invasive cancer, stroma, lymphocyte-rich, and normal-like regions.",
        "- The AD MTG cohort is a strong validation panel because it has an explicit disease/control label plus layer-like spatial structure in the provided metadata.",
        "- The two chosen datasets cover complementary stories: tumor neighborhood biology and disease-control brain architecture.",
        "",
        "## Next step",
        "",
        "- Use the standardized `.h5ad` files as direct inputs for neighborhood, deconvolution, and cross-sample motif analysis.",
        "",
    ]
    analysis_path.write_text("\n".join(analysis_lines), encoding="utf-8")
