from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


@dataclass
class SCRNADataset:
    counts: np.ndarray
    gene_names: np.ndarray
    cell_names: np.ndarray
    labels: np.ndarray | None = None
    batches: np.ndarray | None = None
    name: str = "dataset"


@dataclass
class SCRNAInputSpec:
    dataset_id: str
    dataset_name: str
    input_path: str
    file_format: str
    transpose: bool = False
    obs_path: str | None = None
    var_path: str | None = None
    genes_path: str | None = None
    cells_path: str | None = None
    labels_col: str | None = None
    batches_col: str | None = None
    cell_count: int | None = None
    gene_count: int | None = None
    label_classes: int | None = None
    batch_classes: int | None = None
    fingerprint: str | None = None
    duplicate_of: str | None = None
    selected_for_benchmark: bool = True


def load_scrna_dataset(
    data_path: str,
    *,
    file_format: str = "auto",
    transpose: bool = False,
    obs_path: str | None = None,
    var_path: str | None = None,
    genes_path: str | None = None,
    cells_path: str | None = None,
    labels_col: str | None = None,
    batches_col: str | None = None,
    labels_path: str | None = None,
    batches_path: str | None = None,
    delimiter: str | None = None,
    dataset_name: str | None = None,
    max_cells: int | None = None,
    max_genes: int | None = None,
    random_state: int = 0,
) -> SCRNADataset:
    path = Path(data_path)
    resolved_format = _resolve_format(path=path, file_format=file_format)
    used_sparse_budget_path = False

    if resolved_format in {"csv", "tsv", "txt"}:
        counts, cell_names, gene_names = _load_tabular_matrix(path=path, delimiter=delimiter)
        if transpose:
            counts = counts.T
            cell_names, gene_names = gene_names, cell_names
        labels, batches = _load_sidecars(
            cell_names=cell_names,
            obs_path=obs_path,
            labels_col=labels_col,
            batches_col=batches_col,
            labels_path=labels_path,
            batches_path=batches_path,
            delimiter=delimiter,
        )
    elif resolved_format == "mtx":
        if max_cells is not None or max_genes is not None:
            counts, cell_names, gene_names, labels, batches = _load_mtx_matrix_budgeted(
                matrix_path=path,
                genes_path=genes_path,
                cells_path=cells_path,
                transpose=transpose,
                obs_path=obs_path,
                labels_col=labels_col,
                batches_col=batches_col,
                labels_path=labels_path,
                batches_path=batches_path,
                delimiter=delimiter,
                max_cells=max_cells,
                max_genes=max_genes,
                random_state=random_state,
            )
            used_sparse_budget_path = True
        else:
            counts, cell_names, gene_names = _load_mtx_matrix(
                matrix_path=path,
                genes_path=genes_path,
                cells_path=cells_path,
                transpose=transpose,
            )
            labels, batches = _load_sidecars(
                cell_names=cell_names,
                obs_path=obs_path,
                labels_col=labels_col,
                batches_col=batches_col,
                labels_path=labels_path,
                batches_path=batches_path,
                delimiter=delimiter,
            )
    elif resolved_format == "h5ad":
        if transpose:
            raise ValueError("Transpose is not supported for h5ad inputs because obs/var semantics would be reversed.")
        counts, cell_names, gene_names, h5ad_labels, h5ad_batches = _load_h5ad(
            path=path,
            labels_col=labels_col,
            batches_col=batches_col,
            max_cells=max_cells,
            max_genes=max_genes,
            random_state=random_state,
        )
        labels, batches = h5ad_labels, h5ad_batches
        used_sparse_budget_path = max_cells is not None or max_genes is not None
        if labels_path is not None or batches_path is not None or obs_path is not None:
            sidecar_labels, sidecar_batches = _load_sidecars(
                cell_names=cell_names,
                obs_path=obs_path,
                labels_col=labels_col,
                batches_col=batches_col,
                labels_path=labels_path,
                batches_path=batches_path,
                delimiter=delimiter,
            )
            labels = sidecar_labels if sidecar_labels is not None else labels
            batches = sidecar_batches if sidecar_batches is not None else batches
    else:
        raise ValueError(f"Unsupported file format: {resolved_format}")

    dataset = SCRNADataset(
        counts=_ensure_dense_float32(counts),
        gene_names=np.asarray(gene_names, dtype=object),
        cell_names=np.asarray(cell_names, dtype=object),
        labels=None if labels is None else np.asarray(labels, dtype=object),
        batches=None if batches is None else np.asarray(batches, dtype=object),
        name=dataset_name or path.stem,
    )
    dataset = sanitize_dataset(dataset)
    if max_cells is not None or max_genes is not None:
        if not used_sparse_budget_path:
            dataset = subsample_dataset(
                dataset,
                max_cells=max_cells,
                max_genes=max_genes,
                random_state=random_state,
            )
    return dataset


def _load_mtx_matrix_budgeted(
    *,
    matrix_path: Path,
    genes_path: str | None,
    cells_path: str | None,
    transpose: bool,
    obs_path: str | None,
    labels_col: str | None,
    batches_col: str | None,
    labels_path: str | None,
    batches_path: str | None,
    delimiter: str | None,
    max_cells: int | None,
    max_genes: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    raw_matrix = mmread(matrix_path)
    matrix = sparse.csr_matrix(raw_matrix) if not sparse.issparse(raw_matrix) else raw_matrix.tocsr()

    n_rows, n_cols = matrix.shape
    if transpose:
        gene_names = _read_name_vector(genes_path, expected_len=n_rows) if genes_path else _default_names("gene", n_rows)
        cell_names = _read_name_vector(cells_path, expected_len=n_cols) if cells_path else _default_names("cell", n_cols)
        matrix = matrix.transpose().tocsr()
    else:
        cell_names = _read_name_vector(cells_path, expected_len=n_rows) if cells_path else _default_names("cell", n_rows)
        gene_names = _read_name_vector(genes_path, expected_len=n_cols) if genes_path else _default_names("gene", n_cols)

    labels, batches = _load_sidecars(
        cell_names=cell_names,
        obs_path=obs_path,
        labels_col=labels_col,
        batches_col=batches_col,
        labels_path=labels_path,
        batches_path=batches_path,
        delimiter=delimiter,
    )

    rng = np.random.default_rng(random_state)
    if max_cells is not None and matrix.shape[0] > max_cells:
        strat_labels = labels if labels is not None else batches
        cell_idx = _stratified_subsample_indices(
            labels=strat_labels,
            n_total=matrix.shape[0],
            target_n=max_cells,
            rng=rng,
        )
        matrix = matrix[cell_idx].tocsr()
        cell_names = cell_names[cell_idx]
        labels = None if labels is None else labels[cell_idx]
        batches = None if batches is None else batches[cell_idx]

    if max_genes is not None and matrix.shape[1] > max_genes:
        gene_idx = _select_sparse_gene_indices(matrix=matrix, target_n=max_genes)
        matrix = matrix[:, gene_idx].tocsr()
        gene_names = gene_names[gene_idx]

    return _ensure_dense_float32(matrix), cell_names, gene_names, labels, batches


def _select_sparse_gene_indices(*, matrix: sparse.spmatrix, target_n: int) -> np.ndarray:
    if matrix.shape[1] <= target_n:
        return np.arange(matrix.shape[1], dtype=np.int64)

    n_cells = max(matrix.shape[0], 1)
    sum_per_gene = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    sq_sum_per_gene = np.asarray(matrix.power(2).sum(axis=0)).ravel().astype(np.float64)
    mean_per_gene = sum_per_gene / n_cells
    var_per_gene = np.clip(sq_sum_per_gene / n_cells - mean_per_gene**2, a_min=0.0, a_max=None)
    gene_score = var_per_gene + 0.05 * np.log1p(sum_per_gene)
    selected = np.argsort(gene_score)[-target_n:]
    return np.sort(selected.astype(np.int64))


def sanitize_dataset(dataset: SCRNADataset) -> SCRNADataset:
    counts = np.asarray(dataset.counts, dtype=np.float32)
    counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
    counts[counts < 0] = 0.0

    nonzero_cells = counts.sum(axis=1) > 0
    nonzero_genes = counts.sum(axis=0) > 0

    counts = counts[np.ix_(nonzero_cells, nonzero_genes)]
    cell_names = dataset.cell_names[nonzero_cells]
    gene_names = dataset.gene_names[nonzero_genes]
    labels = None if dataset.labels is None else dataset.labels[nonzero_cells]
    batches = None if dataset.batches is None else dataset.batches[nonzero_cells]

    return SCRNADataset(
        counts=counts,
        gene_names=gene_names,
        cell_names=cell_names,
        labels=labels,
        batches=batches,
        name=dataset.name,
    )


def discover_scrna_input_specs(
    root: str | Path,
    *,
    exclude_dir_prefixes: tuple[str, ...] = ("artifacts",),
) -> list[SCRNAInputSpec]:
    root_path = Path(root).resolve()
    specs: list[SCRNAInputSpec] = []

    h5ad_paths = sorted(
        path for path in root_path.rglob("*.h5ad") if not _should_skip_discovery_path(path, root_path, exclude_dir_prefixes)
    )
    mtx_paths = sorted(
        path for path in root_path.rglob("*.mtx") if not _should_skip_discovery_path(path, root_path, exclude_dir_prefixes)
    )
    tabular_paths = sorted(
        path
        for suffix in ("*.csv", "*.tsv", "*.txt")
        for path in root_path.rglob(suffix)
        if not _should_skip_discovery_path(path, root_path, exclude_dir_prefixes)
    )

    for path in h5ad_paths:
        labels_col, batches_col = _infer_h5ad_annotation_columns(path)
        specs.append(
            SCRNAInputSpec(
                dataset_id=_make_dataset_id(path, root_path),
                dataset_name=_infer_h5ad_dataset_name(path),
                input_path=str(path),
                file_format="h5ad",
                labels_col=labels_col,
                batches_col=batches_col,
            )
        )

    for path in tabular_paths:
        if not _looks_like_count_matrix(path):
            continue
        obs_path = _find_sidecar_file(path.parent, stems=("obs", "metadata"))
        labels_col, batches_col = (None, None)
        if obs_path is not None:
            labels_col, batches_col = _infer_table_annotation_columns(obs_path)
        specs.append(
            SCRNAInputSpec(
                dataset_id=_make_dataset_id(path, root_path),
                dataset_name=path.stem,
                input_path=str(path),
                file_format=_resolve_format(path=path, file_format="auto"),
                obs_path=None if obs_path is None else str(obs_path),
                labels_col=labels_col,
                batches_col=batches_col,
            )
        )

    for path in mtx_paths:
        genes_path = _find_sidecar_file(path.parent, stems=("genes", "features"))
        cells_path = _find_sidecar_file(path.parent, stems=("barcodes", "cells"))
        obs_path = _find_sidecar_file_nearby(path.parent, stems=("obs", "metadata"))
        labels_col, batches_col = (None, None)
        if obs_path is not None:
            labels_col, batches_col = _infer_table_annotation_columns(obs_path)
        specs.append(
            SCRNAInputSpec(
                dataset_id=_make_dataset_id(path, root_path),
                dataset_name=_infer_dataset_name(path),
                input_path=str(path),
                file_format="mtx",
                transpose=_infer_mtx_transpose(path=path, genes_path=genes_path, cells_path=cells_path),
                obs_path=None if obs_path is None else str(obs_path),
                genes_path=None if genes_path is None else str(genes_path),
                cells_path=None if cells_path is None else str(cells_path),
                labels_col=labels_col,
                batches_col=batches_col,
            )
        )

    prioritized_specs = sorted(
        specs,
        key=lambda spec: (
            _dataset_name_key(spec.dataset_name),
            _format_priority(spec.file_format),
            Path(spec.input_path).name.lower(),
        ),
    )
    populated = [_populate_input_spec(spec) for spec in prioritized_specs]

    fingerprint_to_primary: dict[str, str] = {}
    for spec in populated:
        if spec.fingerprint is None:
            continue
        primary_id = fingerprint_to_primary.get(spec.fingerprint)
        if primary_id is None:
            fingerprint_to_primary[spec.fingerprint] = spec.dataset_id
            spec.selected_for_benchmark = True
            continue
        spec.duplicate_of = primary_id
        spec.selected_for_benchmark = False
    return populated


def subsample_dataset(
    dataset: SCRNADataset,
    *,
    max_cells: int | None = None,
    max_genes: int | None = None,
    random_state: int = 0,
) -> SCRNADataset:
    rng = np.random.default_rng(random_state)
    counts = dataset.counts
    cell_idx = np.arange(counts.shape[0])
    gene_idx = np.arange(counts.shape[1])

    if max_cells is not None and counts.shape[0] > max_cells:
        cell_idx = _stratified_subsample_indices(
            labels=dataset.labels,
            n_total=counts.shape[0],
            target_n=max_cells,
            rng=rng,
        )

    counts = counts[cell_idx]
    cell_names = dataset.cell_names[cell_idx]
    labels = None if dataset.labels is None else dataset.labels[cell_idx]
    batches = None if dataset.batches is None else dataset.batches[cell_idx]

    if max_genes is not None and counts.shape[1] > max_genes:
        gene_score = counts.var(axis=0) + 0.05 * np.log1p(counts.sum(axis=0))
        gene_idx = np.argsort(gene_score)[-max_genes:]

    counts = counts[:, gene_idx]
    gene_names = dataset.gene_names[gene_idx]
    return SCRNADataset(
        counts=counts,
        gene_names=gene_names,
        cell_names=cell_names,
        labels=labels,
        batches=batches,
        name=dataset.name,
    )


def _resolve_format(path: Path, file_format: str) -> str:
    if file_format != "auto":
        return file_format.lower()

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".tsv", ".txt"}:
        return "tsv"
    if suffix == ".mtx":
        return "mtx"
    if suffix == ".h5ad":
        return "h5ad"
    raise ValueError(f"Could not infer format from suffix: {suffix}")


def _load_tabular_matrix(path: Path, delimiter: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sep = delimiter if delimiter is not None else ("\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")
    raw_df = pd.read_csv(path, sep=sep)
    if raw_df.empty:
        raise ValueError(f"Empty matrix file: {path}")

    first_col_numeric = pd.to_numeric(raw_df.iloc[:, 0], errors="coerce").notna().mean()
    sample_df = raw_df.iloc[: min(64, len(raw_df)), 1 : min(65, raw_df.shape[1])]
    rest_numeric = pd.to_numeric(sample_df.stack(), errors="coerce").notna().mean() if not sample_df.empty else 1.0

    if raw_df.shape[1] >= 2 and first_col_numeric < 0.5 and rest_numeric > 0.95:
        cell_names = raw_df.iloc[:, 0].astype(str).to_numpy()
        matrix_df = raw_df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        gene_names = matrix_df.columns.astype(str).to_numpy()
    else:
        matrix_df = raw_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        gene_names = matrix_df.columns.astype(str).to_numpy()
        cell_names = np.asarray([f"cell_{idx}" for idx in range(matrix_df.shape[0])], dtype=object)

    return matrix_df.to_numpy(dtype=np.float32), np.asarray(cell_names, dtype=object), np.asarray(gene_names, dtype=object)


def _load_mtx_matrix(
    matrix_path: Path,
    genes_path: str | None,
    cells_path: str | None,
    transpose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = mmread(matrix_path)
    counts = _ensure_dense_float32(matrix)
    n_rows, n_cols = counts.shape
    if transpose:
        gene_names = _read_name_vector(genes_path, expected_len=n_rows) if genes_path else _default_names("gene", n_rows)
        cell_names = _read_name_vector(cells_path, expected_len=n_cols) if cells_path else _default_names("cell", n_cols)
        return counts.T, cell_names, gene_names

    cell_names = _read_name_vector(cells_path, expected_len=n_rows) if cells_path else _default_names("cell", n_rows)
    gene_names = _read_name_vector(genes_path, expected_len=n_cols) if genes_path else _default_names("gene", n_cols)
    return counts, cell_names, gene_names


def _load_h5ad(
    path: Path,
    labels_col: str | None,
    batches_col: str | None,
    max_cells: int | None = None,
    max_genes: int | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if max_cells is None and max_genes is None:
        try:
            import anndata as ad  # type: ignore

            adata = ad.read_h5ad(path)
            counts = _ensure_dense_float32(adata.X)
            cell_names = adata.obs_names.astype(str).to_numpy()
            gene_names = adata.var_names.astype(str).to_numpy()
            labels = adata.obs[labels_col].astype(str).to_numpy() if labels_col and labels_col in adata.obs.columns else None
            batches = adata.obs[batches_col].astype(str).to_numpy() if batches_col and batches_col in adata.obs.columns else None
            return counts, cell_names, gene_names, labels, batches
        except ImportError:
            pass

    with h5py.File(path, "r") as handle:
        obs_df = _read_h5ad_dataframe(handle["obs"])
        var_df = _read_h5ad_dataframe(handle["var"])
        labels = obs_df[labels_col].astype(str).to_numpy() if labels_col and labels_col in obs_df.columns else None
        batches = obs_df[batches_col].astype(str).to_numpy() if batches_col and batches_col in obs_df.columns else None
        if max_cells is not None or max_genes is not None:
            counts, cell_idx, gene_idx = _read_h5ad_matrix_budgeted(
                handle["X"],
                labels=labels,
                batches=batches,
                max_cells=max_cells,
                max_genes=max_genes,
                random_state=random_state,
            )
            obs_df = obs_df.iloc[cell_idx]
            var_df = var_df.iloc[gene_idx]
            labels = None if labels is None else np.asarray(labels)[cell_idx]
            batches = None if batches is None else np.asarray(batches)[cell_idx]
        else:
            counts = _read_h5ad_matrix(handle["X"])

    cell_names = obs_df.index.astype(str).to_numpy()
    gene_names = var_df.index.astype(str).to_numpy()
    return counts, cell_names, gene_names, labels, batches


def _read_h5ad_matrix(node) -> np.ndarray:
    return _ensure_dense_float32(_read_h5ad_matrix_raw(node))


def _read_h5ad_matrix_raw(node):
    if isinstance(node, h5py.Dataset):
        return node[()]

    encoding_type = _decode_scalar(node.attrs.get("encoding-type", ""))
    if encoding_type in {"csr_matrix", "csc_matrix"}:
        data = np.asarray(node["data"][()], dtype=np.float32)
        indices = np.asarray(node["indices"][()], dtype=np.int32)
        indptr = np.asarray(node["indptr"][()], dtype=np.int32)
        if "shape" in node:
            shape = tuple(int(x) for x in node["shape"][()])
        else:
            shape = tuple(int(x) for x in node.attrs["shape"])
        if encoding_type == "csr_matrix":
            matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
        else:
            matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
        return matrix

    raise ValueError(f"Unsupported h5ad matrix encoding: {encoding_type}")


def _read_h5ad_matrix_budgeted(
    node,
    *,
    labels: np.ndarray | None,
    batches: np.ndarray | None,
    max_cells: int | None,
    max_genes: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = _read_h5ad_matrix_raw(node)
    n_cells, n_genes = matrix.shape
    cell_idx = np.arange(n_cells, dtype=np.int64)
    gene_idx = np.arange(n_genes, dtype=np.int64)
    rng = np.random.default_rng(random_state)

    if max_cells is not None and n_cells > max_cells:
        strat_labels = labels if labels is not None else batches
        cell_idx = _stratified_subsample_indices(
            labels=strat_labels,
            n_total=n_cells,
            target_n=max_cells,
            rng=rng,
        )
        matrix = matrix[cell_idx]

    if max_genes is not None and matrix.shape[1] > max_genes:
        if sparse.issparse(matrix):
            selected_gene_idx = _select_sparse_gene_indices(matrix=matrix, target_n=max_genes)
        else:
            dense_matrix = np.asarray(matrix, dtype=np.float32)
            gene_score = dense_matrix.var(axis=0) + 0.05 * np.log1p(dense_matrix.sum(axis=0))
            selected_gene_idx = np.sort(np.argsort(gene_score)[-max_genes:].astype(np.int64))
        matrix = matrix[:, selected_gene_idx]
        gene_idx = gene_idx[selected_gene_idx]

    return _ensure_dense_float32(matrix), cell_idx, gene_idx


def _read_h5ad_dataframe(group: h5py.Group) -> pd.DataFrame:
    index_key = _decode_scalar(group.attrs.get("_index"))
    column_order = group.attrs.get("column-order")
    if column_order is None:
        columns = [key for key in group.keys() if key != index_key]
    else:
        columns = [_decode_scalar(item) for item in column_order]

    if index_key is None or index_key not in group:
        index = pd.Index([f"row_{idx}" for idx in range(len(group[columns[0]]))], dtype=str)
    else:
        index = pd.Index(_read_h5ad_column(group[index_key]).astype(str))

    data = {}
    for column in columns:
        if column not in group:
            continue
        data[column] = _read_h5ad_column(group[column])
    return pd.DataFrame(data=data, index=index)


def _read_h5ad_column(node) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        raw = node[()]
        return _decode_array(raw)

    encoding_type = _decode_scalar(node.attrs.get("encoding-type", ""))
    if encoding_type == "categorical":
        codes = np.asarray(node["codes"][()], dtype=np.int64)
        categories = _read_h5ad_column(node["categories"])
        decoded = np.empty_like(codes, dtype=object)
        for idx, code in enumerate(codes):
            decoded[idx] = None if code < 0 else categories[code]
        return decoded

    raise ValueError(f"Unsupported h5ad column encoding: {encoding_type}")


def _load_sidecars(
    *,
    cell_names: np.ndarray,
    obs_path: str | None,
    labels_col: str | None,
    batches_col: str | None,
    labels_path: str | None,
    batches_path: str | None,
    delimiter: str | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    labels = None
    batches = None

    if obs_path is not None:
        obs_df = _read_table(Path(obs_path), delimiter=delimiter)
        labels = _extract_aligned_column(obs_df, labels_col, cell_names) if labels_col else None
        batches = _extract_aligned_column(obs_df, batches_col, cell_names) if batches_col else None

    if labels_path is not None:
        label_df = _read_table(Path(labels_path), delimiter=delimiter)
        labels = _extract_single_vector(label_df, cell_names, value_name=labels_col or "label")

    if batches_path is not None:
        batch_df = _read_table(Path(batches_path), delimiter=delimiter)
        batches = _extract_single_vector(batch_df, cell_names, value_name=batches_col or "batch")

    return labels, batches


def _read_table(path: Path, delimiter: str | None = None) -> pd.DataFrame:
    sep = delimiter if delimiter is not None else ("\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")
    df = pd.read_csv(path, sep=sep)
    if df.empty:
        raise ValueError(f"Empty annotation file: {path}")

    first_col_numeric = pd.to_numeric(df.iloc[:, 0], errors="coerce").notna().mean()
    if df.shape[1] >= 2 and first_col_numeric < 0.5:
        df = df.set_index(df.columns[0])
    else:
        df.index = [f"cell_{idx}" for idx in range(len(df))]
    df.index = df.index.astype(str)
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="first")]
    return df


def _extract_aligned_column(df: pd.DataFrame, column: str | None, cell_names: np.ndarray) -> np.ndarray | None:
    if column is None:
        return None
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in annotation table.")

    if set(cell_names).issubset(set(df.index.astype(str))):
        series = df.loc[cell_names, column]
        return series.astype(str).to_numpy()

    if len(df) == len(cell_names):
        return df[column].astype(str).to_numpy()

    raise ValueError(f"Could not align column '{column}' to the cells in the count matrix.")


def _extract_single_vector(df: pd.DataFrame, cell_names: np.ndarray, value_name: str) -> np.ndarray:
    if len(df.columns) == 0:
        raise ValueError("Annotation vector file has no value columns.")
    if len(df.columns) == 1:
        value_column = df.columns[0]
    else:
        candidate_columns = [col for col in df.columns if col != value_name]
        value_column = candidate_columns[0] if candidate_columns else df.columns[-1]

    if set(cell_names).issubset(set(df.index.astype(str))):
        return df.loc[cell_names, value_column].astype(str).to_numpy()

    if len(df) == len(cell_names):
        return df[value_column].astype(str).to_numpy()

    raise ValueError("Could not align annotation vector to cells.")


def _read_name_vector(path: str, expected_len: int) -> np.ndarray:
    file_path = Path(path)
    sep = "\t" if file_path.suffix.lower() in {".tsv", ".txt"} else ","
    table = pd.read_csv(file_path, sep=sep, header=None)
    if table.empty:
        raise ValueError(f"Empty name vector file: {path}")
    values = table.iloc[:, 0].astype(str).to_numpy()
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} names in {path}, found {len(values)}.")
    return np.asarray(values, dtype=object)


def _ensure_dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _decode_scalar(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_scalar(value.item())
    return value


def _decode_array(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        return np.asarray([item.decode("utf-8") for item in arr], dtype=object)
    if arr.dtype.kind == "O":
        return np.asarray(
            [item.decode("utf-8") if isinstance(item, bytes) else item for item in arr],
            dtype=object,
        )
    return arr


def _default_names(prefix: str, n: int) -> np.ndarray:
    return np.asarray([f"{prefix}_{idx}" for idx in range(n)], dtype=object)


def _stratified_subsample_indices(
    *,
    labels: np.ndarray | None,
    n_total: int,
    target_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if labels is None:
        return np.sort(rng.choice(n_total, size=target_n, replace=False))

    labels = np.asarray(labels)
    selected = []
    unique, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    per_group = np.maximum(1, np.floor(proportions * target_n).astype(int))
    shortfall = target_n - int(per_group.sum())

    if shortfall > 0:
        order = np.argsort(-(proportions * target_n - per_group))
        for idx in order[:shortfall]:
            per_group[idx] += 1

    for label, k in zip(unique, per_group, strict=False):
        label_idx = np.where(labels == label)[0]
        k = min(k, len(label_idx))
        selected.append(rng.choice(label_idx, size=k, replace=False))

    merged = np.concatenate(selected)
    if len(merged) > target_n:
        merged = rng.choice(merged, size=target_n, replace=False)
    return np.sort(merged)


def _populate_input_spec(spec: SCRNAInputSpec) -> SCRNAInputSpec:
    if spec.file_format == "mtx":
        cell_count, gene_count = _summarize_mtx_input_shape(spec)
        label_classes, batch_classes = _summarize_annotation_classes(
            obs_path=spec.obs_path,
            labels_col=spec.labels_col,
            batches_col=spec.batches_col,
        )
        spec.cell_count = cell_count
        spec.gene_count = gene_count
        spec.label_classes = label_classes
        spec.batch_classes = batch_classes
        spec.fingerprint = _fingerprint_input_spec(spec)
        return spec

    dataset = load_scrna_dataset(
        data_path=spec.input_path,
        file_format=spec.file_format,
        transpose=spec.transpose,
        obs_path=spec.obs_path,
        var_path=spec.var_path,
        genes_path=spec.genes_path,
        cells_path=spec.cells_path,
        labels_col=spec.labels_col,
        batches_col=spec.batches_col,
        dataset_name=spec.dataset_name,
    )
    fingerprint = hashlib.sha1()
    fingerprint.update(np.asarray(dataset.counts, dtype=np.float32).tobytes())
    fingerprint.update("|".join(map(str, dataset.gene_names.tolist())).encode("utf-8"))
    if dataset.labels is not None:
        fingerprint.update("|".join(map(str, dataset.labels.tolist())).encode("utf-8"))
    if dataset.batches is not None:
        fingerprint.update("|".join(map(str, dataset.batches.tolist())).encode("utf-8"))

    spec.cell_count = int(dataset.counts.shape[0])
    spec.gene_count = int(dataset.counts.shape[1])
    spec.label_classes = None if dataset.labels is None else int(len(np.unique(dataset.labels)))
    spec.batch_classes = None if dataset.batches is None else int(len(np.unique(dataset.batches)))
    spec.fingerprint = fingerprint.hexdigest()
    return spec


def _summarize_mtx_input_shape(spec: SCRNAInputSpec) -> tuple[int | None, int | None]:
    shape = _read_matrix_market_shape(Path(spec.input_path))
    if shape is None:
        return None, None
    n_rows, n_cols = shape
    if spec.transpose:
        return int(n_cols), int(n_rows)
    return int(n_rows), int(n_cols)


def _summarize_annotation_classes(
    *,
    obs_path: str | None,
    labels_col: str | None,
    batches_col: str | None,
) -> tuple[int | None, int | None]:
    if obs_path is None or (labels_col is None and batches_col is None):
        return None, None

    table = _read_table(Path(obs_path))
    label_classes = None
    batch_classes = None
    if labels_col is not None and labels_col in table.columns:
        label_classes = int(table[labels_col].astype(str).nunique(dropna=True))
    if batches_col is not None and batches_col in table.columns:
        batch_classes = int(table[batches_col].astype(str).nunique(dropna=True))
    return label_classes, batch_classes


def _fingerprint_input_spec(spec: SCRNAInputSpec) -> str:
    fingerprint = hashlib.sha1()
    for raw_path in (spec.input_path, spec.obs_path, spec.var_path, spec.genes_path, spec.cells_path):
        if raw_path is None:
            continue
        path = Path(raw_path)
        fingerprint.update(str(path.resolve()).encode("utf-8"))
        try:
            stat = path.stat()
        except OSError:
            continue
        fingerprint.update(str(stat.st_size).encode("utf-8"))
        fingerprint.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    fingerprint.update(str(spec.file_format).encode("utf-8"))
    fingerprint.update(str(spec.transpose).encode("utf-8"))
    fingerprint.update(str(spec.labels_col).encode("utf-8"))
    fingerprint.update(str(spec.batches_col).encode("utf-8"))
    return fingerprint.hexdigest()


def _should_skip_discovery_path(path: Path, root: Path, exclude_dir_prefixes: tuple[str, ...]) -> bool:
    try:
        relative_parts = path.resolve().relative_to(root).parts
    except ValueError:
        relative_parts = path.parts

    normalized_parts = [part.lower() for part in relative_parts[:-1]]
    for part in normalized_parts:
        if any(part.startswith(prefix.lower()) for prefix in exclude_dir_prefixes):
            return True
    return False


def _make_dataset_id(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        relative = path
    stem = str(relative.with_suffix("")).replace("\\", "_").replace("/", "_")
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in stem)


def _find_sidecar_file(directory: Path, *, stems: tuple[str, ...]) -> Path | None:
    for stem in stems:
        for suffix in (".csv", ".tsv", ".txt"):
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _find_sidecar_file_nearby(directory: Path, *, stems: tuple[str, ...]) -> Path | None:
    for candidate_dir in (directory, directory.parent):
        candidate = _find_sidecar_file(candidate_dir, stems=stems)
        if candidate is not None:
            return candidate
    return None


def _infer_dataset_name(path: Path) -> str:
    if path.suffix.lower() == ".mtx" and path.stem.lower() == "matrix" and path.parent.name.lower() == "matrix":
        return path.parent.parent.name or path.stem
    return path.stem


def _infer_h5ad_dataset_name(path: Path) -> str:
    generic_stems = {"source", "raw", "data", "dataset", "matrix"}
    if path.stem.lower() in generic_stems and path.parent.name:
        return path.parent.name
    return path.stem


def _infer_mtx_transpose(path: Path, genes_path: Path | None, cells_path: Path | None) -> bool:
    if genes_path is None or cells_path is None:
        return False

    shape = _read_matrix_market_shape(path)
    if shape is None:
        return False

    n_rows, n_cols = shape
    gene_count = _count_lines(genes_path)
    cell_count = _count_lines(cells_path)
    if gene_count is None or cell_count is None:
        return False

    if n_rows == gene_count and n_cols == cell_count:
        return True
    if n_rows == cell_count and n_cols == gene_count:
        return False
    return False


def _read_matrix_market_shape(path: Path) -> tuple[int, int] | None:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    return None
            return None
    return None


def _count_lines(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def _infer_h5ad_annotation_columns(path: Path) -> tuple[str | None, str | None]:
    with h5py.File(path, "r") as handle:
        obs_df = _read_h5ad_dataframe(handle["obs"])
    return _guess_annotation_columns(obs_df.columns.tolist())


def _infer_table_annotation_columns(path: Path) -> tuple[str | None, str | None]:
    table = _read_table(path)
    return _guess_annotation_columns(table.columns.tolist())


def _guess_annotation_columns(columns: list[str]) -> tuple[str | None, str | None]:
    label_priorities = (
        "cell type",
        "cell_type",
        "cell label",
        "cell_label",
        "celltype",
        "label",
        "labels",
        "annotation",
        "annot",
        "clusters",
        "cluster",
        "cluster_id",
    )
    batch_priorities = (
        "batch",
        "batches",
        "sample_id",
        "sample id",
        "donor_id",
        "donor",
        "individual",
        "patient",
        "subject",
        "library",
        "lane",
    )
    lower_to_original = {str(column).lower(): str(column) for column in columns}
    labels_col = _match_annotation_column(lower_to_original, label_priorities)
    batches_col = _match_annotation_column(lower_to_original, batch_priorities)
    return labels_col, batches_col


def _match_annotation_column(lower_to_original: dict[str, str], priorities: tuple[str, ...]) -> str | None:
    for candidate in priorities:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    normalized_to_original = {
        _normalize_annotation_key(key): value
        for key, value in lower_to_original.items()
    }
    for candidate in priorities:
        normalized_candidate = _normalize_annotation_key(candidate)
        if normalized_candidate in normalized_to_original:
            return normalized_to_original[normalized_candidate]
    for candidate in priorities:
        normalized_candidate = _normalize_annotation_key(candidate)
        for key, value in lower_to_original.items():
            normalized_key = _normalize_annotation_key(key)
            if normalized_candidate and normalized_candidate in normalized_key:
                return value
    return None


def _normalize_annotation_key(value: str) -> str:
    normalized = "".join(char if char.isalnum() else " " for char in str(value).lower())
    return " ".join(normalized.split())


def _looks_like_count_matrix(path: Path) -> bool:
    if path.stem.lower() in {"obs", "metadata", "genes", "features", "barcodes", "cells", "labels", "batches"}:
        return False
    try:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        sample = pd.read_csv(path, sep=sep, nrows=24)
    except Exception:
        return False

    if sample.empty or sample.shape[1] < 2:
        return False

    numeric_ratio = pd.to_numeric(sample.iloc[:, 1:].stack(), errors="coerce").notna().mean()
    return bool(numeric_ratio > 0.8)


def _dataset_name_key(name: str) -> str:
    normalized = name.lower()
    for suffix in ("_counts", "_count", "_matrix", "_mtx"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def _format_priority(file_format: str) -> int:
    priority = {
        "h5ad": 0,
        "csv": 1,
        "tsv": 2,
        "txt": 3,
        "mtx": 4,
    }
    return priority.get(file_format.lower(), 9)
