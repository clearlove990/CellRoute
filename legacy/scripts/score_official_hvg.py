from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_PY311_ROOT = REPO_ROOT / ".py311_vendor"
if VENDORED_PY311_ROOT.exists():
    vendor_path = str(VENDORED_PY311_ROOT)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.n_jobs = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official scanpy HVG scorers in the external scanpy environment.")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--counts-path", type=str, required=True)
    parser.add_argument("--batches-path", type=str, default=None)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = np.load(args.counts_path, mmap_mode="r")
    batches = None
    if args.batches_path:
        batches = np.load(args.batches_path, allow_pickle=True)

    scores, metadata = score_method(
        method=args.method,
        counts=counts,
        batches=batches,
        top_k=int(args.top_k),
        strict=bool(args.strict),
    )
    np.save(args.output_path, np.asarray(scores, dtype=np.float64))
    if args.metadata_path:
        Path(args.metadata_path).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def score_method(
    *,
    method: str,
    counts: np.ndarray,
    batches: np.ndarray | None,
    top_k: int,
    strict: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    metadata = {
        "method": method,
        "top_k": int(top_k),
        "worker_strict_mode": bool(strict),
    }

    if method == "triku_hvg":
        return score_triku(
            counts=np.array(counts, dtype=np.float32, copy=True),
            top_k=int(top_k),
            strict=bool(strict),
        )

    adata = ad.AnnData(X=np.array(counts, dtype=np.float32, copy=True))
    if method == "scanpy_seurat_v3_hvg":
        metadata["official_backend"] = "scanpy"
        metadata["scanpy_version"] = str(sc.__version__)
        batch_key = attach_batch_key(adata=adata, batches=batches)
        metadata["used_batch_key"] = batch_key or ""
        run_highly_variable_genes(
            adata,
            flavor="seurat_v3",
            top_k=int(top_k),
            batch_key=batch_key,
            metadata=metadata,
            strict=bool(strict),
        )
        score_key = "variances_norm"
    elif method == "scanpy_cell_ranger_hvg":
        metadata["official_backend"] = "scanpy"
        metadata["scanpy_version"] = str(sc.__version__)
        batch_key = attach_batch_key(adata=adata, batches=batches)
        metadata["used_batch_key"] = batch_key or ""
        run_highly_variable_genes(
            adata,
            flavor="cell_ranger",
            top_k=int(top_k),
            batch_key=batch_key,
            metadata=metadata,
            strict=bool(strict),
        )
        score_key = "dispersions_norm"
    else:
        raise ValueError(f"Unsupported official baseline method: {method}")

    if score_key not in adata.var.columns:
        raise KeyError(f"Expected score column '{score_key}' in adata.var.")

    scores = adata.var[score_key].to_numpy(dtype=np.float64, na_value=np.nan)
    nan_mask = ~np.isfinite(scores)
    if np.any(nan_mask):
        scores[nan_mask] = np.nanmin(scores[np.isfinite(scores)]) - 1.0 if np.any(np.isfinite(scores)) else 0.0
    metadata["score_key"] = score_key
    metadata["highly_variable_count"] = int(np.sum(adata.var["highly_variable"].to_numpy(dtype=bool)))
    return scores, metadata


def attach_batch_key(*, adata: ad.AnnData, batches: np.ndarray | None) -> str | None:
    if batches is None:
        return None
    batch_values = np.asarray(batches, dtype=object)
    if len(np.unique(batch_values)) < 2:
        return None
    adata.obs["benchmark_batch"] = pd.Categorical(batch_values.astype(str))
    return "benchmark_batch"


def run_highly_variable_genes(
    adata: ad.AnnData,
    *,
    flavor: str,
    top_k: int,
    batch_key: str | None,
    metadata: dict[str, object],
    strict: bool,
) -> None:
    base_kwargs = {
        "n_top_genes": int(top_k),
        "flavor": flavor,
        "subset": False,
        "inplace": True,
    }
    try:
        sc.pp.highly_variable_genes(
            adata,
            batch_key=batch_key,
            **base_kwargs,
        )
        metadata["batch_fallback_used"] = False
        metadata["duplicate_bin_fallback_used"] = False
    except ValueError as exc:
        if "Bin edges must be unique" not in str(exc):
            raise
        if strict:
            raise
        duplicate_bin_n_bins = min(10, max(5, adata.n_vars // 50 if adata.n_vars >= 50 else adata.n_vars))
        duplicate_bin_n_bins = max(2, int(duplicate_bin_n_bins))

        fallback_attempts: list[tuple[str | None, dict[str, object]]] = []
        if batch_key is not None:
            fallback_attempts.append(
                (
                    None,
                    {
                        "batch_fallback_used": True,
                        "batch_fallback_reason": "scanpy_duplicate_bin_edges",
                        "duplicate_bin_fallback_used": False,
                    },
                )
            )
        fallback_attempts.append(
            (
                None,
                {
                    "batch_fallback_used": batch_key is not None,
                    "batch_fallback_reason": "scanpy_duplicate_bin_edges" if batch_key is not None else "",
                    "duplicate_bin_fallback_used": True,
                    "duplicate_bin_fallback_n_bins": duplicate_bin_n_bins,
                },
            )
        )

        last_exc: Exception | None = exc
        for retry_batch_key, retry_metadata in fallback_attempts:
            retry_kwargs = dict(base_kwargs)
            if retry_metadata.get("duplicate_bin_fallback_used"):
                retry_kwargs["n_bins"] = duplicate_bin_n_bins
            try:
                sc.pp.highly_variable_genes(
                    adata,
                    batch_key=retry_batch_key,
                    **retry_kwargs,
                )
                metadata.update(retry_metadata)
                return
            except ValueError as retry_exc:
                last_exc = retry_exc
                if "Bin edges must be unique" not in str(retry_exc):
                    raise
        adata_jitter = adata.copy()
        adata_jitter.X = np.asarray(adata_jitter.X, dtype=np.float64) + np.linspace(
            0.0,
            1e-8,
            adata_jitter.n_vars,
            dtype=np.float64,
        )[None, :]
        try:
            sc.pp.highly_variable_genes(
                adata_jitter,
                batch_key=None,
                n_bins=duplicate_bin_n_bins,
                **base_kwargs,
            )
            adata.var = adata_jitter.var.copy()
            metadata.update(
                {
                    "batch_fallback_used": batch_key is not None,
                    "batch_fallback_reason": "scanpy_duplicate_bin_edges" if batch_key is not None else "",
                    "duplicate_bin_fallback_used": True,
                    "duplicate_bin_fallback_n_bins": duplicate_bin_n_bins,
                    "duplicate_bin_jitter_fallback_used": True,
                }
            )
            return
        except ValueError as jitter_exc:
            last_exc = jitter_exc
            if "Bin edges must be unique" not in str(jitter_exc):
                raise
        raise last_exc


def score_triku(
    *,
    counts: np.ndarray,
    top_k: int,
    strict: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    n_cells, n_genes = counts.shape
    metadata: dict[str, object] = {
        "official_backend": "triku",
        "method": "triku_hvg",
        "top_k": int(top_k),
        "used_batch_key": "",
        "use_raw": False,
        "scanpy_n_jobs": int(sc.settings.n_jobs),
        "neighbors_backend": "sklearn",
        "vendor_path": str(VENDORED_PY311_ROOT) if VENDORED_PY311_ROOT.exists() else "",
        "worker_strict_mode": bool(strict),
    }
    try:
        metadata["triku_version"] = importlib_metadata.version("triku")
    except importlib_metadata.PackageNotFoundError:
        metadata["triku_version"] = ""

    if n_cells < 3 or n_genes < 2:
        metadata["score_key"] = "triku_distance"
        metadata["highly_variable_count"] = 0
        metadata["n_neighbors"] = 0
        metadata["n_pcs"] = 0
        metadata["n_features_used"] = 0
        metadata["min_knn"] = 0
        return np.zeros(n_genes, dtype=np.float64), metadata

    import triku as tk

    adata = ad.AnnData(X=np.array(counts, dtype=np.float32, copy=True))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_neighbors = max(2, min(30, n_cells - 1))
    min_knn = max(1, min(6, n_neighbors))
    if min(n_cells, n_genes) > 2:
        n_pcs = max(2, min(50, min(n_cells, n_genes) - 1))
        sc.pp.pca(adata, n_comps=n_pcs)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, transformer="sklearn")
    else:
        n_pcs = 0
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", transformer="sklearn")

    n_features_used = max(1, min(int(top_k), max(n_genes - 1, 1)))
    tk.tl.triku(
        adata,
        n_features=n_features_used,
        use_raw=False,
        min_knn=min_knn,
        verbose="error",
    )

    score_key = "triku_distance"
    if score_key not in adata.var.columns:
        raise KeyError(f"Expected score column '{score_key}' in adata.var after triku.")

    scores = adata.var[score_key].to_numpy(dtype=np.float64, na_value=np.nan)
    nan_mask = ~np.isfinite(scores)
    if np.any(nan_mask):
        scores[nan_mask] = np.nanmin(scores[np.isfinite(scores)]) - 1.0 if np.any(np.isfinite(scores)) else 0.0

    metadata["score_key"] = score_key
    metadata["highly_variable_count"] = int(np.sum(adata.var["highly_variable"].to_numpy(dtype=bool)))
    metadata["n_neighbors"] = int(n_neighbors)
    metadata["n_pcs"] = int(n_pcs)
    metadata["n_features_used"] = int(n_features_used)
    metadata["min_knn"] = int(min_knn)
    return scores, metadata


if __name__ == "__main__":
    main()
