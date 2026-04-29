from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmwrite

from .baselines import (
    score_mean_var_residual,
    score_scran_like_model_gene_var,
    score_seurat_v3_like_hvg,
)


DEFAULT_OFFICIAL_SCANPY_PYTHON = r"D:\code_py\hvg\.conda_stage5\python.exe"
DEFAULT_OFFICIAL_TRIKU_PYTHON = DEFAULT_OFFICIAL_SCANPY_PYTHON
DEFAULT_OFFICIAL_RSCRIPT = "Rscript"
REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_WORKER_SCRIPT = REPO_ROOT / "scripts" / "score_official_hvg.py"
R_WORKER_SCRIPT = REPO_ROOT / "scripts" / "score_official_hvg.R"
VENDORED_PY311_ROOT = REPO_ROOT / ".py311_vendor"
_DATA_CACHE: dict[tuple[int, int], dict[str, str]] = {}
_CACHE_ROOT = REPO_ROOT / ".official_hvg_cache"


class StrictBaselineUnavailableError(RuntimeError):
    """Raised when a strict baseline cannot run without fallback."""


def score_scanpy_seurat_v3_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    top_k: int,
    allow_fallback: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        scores, metadata = _score_with_external_python_worker(
            method="scanpy_seurat_v3_hvg",
            counts=counts,
            batches=batches,
            top_k=top_k,
            python_env_var="HVG_OFFICIAL_SCANPY_PYTHON",
            default_python=DEFAULT_OFFICIAL_SCANPY_PYTHON,
            include_vendor_path=False,
            strict_worker=not allow_fallback,
        )
        metadata.update(
            {
                "implementation_source": "scanpy.pp.highly_variable_genes(flavor='seurat_v3')",
                "official_requested_backend": "scanpy",
                "official_fallback_used": False,
                "batch_aware": True,
                "official_execution_mode": "strict" if not allow_fallback else "fallback_allowed",
            }
        )
        return scores, metadata
    except Exception as exc:
        if not allow_fallback:
            raise StrictBaselineUnavailableError(
                f"Strict official baseline unavailable for scanpy_seurat_v3_hvg: {exc}"
            ) from exc
        return _fallback_local_scores(
            method="scanpy_seurat_v3_hvg",
            counts=counts,
            fallback_method="seurat_v3_like_hvg",
            fallback_backend="python_seurat_v3_like_fallback",
            fallback_scorer=score_seurat_v3_like_hvg,
            top_k=top_k,
            source_exception=exc,
            implementation_source="scanpy.pp.highly_variable_genes(flavor='seurat_v3')",
            batch_aware=True,
        )


def score_scanpy_cell_ranger_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    top_k: int,
    allow_fallback: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        scores, metadata = _score_with_external_python_worker(
            method="scanpy_cell_ranger_hvg",
            counts=counts,
            batches=batches,
            top_k=top_k,
            python_env_var="HVG_OFFICIAL_SCANPY_PYTHON",
            default_python=DEFAULT_OFFICIAL_SCANPY_PYTHON,
            include_vendor_path=False,
            strict_worker=not allow_fallback,
        )
        metadata.update(
            {
                "implementation_source": "scanpy.pp.highly_variable_genes(flavor='cell_ranger')",
                "official_requested_backend": "scanpy",
                "official_fallback_used": False,
                "batch_aware": True,
                "official_execution_mode": "strict" if not allow_fallback else "fallback_allowed",
            }
        )
        return scores, metadata
    except Exception as exc:
        if not allow_fallback:
            raise StrictBaselineUnavailableError(
                f"Strict official baseline unavailable for scanpy_cell_ranger_hvg: {exc}"
            ) from exc
        return _fallback_local_scores(
            method="scanpy_cell_ranger_hvg",
            counts=counts,
            fallback_method="mv_residual",
            fallback_backend="python_mv_residual_fallback",
            fallback_scorer=score_mean_var_residual,
            top_k=top_k,
            source_exception=exc,
            implementation_source="scanpy.pp.highly_variable_genes(flavor='cell_ranger')",
            batch_aware=True,
        )


def score_triku_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    top_k: int,
    allow_fallback: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        scores, metadata = _score_with_external_python_worker(
            method="triku_hvg",
            counts=counts,
            batches=batches,
            top_k=top_k,
            python_env_var="HVG_OFFICIAL_TRIKU_PYTHON",
            default_python=DEFAULT_OFFICIAL_TRIKU_PYTHON,
            include_vendor_path=True,
            strict_worker=not allow_fallback,
        )
        metadata.update(
            {
                "implementation_source": "triku.tl.triku",
                "official_requested_backend": "triku",
                "official_fallback_used": False,
                "batch_aware": False,
                "official_execution_mode": "strict" if not allow_fallback else "fallback_allowed",
            }
        )
        return scores, metadata
    except Exception as exc:
        if not allow_fallback:
            raise StrictBaselineUnavailableError(
                f"Strict official baseline unavailable for triku_hvg: {exc}"
            ) from exc
        return _fallback_local_scores(
            method="triku_hvg",
            counts=counts,
            fallback_method="mv_residual",
            fallback_backend="python_mv_residual_fallback",
            fallback_scorer=score_mean_var_residual,
            top_k=top_k,
            source_exception=exc,
            implementation_source="triku.tl.triku",
            batch_aware=False,
        )


def score_seurat_r_vst_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    top_k: int,
    allow_fallback: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    del batches
    try:
        scores, metadata = _score_with_external_r_worker(
            method="seurat_r_vst_hvg",
            counts=counts,
            batches=None,
            top_k=top_k,
        )
        metadata.update(
            {
                "implementation_source": "Seurat::FindVariableFeatures(selection.method='vst')",
                "official_requested_backend": "Seurat/R",
                "official_fallback_used": False,
                "batch_aware": False,
                "official_execution_mode": "strict" if not allow_fallback else "fallback_allowed",
            }
        )
        return scores, metadata
    except Exception as exc:
        if not allow_fallback:
            raise StrictBaselineUnavailableError(
                f"Strict official baseline unavailable for seurat_r_vst_hvg: {exc}"
            ) from exc
        try:
            scores, metadata = score_scanpy_seurat_v3_hvg(counts=counts, batches=None, top_k=top_k)
            metadata = dict(metadata)
            metadata.update(
                _fallback_metadata(
                    method="seurat_r_vst_hvg",
                    fallback_method="scanpy_seurat_v3_hvg",
                    fallback_backend="scanpy_seurat_v3_fallback",
                    source_exception=exc,
                    implementation_source="Seurat::FindVariableFeatures(selection.method='vst')",
                    batch_aware=False,
                )
            )
            return scores, metadata
        except Exception as fallback_exc:
            return _fallback_local_scores(
                method="seurat_r_vst_hvg",
                counts=counts,
                fallback_method="seurat_v3_like_hvg",
                fallback_backend="python_seurat_v3_like_fallback",
                fallback_scorer=score_seurat_v3_like_hvg,
                top_k=top_k,
                source_exception=fallback_exc,
                upstream_exception=exc,
                implementation_source="Seurat::FindVariableFeatures(selection.method='vst')",
                batch_aware=False,
            )


def score_scran_model_gene_var_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    top_k: int,
    allow_fallback: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        scores, metadata = _score_with_external_r_worker(
            method="scran_model_gene_var_hvg",
            counts=counts,
            batches=batches,
            top_k=top_k,
        )
        metadata.update(
            {
                "implementation_source": "scran::modelGeneVar",
                "official_requested_backend": "scran/R",
                "official_fallback_used": False,
                "batch_aware": True,
                "official_execution_mode": "strict" if not allow_fallback else "fallback_allowed",
            }
        )
        return scores, metadata
    except Exception as exc:
        if not allow_fallback:
            raise StrictBaselineUnavailableError(
                f"Strict official baseline unavailable for scran_model_gene_var_hvg: {exc}"
            ) from exc
        return _fallback_local_scores(
            method="scran_model_gene_var_hvg",
            counts=counts,
            fallback_method="scran_like_model_gene_var_hvg",
            fallback_backend="python_scran_like_fallback",
            fallback_scorer=score_scran_like_model_gene_var,
            top_k=top_k,
            source_exception=exc,
            implementation_source="scran::modelGeneVar",
            batch_aware=batches is not None and len(np.unique(np.asarray(batches, dtype=object))) >= 2,
        )


def has_scran_worker() -> bool:
    return _resolve_executable(os.environ.get("HVG_OFFICIAL_RSCRIPT", DEFAULT_OFFICIAL_RSCRIPT)) is not None and R_WORKER_SCRIPT.exists()


def has_seurat_r_worker() -> bool:
    return has_scran_worker()


def _score_with_external_python_worker(
    *,
    method: str,
    counts: np.ndarray,
    batches: np.ndarray | None,
    top_k: int,
    python_env_var: str,
    default_python: str,
    include_vendor_path: bool,
    strict_worker: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    python_candidate = os.environ.get(python_env_var, default_python)
    python_exe = _resolve_executable(python_candidate)
    if python_exe is None:
        raise FileNotFoundError(
            f"Official baseline interpreter not found: {python_candidate}. "
            f"Set {python_env_var} to a Python with the required baseline dependencies installed."
        )
    if not PYTHON_WORKER_SCRIPT.exists():
        raise FileNotFoundError(f"Official Python worker script not found: {PYTHON_WORKER_SCRIPT}")

    cache_entry = _materialize_inputs(counts=counts, batches=batches)
    cache_dir = Path(cache_entry["cache_dir"])
    output_path = cache_dir / f"{method}_top{int(top_k)}_scores.npy"
    metadata_path = cache_dir / f"{method}_top{int(top_k)}_metadata.json"

    command = [
        python_exe,
        str(PYTHON_WORKER_SCRIPT),
        "--method",
        method,
        "--counts-path",
        cache_entry["counts_path"],
        "--top-k",
        str(int(top_k)),
        "--output-path",
        str(output_path),
        "--metadata-path",
        str(metadata_path),
    ]
    if cache_entry["batches_path"]:
        command.extend(["--batches-path", cache_entry["batches_path"]])
    if strict_worker:
        command.append("--strict")

    env = os.environ.copy()
    if include_vendor_path and VENDORED_PY311_ROOT.exists():
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(VENDORED_PY311_ROOT)
            if not existing_pythonpath
            else os.pathsep.join((str(VENDORED_PY311_ROOT), existing_pythonpath))
        )

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Official Python baseline failed.\n"
            f"method={method}\n"
            f"stdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )
    scores = np.load(output_path)
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if strict_worker and _worker_metadata_uses_fallback(metadata):
        raise RuntimeError(f"Strict worker unexpectedly reported fallback metadata for {method}.")
    metadata["official_worker_python"] = python_exe
    metadata["official_worker_script"] = str(PYTHON_WORKER_SCRIPT)
    metadata["official_worker_strict"] = bool(strict_worker)
    if include_vendor_path and VENDORED_PY311_ROOT.exists():
        metadata["official_worker_pythonpath"] = str(VENDORED_PY311_ROOT)
    return np.asarray(scores, dtype=np.float64), metadata


def _score_with_external_r_worker(
    *,
    method: str,
    counts: np.ndarray,
    batches: np.ndarray | None,
    top_k: int,
) -> tuple[np.ndarray, dict[str, object]]:
    rscript_candidate = os.environ.get("HVG_OFFICIAL_RSCRIPT", DEFAULT_OFFICIAL_RSCRIPT)
    rscript_exe = _resolve_executable(rscript_candidate)
    if rscript_exe is None:
        raise FileNotFoundError(
            f"Official R worker not found: {rscript_candidate}. "
            "Set HVG_OFFICIAL_RSCRIPT to an Rscript executable with scran/scuttle/SingleCellExperiment installed."
        )
    if not R_WORKER_SCRIPT.exists():
        raise FileNotFoundError(f"Official R worker script not found: {R_WORKER_SCRIPT}")

    cache_entry = _materialize_inputs(counts=counts, batches=batches)
    cache_dir = Path(cache_entry["cache_dir"])
    output_path = cache_dir / f"{method}_top{int(top_k)}_scores.txt"
    metadata_path = cache_dir / f"{method}_top{int(top_k)}_metadata.json"

    command = [
        rscript_exe,
        str(R_WORKER_SCRIPT),
        "--method",
        method,
        "--counts-mtx-path",
        cache_entry["counts_mtx_path"],
        "--top-k",
        str(int(top_k)),
        "--output-path",
        str(output_path),
        "--metadata-path",
        str(metadata_path),
    ]
    if cache_entry["batches_tsv_path"]:
        command.extend(["--batches-path", cache_entry["batches_tsv_path"]])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Official R baseline failed.\n"
            f"method={method}\n"
            f"stdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )

    scores = np.loadtxt(output_path, dtype=np.float64)
    if np.ndim(scores) == 0:
        scores = np.asarray([float(scores)], dtype=np.float64)

    metadata: dict[str, object] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["official_worker_rscript"] = rscript_exe
    metadata["official_worker_script"] = str(R_WORKER_SCRIPT)
    return np.asarray(scores, dtype=np.float64), metadata


def _fallback_local_scores(
    *,
    method: str,
    counts: np.ndarray,
    fallback_method: str,
    fallback_backend: str,
    fallback_scorer,
    top_k: int,
    source_exception: Exception,
    implementation_source: str,
    batch_aware: bool,
    upstream_exception: Exception | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    del top_k
    scores = np.asarray(fallback_scorer(np.asarray(counts, dtype=np.float32)), dtype=np.float64)
    metadata = _fallback_metadata(
        method=method,
        fallback_method=fallback_method,
        fallback_backend=fallback_backend,
        source_exception=source_exception,
        implementation_source=implementation_source,
        batch_aware=batch_aware,
        upstream_exception=upstream_exception,
    )
    metadata["official_backend"] = fallback_backend
    return scores, metadata


def _fallback_metadata(
    *,
    method: str,
    fallback_method: str,
    fallback_backend: str,
    source_exception: Exception,
    implementation_source: str,
    batch_aware: bool,
    upstream_exception: Exception | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "method": method,
        "implementation_source": implementation_source,
        "official_backend": fallback_backend,
        "official_requested_backend": implementation_source,
        "official_fallback_used": True,
        "official_fallback_target": fallback_method,
        "official_fallback_reason": type(source_exception).__name__,
        "official_fallback_message": str(source_exception),
        "batch_aware": bool(batch_aware),
        "fallback_impl_version": "repo-local",
        "official_execution_mode": "fallback_allowed",
    }
    if upstream_exception is not None:
        metadata["official_upstream_exception_type"] = type(upstream_exception).__name__
        metadata["official_upstream_exception_message"] = str(upstream_exception)
    return metadata


def _materialize_inputs(*, counts: np.ndarray, batches: np.ndarray | None) -> dict[str, str]:
    counts_array = np.asarray(counts, dtype=np.float32)
    cache_key = (int(counts_array.__array_interface__["data"][0]), int(counts_array.shape[0] * counts_array.shape[1]))
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_dir = _CACHE_ROOT / f"{abs(cache_key[0])}_{cache_key[1]}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts_path = cache_dir / "counts.npy"
    if not counts_path.exists():
        np.save(counts_path, counts_array)
    counts_mtx_path = cache_dir / "counts.mtx"
    if not counts_mtx_path.exists():
        mmwrite(str(counts_mtx_path), sparse.coo_matrix(counts_array))
    batches_path = ""
    batches_tsv_path = ""
    if batches is not None:
        batches_array = np.asarray(batches, dtype=object)
        batches_path = str(cache_dir / "batches.npy")
        if not Path(batches_path).exists():
            np.save(batches_path, batches_array, allow_pickle=True)
        batches_tsv_path = str(cache_dir / "batches.tsv")
        if not Path(batches_tsv_path).exists():
            Path(batches_tsv_path).write_text("\n".join(str(value) for value in batches_array.tolist()), encoding="utf-8")
    payload = {
        "cache_dir": str(cache_dir),
        "counts_path": str(counts_path),
        "counts_mtx_path": str(counts_mtx_path),
        "batches_path": batches_path,
        "batches_tsv_path": batches_tsv_path,
    }
    _DATA_CACHE[cache_key] = payload
    return payload


def _resolve_executable(candidate: str) -> str | None:
    path = Path(candidate)
    if path.exists():
        return str(path)
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    return None


def _worker_metadata_uses_fallback(metadata: dict[str, object]) -> bool:
    fallback_flags = (
        "batch_fallback_used",
        "duplicate_bin_fallback_used",
        "duplicate_bin_jitter_fallback_used",
    )
    return any(bool(metadata.get(flag, False)) for flag in fallback_flags)
