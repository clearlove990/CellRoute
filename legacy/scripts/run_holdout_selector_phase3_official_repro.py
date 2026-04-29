from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for candidate in (str(SRC), str(SCRIPTS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hvg_research import build_default_method_registry, choose_torch_device, discover_scrna_input_specs, evaluate_real_selection, load_scrna_dataset
from hvg_research.eval import timed_call
from hvg_research.holdout_selector import RiskControlledSelectorConfig
from hvg_research.official_baselines import (
    DEFAULT_OFFICIAL_RSCRIPT,
    DEFAULT_OFFICIAL_SCANPY_PYTHON,
    VENDORED_PY311_ROOT,
)

import build_topconf_selector_round2_package as pkg
import run_holdout_selector_upgrade as phase1
import run_real_inputs_round1 as rr1


DEFAULT_OUTPUT_DIR = ROOT / "artifacts_codex_selector_phase3_official_repro"
SAFE_ANCHOR_METHOD = "adaptive_hybrid_hvg"
BANK_METHODS = (
    "scran_model_gene_var_hvg",
    "seurat_r_vst_hvg",
    "scanpy_cell_ranger_hvg",
    "scanpy_seurat_v3_hvg",
    "triku_hvg",
    "multinomial_deviance_hvg",
)
STRICT_SELECTOR_LABELS = {
    "strict_official_available",
    "strict_reproduced_from_paper_or_release",
}


@dataclass(frozen=True)
class MethodNote:
    legacy_label: str
    source: str
    dependencies: str
    parameters: str
    preprocessing: str
    difference_from_repo_local: str
    note: str = ""


METHOD_NOTES: dict[str, MethodNote] = {
    "scran_model_gene_var_hvg": MethodNote(
        legacy_label="official_declared_but_fallback_used",
        source="scran::modelGeneVar via the repo R worker (`scripts/score_official_hvg.R`).",
        dependencies="`Rscript` plus `scran`, `scuttle`, `SingleCellExperiment`, `Matrix`.",
        parameters="`top_k` is passed through for bookkeeping; score uses the `bio` column from `modelGeneVar`; `block` is enabled when at least two batches are present.",
        preprocessing="Raw count matrix is transposed into genes x cells for R, wrapped in `SingleCellExperiment`, then normalized with `scuttle::logNormCounts` before `scran::modelGeneVar`.",
        difference_from_repo_local="Legacy execution silently fell back to `score_scran_like_model_gene_var`; phase3 strict mode forbids that fallback and reports unavailable when the R worker is missing.",
    ),
    "seurat_r_vst_hvg": MethodNote(
        legacy_label="official_declared_but_fallback_used",
        source="Seurat `FindVariableFeatures(selection.method='vst')` via the repo R worker (`scripts/score_official_hvg.R`).",
        dependencies="`Rscript` plus `Seurat` and `Matrix`.",
        parameters="`nfeatures=top_k`; score comes from `HVFInfo` standardized-variance columns when present.",
        preprocessing="Raw count matrix is transposed into genes x cells for R and wrapped in a Seurat object with assay `RNA`; no batch wrapper is applied in the current strict path.",
        difference_from_repo_local="Legacy execution silently fell back first to `scanpy_seurat_v3_hvg` and then to `seurat_v3_like_hvg`; phase3 strict mode forbids both fallbacks and reports unavailable when Seurat/R is absent.",
    ),
    "scanpy_cell_ranger_hvg": MethodNote(
        legacy_label="strict_official_available",
        source="`scanpy.pp.highly_variable_genes(flavor='cell_ranger')` via `scripts/score_official_hvg.py`.",
        dependencies=f"`{DEFAULT_OFFICIAL_SCANPY_PYTHON}` with Scanpy installed.",
        parameters="`n_top_genes=top_k`, `subset=False`, `inplace=True`; `batch_key` is used only when at least two batches are present.",
        preprocessing="Raw counts are loaded into AnnData inside the external Scanpy worker; the worker reads the raw matrix directly and does not log-transform first.",
        difference_from_repo_local="Phase3 strict mode disables the worker-side duplicate-bin retries, batch dropping, and jitter fallback. If the official call fails, the method is marked unavailable instead of falling back to `mv_residual`.",
    ),
    "scanpy_seurat_v3_hvg": MethodNote(
        legacy_label="strict_official_available",
        source="`scanpy.pp.highly_variable_genes(flavor='seurat_v3')` via `scripts/score_official_hvg.py`.",
        dependencies=f"`{DEFAULT_OFFICIAL_SCANPY_PYTHON}` with Scanpy installed.",
        parameters="`n_top_genes=top_k`, `subset=False`, `inplace=True`; `batch_key` is used only when at least two batches are present.",
        preprocessing="Raw counts are loaded into AnnData inside the external Scanpy worker; the worker reads the raw matrix directly and does not log-transform first.",
        difference_from_repo_local="Phase3 strict mode disables the worker-side duplicate-bin retries, batch dropping, and jitter fallback. If the official call fails, the method is marked unavailable instead of falling back to `seurat_v3_like_hvg`.",
    ),
    "triku_hvg": MethodNote(
        legacy_label="strict_official_available",
        source="`triku.tl.triku` via `scripts/score_official_hvg.py` with the vendored Triku package path when needed.",
        dependencies=f"`{DEFAULT_OFFICIAL_SCANPY_PYTHON}` plus the vendored Triku path `{VENDORED_PY311_ROOT}`.",
        parameters="`n_features=top_k` clipped by gene count, `use_raw=False`, `min_knn` and PCA/neighbors are chosen from dataset size in the worker.",
        preprocessing="Worker normalizes counts with `scanpy.pp.normalize_total`, applies `log1p`, computes PCA/neighbors, then calls `triku.tl.triku` on the AnnData object.",
        difference_from_repo_local="Phase3 continues to use the actual Triku package path and forbids any fallback to `mv_residual` when the worker fails.",
    ),
    "multinomial_deviance_hvg": MethodNote(
        legacy_label="repo_local_published_only",
        source="Repo-local paper-style multinomial/binomial deviance reproduction in `src/hvg_research/baselines.py`.",
        dependencies="NumPy plus optional PyTorch CUDA acceleration; no external R/Python baseline package is required.",
        parameters="`gene_chunk_size=512` by default; scores are z-scored gene-wise deviances under the global multinomial null.",
        preprocessing="Method consumes raw counts directly, estimates gene probabilities from total counts, then computes binomial deviance terms using per-cell library sizes.",
        difference_from_repo_local="Phase3 does not swap in a different heuristic. It promotes the existing count-model implementation into an explicit strict-reproduced tier with no official wrapper claim and no fallback path.",
        note="This remains a repo-local reproduction rather than an official package call, so the defended claim is 'strict reproduced' rather than 'strict official'.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase3 strict/reproduced bank audit and selector evaluation.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--real-data-root", type=str, default="data/real_inputs")
    parser.add_argument("--gate-model-path", type=str, default=rr1.DEFAULT_GATE_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=1)
    parser.add_argument("--refine-epochs", type=int, default=6)
    parser.add_argument("--robust-seeds", type=str, default="7,17,27")
    parser.add_argument("--robust-topks", type=str, default="100,200,500")
    return parser.parse_args()


def parse_int_list(spec: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in spec.split(",") if part.strip())


def save_device_info(*, output_dir: Path) -> None:
    cuda_available = torch.cuda.is_available()
    payload: dict[str, object] = {
        "device": str(choose_torch_device()),
        "cuda_available": bool(cuda_available),
        "cuda_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        payload["cuda_devices"] = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
    (output_dir / "device_info.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_round2_specs(*, real_data_root: Path) -> list:
    spec_map = {spec.dataset_name: spec for spec in discover_scrna_input_specs(real_data_root)}
    missing = sorted(set(rr1.ROUND2_PLANS) - set(spec_map))
    if missing:
        raise FileNotFoundError(f"Missing round2 dataset specs: {missing}")
    return [spec_map[name] for name in sorted(rr1.ROUND2_PLANS)]


def safe_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    try:
        return bool(value)
    except Exception:
        return False


def join_unique(values: list[object]) -> str:
    seen: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.append(text)
    return "; ".join(seen)


def method_base_row(*, dataset, dataset_id: str, spec, method_name: str, top_k: int, seed: int) -> dict[str, object]:
    return {
        "dataset": dataset.name,
        "dataset_id": dataset_id,
        "dataset_name": spec.dataset_name,
        "input_path": spec.input_path,
        "file_format": spec.file_format,
        "transpose": spec.transpose,
        "labels_col": spec.labels_col or "",
        "batches_col": spec.batches_col or "",
        "cells": int(dataset.counts.shape[0]),
        "genes": int(dataset.counts.shape[1]),
        "method": method_name,
        "seed": int(seed),
        "top_k": int(top_k),
    }


def evaluate_dataset_methods(
    *,
    dataset,
    dataset_id: str,
    spec,
    method_names: tuple[str, ...],
    gate_model_path: str | None,
    refine_epochs: int,
    top_k: int,
    seed: int,
    bootstrap_samples: int,
    official_baseline_strict: bool,
) -> pd.DataFrame:
    registry = build_default_method_registry(
        top_k=top_k,
        refine_epochs=refine_epochs,
        random_state=seed,
        gate_model_path=gate_model_path,
        official_baseline_strict=official_baseline_strict,
    )
    current_top_k = min(int(top_k), dataset.counts.shape[1])
    labels = None if dataset.labels is None else np.asarray(dataset.labels, dtype=object)
    markers: dict[str, set[int]] = {}
    class_weights: dict[str, float] = {}
    if labels is not None:
        markers, class_weights = pkg.compute_one_vs_rest_markers(
            counts=dataset.counts,
            labels=labels,
            top_n=50,
        )

    rows: list[dict[str, object]] = []
    for method_name in method_names:
        method_fn = registry[method_name]
        row = method_base_row(
            dataset=dataset,
            dataset_id=dataset_id,
            spec=spec,
            method_name=method_name,
            top_k=current_top_k,
            seed=seed,
        )
        row["analysis_group"] = "anchor" if method_name == SAFE_ANCHOR_METHOD else "strict_repro_bank"
        row["strict_bank_requested"] = bool(method_name in BANK_METHODS)
        row["result_available"] = False
        row["official_fallback_used"] = False
        row["error_type"] = ""
        row["error_message"] = ""
        try:
            scores, elapsed = timed_call(method_fn, dataset.counts, dataset.batches, current_top_k)
            selected = np.argsort(scores)[-current_top_k:]
            metadata = rr1.extract_method_metadata(method_fn)
            metrics = evaluate_real_selection(
                counts=dataset.counts,
                selected_genes=selected,
                scorer_fn=lambda subset_counts, subset_batches, fn=method_fn, local_top_k=current_top_k: fn(
                    subset_counts,
                    subset_batches,
                    local_top_k,
                ),
                labels=dataset.labels,
                batches=dataset.batches,
                top_k=current_top_k,
                random_state=seed,
                n_bootstrap=bootstrap_samples,
            )
            row["runtime_sec"] = float(elapsed)
            row["result_available"] = True
            row.update({key: float(value) for key, value in metrics.items()})
            if markers:
                recall, weighted, rare = pkg.marker_recovery(
                    selected=selected,
                    marker_sets=markers,
                    class_weights=class_weights,
                )
                row["marker_recall_at_50"] = float(recall)
                row["weighted_marker_recall_at_50"] = float(weighted)
                row["rare_marker_recall_at_50"] = float(rare)
            else:
                row["marker_recall_at_50"] = np.nan
                row["weighted_marker_recall_at_50"] = np.nan
                row["rare_marker_recall_at_50"] = np.nan
            row.update(metadata)
            row["official_fallback_used"] = safe_bool(row.get("official_fallback_used", False))
            row["status"] = "available"
        except Exception as exc:  # pragma: no cover - experiment fault capture
            row["runtime_sec"] = np.nan
            row["marker_recall_at_50"] = np.nan
            row["weighted_marker_recall_at_50"] = np.nan
            row["rare_marker_recall_at_50"] = np.nan
            row["status"] = "failed"
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)
        rows.append(row)

    available_df = pd.DataFrame([row for row in rows if row["status"] == "available"])
    score_lookup: dict[str, dict[str, object]] = {}
    if not available_df.empty:
        scored_df = rr1.add_run_level_scores(available_df)
        score_lookup = scored_df.set_index("method", drop=False)[["overall_score", "overall_rank", "is_winner"]].to_dict(orient="index")
    for row in rows:
        scored = score_lookup.get(row["method"])
        if scored is None:
            row["overall_score"] = np.nan
            row["overall_rank"] = np.nan
            row["is_winner"] = False
        else:
            row["overall_score"] = float(scored["overall_score"])
            row["overall_rank"] = int(scored["overall_rank"])
            row["is_winner"] = bool(scored["is_winner"])
    return pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)


def run_main_bank_benchmark(
    *,
    output_dir: Path,
    real_data_root: Path,
    gate_model_path: str | None,
    refine_epochs: int,
    top_k: int,
    seed: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    final_path = output_dir / "strict_repro_benchmark_raw.csv"
    if final_path.exists():
        return pd.read_csv(final_path)
    method_names = tuple([SAFE_ANCHOR_METHOD, *BANK_METHODS])
    specs = load_round2_specs(real_data_root=real_data_root)
    partial_path = output_dir / "_phase3_bank_partial.csv"
    frames: list[pd.DataFrame] = []
    for spec in specs:
        plan = rr1.ROUND2_PLANS[spec.dataset_name]
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
            max_cells=plan.max_cells,
            max_genes=plan.max_genes,
            random_state=seed,
        )
        dataset_df = evaluate_dataset_methods(
            dataset=dataset,
            dataset_id=str(spec.dataset_id),
            spec=spec,
            method_names=method_names,
            gate_model_path=gate_model_path,
            refine_epochs=refine_epochs,
            top_k=top_k,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            official_baseline_strict=True,
        )
        frames.append(dataset_df)
        pd.concat(frames, ignore_index=True).to_csv(partial_path, index=False)
    benchmark_df = pd.concat(frames, ignore_index=True).sort_values(["dataset", "method"]).reset_index(drop=True)
    benchmark_df.to_csv(final_path, index=False)
    return benchmark_df


def build_method_availability_table(*, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "method",
        "status",
        "result_available",
        "runtime_sec",
        "official_execution_mode",
        "official_fallback_used",
        "error_type",
        "error_message",
        "implementation_source",
        "official_backend",
        "official_requested_backend",
    ]
    existing = [column for column in columns if column in benchmark_df.columns]
    return benchmark_df[benchmark_df["method"].isin(BANK_METHODS)][existing].copy().sort_values(["method", "dataset"]).reset_index(drop=True)


def build_holdout_results(*, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    available = benchmark_df[benchmark_df["status"] == "available"].copy()
    anchor = available[available["method"] == SAFE_ANCHOR_METHOD][
        [
            "dataset",
            "overall_score",
            "weighted_marker_recall_at_50",
            "runtime_sec",
        ]
    ].rename(
        columns={
            "overall_score": "anchor_overall_score",
            "weighted_marker_recall_at_50": "anchor_biology_proxy",
            "runtime_sec": "anchor_runtime_sec",
        }
    )
    bank = available[available["method"].isin(BANK_METHODS)].copy()
    results = bank.merge(anchor, on="dataset", how="left")
    results["overall_delta_vs_anchor"] = results["overall_score"] - results["anchor_overall_score"]
    results["biology_delta_vs_anchor"] = results["weighted_marker_recall_at_50"] - results["anchor_biology_proxy"]
    results["runtime_delta_vs_anchor"] = results["runtime_sec"] - results["anchor_runtime_sec"]
    return results.sort_values(["method", "dataset"]).reset_index(drop=True)


def build_holdout_summary(*, benchmark_df: pd.DataFrame, holdout_results: pd.DataFrame) -> pd.DataFrame:
    total_datasets = int(benchmark_df["dataset"].nunique())
    rows: list[dict[str, object]] = []
    anchor_rows = benchmark_df[
        (benchmark_df["method"] == SAFE_ANCHOR_METHOD)
        & (benchmark_df["status"] == "available")
    ]
    rows.append(
        {
            "method": SAFE_ANCHOR_METHOD,
            "is_anchor": True,
            "available_dataset_count": int(len(anchor_rows)),
            "dataset_count": total_datasets,
            "availability_rate": float(len(anchor_rows) / max(total_datasets, 1)),
            "mean_overall_score": float(np.nanmean(anchor_rows["overall_score"].to_numpy(dtype=np.float64))),
            "mean_biology_proxy": float(np.nanmean(anchor_rows["weighted_marker_recall_at_50"].to_numpy(dtype=np.float64))),
            "mean_runtime_sec": float(np.nanmean(anchor_rows["runtime_sec"].to_numpy(dtype=np.float64))),
            "mean_delta_vs_anchor": 0.0,
            "mean_biology_delta_vs_anchor": 0.0,
            "mean_runtime_delta_vs_anchor": 0.0,
            "fallback_use_count": 0,
            "fallback_use_rate": 0.0,
            "failed_dataset_count": 0,
            "failed_datasets": "",
            "failure_types": "",
        }
    )
    for method_name in BANK_METHODS:
        method_rows = benchmark_df[benchmark_df["method"] == method_name].copy()
        available_rows = method_rows[method_rows["status"] == "available"]
        result_rows = holdout_results[holdout_results["method"] == method_name]
        failed_rows = method_rows[method_rows["status"] != "available"]
        fallback_count = int(np.sum(method_rows["official_fallback_used"].fillna(False).astype(bool))) if "official_fallback_used" in method_rows.columns else 0
        rows.append(
            {
                "method": method_name,
                "is_anchor": False,
                "available_dataset_count": int(len(available_rows)),
                "dataset_count": total_datasets,
                "availability_rate": float(len(available_rows) / max(total_datasets, 1)),
                "mean_overall_score": float(np.nanmean(available_rows["overall_score"].to_numpy(dtype=np.float64))) if not available_rows.empty else np.nan,
                "mean_biology_proxy": float(np.nanmean(available_rows["weighted_marker_recall_at_50"].to_numpy(dtype=np.float64))) if not available_rows.empty else np.nan,
                "mean_runtime_sec": float(np.nanmean(available_rows["runtime_sec"].to_numpy(dtype=np.float64))) if not available_rows.empty else np.nan,
                "mean_delta_vs_anchor": float(np.nanmean(result_rows["overall_delta_vs_anchor"].to_numpy(dtype=np.float64))) if not result_rows.empty else np.nan,
                "mean_biology_delta_vs_anchor": float(np.nanmean(result_rows["biology_delta_vs_anchor"].to_numpy(dtype=np.float64))) if not result_rows.empty else np.nan,
                "mean_runtime_delta_vs_anchor": float(np.nanmean(result_rows["runtime_delta_vs_anchor"].to_numpy(dtype=np.float64))) if not result_rows.empty else np.nan,
                "fallback_use_count": fallback_count,
                "fallback_use_rate": float(fallback_count / max(len(available_rows), 1)) if len(available_rows) > 0 else 0.0,
                "failed_dataset_count": int(len(failed_rows)),
                "failed_datasets": join_unique(failed_rows["dataset"].astype(str).tolist()),
                "failure_types": join_unique(failed_rows["error_type"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["is_anchor", "method"], ascending=[False, True]).reset_index(drop=True)


def phase3_label_for_method(*, method_name: str, available_dataset_count: int) -> str:
    if method_name == "multinomial_deviance_hvg":
        return "strict_reproduced_from_paper_or_release"
    if method_name in {"scanpy_cell_ranger_hvg", "scanpy_seurat_v3_hvg", "triku_hvg"}:
        return "strict_official_available" if available_dataset_count > 0 else "unavailable_under_strict_mode"
    if method_name in {"scran_model_gene_var_hvg", "seurat_r_vst_hvg"}:
        return "strict_official_available" if available_dataset_count > 0 else "unavailable_under_strict_mode"
    return "repo_local_published_only"


def backend_ready_note(method_name: str) -> str:
    if method_name == "multinomial_deviance_hvg":
        return "repo-local reproduced implementation ready"
    if method_name in {"scran_model_gene_var_hvg", "seurat_r_vst_hvg"}:
        return "R worker detected" if shutil.which(os.environ.get("HVG_OFFICIAL_RSCRIPT", DEFAULT_OFFICIAL_RSCRIPT)) else "R worker missing"
    if method_name == "triku_hvg":
        python_ready = Path(os.environ.get("HVG_OFFICIAL_TRIKU_PYTHON", DEFAULT_OFFICIAL_SCANPY_PYTHON)).exists()
        vendor_ready = VENDORED_PY311_ROOT.exists()
        if python_ready and vendor_ready:
            return "Python worker and vendored Triku path detected"
        if python_ready:
            return "Python worker detected but vendored Triku path missing"
        return "Triku Python worker missing"
    python_ready = Path(os.environ.get("HVG_OFFICIAL_SCANPY_PYTHON", DEFAULT_OFFICIAL_SCANPY_PYTHON)).exists()
    return "Python worker detected" if python_ready else "Python worker missing"


def build_bank_audit(*, holdout_summary: pd.DataFrame) -> pd.DataFrame:
    total_datasets = int(holdout_summary["dataset_count"].max())
    rows: list[dict[str, object]] = []
    for method_name in BANK_METHODS:
        summary_row = holdout_summary[holdout_summary["method"] == method_name].iloc[0]
        phase3_label = phase3_label_for_method(
            method_name=method_name,
            available_dataset_count=int(summary_row["available_dataset_count"]),
        )
        selector_admissible = (
            phase3_label in STRICT_SELECTOR_LABELS
            and int(summary_row["available_dataset_count"]) == total_datasets
            and int(summary_row["fallback_use_count"]) == 0
        )
        exclusion_reason = ""
        if not selector_admissible:
            if phase3_label not in STRICT_SELECTOR_LABELS:
                exclusion_reason = phase3_label
            elif int(summary_row["available_dataset_count"]) != total_datasets:
                exclusion_reason = (
                    f"incomplete_dataset_coverage {int(summary_row['available_dataset_count'])}/{total_datasets}"
                )
            elif int(summary_row["fallback_use_count"]) > 0:
                exclusion_reason = "strict_mode_fallback_detected"
        rows.append(
            {
                "method": method_name,
                "legacy_label": METHOD_NOTES[method_name].legacy_label,
                "phase3_label": phase3_label,
                "strict_backend_ready": backend_ready_note(method_name),
                "available_dataset_count": int(summary_row["available_dataset_count"]),
                "dataset_count": total_datasets,
                "availability_rate": float(summary_row["availability_rate"]),
                "fallback_use_count": int(summary_row["fallback_use_count"]),
                "failed_dataset_count": int(summary_row["failed_dataset_count"]),
                "failed_datasets": str(summary_row["failed_datasets"]),
                "failure_types": str(summary_row["failure_types"]),
                "selector_admissible": bool(selector_admissible),
                "selector_reliability": 1.0 if selector_admissible else 0.0,
                "selector_exclusion_reason": exclusion_reason,
                "source": METHOD_NOTES[method_name].source,
                "dependencies": METHOD_NOTES[method_name].dependencies,
                "parameters": METHOD_NOTES[method_name].parameters,
                "preprocessing": METHOD_NOTES[method_name].preprocessing,
                "difference_from_repo_local": METHOD_NOTES[method_name].difference_from_repo_local,
                "note": METHOD_NOTES[method_name].note,
            }
        )
    return pd.DataFrame(rows).sort_values(["selector_admissible", "phase3_label", "method"], ascending=[False, True, True]).reset_index(drop=True)


def write_bank_audit_md(*, output_dir: Path, audit_df: pd.DataFrame) -> None:
    header = "| method | legacy_label | phase3_label | available/total | selector_admissible | failures |"
    sep = "|---|---|---|---:|---:|---|"
    table_lines = [header, sep]
    for _, row in audit_df.iterrows():
        table_lines.append(
            "| {method} | {legacy_label} | {phase3_label} | {avail}/{total} | {admissible} | {failures} |".format(
                method=row["method"],
                legacy_label=row["legacy_label"],
                phase3_label=row["phase3_label"],
                avail=int(row["available_dataset_count"]),
                total=int(row["dataset_count"]),
                admissible="yes" if bool(row["selector_admissible"]) else "no",
                failures=row["failed_datasets"] or "",
            )
        )
    strict_official = audit_df[audit_df["phase3_label"] == "strict_official_available"]["method"].astype(str).tolist()
    strict_repro = audit_df[audit_df["phase3_label"] == "strict_reproduced_from_paper_or_release"]["method"].astype(str).tolist()
    unavailable = audit_df[audit_df["phase3_label"] == "unavailable_under_strict_mode"]["method"].astype(str).tolist()
    selector_bank = audit_df[audit_df["selector_admissible"]]["method"].astype(str).tolist()
    lines = [
        "# Phase3 Bank Audit",
        "",
        "- Strict mode was executed with official worker fallback disabled. Any worker failure is reported as unavailable rather than redirected to a repo-local approximation.",
        f"- Strict official methods available in the current local run: {', '.join(strict_official) if strict_official else 'none'}.",
        f"- Strict reproduced methods available in the current local run: {', '.join(strict_repro) if strict_repro else 'none'}.",
        f"- Methods unavailable under strict mode: {', '.join(unavailable) if unavailable else 'none'}.",
        f"- Selector-admissible strict/reproduced bank after availability screening: {', '.join(selector_bank) if selector_bank else 'none'}.",
        "",
        *table_lines,
    ]
    (output_dir / "bank_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_reproduction_notes(*, output_dir: Path, audit_df: pd.DataFrame) -> None:
    sections = ["# Reproduction Notes", ""]
    ordered_methods = (
        "scran_model_gene_var_hvg",
        "seurat_r_vst_hvg",
        "multinomial_deviance_hvg",
        "scanpy_cell_ranger_hvg",
        "scanpy_seurat_v3_hvg",
        "triku_hvg",
    )
    for method_name in ordered_methods:
        note = METHOD_NOTES[method_name]
        audit_row = audit_df[audit_df["method"] == method_name].iloc[0]
        sections.extend(
            [
                f"## {method_name}",
                f"- Phase3 label: {audit_row['phase3_label']}",
                f"- Legacy label: {audit_row['legacy_label']}",
                f"- Source: {note.source}",
                f"- Dependencies: {note.dependencies}",
                f"- Parameters: {note.parameters}",
                f"- Preprocessing assumptions: {note.preprocessing}",
                f"- Difference vs repo-local approximation path: {note.difference_from_repo_local}",
                f"- Current local outcome: available on {int(audit_row['available_dataset_count'])}/{int(audit_row['dataset_count'])} datasets; fallback count in strict mode = {int(audit_row['fallback_use_count'])}.",
            ]
        )
        if note.note:
            sections.append(f"- Note: {note.note}")
        sections.append("")
    (output_dir / "reproduction_notes.md").write_text("\n".join(sections), encoding="utf-8")


def selector_tables_from_benchmark(
    *,
    benchmark_df: pd.DataFrame,
    candidate_methods: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = benchmark_df[benchmark_df["status"] == "available"].copy()
    selected_methods = [SAFE_ANCHOR_METHOD, *candidate_methods]
    selected_df = available[available["method"].isin(selected_methods)].copy()
    feature_df = phase1.build_profile_df(selected_df)
    overall_df = selected_df.pivot(index="dataset", columns="method", values="overall_score").sort_index()
    runtime_df = selected_df.pivot(index="dataset", columns="method", values="runtime_sec").sort_index()
    observed_bio_df = selected_df.pivot(index="dataset", columns="method", values="weighted_marker_recall_at_50").sort_index()
    biology_df = phase1.build_biology_model_df(
        observed_biology_df=observed_bio_df,
        overall_df=overall_df,
        safe_anchor=SAFE_ANCHOR_METHOD,
    )
    return feature_df, overall_df, biology_df, runtime_df


def run_phase3_selector(
    *,
    benchmark_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    candidate_methods = tuple(
        audit_df[
            audit_df["selector_admissible"]
        ]["method"].astype(str).tolist()
    )
    if not candidate_methods:
        return None
    feature_df, overall_df, biology_df, runtime_df = selector_tables_from_benchmark(
        benchmark_df=benchmark_df,
        candidate_methods=candidate_methods,
    )
    selector_audit = audit_df[["method", "selector_admissible", "selector_reliability", "selector_exclusion_reason"]].copy()
    config = RiskControlledSelectorConfig(
        safe_anchor=SAFE_ANCHOR_METHOD,
        candidate_methods=candidate_methods,
    )
    results_df, summary_df = phase1.run_leave_one_dataset_out(
        feature_df=feature_df,
        overall_df=overall_df,
        biology_df=biology_df,
        runtime_df=runtime_df,
        official_audit=selector_audit,
        config=config,
    )
    results_df["released"] = 1.0 - results_df["abstained"].to_numpy(dtype=np.float64)
    results_df["selector_bank_methods"] = ",".join(candidate_methods)
    summary_df["selector_bank_methods"] = ",".join(candidate_methods)
    summary_df["selector_bank_size"] = int(len(candidate_methods))
    return results_df, summary_df


def run_selector_topk_seed_robustness(
    *,
    output_dir: Path,
    real_data_root: Path,
    gate_model_path: str | None,
    refine_epochs: int,
    bootstrap_samples: int,
    seeds: tuple[int, ...],
    topks: tuple[int, ...],
    audit_df: pd.DataFrame,
    selector_summary: pd.DataFrame,
) -> pd.DataFrame:
    final_path = output_dir / "strict_repro_topk_seed_summary.csv"
    if final_path.exists():
        return pd.read_csv(final_path)
    base_candidates = tuple(audit_df[audit_df["selector_admissible"]]["method"].astype(str).tolist())
    selector_row = selector_summary[selector_summary["policy"] == "holdout_risk_selector"].iloc[0]
    if float(selector_row["override_rate"]) <= 0.0:
        skipped_rows = [
            {
                "seed": int(seed),
                "top_k": int(top_k),
                "selector_bank_methods": str(selector_row.get("selector_bank_methods", ",".join(base_candidates))),
                "selector_bank_size": int(selector_row.get("selector_bank_size", len(base_candidates))),
                "release_coverage": 0.0,
                "mean_overall_score": np.nan,
                "anchor_mean_overall_score": np.nan,
                "mean_delta_vs_anchor": np.nan,
                "mean_biology_proxy": np.nan,
                "anchor_mean_biology_proxy": np.nan,
                "mean_biology_delta_vs_anchor": np.nan,
                "mean_runtime_sec": np.nan,
                "anchor_mean_runtime_sec": np.nan,
                "evaluated": False,
                "skip_reason": "main_setting_zero_release_coverage",
            }
            for seed in seeds
            for top_k in topks
        ]
        skipped_df = pd.DataFrame(skipped_rows).sort_values(["top_k", "seed"]).reset_index(drop=True)
        skipped_df.to_csv(final_path, index=False)
        return skipped_df
    method_names = tuple([SAFE_ANCHOR_METHOD, *base_candidates])
    partial_path = output_dir / "_phase3_topk_seed_partial.csv"
    summary_rows: list[dict[str, object]] = []
    completed_keys: set[tuple[int, int]] = set()
    if partial_path.exists():
        partial_df = pd.read_csv(partial_path)
        summary_rows = partial_df.to_dict(orient="records")
        completed_keys = {
            (int(row["seed"]), int(row["top_k"]))
            for _, row in partial_df.iterrows()
        }
    specs = load_round2_specs(real_data_root=real_data_root)
    for seed in seeds:
        for top_k in topks:
            if (int(seed), int(top_k)) in completed_keys:
                continue
            setting_frames: list[pd.DataFrame] = []
            for spec in specs:
                plan = rr1.ROUND2_PLANS[spec.dataset_name]
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
                    max_cells=plan.max_cells,
                    max_genes=plan.max_genes,
                    random_state=int(seed),
                )
                setting_frames.append(
                    evaluate_dataset_methods(
                        dataset=dataset,
                        dataset_id=str(spec.dataset_id),
                        spec=spec,
                        method_names=method_names,
                        gate_model_path=gate_model_path,
                        refine_epochs=refine_epochs,
                        top_k=int(top_k),
                        seed=int(seed),
                        bootstrap_samples=bootstrap_samples,
                        official_baseline_strict=True,
                    )
                )
            setting_df = pd.concat(setting_frames, ignore_index=True)
            setting_summary = build_holdout_summary(
                benchmark_df=setting_df,
                holdout_results=build_holdout_results(benchmark_df=setting_df),
            )
            total_datasets = int(setting_summary["dataset_count"].max())
            setting_candidates = tuple(
                row["method"]
                for _, row in setting_summary.iterrows()
                if (
                    row["method"] in base_candidates
                    and int(row["available_dataset_count"]) == total_datasets
                )
            )
            if setting_candidates:
                setting_audit = audit_df.copy()
                setting_audit["selector_admissible"] = setting_audit["method"].isin(setting_candidates)
                selector_run = run_phase3_selector(
                    benchmark_df=setting_df,
                    audit_df=setting_audit,
                )
                if selector_run is None:
                    selector_row = None
                else:
                    _, summary_df = selector_run
                    selector_row = summary_df[summary_df["policy"] == "holdout_risk_selector"].iloc[0]
                    anchor_row = summary_df[summary_df["policy"] == "adaptive_hybrid_anchor"].iloc[0]
            else:
                selector_row = None
                anchor_rows = setting_df[
                    (setting_df["method"] == SAFE_ANCHOR_METHOD)
                    & (setting_df["status"] == "available")
                ]
                anchor_mean_score = float(np.nanmean(anchor_rows["overall_score"].to_numpy(dtype=np.float64)))
                anchor_mean_bio = float(np.nanmean(anchor_rows["weighted_marker_recall_at_50"].to_numpy(dtype=np.float64)))
                anchor_mean_runtime = float(np.nanmean(anchor_rows["runtime_sec"].to_numpy(dtype=np.float64)))
                anchor_row = {
                    "mean_overall_score": anchor_mean_score,
                    "mean_biology_proxy": anchor_mean_bio,
                    "mean_runtime_sec": anchor_mean_runtime,
                }
            summary_row = {
                "seed": int(seed),
                "top_k": int(top_k),
                "selector_bank_methods": ",".join(setting_candidates),
                "selector_bank_size": int(len(setting_candidates)),
                "release_coverage": 0.0 if selector_row is None else float(selector_row["override_rate"]),
                "mean_overall_score": float(anchor_row["mean_overall_score"]) if selector_row is None else float(selector_row["mean_overall_score"]),
                "anchor_mean_overall_score": float(anchor_row["mean_overall_score"]),
                "mean_delta_vs_anchor": 0.0 if selector_row is None else float(selector_row["mean_score_delta_vs_anchor"]),
                "mean_biology_proxy": float(anchor_row["mean_biology_proxy"]) if selector_row is None else float(selector_row["mean_biology_proxy"]),
                "anchor_mean_biology_proxy": float(anchor_row["mean_biology_proxy"]),
                "mean_biology_delta_vs_anchor": 0.0 if selector_row is None else float(selector_row["mean_biology_delta_vs_anchor"]),
                "mean_runtime_sec": float(anchor_row["mean_runtime_sec"]) if selector_row is None else float(selector_row["mean_runtime_sec"]),
                "anchor_mean_runtime_sec": float(anchor_row["mean_runtime_sec"]),
                "evaluated": True,
                "skip_reason": "",
            }
            summary_rows.append(summary_row)
            pd.DataFrame(summary_rows).to_csv(partial_path, index=False)
    summary_df = pd.DataFrame(summary_rows).sort_values(["top_k", "seed"]).reset_index(drop=True)
    summary_df.to_csv(final_path, index=False)
    return summary_df


def build_go_no_go(
    *,
    audit_df: pd.DataFrame,
    holdout_summary: pd.DataFrame,
    selector_results: pd.DataFrame | None,
    selector_summary: pd.DataFrame | None,
    robustness_summary: pd.DataFrame | None,
) -> str:
    selector_row = None
    anchor_row = None
    if selector_summary is not None and not selector_summary.empty:
        selector_row = selector_summary[selector_summary["policy"] == "holdout_risk_selector"].iloc[0]
        anchor_row = selector_summary[selector_summary["policy"] == "adaptive_hybrid_anchor"].iloc[0]

    strict_bank_methods = audit_df[audit_df["selector_admissible"]]["method"].astype(str).tolist()
    release_coverage = 0.0 if selector_row is None else float(selector_row["override_rate"])
    selector_delta = np.nan if selector_row is None else float(selector_row["mean_score_delta_vs_anchor"])
    robustness_delta = np.nan
    robustness_skip_reason = ""
    if robustness_summary is not None and not robustness_summary.empty:
        if "evaluated" in robustness_summary.columns and not bool(np.any(robustness_summary["evaluated"].astype(bool))):
            robustness_skip_reason = join_unique(robustness_summary["skip_reason"].astype(str).tolist())
        else:
            robustness_delta = float(np.nanmean(robustness_summary["mean_delta_vs_anchor"].to_numpy(dtype=np.float64)))

    best_bank_row = holdout_summary[
        (~holdout_summary["is_anchor"].astype(bool))
        & np.isfinite(holdout_summary["mean_delta_vs_anchor"])
    ].sort_values(["mean_delta_vs_anchor", "availability_rate"], ascending=[False, False]).head(1)
    best_bank_method = ""
    best_bank_delta = np.nan
    if not best_bank_row.empty:
        best_bank_method = str(best_bank_row.iloc[0]["method"])
        best_bank_delta = float(best_bank_row.iloc[0]["mean_delta_vs_anchor"])

    if selector_row is None:
        verdict = "no-go"
    elif release_coverage <= 0.0 or selector_delta <= 0.0:
        verdict = "no-go"
    elif np.isfinite(robustness_delta) and robustness_delta < -0.02:
        verdict = "no-go"
    else:
        verdict = "go"

    if selector_row is None:
        bottleneck = "official reproduction not established" if audit_df["phase3_label"].eq("unavailable_under_strict_mode").any() else "bank quality"
    elif release_coverage <= 0.0 and np.isfinite(best_bank_delta) and best_bank_delta > 0.0:
        bottleneck = "selector feature / uncertainty calibration"
    elif release_coverage <= 0.0:
        bottleneck = "bank quality"
    elif selector_delta <= 0.0 and np.isfinite(best_bank_delta) and best_bank_delta > 0.0:
        bottleneck = "selector feature / uncertainty calibration"
    elif selector_delta <= 0.0:
        bottleneck = "bank quality"
    elif np.isfinite(robustness_delta) and robustness_delta < 0.0:
        bottleneck = "the line itself is not worth continuing"
    else:
        bottleneck = "none"

    q1 = (
        f"Yes. Selector release coverage under the strict/reproduced bank is {release_coverage:.3f}."
        if selector_row is not None and release_coverage > 0.0
        else "No. The strict/reproduced bank did not yield positive release coverage."
    )
    q2 = (
        f"Yes. Held-out mean delta vs anchor is {selector_delta:.4f}."
        if selector_row is not None and selector_delta > 0.0
        else (
            f"No. Held-out mean delta vs anchor is {selector_delta:.4f}."
            if selector_row is not None
            else "No selector refit was executed because the strict/reproduced bank was not selector-admissible."
        )
    )
    q3 = (
        f"Yes. Mean robustness overall delta is {robustness_delta:.4f}."
        if np.isfinite(robustness_delta) and robustness_delta >= 0.0
        else (
            f"No. Mean robustness overall delta is {robustness_delta:.4f}."
            if np.isfinite(robustness_delta)
            else (
                "Robustness was skipped because the calibrated main-setting selector already had zero release coverage."
                if robustness_skip_reason == "main_setting_zero_release_coverage"
                else "Robustness was not run because selector refit did not produce an admissible phase3 bank."
            )
        )
    )
    q4 = (
        "Yes. The contribution now survives as a defended strict/reproduced-bank result."
        if verdict == "go"
        else "No. The result still does not upgrade beyond a calibrated operating-point effect."
    )
    lines = [
        "# Phase3 Go / No-Go",
        "",
        f"Decision: `{verdict}`",
        "",
        "## Findings",
        f"- Selector-admissible strict/reproduced bank: {', '.join(strict_bank_methods) if strict_bank_methods else 'none'}.",
        f"- Best single strict/reproduced expert vs anchor in the bank benchmark: {best_bank_method or 'none'} (delta={best_bank_delta:.4f})." if best_bank_method else "- No strict/reproduced expert completed the bank benchmark with a valid delta vs anchor.",
        (
            f"- Held-out selector mean overall score={float(selector_row['mean_overall_score']):.4f}, anchor={float(anchor_row['mean_overall_score']):.4f}, delta={selector_delta:.4f}, release_coverage={release_coverage:.3f}."
            if selector_row is not None and anchor_row is not None
            else "- Selector refit was skipped because no method cleared the strict/reproduced selector-admissibility screen."
        ),
        (
            f"- Mean top-k/seed robustness delta vs anchor={robustness_delta:.4f}."
            if np.isfinite(robustness_delta)
            else (
                "- Top-k/seed robustness was skipped after refit because the calibrated main-setting selector already had zero release coverage."
                if robustness_skip_reason == "main_setting_zero_release_coverage"
                else "- No top-k/seed robustness run was emitted because selector refit was skipped."
            )
        ),
        "",
        "## Answers",
        f"1. Non-zero release coverage under the strict/reproduced bank? {q1}",
        f"2. Positive held-out mean delta vs anchor under the strict/reproduced bank? {q2}",
        f"3. Is the mean top-k/seed overall delta at least not clearly negative? {q3}",
        f"4. Can the main conclusion now be upgraded to a more defended method-level contribution? {q4}",
        f"5. Largest bottleneck: {bottleneck}",
        "",
        "## Notes",
        "- Phase3 strict mode disables silent fallback in the official wrappers. Unavailable methods are now reported as unavailable instead of being redirected to repo-local approximations.",
        "- `adaptive_hybrid_hvg` remains the safe anchor. The only selector story tested here is safe-anchor release with uncertainty-aware abstention and the existing soft biology term.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_device_info(output_dir=output_dir)

    real_data_root = (ROOT / args.real_data_root).resolve()
    if not real_data_root.exists():
        raise FileNotFoundError(f"Real data root not found: {real_data_root}")

    gate_model_path = rr1.resolve_gate_model_path(args.gate_model_path, (SAFE_ANCHOR_METHOD,))
    benchmark_df = run_main_bank_benchmark(
        output_dir=output_dir,
        real_data_root=real_data_root,
        gate_model_path=gate_model_path,
        refine_epochs=int(args.refine_epochs),
        top_k=int(args.top_k),
        seed=int(args.seed),
        bootstrap_samples=int(args.bootstrap_samples),
    )
    availability_df = build_method_availability_table(benchmark_df=benchmark_df)
    availability_df.to_csv(output_dir / "strict_repro_method_availability.csv", index=False)

    holdout_results = build_holdout_results(benchmark_df=benchmark_df)
    holdout_summary = build_holdout_summary(
        benchmark_df=benchmark_df,
        holdout_results=holdout_results,
    )
    audit_df = build_bank_audit(holdout_summary=holdout_summary)
    holdout_results = holdout_results.merge(
        audit_df[["method", "phase3_label"]],
        on="method",
        how="left",
    )
    holdout_summary = holdout_summary.merge(
        audit_df[["method", "phase3_label", "selector_admissible", "selector_exclusion_reason"]],
        on="method",
        how="left",
    )
    holdout_results.to_csv(output_dir / "strict_repro_holdout_results.csv", index=False)
    holdout_summary.to_csv(output_dir / "strict_repro_holdout_summary.csv", index=False)
    audit_df.to_csv(output_dir / "bank_audit.csv", index=False)
    write_bank_audit_md(output_dir=output_dir, audit_df=audit_df)
    write_reproduction_notes(output_dir=output_dir, audit_df=audit_df)

    selector_run = run_phase3_selector(
        benchmark_df=benchmark_df,
        audit_df=audit_df,
    )
    selector_results: pd.DataFrame | None = None
    selector_summary: pd.DataFrame | None = None
    robustness_summary: pd.DataFrame | None = None
    if selector_run is not None:
        selector_results, selector_summary = selector_run
        selector_results.to_csv(output_dir / "strict_repro_selector_results.csv", index=False)
        selector_summary.to_csv(output_dir / "strict_repro_selector_summary.csv", index=False)
        robustness_summary = run_selector_topk_seed_robustness(
            output_dir=output_dir,
            real_data_root=real_data_root,
            gate_model_path=gate_model_path,
            refine_epochs=int(args.refine_epochs),
            bootstrap_samples=int(args.bootstrap_samples),
            seeds=parse_int_list(args.robust_seeds),
            topks=parse_int_list(args.robust_topks),
            audit_df=audit_df,
            selector_summary=selector_summary,
        )
        robustness_summary.to_csv(output_dir / "strict_repro_topk_seed_summary.csv", index=False)

    go_no_go = build_go_no_go(
        audit_df=audit_df,
        holdout_summary=holdout_summary,
        selector_results=selector_results,
        selector_summary=selector_summary,
        robustness_summary=robustness_summary,
    )
    (output_dir / "phase3_go_no_go.md").write_text(go_no_go, encoding="utf-8")


if __name__ == "__main__":
    main()
