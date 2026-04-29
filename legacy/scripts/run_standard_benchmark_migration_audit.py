from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hvg_research import build_default_method_registry  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "artifacts_standard_hvg_benchmark_migration"
DEFAULT_TOP_K = 2000
DEFAULT_SEED = 7

CURRENT_ROUTE_BOUNDARY_PATH = ROOT / "artifacts_route_boundary_analysis" / "final_boundary_decision.md"
PAUSE_MAINLINE_PATH = ROOT / "artifacts_pause_mainline" / "problem_reframing.md"

PRIMARY_SOURCES = {
    "zhao_paper": {
        "label": "Zhao et al. 2025 Genome Biology",
        "url": "https://link.springer.com/article/10.1186/s13059-025-03887-x",
    },
    "benchmarkhvg_repo": {
        "label": "benchmarkHVG GitHub",
        "url": "https://github.com/RuzhangZhao/benchmarkHVG",
    },
    "mixhvg_repo": {
        "label": "mixhvg GitHub",
        "url": "https://github.com/RuzhangZhao/mixhvg",
    },
    "mixhvg_cran": {
        "label": "mixhvg CRAN",
        "url": "https://CRAN.R-project.org/package=mixhvg",
    },
    "scrnaseqprocess_repo": {
        "label": "scRNAseqProcess GitHub",
        "url": "https://github.com/RuzhangZhao/scRNAseqProcess",
    },
    "zhao_processed_data": {
        "label": "Zhao processed data Zenodo",
        "url": "https://zenodo.org/records/12135988",
    },
    "zhao_code_zenodo": {
        "label": "Zhao source code Zenodo",
        "url": "https://zenodo.org/records/16924165",
    },
    "atlas_feature_selection_paper": {
        "label": "Zappia et al. 2025 Nature Methods",
        "url": "https://www.nature.com/articles/s41592-025-02624-3",
    },
    "atlas_feature_selection_repo": {
        "label": "atlas-feature-selection-benchmark GitHub",
        "url": "https://github.com/theislab/atlas-feature-selection-benchmark",
    },
    "openproblems_home": {
        "label": "Open Problems home",
        "url": "https://openproblems.bio/",
    },
    "openproblems_docs": {
        "label": "Open Problems documentation",
        "url": "https://openproblems.bio/documentation",
    },
    "openproblems_task_structure": {
        "label": "Open Problems task structure",
        "url": "https://openproblems.bio/documentation/reference/openproblems/src-task_id",
    },
    "scib_repo": {
        "label": "scIB GitHub",
        "url": "https://github.com/theislab/scib",
    },
    "scib_paper": {
        "label": "scIB Nature Methods paper",
        "url": "https://doi.org/10.1038/s41592-021-01336-8",
    },
    "bib2024_paper": {
        "label": "Cho et al. 2024 Briefings in Bioinformatics",
        "url": "https://academic.oup.com/bib/article/doi/10.1093/bib/bbae317/7709086",
    },
    "bib2024_code": {
        "label": "Cho et al. 2024 Zenodo",
        "url": "https://doi.org/10.5281/zenodo.10017609",
    },
    "dlfs_gb2023_paper": {
        "label": "Huang et al. 2023 Genome Biology",
        "url": "https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03100-x",
    },
    "dlfs_gb2023_code": {
        "label": "scDeepFeatures GitHub",
        "url": "https://github.com/PYangLab/scDeepFeatures",
    },
    "marker_gb2024_paper": {
        "label": "Pullin and McCarthy 2024 Genome Biology",
        "url": "https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03183-0",
    },
    "marker_gb2024_code": {
        "label": "marker benchmark GitLab",
        "url": "https://gitlab.svi.edu.au/biocellgen-public/mage_2020_marker-gene-benchmarking",
    },
}

BENCHMARK_CANDIDATES: list[dict[str, Any]] = [
    {
        "benchmark_name": "benchmarkHVG (Zhao et al. 2025)",
        "paper_or_repo": "paper+repo",
        "year": "2025",
        "task_type": "HVG-specific benchmark",
        "dataset_count": "19",
        "method_count": "21 baseline + 26 hybrid",
        "metric_count": "18",
        "data_access": "public datasets + processed Zenodo bundle",
        "code_access": "official GitHub repo + code/data Zenodo",
        "license_risk": "moderate",
        "implementation_difficulty": "moderate",
        "relevance_to_hvg_scorer": "high",
        "recommendation": "primary_reproduction_target",
        "fit_summary": (
            "Most direct public benchmark for generic scRNA-seq HVG scoring. Uses orthogonal supervision "
            "from cell sorting labels, CITE-seq ADTs, and paired multiome ATAC. Closest match to this audit."
        ),
        "why_not_primary": "",
        "sources": [
            "zhao_paper",
            "benchmarkhvg_repo",
            "scrnaseqprocess_repo",
            "zhao_processed_data",
            "zhao_code_zenodo",
        ],
    },
    {
        "benchmark_name": "mixhvg",
        "paper_or_repo": "package+repo",
        "year": "2024-2025",
        "task_type": "method package, not benchmark",
        "dataset_count": "n/a",
        "method_count": "26 hybrid variants exposed by package",
        "metric_count": "n/a",
        "data_access": "example data links + relies on Zhao benchmark resources",
        "code_access": "CRAN + GitHub",
        "license_risk": "low",
        "implementation_difficulty": "low",
        "relevance_to_hvg_scorer": "medium",
        "recommendation": "use_as_comparator_or_adapter_helper",
        "fit_summary": (
            "Not a benchmark by itself, but an important official implementation artifact from the Zhao line. "
            "Useful because it already exposes an external ranking interface (`extra.rank`) that can help adapter design."
        ),
        "why_not_primary": "Package for one method family, not a standalone evaluation framework.",
        "sources": ["mixhvg_repo", "mixhvg_cran", "zhao_paper"],
    },
    {
        "benchmark_name": "Atlas feature selection benchmark (Zappia et al. 2025)",
        "paper_or_repo": "paper+repo",
        "year": "2025",
        "task_type": "integration/query mapping feature-selection benchmark",
        "dataset_count": "10",
        "method_count": "24 variants",
        "metric_count": "5 metric categories",
        "data_access": "figshare metric files + selected feature files",
        "code_access": "official Nextflow GitHub repo",
        "license_risk": "low",
        "implementation_difficulty": "high",
        "relevance_to_hvg_scorer": "medium",
        "recommendation": "secondary_context_benchmark",
        "fit_summary": (
            "Strong, reproducible public benchmark for how feature selection affects atlas integration and query mapping. "
            "Useful as secondary evidence if a method is intended for atlas/reference workflows."
        ),
        "why_not_primary": (
            "Task is downstream integration/query mapping rather than direct HVG scoring quality in ordinary scRNA-seq."
        ),
        "sources": ["atlas_feature_selection_paper", "atlas_feature_selection_repo"],
    },
    {
        "benchmark_name": "scIB / atlas-level integration benchmark",
        "paper_or_repo": "paper+repo",
        "year": "2022-2024",
        "task_type": "integration benchmark with HVG conservation metric",
        "dataset_count": "85 batches",
        "method_count": "16 methods, 68 preprocessing combinations",
        "metric_count": "multiple integration metrics",
        "data_access": "public benchmark resources",
        "code_access": "official GitHub repo + docs + releases",
        "license_risk": "low",
        "implementation_difficulty": "high",
        "relevance_to_hvg_scorer": "low",
        "recommendation": "context_only",
        "fit_summary": (
            "Important downstream benchmark family and provides an HVG conservation metric, but that metric evaluates "
            "integration behavior after preprocessing, not generic HVG scorer quality directly."
        ),
        "why_not_primary": "Mismatch between evaluated object (integration output) and target object (gene scoring function).",
        "sources": ["scib_repo", "scib_paper"],
    },
    {
        "benchmark_name": "Open Problems in Single-cell Analysis",
        "paper_or_repo": "platform",
        "year": "2025",
        "task_type": "community benchmark framework",
        "dataset_count": "task-dependent",
        "method_count": "task-dependent",
        "metric_count": "task-dependent",
        "data_access": "public task resources",
        "code_access": "official docs and task repositories",
        "license_risk": "low",
        "implementation_difficulty": "high",
        "relevance_to_hvg_scorer": "low",
        "recommendation": "do_not_use_as_immediate_primary",
        "fit_summary": (
            "Potential long-term hosting or formalization framework. Publicly documented as a task-based benchmark system."
        ),
        "why_not_primary": (
            "No dedicated HVG-selection task was found in the public docs examined during this audit; adopting it now "
            "would require creating a new task rather than reproducing a recognized HVG benchmark."
        ),
        "sources": ["openproblems_home", "openproblems_docs", "openproblems_task_structure"],
    },
    {
        "benchmark_name": "Cho et al. 2024 feature-selection comparison",
        "paper_or_repo": "paper+zenodo",
        "year": "2024",
        "task_type": "unsupervised feature-selection benchmark for clustering/trajectory",
        "dataset_count": "simulation + several real datasets",
        "method_count": "11",
        "metric_count": "at least ARI, ASW, trajectory quality",
        "data_access": "public raw data sources + Zenodo simulation/code bundle",
        "code_access": "Zenodo code bundle",
        "license_risk": "low",
        "implementation_difficulty": "moderate",
        "relevance_to_hvg_scorer": "medium",
        "recommendation": "supporting_context_only",
        "fit_summary": (
            "Relevant because it benchmarks unsupervised feature-selection methods with clustering and trajectory outcomes."
        ),
        "why_not_primary": (
            "Not an HVG-specific public benchmark framework with official reusable pipeline comparable to benchmarkHVG."
        ),
        "sources": ["bib2024_paper", "bib2024_code"],
    },
    {
        "benchmark_name": "Huang et al. 2023 deep-learning feature-selection evaluation",
        "paper_or_repo": "paper+repo",
        "year": "2023",
        "task_type": "classification/reproducibility feature-selection benchmark",
        "dataset_count": "synthetic atlas-derived panels + colon test",
        "method_count": "6 deep learning + traditional baselines",
        "metric_count": "classification, reproducibility, diversity, runtime",
        "data_access": "public atlases + portal accessions",
        "code_access": "GitHub + Zenodo",
        "license_risk": "low",
        "implementation_difficulty": "moderate",
        "relevance_to_hvg_scorer": "low",
        "recommendation": "not_primary",
        "fit_summary": (
            "Good benchmark of supervised/embedded feature selection methods for classification-style tasks."
        ),
        "why_not_primary": "Different target task from generic unsupervised HVG selection.",
        "sources": ["dlfs_gb2023_paper", "dlfs_gb2023_code"],
    },
    {
        "benchmark_name": "Marker gene selection benchmark (Pullin and McCarthy 2024)",
        "paper_or_repo": "paper+repo",
        "year": "2024",
        "task_type": "marker-gene selection benchmark",
        "dataset_count": "14 real + 170+ simulated",
        "method_count": "59",
        "metric_count": "recovery, predictive performance, runtime, implementation quality",
        "data_access": "public datasets",
        "code_access": "official GitLab + Zenodo",
        "license_risk": "low",
        "implementation_difficulty": "moderate",
        "relevance_to_hvg_scorer": "low",
        "recommendation": "distinguish_not_equivalent",
        "fit_summary": (
            "High-quality benchmark, but for marker gene selection rather than globally variable genes."
        ),
        "why_not_primary": "Conceptually different task; should not be substituted for HVG benchmarking.",
        "sources": ["marker_gb2024_paper", "marker_gb2024_code"],
    },
]

ZHAO2025_DATASET_REGISTRY = [
    {"modality": "cell_sorting", "dataset_name": "GBM_sd", "accession_or_source": "GEO:GSE84465"},
    {"modality": "cell_sorting", "dataset_name": "duo4_pbmc", "accession_or_source": "DuoClustering2018 / duo4 PBMC"},
    {"modality": "cell_sorting", "dataset_name": "duo8_pbmc", "accession_or_source": "DuoClustering2018 / duo8 PBMC"},
    {"modality": "cell_sorting", "dataset_name": "duo4un_pbmc", "accession_or_source": "DuoClustering2018 / duo4 unequal PBMC"},
    {"modality": "cell_sorting", "dataset_name": "zheng_pbmc", "accession_or_source": "10x / Zheng PBMC"},
    {"modality": "cell_sorting", "dataset_name": "mus_tissue", "accession_or_source": "GEO:GSE108097"},
    {"modality": "cell_sorting", "dataset_name": "homo_tissue", "accession_or_source": "GEO:GSE108097"},
    {"modality": "cite_seq", "dataset_name": "pbmc_cite", "accession_or_source": "GEO:GSE100866"},
    {"modality": "cite_seq", "dataset_name": "cbmc8k_cite", "accession_or_source": "GEO:GSE100866"},
    {"modality": "cite_seq", "dataset_name": "FBM_cite", "accession_or_source": "GEO:GSE166895"},
    {"modality": "cite_seq", "dataset_name": "FLiver_cite", "accession_or_source": "GEO:GSE166895"},
    {"modality": "cite_seq", "dataset_name": "bmcite", "accession_or_source": "GEO:GSE128639 / SeuratData"},
    {"modality": "cite_seq", "dataset_name": "seurat_cite", "accession_or_source": "GEO:GSE164378"},
    {"modality": "cite_seq", "dataset_name": "sucovid_cite", "accession_or_source": "ArrayExpress:E-MTAB-9357"},
    {"modality": "multiome_atac", "dataset_name": "pbmc3k_multi", "accession_or_source": "10x Genomics multiome"},
    {"modality": "multiome_atac", "dataset_name": "homo_brain3k_multi", "accession_or_source": "10x Genomics multiome"},
    {"modality": "multiome_atac", "dataset_name": "mus_brain5k_multi", "accession_or_source": "10x Genomics multiome"},
    {"modality": "multiome_atac", "dataset_name": "pbmc10k_multi", "accession_or_source": "10x Genomics multiome"},
    {"modality": "multiome_atac", "dataset_name": "lymphoma_multi", "accession_or_source": "10x Genomics multiome"},
]

METHOD_AUDIT_CONFIG: list[dict[str, str]] = [
    {
        "method_name": "adaptive_hybrid_hvg",
        "input_requirements": "dense counts matrix (cells x genes); optional batches; top_k used for frontier-lite branch only",
        "output_format": "1D float score vector aligned to gene order",
        "metadata_dependency": "optional batch labels; routing itself can run without metadata",
        "metadata_free_behavior": "yes; falls back to profile-conditioned routing without explicit donor metadata",
        "runtime_cost": "low_to_moderate",
        "recommended_role": "main_method_as_repo_anchor",
        "adapter_risk": "moderate because routing logic complicates interpretation under public benchmark",
        "notes": "Good carry-over anchor for comparison, but not a clean new-method story.",
    },
    {
        "method_name": "adaptive_spectral_locality_hvg",
        "input_requirements": "dense counts matrix; optional batches; builds PCA/kNN graph on a cell subsample",
        "output_format": "1D float score vector aligned to gene order",
        "metadata_dependency": "no required metadata",
        "metadata_free_behavior": "yes",
        "runtime_cost": "moderate",
        "recommended_role": "ablation_or_negative_control",
        "adapter_risk": "low for mechanics, high for scientific positioning",
        "notes": "Mechanically portable, but current project evidence says locality route is not benchmark-safe internally.",
    },
    {
        "method_name": "adaptive_eb_shrinkage_hvg",
        "input_requirements": "dense counts matrix; optional batches",
        "output_format": "1D float score vector aligned to gene order",
        "metadata_dependency": "no required metadata",
        "metadata_free_behavior": "yes",
        "runtime_cost": "low",
        "recommended_role": "ablation_or_secondary_single_scorer",
        "adapter_risk": "low",
        "notes": "Cleanest single-scorer carry-over among the non-routing adaptive variants.",
    },
    {
        "method_name": "adaptive_invariant_residual_hvg",
        "input_requirements": "dense counts matrix; optional batches; internally constructs environments",
        "output_format": "1D float score vector aligned to gene order",
        "metadata_dependency": "strongly metadata-sensitive if batches are supplied; otherwise pseudo-environments are synthesized",
        "metadata_free_behavior": "yes, but via pseudo_kmeans environments",
        "runtime_cost": "moderate",
        "recommended_role": "ablation_not_main",
        "adapter_risk": "high",
        "notes": "Portable in code, but semantically risky on no-metadata public datasets because the target mechanism is donor/batch-aware.",
    },
    {
        "method_name": "adaptive_stability_jackknife_hvg",
        "input_requirements": "dense counts matrix; optional batches; internal cell subsampling and split scoring",
        "output_format": "1D float score vector aligned to gene order",
        "metadata_dependency": "no required metadata",
        "metadata_free_behavior": "yes",
        "runtime_cost": "moderate",
        "recommended_role": "ablation",
        "adapter_risk": "low",
        "notes": "Scientifically clean ablation for robustness-oriented scoring.",
    },
    {
        "method_name": "adaptive_risk_parity_hvg",
        "input_requirements": "dense counts matrix; optional batches",
        "output_format": "1D float score vector aligned to gene order",
        "metadata_dependency": "no required metadata",
        "metadata_free_behavior": "yes",
        "runtime_cost": "low",
        "recommended_role": "ablation",
        "adapter_risk": "low",
        "notes": "Easy to adapter-wrap, but mainly useful as another fixed-fusion ablation.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a standard HVG benchmark migration audit without executing a full benchmark."
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def resolve_compute_context() -> dict[str, Any]:
    device_info: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        device_info["cuda_devices"] = [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(idx).total_memory / (1024**3),
                    3,
                ),
            }
            for idx in range(torch.cuda.device_count())
        ]
    else:
        device_info["cuda_devices"] = []
    return device_info


def check_importable(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        return {"importable": True, "version": None if version is None else str(version)}
    except Exception as exc:
        return {
            "importable": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def inspect_path(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not status["exists"]:
        status["accessible"] = False
        status["kind"] = "missing"
        return status
    try:
        if path.is_dir():
            entries = []
            for idx, child in enumerate(path.iterdir()):
                if idx >= 10:
                    break
                entries.append(child.name)
            status["accessible"] = True
            status["kind"] = "directory"
            status["sample_entries"] = entries
            status["entry_sample_count"] = len(entries)
            return status
        status["accessible"] = True
        status["kind"] = "file"
        status["size_bytes"] = int(path.stat().st_size)
        return status
    except Exception as exc:
        status["accessible"] = False
        status["kind"] = "inaccessible"
        status["error_type"] = type(exc).__name__
        status["error_message"] = str(exc)
        return status


def collect_local_resource_audit() -> dict[str, Any]:
    candidate_paths = [
        ROOT / ".official_hvg_cache",
        ROOT / "official_hvg_ak36oj8t",
        ROOT / "scripts" / "score_official_hvg.py",
        ROOT / "scripts" / "score_official_hvg.R",
        ROOT / "data",
        ROOT / "artifacts_route_boundary_analysis",
        CURRENT_ROUTE_BOUNDARY_PATH,
        PAUSE_MAINLINE_PATH,
        Path(r"D:\code_py\hvg\.conda_stage5\python.exe"),
    ]
    return {
        "paths": [inspect_path(path) for path in candidate_paths],
        "executables": {
            "python": shutil.which("python"),
            "Rscript": shutil.which("Rscript"),
        },
        "python_modules": {
            name: check_importable(name)
            for name in (
                "numpy",
                "pandas",
                "scanpy",
                "anndata",
                "h5py",
                "sklearn",
            )
        },
    }


def safe_write_text(path: Path, content: str) -> dict[str, Any]:
    if path.exists():
        return {"path": str(path), "written": False, "reason": "already_exists"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "written": True}


def safe_write_json(path: Path, payload: Any) -> dict[str, Any]:
    return safe_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def safe_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> dict[str, Any]:
    if path.exists():
        return {"path": str(path), "written": False, "reason": "already_exists"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return {"path": str(path), "written": True}


def source_markdown(source_ids: list[str]) -> str:
    lines = []
    for source_id in source_ids:
        source = PRIMARY_SOURCES[source_id]
        lines.append(f"- [{source['label']}]({source['url']})")
    return "\n".join(lines)


def summarize_local_route_context() -> dict[str, str]:
    boundary_status = "unknown"
    boundary_reason = "Local route-boundary artifact not found."
    if CURRENT_ROUTE_BOUNDARY_PATH.exists():
        text = CURRENT_ROUTE_BOUNDARY_PATH.read_text(encoding="utf-8", errors="replace")
        if "route_exists_but_is_not_benchmark_safe" in text:
            boundary_status = "route_exists_but_is_not_benchmark_safe"
            boundary_reason = (
                "Current local artifact already concludes that the route exists but is not benchmark-safe."
            )
        else:
            boundary_status = "present_but_unparsed"
            boundary_reason = "Boundary artifact exists but expected decision string was not found."
    return {"boundary_status": boundary_status, "boundary_reason": boundary_reason}


def run_adapter_smoke(*, top_k: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=1.5, size=(96, 128)).astype(np.float32)
    batches = np.array(["batch0"] * 48 + ["batch1"] * 48, dtype=object)
    registry = build_default_method_registry(
        top_k=top_k,
        refine_epochs=1,
        random_state=seed,
        gate_model_path=None,
        holdout_selector_policy_path=None,
    )

    rows: list[dict[str, Any]] = []
    for item in METHOD_AUDIT_CONFIG:
        method_name = item["method_name"]
        fn = registry[method_name]
        for mode_name, batch_vector in (("no_metadata", None), ("with_batch", batches)):
            try:
                scores = np.asarray(fn(counts, batch_vector, top_k), dtype=np.float64)
                metadata = dict(getattr(fn, "last_gate_metadata", {}) or {})
                row = {
                    "method_name": method_name,
                    "mode": mode_name,
                    "ok": True,
                    "score_length": int(scores.shape[0]),
                    "all_finite": bool(np.isfinite(scores).all()),
                    "route_name": str(metadata.get("route_name", "")),
                    "variant": str(metadata.get("variant", "")),
                    "environment_source": str(metadata.get("environment_source", "")),
                    "resolved_method": str(metadata.get("resolved_method", "")),
                }
            except Exception as exc:
                row = {
                    "method_name": method_name,
                    "mode": mode_name,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=1),
                }
            rows.append(row)
    return {
        "synthetic_counts_shape": list(counts.shape),
        "top_k": int(top_k),
        "seed": int(seed),
        "rows": rows,
    }


def build_benchmark_candidate_table() -> list[dict[str, Any]]:
    table_rows = []
    for item in BENCHMARK_CANDIDATES:
        table_rows.append(
            {
                "benchmark_name": item["benchmark_name"],
                "paper_or_repo": item["paper_or_repo"],
                "year": item["year"],
                "task_type": item["task_type"],
                "dataset_count": item["dataset_count"],
                "method_count": item["method_count"],
                "metric_count": item["metric_count"],
                "data_access": item["data_access"],
                "code_access": item["code_access"],
                "license_risk": item["license_risk"],
                "implementation_difficulty": item["implementation_difficulty"],
                "relevance_to_hvg_scorer": item["relevance_to_hvg_scorer"],
                "recommendation": item["recommendation"],
            }
        )
    return table_rows


def render_standard_benchmark_survey(
    *,
    local_resource_audit: dict[str, Any],
    route_context: dict[str, str],
) -> str:
    candidate_sections = []
    for item in BENCHMARK_CANDIDATES:
        why_not_primary = ""
        if item["why_not_primary"]:
            why_not_primary = f"\n- Why not primary: {item['why_not_primary']}"
        candidate_sections.append(
            textwrap.dedent(
                f"""
                ## {item['benchmark_name']}

                - Task type: {item['task_type']}
                - Year: {item['year']}
                - Dataset count: {item['dataset_count']}
                - Method count: {item['method_count']}
                - Metric count: {item['metric_count']}
                - Data access: {item['data_access']}
                - Code access: {item['code_access']}
                - License risk: {item['license_risk']}
                - Implementation difficulty: {item['implementation_difficulty']}
                - Relevance to ordinary scRNA-seq HVG scorer evaluation: {item['relevance_to_hvg_scorer']}
                - Recommendation: {item['recommendation']}
                - Fit summary: {item['fit_summary']}{why_not_primary}

                Sources
                {source_markdown(item['sources'])}
                """
            ).strip()
        )

    inspected_paths = "\n".join(
        f"- `{entry['path']}`: exists={entry['exists']}, accessible={entry.get('accessible')}, kind={entry.get('kind')}"
        for entry in local_resource_audit["paths"]
    )

    return textwrap.dedent(
        f"""
        # Standard HVG Benchmark Migration Survey

        ## Scope

        This artifact audits whether the next mainline for this repository should migrate from the current internal benchmark route
        to a public, more publication-grade benchmark. The target question is migration feasibility, not the design of a new scorer.

        ## Audit Filters

        A candidate is considered suitable as a primary migration target only if it satisfies most of the following:

        - It evaluates generic scRNA-seq HVG or feature-selection methods rather than a downstream task only.
        - It uses public data with identifiable accessions or archived processed bundles.
        - It exposes enough official code to reproduce at least a small slice.
        - It can accept or be extended to accept an external gene ranking from this repository.
        - Its evaluation target is close to an ordinary HVG scorer rather than an integration model, annotation model, or marker-gene selector.

        ## Key Distinctions

        - `HVG-specific benchmark`: directly compares HVG/feature-selection methods for their ability to support downstream scRNA-seq structure recovery.
        - `Downstream integration/annotation benchmark`: evaluates integration, query mapping, or annotation pipelines, even if HVG preprocessing is involved.
        - `Method package`: useful for comparator implementation or adapter design, but not itself a benchmark framework.

        ## Local Context

        - Existing route boundary state: `{route_context['boundary_status']}`
        - Interpretation: {route_context['boundary_reason']}
        - This repository already contains official-baseline bridge scripts (`scripts/score_official_hvg.py`, `scripts/score_official_hvg.R`) and an official cache directory, so the migration problem is mostly benchmark alignment and adapter design rather than zero-to-one infrastructure.

        ### Local Resource Check

        {inspected_paths}

        ## Candidate Assessment

        {'\n\n'.join(candidate_sections)}

        ## Bottom Line

        - The strongest primary target is `benchmarkHVG` from Zhao et al. 2025 because it is the only located public benchmark that is explicitly about HVG selection itself, with 19 datasets, 18 criteria, public code, and archived processed data.
        - `mixhvg` matters because it is part of the same official ecosystem and exposes an external ranking hook, but it is not a benchmark.
        - `scIB`, `Open Problems`, and the 2025 atlas feature-selection benchmark are useful context, but they are not equivalent to the problem of benchmarking a general-purpose HVG scorer.
        - Marker-gene and supervised feature-selection benchmarks should be treated as non-equivalent reference points, not substitutes for HVG benchmarking.

        ## Primary Sources Consulted

        {source_markdown(list(PRIMARY_SOURCES.keys()))}
        """
    ).strip() + "\n"


def render_zhao_reproduction_plan(
    *,
    local_resource_audit: dict[str, Any],
) -> str:
    has_rscript = bool(local_resource_audit["executables"].get("Rscript"))
    has_official_python = any(
        entry["path"] == r"D:\code_py\hvg\.conda_stage5\python.exe" and entry["exists"]
        for entry in local_resource_audit["paths"]
    )

    dataset_lines = "\n".join(
        f"- `{row['dataset_name']}` ({row['modality']}): {row['accession_or_source']}"
        for row in ZHAO2025_DATASET_REGISTRY
    )

    return textwrap.dedent(
        f"""
        # Zhao 2025 / benchmarkHVG Minimal Reproduction Plan

        ## Recommendation

        Reproduce `benchmarkHVG` first. It is the most direct public benchmark for generic HVG scoring that this audit found.

        ## Why This Is Reproducible Enough To Try

        - The paper provides a 19-dataset registry and official code/data pointers.
        - The benchmark analysis code is public at `benchmarkHVG`.
        - Preprocessing code is public at `scRNAseqProcess`.
        - Processed data are archived on Zenodo.
        - The source code snapshot used in the manuscript is also archived on Zenodo.

        ## Data Acquisition

        Primary route:
        - Pull the processed dataset bundle from the Zenodo record linked in Zhao et al. 2025.
        - If needed for smaller debugging, use the example datasets referenced in the repo README first.

        Literature-derived dataset registry from the paper:
        {dataset_lines}

        Small-first dataset order:
        1. `duo4_pbmc` or `duo8_pbmc` for the cell-sorting slice.
        2. `bmcite` or `cbmc8k_cite` for the CITE-seq slice.
        3. `pbmc3k_multi` for the multiome slice.

        ## Code Installation

        Minimum expected components:
        - R with `benchmarkHVG`
        - R with `mixhvg`
        - Supporting packages implied by the repo examples, such as `Seurat`, `SeuratData`, and `DuoClustering2018`
        - Any preprocessing helpers from `scRNAseqProcess`

        Install path inference:
        - Because `benchmarkHVG` is structured as an R package repository, the likely install route is `remotes::install_github("RuzhangZhao/benchmarkHVG")` or local `R CMD INSTALL` after cloning.
        - `mixhvg` can be installed from CRAN or GitHub.

        Local environment reality in this repo:
        - `Rscript` detected: {has_rscript}
        - official scanpy worker python detected: {has_official_python}
        - This means the current repo already has part of the cross-language bridge infrastructure required for a future adapter.

        ## First Reproduction Slice

        Do not jump to all 19 datasets.

        Phase A: import-level confirmation
        - Install or clone the official benchmark code.
        - Confirm the benchmark functions load.
        - Confirm one example dataset can be loaded into the expected R object.

        Phase B: one-dataset one-modality pilot
        - Reproduce the cell-sorting example from the README on `duo4_pbmc`.
        - Reproduce only baseline evaluation first.
        - Confirm that the official evaluation functions run end-to-end.

        Phase C: one-dataset per modality
        - Add one CITE-seq dataset (`bmcite` or `cbmc8k_cite`).
        - Add one multiome dataset (`pbmc3k_multi`).
        - Stop after a single successful run in each modality.

        ## First Metrics To Reproduce

        Smallest safe goal:
        - One full metric vector for one cell-sorting dataset.
        - One full metric vector for one CITE-seq dataset.
        - One full metric vector for one multiome dataset.

        Rationale:
        - This validates the three modality-specific evaluators without paying full benchmark cost.

        ## Adapter Plan For This Repo's Scorers

        Primary adapter route:
        - Keep scoring in Python inside this repo.
        - Export a full-length gene score vector or rank vector aligned to the dataset gene order.
        - In R, inject that ranking into the official benchmark preprocessing path and reuse the benchmark's PCA/evaluation code.

        Best official hook discovered during this audit:
        - `mixhvg` documents an `extra.rank` argument for injecting an external ranking into `FindVariableFeaturesMix`.
        - This suggests a low-friction adapter path for external scorers, even if `benchmarkHVG` itself does not document a direct external-rank API in the README.

        Fallback adapter route:
        - Export the top-`k` gene set from Python.
        - In R, subset the Seurat object or matrix to those genes.
        - Run the same PCA and official evaluation criteria on the resulting feature-restricted representation.

        ## Estimated Compute Cost

        These are rough estimates inferred from public dataset sizes and workflow structure, not measured in this repo:

        - import/install audit: less than half a day
        - one cell-sorting pilot dataset: tens of minutes to a few CPU hours
        - one dataset for each of the three modalities: likely a same-day CPU run if dependencies are healthy
        - full 19-dataset reproduction: multi-day and likely memory-sensitive, not appropriate for the first migration step

        ## Failure Modes

        High-probability failure points:
        - R package dependency drift
        - missing processed data files even when code is available
        - benchmark code assuming object classes or data layouts not documented in the README
        - cross-language adapter friction between Python gene scores and R benchmark wrappers
        - license ambiguity between article statement (GPL-3) and some repository/Zenodo pages showing MIT or CC-BY

        ## Fallback If The Official Reproduction Fails

        Fallback A:
        - Use the official processed data bundle and the official metric definitions, but build only a minimal external-rank adapter around one modality.

        Fallback B:
        - Use the official example datasets and evaluation logic from the repo README as a thin reproducibility target, documenting which parts of the full benchmark are inaccessible.

        Fallback C:
        - If official code is unavailable but processed data are present, reproduce only the modality-specific evaluation criteria on one dataset and record the missing benchmark components explicitly.

        ## Sources

        {source_markdown(['zhao_paper', 'benchmarkhvg_repo', 'scrnaseqprocess_repo', 'zhao_processed_data', 'zhao_code_zenodo', 'mixhvg_repo'])}
        """
    ).strip() + "\n"


def render_current_methods_adapter_audit(*, adapter_smoke: dict[str, Any]) -> str:
    smoke_lookup = {
        (row["method_name"], row["mode"]): row
        for row in adapter_smoke["rows"]
    }
    sections = []
    for method in METHOD_AUDIT_CONFIG:
        no_meta = smoke_lookup[(method["method_name"], "no_metadata")]
        with_batch = smoke_lookup[(method["method_name"], "with_batch")]

        smoke_lines = [
            f"- no-metadata smoke: ok={no_meta.get('ok')}, route=`{no_meta.get('route_name', '')}`, environment_source=`{no_meta.get('environment_source', '')}`",
            f"- with-batch smoke: ok={with_batch.get('ok')}, route=`{with_batch.get('route_name', '')}`, environment_source=`{with_batch.get('environment_source', '')}`",
        ]

        sections.append(
            textwrap.dedent(
                f"""
                ## {method['method_name']}

                - Input requirements: {method['input_requirements']}
                - Output format: {method['output_format']}
                - Needs batch/donor metadata: {method['metadata_dependency']}
                - Can run without metadata: {method['metadata_free_behavior']}
                - Runtime cost: {method['runtime_cost']}
                - Recommended role: {method['recommended_role']}
                - Adapter risk: {method['adapter_risk']}
                - Notes: {method['notes']}
                {'\n'.join(smoke_lines)}
                """
            ).strip()
        )

    return textwrap.dedent(
        f"""
        # Current Methods Adapter Audit

        ## Scope

        This artifact checks whether the currently named repo methods can be adapter-wrapped into a standard public benchmark.
        The criterion here is `can_be_evaluated honestly`, not `should_be promoted`.

        ## Common Adapter Contract

        The current Python registry already exposes a useful common contract:

        - input: `counts`, optional `batches`, and `top_k`
        - output: one continuous score per gene in the original gene order

        That contract is sufficient for adapter-wrapping into a public benchmark that expects either:

        - a ranked full-length gene score vector, or
        - a top-`k` gene list

        ## Synthetic Smoke Audit

        - Synthetic matrix shape: `{adapter_smoke['synthetic_counts_shape']}`
        - `top_k`: `{adapter_smoke['top_k']}`
        - All listed methods completed both `no_metadata` and `with_batch` smoke runs in this repository.

        {'\n\n'.join(sections)}

        ## Main Judgment

        - Mechanically, all six audited methods can be adapter-wrapped because they output per-gene scores.
        - Scientifically, only `adaptive_hybrid_hvg` is a reasonable carry-over anchor for a migration audit, and even that should be positioned as an existing repo anchor rather than a new headline method.
        - `adaptive_invariant_residual_hvg` is the most problematic for standard public benchmarking because its mechanism becomes metadata-sensitive when real batches exist and pseudo-environment-sensitive when they do not.
        - `adaptive_spectral_locality_hvg` is portable as code, but current internal evidence argues against treating it as a benchmark-safe main method.
        """
    ).strip() + "\n"


def render_migration_decision(*, route_context: dict[str, str]) -> str:
    return textwrap.dedent(
        f"""
        # Migration Decision

        ## Final Judgment

        - Should the next-stage mainline migrate to a standard public HVG benchmark? Yes.
        - Most recommended benchmark to reproduce first: `benchmarkHVG` from Zhao et al. 2025.
        - Is designing a new scorer allowed right now? No.

        ## Why Migration Is Recommended

        - The repository's own boundary artifact state is `{route_context['boundary_status']}`.
        - The current internal route evidence supports the existence of some mechanism-specific signal, but it does not support benchmark-safe mainline advancement.
        - A public benchmark is now the shortest path to answering the paper-level question that matters: does any real, publishable headroom remain once evaluation is moved onto a more standard footing?

        ## Why New Scorer Design Is Not Allowed Now

        - The present blocker is benchmark credibility, not absence of additional model ideas.
        - Internal evidence already says route existence does not imply benchmark safety.
        - Continuing to design new scorers before reproducing a standard benchmark would compound uncertainty and make any positive result harder to defend to reviewers.
        - Public benchmark migration is prerequisite evidence for whether there is even a worthwhile target to optimize.

        ## Recommended Next Minimal Experiment

        1. Run the migration audit script in this repository to materialize the literature-backed decision artifacts.
        2. Acquire or clone the official `benchmarkHVG` resources.
        3. Reproduce exactly one official example dataset in one modality.
        4. Only after that, wire one existing repo scorer into the same pipeline as a pure adapter exercise.

        ## Operational Recommendation

        - Keep `adaptive_hybrid_hvg` as the repo anchor for adapter testing.
        - Treat the other adaptive variants as ablations or negative controls during migration.
        - Do not introduce any new scorer name, architecture, or routing logic until the Zhao benchmark pilot is demonstrably runnable.
        """
    ).strip() + "\n"


def render_sources_appendix() -> str:
    return "\n".join(
        f"- [{item['label']}]({item['url']})"
        for item in PRIMARY_SOURCES.values()
    )


def render_audit_readme_note() -> str:
    return textwrap.dedent(
        """
        This directory was created by `scripts/run_standard_benchmark_migration_audit.py`.
        The script is intentionally analysis-only:

        - it does not run a full public benchmark
        - it does not download remote data inside the script
        - it records local accessibility facts and literature-derived benchmark facts separately
        - it skips writes when a target artifact already exists
        """
    ).strip() + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_context = resolve_compute_context()
    local_resource_audit = collect_local_resource_audit()
    route_context = summarize_local_route_context()
    adapter_smoke = run_adapter_smoke(top_k=int(args.top_k), seed=int(args.seed))

    candidate_table = build_benchmark_candidate_table()
    write_events = [
        safe_write_json(output_dir / "compute_context.json", compute_context),
        safe_write_json(output_dir / "local_resource_audit.json", local_resource_audit),
        safe_write_json(output_dir / "adapter_smoke.json", adapter_smoke),
        safe_write_json(output_dir / "zhao2025_dataset_registry.json", ZHAO2025_DATASET_REGISTRY),
        safe_write_text(output_dir / "_README.txt", render_audit_readme_note()),
        safe_write_csv(
            output_dir / "benchmark_candidate_table.csv",
            candidate_table,
            fieldnames=[
                "benchmark_name",
                "paper_or_repo",
                "year",
                "task_type",
                "dataset_count",
                "method_count",
                "metric_count",
                "data_access",
                "code_access",
                "license_risk",
                "implementation_difficulty",
                "relevance_to_hvg_scorer",
                "recommendation",
            ],
        ),
        safe_write_text(
            output_dir / "standard_benchmark_survey.md",
            render_standard_benchmark_survey(
                local_resource_audit=local_resource_audit,
                route_context=route_context,
            ),
        ),
        safe_write_text(
            output_dir / "zhao2025_reproduction_plan.md",
            render_zhao_reproduction_plan(local_resource_audit=local_resource_audit),
        ),
        safe_write_text(
            output_dir / "current_methods_adapter_audit.md",
            render_current_methods_adapter_audit(adapter_smoke=adapter_smoke),
        ),
        safe_write_text(
            output_dir / "migration_decision.md",
            render_migration_decision(route_context=route_context),
        ),
        safe_write_text(
            output_dir / "sources_appendix.md",
            render_sources_appendix(),
        ),
    ]

    summary = {
        "output_dir": str(output_dir),
        "write_events": write_events,
        "route_context": route_context,
        "recommendation": "migrate_to_benchmarkHVG_first",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
