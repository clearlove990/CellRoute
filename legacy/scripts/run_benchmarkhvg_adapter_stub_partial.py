from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None

import numpy as np
from scipy import io as spio
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hvg_research.methods import build_default_method_registry


DEFAULT_R_ENV_ROOT = ROOT / ".conda_benchmarkhvg_bootstrap_r44"
DATA_PROCESSED_DIR = ROOT / "data" / "standard_benchmarkhvg_processed"
R_HELPER = ROOT / "scripts" / "run_benchmarkhvg_adapter_stub_partial.R"
BOOTSTRAP_DECISION = ROOT / "artifacts_benchmarkhvg_bootstrap_audit_r44_v4" / "final_bootstrap_decision.md"
BOOTSTRAP_PATCH_AUDIT = ROOT / "artifacts_benchmarkhvg_bootstrap_audit_r44_v4" / "official_wrapper_patch_audit.md"

DATASET_REGISTRY: dict[str, dict[str, Path]] = {
    "duo4_pbmc": {
        "expr_rds": DATA_PROCESSED_DIR / "duo4_pbmc" / "duo4_pbmc" / "duo4_expr.rds",
        "annotation_rds": DATA_PROCESSED_DIR / "duo4_pbmc" / "duo4_pbmc" / "duo4_label.rds",
        "evaluation_mode": "discrete",
    },
    "duo4un_pbmc": {
        "expr_rds": DATA_PROCESSED_DIR / "duo4un_pbmc" / "duo4un_pbmc" / "duo4un_expr.rds",
        "annotation_rds": DATA_PROCESSED_DIR / "duo4un_pbmc" / "duo4un_pbmc" / "duo4un_label.rds",
        "evaluation_mode": "discrete",
    },
    "duo8_pbmc": {
        "expr_rds": DATA_PROCESSED_DIR / "duo8_pbmc" / "duo8_pbmc" / "duo8_expr.rds",
        "annotation_rds": DATA_PROCESSED_DIR / "duo8_pbmc" / "duo8_pbmc" / "duo8_label.rds",
        "evaluation_mode": "discrete",
    },
    "pbmc3k_multi": {
        "expr_rds": DATA_PROCESSED_DIR / "pbmc3k_multi" / "pbmc3k_multi" / "pbmc3k_rna.mat.rds",
        "annotation_rds": DATA_PROCESSED_DIR / "pbmc3k_multi" / "pbmc3k_multi" / "pbmc3k_lsistdev.rds",
        "evaluation_mode": "continuous",
        "continuous_input": "MultiomeATAC",
    },
    "pbmc_cite": {
        "expr_rds": DATA_PROCESSED_DIR / "pbmc_cite" / "pbmc_cite" / "cbmc_pbmc_rna_filter.rds",
        "annotation_rds": DATA_PROCESSED_DIR / "pbmc_cite" / "pbmc_cite" / "cbmc_pbmc_pro_filter.rds",
        "evaluation_mode": "continuous",
        "continuous_input": "CITEseq",
    },
}

SUPPORTED_MAINLINE_METHODS = (
    "adaptive_hybrid_hvg",
    "adaptive_core_consensus_hvg",
    "adaptive_risk_parity_hvg",
    "adaptive_risk_parity_safe_hvg",
    "adaptive_risk_parity_ultrasafe_hvg",
    "adaptive_eb_shrinkage_hvg",
    "adaptive_spectral_locality_hvg",
    "sigma_safe_core_v5_hvg",
)


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal benchmarkHVG extra.rank partial experiment for a selected repo HVG method on a selected pilot dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / f"artifacts_benchmarkhvg_adapter_stub_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--r-env-root", type=Path, default=DEFAULT_R_ENV_ROOT)
    parser.add_argument("--dataset", type=str, default="duo4_pbmc", choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--method", type=str, default="adaptive_hybrid_hvg", choices=SUPPORTED_MAINLINE_METHODS)
    parser.add_argument("--nfeatures", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--refine-epochs", type=int, default=0)
    parser.add_argument(
        "--method-list",
        type=str,
        default="mv_lognc,logmv_lognc,scran_pos,seuratv1,mean_max_nc",
        help="Comma-separated official mixhvg method list to use in a single mixture.",
    )
    return parser.parse_args()


def _ensure_new_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact directory: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _run_command(
    args: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    timeout: int = 120_000,
) -> CommandResult:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command=args,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return CommandResult(
            command=args,
            returncode=1,
            stdout="",
            stderr="",
            error=f"{type(exc).__name__}: {exc}",
        )


def _r_runtime_env(r_env_root: Path) -> tuple[Path | None, dict[str, str] | None]:
    candidates = [
        r_env_root / "lib" / "R" / "bin" / "x64" / "Rscript.exe",
        r_env_root / "Scripts" / "Rscript.exe",
    ]
    rscript = next((path for path in candidates if path.exists()), None)
    if rscript is None:
        return None, None
    env = os.environ.copy()
    env["R_HOME"] = str(r_env_root / "lib" / "R")
    path_parts = [
        r_env_root / "lib" / "R" / "bin" / "x64",
        r_env_root / "lib" / "R" / "bin",
        r_env_root / "Library" / "mingw-w64" / "bin",
        r_env_root / "Library" / "usr" / "bin",
        r_env_root / "Library" / "bin",
        r_env_root / "Scripts",
        r_env_root / "bin",
    ]
    existing = [str(path) for path in path_parts if path.exists()]
    env["PATH"] = ";".join(existing + [env.get("PATH", "")])
    return rscript, env


def _load_lines(path: Path) -> list[str]:
    return [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]


def _nonfinite_to_floor(scores: np.ndarray) -> np.ndarray:
    fixed = np.asarray(scores, dtype=np.float64).copy()
    finite = np.isfinite(fixed)
    if np.any(~finite):
        replacement = float(np.min(fixed[finite]) - 1.0) if np.any(finite) else 0.0
        fixed[~finite] = replacement
    return fixed


def _scores_to_rank(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return ranks


def _method_metadata_dict(method_fn: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for attr in ("last_gate_metadata", "last_dataset_stats"):
        value = getattr(method_fn, attr, None)
        if isinstance(value, dict):
            payload[attr] = value
    gate_source = getattr(method_fn, "last_gate_source", None)
    if gate_source is not None:
        payload["last_gate_source"] = gate_source
    gate = getattr(method_fn, "last_gate", None)
    if gate is not None:
        payload["last_gate"] = np.asarray(gate, dtype=np.float64).tolist()
    return payload


def _compute_context(r_env_root: Path, selected_device_note: str) -> dict[str, Any]:
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(ROOT),
        "torch_available": torch is not None,
        "torch_version": None if torch is None else torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "selected_device_note": selected_device_note,
        "r_env_root": str(r_env_root),
        "r_helper_script": str(R_HELPER),
        "bootstrap_decision_artifact": str(BOOTSTRAP_DECISION),
        "bootstrap_patch_audit_artifact": str(BOOTSTRAP_PATCH_AUDIT),
    }


def _metric_columns_for_mode(evaluation_mode: str) -> list[str]:
    if evaluation_mode == "discrete":
        return ["var_ratio", "ari", "nmi", "lisi"]
    if evaluation_mode == "continuous":
        return ["var_ratio", "dist_cor", "knn_ratio", "three_nn", "max_ari", "max_nmi"]
    raise ValueError(f"Unsupported evaluation mode: {evaluation_mode}")


def _summary_metric_lines(eval_payload: dict[str, Any]) -> list[str]:
    evaluation_mode = str(eval_payload["evaluation_mode"])
    baseline = eval_payload["baseline"]
    adapter = eval_payload["adapter"]
    if evaluation_mode == "discrete":
        return [
            f"- baseline mix ARI / NMI / LISI: `{baseline['ari']:.4f}` / `{baseline['nmi']:.4f}` / `{baseline['lisi']:.4f}`",
            f"- adapter mix ARI / NMI / LISI: `{adapter['ari']:.4f}` / `{adapter['nmi']:.4f}` / `{adapter['lisi']:.4f}`",
            f"- baseline mix var_ratio: `{baseline['var_ratio']:.4f}`",
            f"- adapter mix var_ratio: `{adapter['var_ratio']:.4f}`",
        ]
    if evaluation_mode == "continuous":
        return [
            f"- baseline mix max_ARI / max_NMI: `{baseline['max_ari']:.4f}` / `{baseline['max_nmi']:.4f}`",
            f"- adapter mix max_ARI / max_NMI: `{adapter['max_ari']:.4f}` / `{adapter['max_nmi']:.4f}`",
            f"- baseline mix var_ratio / dist_cor / knn_ratio / 3nn: `{baseline['var_ratio']:.4f}` / `{baseline['dist_cor']:.4f}` / `{baseline['knn_ratio']:.4f}` / `{baseline['three_nn']:.4f}`",
            f"- adapter mix var_ratio / dist_cor / knn_ratio / 3nn: `{adapter['var_ratio']:.4f}` / `{adapter['dist_cor']:.4f}` / `{adapter['knn_ratio']:.4f}` / `{adapter['three_nn']:.4f}`",
        ]
    raise ValueError(f"Unsupported evaluation mode: {evaluation_mode}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    _ensure_new_dir(output_dir)
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=False)

    rscript, r_env = _r_runtime_env(args.r_env_root.resolve())
    if rscript is None or r_env is None:
        raise FileNotFoundError(f"Rscript not found under {args.r_env_root}")

    dataset_paths = DATASET_REGISTRY[args.dataset]
    expr_rds = dataset_paths["expr_rds"]
    annotation_rds = dataset_paths["annotation_rds"]
    evaluation_mode = str(dataset_paths["evaluation_mode"])
    continuous_input = str(dataset_paths.get("continuous_input", "MultiomeATAC"))
    dataset_prefix = args.dataset
    if not expr_rds.exists() or not annotation_rds.exists():
        raise FileNotFoundError(f"{args.dataset} processed RDS files are missing.")

    export_json = output_dir / f"{dataset_prefix}_export_metadata.json"
    counts_mtx = output_dir / f"{dataset_prefix}_counts.mtx"
    gene_names_path = output_dir / f"{dataset_prefix}_gene_names.tsv"
    cell_names_path = output_dir / f"{dataset_prefix}_cell_names.tsv"
    annotation_out = output_dir / f"{dataset_prefix}_annotation.tsv"
    export_cmd = [
        str(rscript),
        "--vanilla",
        str(R_HELPER),
        "--mode",
        "export_dataset",
        "--expr-rds",
        str(expr_rds),
        "--annotation-rds",
        str(annotation_rds),
        "--annotation-mode",
        evaluation_mode,
        "--counts-mtx",
        str(counts_mtx),
        "--gene-names-out",
        str(gene_names_path),
        "--cell-names-out",
        str(cell_names_path),
        "--annotation-out",
        str(annotation_out),
        "--metadata-out",
        str(export_json),
    ]
    export_result = _run_command(export_cmd, env=r_env, timeout=900_000)
    if not export_result.ok or not export_json.exists():
        raise RuntimeError(
            f"Failed to export {args.dataset} for the adapter stub partial experiment.\n"
            f"stdout:\n{export_result.stdout}\n\nstderr:\n{export_result.stderr}\n\nerror:\n{export_result.error}"
        )
    export_metadata = json.loads(export_json.read_text(encoding="utf-8"))

    counts_raw = spio.mmread(str(counts_mtx))
    if sparse.issparse(counts_raw):
        counts_gene_by_cell = counts_raw.tocsr()
        counts = np.asarray(counts_gene_by_cell.transpose().toarray(), dtype=np.float32)
    else:
        counts = np.asarray(counts_raw.T, dtype=np.float32)

    gene_names = _load_lines(gene_names_path)
    cell_names = _load_lines(cell_names_path)
    if counts.shape != (len(cell_names), len(gene_names)):
        raise ValueError(
            f"Count matrix shape mismatch after transpose: got {counts.shape}, "
            f"expected {(len(cell_names), len(gene_names))}"
        )

    method_prefix = args.method
    selected_device_note = (
        f"cuda_available_but_current_route_may_remain_cpu_only_if_{method_prefix}_resolves_to_statistical_branch"
        if torch is not None and torch.cuda.is_available()
        else "cpu_fallback"
    )
    compute_context = _compute_context(args.r_env_root.resolve(), selected_device_note)
    _write_json(output_dir / "compute_context.json", compute_context)

    registry = build_default_method_registry(
        top_k=args.nfeatures,
        refine_epochs=args.refine_epochs,
        random_state=args.random_state,
        gate_model_path=None,
    )
    method_fn = registry[args.method]
    raw_scores = np.asarray(method_fn(counts, None, args.nfeatures), dtype=np.float64)
    if raw_scores.ndim != 1:
        raise ValueError(f"Expected a 1D score vector from {args.method}, got shape {raw_scores.shape}")
    if raw_scores.shape[0] != len(gene_names):
        raise ValueError(
            f"Score vector length mismatch: expected {len(gene_names)} genes, got {raw_scores.shape[0]}"
        )
    scores = _nonfinite_to_floor(raw_scores)
    ranks = _scores_to_rank(scores)

    method_metadata = _method_metadata_dict(method_fn)
    method_metadata.update(
        {
            "method": args.method,
            "dataset": args.dataset,
            "evaluation_mode": evaluation_mode,
            "nfeatures": args.nfeatures,
            "random_state": args.random_state,
            "refine_epochs": args.refine_epochs,
            "score_length": int(scores.shape[0]),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        }
    )
    method_metadata_name = f"{method_prefix}_method_metadata.json"
    _write_json(output_dir / method_metadata_name, method_metadata)

    order = np.argsort(ranks)
    rank_rows = [
        {
            "gene": gene_names[idx],
            "rank": int(ranks[idx]),
            "score": float(scores[idx]),
        }
        for idx in order
    ]
    rank_tsv_name = f"{method_prefix}_full_rank.tsv"
    rank_tsv = output_dir / rank_tsv_name
    _write_tsv(rank_tsv, rank_rows, fieldnames=["gene", "rank", "score"])

    eval_json = output_dir / "official_extra_rank_eval.json"
    eval_cmd = [
        str(rscript),
        "--vanilla",
        str(R_HELPER),
        "--mode",
        "run_adapter_eval",
        "--expr-rds",
        str(expr_rds),
        "--annotation-rds",
        str(annotation_rds),
        "--annotation-mode",
        evaluation_mode,
        "--rank-tsv",
        str(rank_tsv),
        "--output-json",
        str(eval_json),
        "--nfeatures",
        str(args.nfeatures),
        "--method-list-csv",
        args.method_list,
    ]
    if evaluation_mode == "continuous":
        eval_cmd.extend(["--continuous-input", continuous_input])
    eval_result = _run_command(eval_cmd, env=r_env, timeout=3_600_000)
    if not eval_result.ok or not eval_json.exists():
        raise RuntimeError(
            "Failed to run the official extra.rank partial evaluation.\n"
            f"stdout:\n{eval_result.stdout}\n\nstderr:\n{eval_result.stderr}\n\nerror:\n{eval_result.error}"
        )
    eval_payload = json.loads(eval_json.read_text(encoding="utf-8"))

    metric_columns = _metric_columns_for_mode(str(eval_payload["evaluation_mode"]))
    metrics_rows: list[dict[str, Any]] = []
    for setting_name, key in (
        ("baseline_mix", "baseline"),
        (f"adapter_mix_with_{method_prefix}_rank", "adapter"),
    ):
        row: dict[str, Any] = {"setting": setting_name}
        for metric_name in metric_columns:
            row[metric_name] = eval_payload[key][metric_name]
        row["hvg_count"] = eval_payload[key]["hvg_count"]
        row["pca_rows"] = eval_payload[key]["pca_dim"][0]
        row["pca_cols"] = eval_payload[key]["pca_dim"][1]
        metrics_rows.append(row)
    with (output_dir / "partial_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["setting", *metric_columns, "hvg_count", "pca_rows", "pca_cols"],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    benchmark_layer = (
        "`benchmarkHVG::mixture_hvg_pca(..., extra.rank = ...)` plus `benchmarkHVG::evaluate_hvg_discrete()`"
        if evaluation_mode == "discrete"
        else "`benchmarkHVG::mixture_hvg_pca(..., extra.rank = ...)` plus `benchmarkHVG::evaluate_hvg_continuous()`"
    )
    summary_metric_lines = _summary_metric_lines(eval_payload)

    summary_lines = [
        "# benchmarkHVG Adapter Stub Partial Summary",
        "",
        "## Scope",
        "",
        f"- dataset: `{args.dataset}` only",
        f"- method: `{args.method}` only",
        f"- evaluation mode: `{evaluation_mode}`",
        f"- benchmark layer: {benchmark_layer}",
        "- non-goals: no full benchmark, no multi-dataset sweep, no new scorer design",
        "",
        "## Bootstrap Dependency",
        "",
        f"- bootstrap decision artifact: `{BOOTSTRAP_DECISION}`",
        f"- local patch audit artifact: `{BOOTSTRAP_PATCH_AUDIT}`",
        "- interpretation boundary: this partial experiment depends on the same local compatibility patches already documented in the bootstrap audit; it does not claim untouched upstream reproduction.",
        "",
        "## Adapter Result",
        "",
        f"- export layer ok: `{export_metadata.get('ok', False)}`",
        f"- external full-rank vector generated: `True`",
        f"- rank length: `{len(rank_rows)}`",
        f"- rank contract contiguous: `{eval_payload['rank_contract']['contiguous']}`",
        *summary_metric_lines,
        f"- adapter vs baseline HVG overlap: `{eval_payload['overlap']['adapter_vs_baseline_hvg_overlap']}` / `{args.nfeatures}`",
        f"- adapter vs external top overlap: `{eval_payload['overlap']['adapter_vs_external_top_overlap']}` / `{args.nfeatures}`",
        "",
        "## Interpretation",
        "",
        f"- This run proves the current repo can emit a full-length ranked gene vector and inject it into the official `extra.rank` hook on the `{args.dataset}` pilot.",
        "- This is only an adapter-stub-level partial experiment. It is not evidence for publication-scale superiority or robustness.",
        "- The next safe expansion is a slightly cleaner adapter wrapper or one additional pilot dataset, not a full benchmark sweep.",
    ]
    _write_text(output_dir / "adapter_stub_partial_summary.md", "\n".join(summary_lines) + "\n")

    command_lines = [
        "# Adapter Stub Partial Log",
        "",
        "## Export Command",
        "",
        "```powershell",
        subprocess.list2cmdline(export_cmd),
        "```",
        "",
        f"- returncode: `{export_result.returncode}`",
        "",
        "## Official Eval Command",
        "",
        "```powershell",
        subprocess.list2cmdline(eval_cmd),
        "```",
        "",
        f"- returncode: `{eval_result.returncode}`",
        "",
        "## Method Route",
        "",
        f"- `last_gate_source`: `{method_metadata.get('last_gate_source', '')}`",
        f"- `resolved_method`: `{method_metadata.get('last_gate_metadata', {}).get('resolved_method', '')}`",
        f"- `route_name`: `{method_metadata.get('last_gate_metadata', {}).get('route_name', '')}`",
        "",
        "## Notes",
        "",
        "- The Python side records CUDA availability but does not force synthetic GPU work when the resolved adaptive route is a statistical branch.",
        "- The R side is responsible for the official wrapper call and remains the main runtime bottleneck for this partial experiment.",
    ]
    _write_text(output_dir / "adapter_stub_partial_log.md", "\n".join(command_lines) + "\n")

    manifest = {
        "ok": True,
        "dataset": args.dataset,
        "method": args.method,
        "output_dir": str(output_dir),
        "artifacts": {
            "compute_context": "compute_context.json",
            "export_metadata": export_json.name,
            "method_metadata": method_metadata_name,
            "rank_tsv": rank_tsv_name,
            "official_eval": eval_json.name,
            "metrics_csv": "partial_metrics.csv",
            "summary_md": "adapter_stub_partial_summary.md",
            "log_md": "adapter_stub_partial_log.md",
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
