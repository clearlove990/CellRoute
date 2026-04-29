from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from hvg_research import (
    build_default_method_registry,
    discover_scrna_input_specs,
    evaluate_real_selection,
    load_scrna_dataset,
)
from hvg_research.eval import timed_call
from hvg_research.methods import FRONTIER_LITE_METHOD, canonicalize_method_names


DEFAULT_GATE_MODEL_PATH = os.path.join(ROOT, "artifacts_gate_learning_v_next9_multireal", "learnable_gate.pt")
DEFAULT_METHODS = (
    "variance",
    "fano",
    "mv_residual",
    "analytic_pearson_residual_hvg",
    "scanpy_cell_ranger_hvg",
    "scanpy_seurat_v3_hvg",
    "triku_hvg",
    "seurat_v3_like_hvg",
    "multinomial_deviance_hvg",
    "adaptive_stat_hvg",
    "adaptive_hybrid_hvg",
    FRONTIER_LITE_METHOD,
)
CHECKPOINT_FREE_METHODS = {
    "variance",
    "fano",
    "mv_residual",
    "analytic_pearson_residual_hvg",
    "scanpy_cell_ranger_hvg",
    "scanpy_seurat_v3_hvg",
    "triku_hvg",
    "scran_model_gene_var_hvg",
    "seurat_v3_like_hvg",
    "multinomial_deviance_hvg",
    "adaptive_stat_hvg",
    "adaptive_hybrid_hvg",
    "refine_moe_hvg",
    "ablation_no_moe",
    "ablation_no_refine",
}
BASE_EXPERT_NAMES = ("variance", "mv_residual", "fano")


@dataclass(frozen=True)
class DatasetPlan:
    dataset_name: str
    max_cells: int | None
    max_genes: int | None
    mode: str
    rationale: str


ROUND1_PLANS: dict[str, DatasetPlan] = {
    "cellxgene_human_kidney_nonpt": DatasetPlan(
        "cellxgene_human_kidney_nonpt",
        None,
        None,
        "full",
        "Moderate multi-donor kidney panel from CELLxGENE; full first-round benchmark is safe.",
    ),
    "cellxgene_immune_five_donors": DatasetPlan(
        "cellxgene_immune_five_donors",
        None,
        None,
        "full",
        "Compact multi-donor immune panel from CELLxGENE; full first-round benchmark is safe.",
    ),
    "cellxgene_mouse_kidney_aging_10x": DatasetPlan(
        "cellxgene_mouse_kidney_aging_10x",
        18000,
        12000,
        "sampled",
        "Larger multi-donor mouse kidney panel; use a conservative first-round sampled benchmark.",
    ),
    "cellxgene_unciliated_epithelial_five_donors": DatasetPlan(
        "cellxgene_unciliated_epithelial_five_donors",
        None,
        None,
        "full",
        "Compact epithelial multi-donor panel from CELLxGENE; full first-round benchmark is safe.",
    ),
    "paul15": DatasetPlan("paul15", None, None, "full", "Small curated dataset; safe to benchmark at full size."),
    "E-MTAB-4388": DatasetPlan("E-MTAB-4388", None, None, "full", "Moderate size; full first-round benchmark is safe."),
    "E-MTAB-4888": DatasetPlan("E-MTAB-4888", None, None, "full", "Moderate size; full first-round benchmark is safe."),
    "E-MTAB-5061": DatasetPlan("E-MTAB-5061", None, None, "full", "Medium size; full first-round benchmark is still manageable."),
    "GBM_sd": DatasetPlan("GBM_sd", None, None, "full", "Compact labeled dataset; full benchmark is safe."),
    "FBM_cite": DatasetPlan(
        "FBM_cite",
        None,
        16000,
        "near_full",
        "Keep all cells but cap genes to control PCA and evaluation memory.",
    ),
    "mus_tissue": DatasetPlan(
        "mus_tissue",
        12000,
        8000,
        "sampled",
        "Large atlas; use stratified cell subsampling plus gene budget for round 1.",
    ),
    "homo_tissue": DatasetPlan(
        "homo_tissue",
        15000,
        8000,
        "sampled",
        "Largest atlas; use conservative sampled round before any full-scale attempt.",
    ),
}

ROUND2_PLANS: dict[str, DatasetPlan] = {
    "cellxgene_human_kidney_nonpt": DatasetPlan(
        "cellxgene_human_kidney_nonpt",
        None,
        None,
        "carryover",
        "Reuse round-1 full result.",
    ),
    "cellxgene_immune_five_donors": DatasetPlan(
        "cellxgene_immune_five_donors",
        None,
        None,
        "carryover",
        "Reuse round-1 full result.",
    ),
    "cellxgene_mouse_kidney_aging_10x": DatasetPlan(
        "cellxgene_mouse_kidney_aging_10x",
        None,
        None,
        "full",
        "Second round lifts the conservative budget and runs the full mouse kidney panel.",
    ),
    "cellxgene_unciliated_epithelial_five_donors": DatasetPlan(
        "cellxgene_unciliated_epithelial_five_donors",
        None,
        None,
        "carryover",
        "Reuse round-1 full result.",
    ),
    "paul15": DatasetPlan("paul15", None, None, "carryover", "Reuse round-1 result; no need to expand this small dataset yet."),
    "E-MTAB-4388": DatasetPlan("E-MTAB-4388", None, None, "carryover", "Reuse round-1 full result."),
    "E-MTAB-4888": DatasetPlan("E-MTAB-4888", None, None, "carryover", "Reuse round-1 full result."),
    "E-MTAB-5061": DatasetPlan("E-MTAB-5061", None, None, "carryover", "Reuse round-1 full result."),
    "GBM_sd": DatasetPlan("GBM_sd", None, None, "carryover", "Reuse round-1 full result."),
    "FBM_cite": DatasetPlan(
        "FBM_cite",
        None,
        None,
        "full",
        "Second round lifts the gene cap and runs the full matrix.",
    ),
    "mus_tissue": DatasetPlan(
        "mus_tissue",
        18000,
        12000,
        "expanded_sample",
        "Second round widens both cell and gene budgets while keeping a memory-safe sparse load.",
    ),
    "homo_tissue": DatasetPlan(
        "homo_tissue",
        20000,
        12000,
        "expanded_sample",
        "Second round widens the atlas budget conservatively before any larger attempt.",
    ),
}

PLAN_PROFILES: dict[str, dict[str, DatasetPlan]] = {
    "round1": ROUND1_PLANS,
    "round2": ROUND2_PLANS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a memory-safe benchmark on data/real_inputs.")
    parser.add_argument("--real-data-root", type=str, default="data/real_inputs")
    parser.add_argument("--output-dir", type=str, default="artifacts_real_inputs_round1")
    parser.add_argument("--gate-model-path", type=str, default=DEFAULT_GATE_MODEL_PATH)
    parser.add_argument("--plan-profile", type=str, default="round1", choices=sorted(PLAN_PROFILES))
    parser.add_argument("--datasets", type=str, default=",".join(sorted(ROUND1_PLANS)))
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=2)
    parser.add_argument("--refine-epochs", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_root = output_dir / "datasets"
    dataset_output_root.mkdir(parents=True, exist_ok=True)

    device_info = resolve_device_info()
    save_json(output_dir / "device_info.json", device_info)
    print(json.dumps(device_info, indent=2))

    real_data_root = (Path(ROOT) / args.real_data_root).resolve()
    if not real_data_root.exists():
        raise FileNotFoundError(f"Real data root not found: {real_data_root}")

    selected_methods = canonicalize_method_names([part.strip() for part in args.methods.split(",") if part.strip()])
    gate_model_path = resolve_gate_model_path(args.gate_model_path, selected_methods)
    selected_dataset_names = tuple(part.strip() for part in args.datasets.split(",") if part.strip())
    plan_by_name = PLAN_PROFILES[args.plan_profile]

    specs = discover_scrna_input_specs(real_data_root)
    specs = [spec for spec in specs if spec.dataset_name in plan_by_name and spec.dataset_name in selected_dataset_names]
    specs = sorted(specs, key=lambda spec: spec.dataset_name)
    if len(specs) != len(selected_dataset_names):
        missing = sorted(set(selected_dataset_names) - {spec.dataset_name for spec in specs})
        raise FileNotFoundError(f"Missing benchmark dataset specs: {missing}")

    registry_df = load_registry(real_data_root)
    manifest_df = build_manifest(
        specs=specs,
        registry_df=registry_df,
        top_k=args.top_k,
        plan_by_name=plan_by_name,
    )
    manifest_df.to_csv(output_dir / "dataset_manifest.csv", index=False)
    save_json(output_dir / "benchmark_plan.json", manifest_df.to_dict(orient="records"))

    all_rows: list[dict[str, object]] = []
    dataset_status_rows: list[dict[str, object]] = []
    success_names: list[str] = []
    failure_rows: list[dict[str, object]] = []

    for spec in specs:
        plan = plan_by_name[spec.dataset_name]
        dataset_dir = dataset_output_root / spec.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        raw_results_path = dataset_dir / "raw_results.csv"
        dataset_info_path = dataset_dir / "dataset_info.json"
        error_path = dataset_dir / "error.json"
        if args.resume and raw_results_path.exists() and dataset_info_path.exists():
            dataset_df = pd.read_csv(raw_results_path)
            method_summary = summarize_by_keys(dataset_df, keys=["dataset", "dataset_id", "method"])
            method_summary = rank_within_group(method_summary, group_cols=["dataset", "dataset_id"], rank_col="dataset_rank")
            method_summary.to_csv(dataset_dir / "method_summary.csv", index=False)
            all_rows.extend(dataset_df.to_dict(orient="records"))
            success_names.append(spec.dataset_name)
            dataset_status_rows.append(
                {
                    "dataset_name": spec.dataset_name,
                    "status": "success",
                    "mode": plan.mode,
                    "cells_loaded": int(dataset_df["cells"].iloc[0]),
                    "genes_loaded": int(dataset_df["genes"].iloc[0]),
                    "methods_run": int(dataset_df["method"].nunique()),
                    "best_method": str(method_summary.sort_values("dataset_rank").iloc[0]["method"]),
                    "best_overall_score": float(method_summary.sort_values("dataset_rank").iloc[0]["overall_score"]),
                }
            )
            print(f"Resuming dataset={spec.dataset_name} from existing results.")
            continue
        print(
            f"Loading dataset={spec.dataset_name} mode={plan.mode} "
            f"max_cells={plan.max_cells} max_genes={plan.max_genes}"
        )
        try:
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
                random_state=args.seed,
            )
            dataset_info = {
                "dataset_name": spec.dataset_name,
                "dataset_id": spec.dataset_id,
                "input_path": spec.input_path,
                "file_format": spec.file_format,
                "transpose": spec.transpose,
                "labels_col": spec.labels_col,
                "batches_col": spec.batches_col,
                "mode": plan.mode,
                "rationale": plan.rationale,
                "cells_original": spec.cell_count,
                "genes_original": spec.gene_count,
                "cells_loaded": int(dataset.counts.shape[0]),
                "genes_loaded": int(dataset.counts.shape[1]),
                "label_classes_loaded": None if dataset.labels is None else int(len(np.unique(dataset.labels))),
                "batch_classes_loaded": None if dataset.batches is None else int(len(np.unique(dataset.batches))),
            }
            save_json(dataset_dir / "dataset_info.json", dataset_info)

            rows = run_round1_dataset_benchmark(
                dataset=dataset,
                dataset_id=spec.dataset_id,
                spec=spec,
                method_names=selected_methods,
                gate_model_path=gate_model_path,
                refine_epochs=args.refine_epochs,
                top_k=args.top_k,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
            )
            dataset_df = pd.DataFrame(rows)
            dataset_df = add_run_level_scores(dataset_df)
            dataset_df.to_csv(dataset_dir / "raw_results.csv", index=False)
            if error_path.exists():
                error_path.unlink()

            method_summary = summarize_by_keys(dataset_df, keys=["dataset", "dataset_id", "method"])
            method_summary = rank_within_group(method_summary, group_cols=["dataset", "dataset_id"], rank_col="dataset_rank")
            method_summary.to_csv(dataset_dir / "method_summary.csv", index=False)

            all_rows.extend(dataset_df.to_dict(orient="records"))
            success_names.append(spec.dataset_name)
            dataset_status_rows.append(
                {
                    "dataset_name": spec.dataset_name,
                    "status": "success",
                    "mode": plan.mode,
                    "cells_loaded": int(dataset.counts.shape[0]),
                    "genes_loaded": int(dataset.counts.shape[1]),
                    "methods_run": len(selected_methods),
                    "best_method": str(method_summary.sort_values("dataset_rank").iloc[0]["method"]),
                    "best_overall_score": float(method_summary.sort_values("dataset_rank").iloc[0]["overall_score"]),
                }
            )
        except Exception as exc:
            error_payload = {
                "dataset_name": spec.dataset_name,
                "status": "failed",
                "mode": plan.mode,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            save_json(error_path, error_payload)
            failure_rows.append(error_payload)
            dataset_status_rows.append(
                {
                    "dataset_name": spec.dataset_name,
                    "status": "failed",
                    "mode": plan.mode,
                    "cells_loaded": None,
                    "genes_loaded": None,
                    "methods_run": 0,
                    "best_method": None,
                    "best_overall_score": None,
                }
            )
            if args.fail_fast:
                raise

    status_df = pd.DataFrame(dataset_status_rows)
    status_df.to_csv(output_dir / "dataset_status.csv", index=False)

    if all_rows:
        all_results = pd.DataFrame(all_rows)
        all_results.to_csv(output_dir / "benchmark_raw_results.csv", index=False)

        dataset_summary = summarize_by_keys(all_results, keys=["dataset", "dataset_id", "method"])
        dataset_summary = rank_within_group(dataset_summary, group_cols=["dataset", "dataset_id"], rank_col="dataset_rank")
        dataset_summary.to_csv(output_dir / "benchmark_dataset_summary.csv", index=False)

        global_summary = summarize_by_keys(all_results, keys=["method"])
        global_summary = rank_within_group(global_summary, group_cols=[], rank_col="global_rank")
        global_summary.to_csv(output_dir / "benchmark_global_summary.csv", index=False)
    else:
        all_results = pd.DataFrame()
        dataset_summary = pd.DataFrame()
        global_summary = pd.DataFrame()

    next_steps = build_next_steps_report(
        manifest_df=manifest_df,
        status_df=status_df,
        dataset_summary=dataset_summary,
        global_summary=global_summary,
        successful_datasets=success_names,
        failure_rows=failure_rows,
        methods=selected_methods,
        gate_model_path=gate_model_path,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        refine_epochs=args.refine_epochs,
        plan_profile=args.plan_profile,
    )
    (output_dir / "benchmark_next_steps.md").write_text(next_steps, encoding="utf-8")
    print(next_steps)


def resolve_device_info() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    info: dict[str, object] = {
        "device": "cuda" if cuda_available else "cpu",
        "cuda_available": cuda_available,
        "cuda_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        info["cuda_devices"] = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
    return info


def resolve_gate_model_path(candidate_path: str, methods: tuple[str, ...]) -> str | None:
    requires_gate = any(name not in CHECKPOINT_FREE_METHODS for name in methods)
    path = Path(candidate_path)
    if path.exists():
        return str(path.resolve())
    if requires_gate:
        raise FileNotFoundError(f"Requested gate-based methods but checkpoint was not found: {candidate_path}")
    return None


def load_registry(real_data_root: Path) -> pd.DataFrame:
    registry_path = real_data_root / "registry.csv"
    if not registry_path.exists():
        return pd.DataFrame()
    return pd.read_csv(registry_path)


def build_manifest(*, specs: list, registry_df: pd.DataFrame, top_k: int, plan_by_name: dict[str, DatasetPlan]) -> pd.DataFrame:
    registry_by_name = registry_df.set_index("dataset_name", drop=False) if not registry_df.empty else None
    rows: list[dict[str, object]] = []
    for spec in specs:
        plan = plan_by_name[spec.dataset_name]
        registry_row = None if registry_by_name is None or spec.dataset_name not in registry_by_name.index else registry_by_name.loc[spec.dataset_name]
        benchmark_cells = int(spec.cell_count) if plan.max_cells is None else min(int(spec.cell_count), plan.max_cells)
        benchmark_genes = int(spec.gene_count) if plan.max_genes is None else min(int(spec.gene_count), plan.max_genes)
        rows.append(
            {
                "dataset_name": spec.dataset_name,
                "dataset_id": spec.dataset_id,
                "input_path": spec.input_path,
                "file_format": spec.file_format,
                "transpose": spec.transpose,
                "labels_col": spec.labels_col,
                "batches_col": spec.batches_col,
                "cells": spec.cell_count,
                "genes": spec.gene_count,
                "label_classes": spec.label_classes,
                "batch_classes": spec.batch_classes,
                "matrix_gb": None if registry_row is None else float(registry_row["matrix_gb"]),
                "benchmark_mode": plan.mode,
                "benchmark_max_cells": plan.max_cells,
                "benchmark_max_genes": plan.max_genes,
                "benchmark_cells": benchmark_cells,
                "benchmark_genes": benchmark_genes,
                "top_k": top_k,
                "suitable_for_first_round": plan.mode in {"full", "near_full"},
                "rationale": plan.rationale,
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_name").reset_index(drop=True)


def run_round1_dataset_benchmark(
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
) -> list[dict[str, object]]:
    methods = build_default_method_registry(
        top_k=top_k,
        refine_epochs=refine_epochs,
        random_state=seed,
        gate_model_path=gate_model_path,
    )
    missing_methods = [name for name in method_names if name not in methods]
    if missing_methods:
        raise KeyError(f"Method(s) not available in registry: {missing_methods}")

    current_top_k = min(top_k, dataset.counts.shape[1])
    rows: list[dict[str, object]] = []
    for method_name in method_names:
        method_fn = methods[method_name]
        scores, elapsed = timed_call(method_fn, dataset.counts, dataset.batches, current_top_k)
        selected = np.argsort(scores)[-current_top_k:]
        primary_metadata = extract_method_metadata(method_fn)
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
        row: dict[str, object] = {
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
            "top_k": int(current_top_k),
            "runtime_sec": float(elapsed),
        }
        row.update({key: float(value) for key, value in metrics.items()})
        row.update(primary_metadata)
        rows.append(row)
    return rows


def extract_method_metadata(method_fn: object) -> dict[str, float | str]:
    metadata: dict[str, float | str] = {}
    gate_source = getattr(method_fn, "last_gate_source", None)
    if gate_source is not None:
        metadata["gate_source"] = str(gate_source)

    dataset_stats = getattr(method_fn, "last_dataset_stats", None)
    if isinstance(dataset_stats, dict):
        for key, value in dataset_stats.items():
            try:
                metadata[f"stat_{key}"] = float(value)
            except (TypeError, ValueError):
                continue

    gate = getattr(method_fn, "last_gate", None)
    if gate is not None:
        gate_values = np.asarray(gate, dtype=np.float64).ravel()
        if gate_values.size == len(BASE_EXPERT_NAMES):
            for expert_name, weight in zip(BASE_EXPERT_NAMES, gate_values, strict=False):
                metadata[f"weight_{expert_name}"] = float(weight)
        else:
            metadata["selector_gate_dim"] = float(gate_values.size)

    gate_metadata = getattr(method_fn, "last_gate_metadata", None)
    if isinstance(gate_metadata, dict):
        for key, value in gate_metadata.items():
            try:
                metadata[key] = float(value)
            except (TypeError, ValueError):
                metadata[key] = str(value)
    return metadata


def add_run_level_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    positive_metrics = (
        "ari",
        "nmi",
        "label_silhouette",
        "batch_mixing",
        "neighbor_preservation",
        "cluster_silhouette",
        "stability",
    )
    negative_metrics = ("runtime_sec",)
    frames = []
    for _, group in df.groupby(["dataset", "dataset_id", "seed", "top_k"], sort=False):
        scored = group.copy()
        score = np.zeros(len(scored), dtype=np.float64)
        for metric in positive_metrics:
            if metric in scored.columns:
                score += min_max_scale(scored[metric].to_numpy(dtype=np.float64))
        for metric in negative_metrics:
            if metric in scored.columns:
                score -= 0.15 * min_max_scale(scored[metric].to_numpy(dtype=np.float64))
        scored["overall_score"] = score
        scored = scored.sort_values(["overall_score", "runtime_sec", "method"], ascending=[False, True, True]).reset_index(drop=True)
        scored["overall_rank"] = np.arange(1, len(scored) + 1)
        best_score = float(scored["overall_score"].max())
        scored["is_winner"] = np.abs(scored["overall_score"] - best_score) <= 1e-12
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


def summarize_by_keys(df: pd.DataFrame, *, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    tracked_metrics = [
        "ari",
        "nmi",
        "label_silhouette",
        "batch_mixing",
        "cluster_silhouette",
        "neighbor_preservation",
        "stability",
        "runtime_sec",
        "overall_score",
        "overall_rank",
    ]
    available_metrics = [metric for metric in tracked_metrics if metric in df.columns]
    rows: list[dict[str, object]] = []
    for group_key, group in df.groupby(keys, dropna=False, sort=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {key: value for key, value in zip(keys, key_tuple, strict=False)}
        for metric in available_metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            row[metric] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=0))
        row["run_count"] = int(len(group))
        row["win_count"] = int(group["is_winner"].sum()) if "is_winner" in group.columns else 0
        row["win_rate"] = float(row["win_count"] / max(row["run_count"], 1))
        rows.append(row)
    return pd.DataFrame(rows)


def rank_within_group(df: pd.DataFrame, *, group_cols: list[str], rank_col: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    frames = []
    iterator = df.groupby(group_cols, dropna=False, sort=False) if group_cols else [((), df)]
    for _, group in iterator:
        ranked = group.sort_values(
            ["overall_rank", "win_rate", "overall_score", "runtime_sec", "method"],
            ascending=[True, False, False, True, True],
        ).reset_index(drop=True)
        ranked[rank_col] = np.arange(1, len(ranked) + 1)
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True)


def min_max_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if len(values) == 0 or int(finite_mask.sum()) == 0:
        return np.zeros_like(values)
    finite_values = values[finite_mask]
    if np.allclose(finite_values.max(), finite_values.min()):
        return np.zeros_like(values)
    scaled = np.zeros_like(values)
    scaled[finite_mask] = (finite_values - finite_values.min()) / (finite_values.max() - finite_values.min())
    return scaled


def build_next_steps_report(
    *,
    manifest_df: pd.DataFrame,
    status_df: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    successful_datasets: list[str],
    failure_rows: list[dict[str, object]],
    methods: tuple[str, ...],
    gate_model_path: str | None,
    top_k: int,
    bootstrap_samples: int,
    refine_epochs: int,
    plan_profile: str,
) -> str:
    lines = [
        f"# {plan_profile.capitalize()} Real Input Benchmark",
        "",
        f"- Real datasets targeted: {len(manifest_df)}",
        f"- Successful datasets: {len(successful_datasets)}",
        f"- Failed datasets: {len(failure_rows)}",
        f"- Methods: {', '.join(methods)}",
        f"- Gate checkpoint: {gate_model_path or 'not used'}",
        f"- top_k: {top_k}",
        f"- bootstrap_samples: {bootstrap_samples}",
        f"- refine_epochs: {refine_epochs}",
        "",
        "## Successes",
    ]
    if successful_datasets:
        for dataset_name in successful_datasets:
            lines.append(f"- {dataset_name}")
    else:
        lines.append("- None")

    lines.extend(["", "## Failures"])
    if failure_rows:
        for row in failure_rows:
            lines.append(f"- {row['dataset_name']}: {row['error_type']} - {row['error_message']}")
    else:
        lines.append("- None")

    lines.extend(["", "## First-Round Plan"])
    for _, row in manifest_df.iterrows():
        lines.append(
            f"- {row['dataset_name']}: mode={row['benchmark_mode']} "
            f"benchmark_cells={row['benchmark_cells']} benchmark_genes={row['benchmark_genes']} "
            f"rationale={row['rationale']}"
        )

    if not dataset_summary.empty:
        lines.extend(["", "## Per-Dataset Winners"])
        winners = dataset_summary.sort_values(["dataset", "dataset_rank"]).groupby("dataset", sort=False).head(1)
        for _, row in winners.iterrows():
            lines.append(
                f"- {row['dataset']}: winner={row['method']} "
                f"overall_score={row['overall_score']:.4f} runtime_sec={row['runtime_sec']:.2f}"
            )

    if not global_summary.empty:
        lines.extend(["", "## Global Ranking"])
        for _, row in global_summary.sort_values("global_rank").iterrows():
            lines.append(
                f"- rank {int(row['global_rank'])}: {row['method']} "
                f"overall_score={row['overall_score']:.4f} runtime_sec={row['runtime_sec']:.2f}"
            )

    lines.extend(["", "## Next Recommendations"])
    sampled_rows = manifest_df[manifest_df["benchmark_mode"] == "sampled"]
    if not sampled_rows.empty:
        for _, row in sampled_rows.iterrows():
            lines.append(
                f"- {row['dataset_name']}: if round-1 results are stable, next try "
                f"max_cells={int(row['benchmark_cells']) * 2} and max_genes={min(int(row['benchmark_genes']) * 2, int(row['genes']))}."
            )
    near_full_rows = manifest_df[manifest_df["benchmark_mode"] == "near_full"]
    if not near_full_rows.empty:
        for _, row in near_full_rows.iterrows():
            lines.append(
                f"- {row['dataset_name']}: consider lifting max_genes from {row['benchmark_genes']} toward full {row['genes']} if memory headroom is acceptable."
            )
    if failure_rows:
        lines.append("- For failed datasets, keep the sparse-loading path and add stricter per-dataset budgets before rerunning.")
    else:
        lines.append("- The next full benchmark pass can keep the same script and widen only the sampled datasets.")
    return "\n".join(lines) + "\n"


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
