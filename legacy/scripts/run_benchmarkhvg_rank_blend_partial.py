from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import io as spio
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hvg_research.methods import build_default_method_registry
from run_benchmarkhvg_adapter_stub_partial import (
    DATASET_REGISTRY,
    DEFAULT_R_ENV_ROOT,
    R_HELPER,
    _compute_context,
    _load_lines,
    _metric_columns_for_mode,
    _nonfinite_to_floor,
    _r_runtime_env,
    _run_command,
    _scores_to_rank,
    _summary_metric_lines,
    _write_json,
    _write_text,
    _write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official partial benchmarkHVG evaluation for blended rank vectors.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--r-env-root", type=Path, default=DEFAULT_R_ENV_ROOT)
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--anchor-method", type=str, default="adaptive_hybrid_hvg")
    parser.add_argument("--candidate-method", type=str, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--nfeatures", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--refine-epochs", type=int, default=0)
    parser.add_argument(
        "--method-list",
        type=str,
        default="mv_lognc,logmv_lognc,scran_pos,seuratv1,mean_max_nc",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact directory: {output_dir}")
    output_dir.mkdir(parents=True)
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True)

    rscript, r_env = _r_runtime_env(args.r_env_root.resolve())
    if rscript is None or r_env is None:
        raise FileNotFoundError(f"Could not locate Rscript under {args.r_env_root}")

    dataset_info = DATASET_REGISTRY[args.dataset]
    expr_rds = dataset_info["expr_rds"]
    annotation_rds = dataset_info["annotation_rds"]
    evaluation_mode = str(dataset_info["evaluation_mode"])
    continuous_input = str(dataset_info.get("continuous_input", ""))

    export_json = output_dir / f"{args.dataset}_export_metadata.json"
    counts_mtx = output_dir / f"{args.dataset}_counts.mtx"
    gene_names_path = output_dir / f"{args.dataset}_gene_names.tsv"
    cell_names_path = output_dir / f"{args.dataset}_cell_names.tsv"
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
        "--metadata-out",
        str(export_json),
    ]
    export_result = _run_command(export_cmd, env=r_env, timeout=1_200_000)
    if not export_result.ok or not export_json.exists():
        raise RuntimeError(f"Export failed: {export_result.stderr}\n{export_result.error}")
    export_metadata = json.loads(export_json.read_text(encoding="utf-8"))

    counts_raw = spio.mmread(str(counts_mtx))
    if sparse.issparse(counts_raw):
        counts = np.asarray(counts_raw.tocsr().transpose().toarray(), dtype=np.float32)
    else:
        counts = np.asarray(counts_raw.T, dtype=np.float32)
    gene_names = _load_lines(gene_names_path)
    cell_names = _load_lines(cell_names_path)
    if counts.shape != (len(cell_names), len(gene_names)):
        raise ValueError(f"Count matrix shape mismatch: {counts.shape}")

    registry = build_default_method_registry(
        top_k=args.nfeatures,
        refine_epochs=args.refine_epochs,
        random_state=args.random_state,
        gate_model_path=None,
    )
    missing = [name for name in (args.anchor_method, args.candidate_method) if name not in registry]
    if missing:
        raise KeyError(f"Missing methods in registry: {missing}")
    anchor_scores = _nonfinite_to_floor(np.asarray(registry[args.anchor_method](counts, None, args.nfeatures), dtype=np.float64))
    candidate_scores = _nonfinite_to_floor(np.asarray(registry[args.candidate_method](counts, None, args.nfeatures), dtype=np.float64))
    anchor_rank = _scores_to_rank(anchor_scores).astype(np.float64)
    candidate_rank = _scores_to_rank(candidate_scores).astype(np.float64)
    blended_rank_score = -((1.0 - args.alpha) * anchor_rank + args.alpha * candidate_rank)
    blended_rank = _scores_to_rank(blended_rank_score)

    method_prefix = f"blend_{args.anchor_method}__{args.candidate_method}__alpha_{args.alpha:.2f}".replace(".", "p")
    order = np.argsort(blended_rank)
    rank_rows = [
        {"gene": gene_names[idx], "rank": int(blended_rank[idx]), "score": float(blended_rank_score[idx])}
        for idx in order
    ]
    rank_tsv = output_dir / f"{method_prefix}_full_rank.tsv"
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
        raise RuntimeError(f"Official eval failed: {eval_result.stderr}\n{eval_result.error}")
    eval_payload = json.loads(eval_json.read_text(encoding="utf-8"))

    metric_columns = _metric_columns_for_mode(str(eval_payload["evaluation_mode"]))
    metrics_rows: list[dict[str, Any]] = []
    for setting_name, key in (("baseline_mix", "baseline"), (f"adapter_mix_with_{method_prefix}_rank", "adapter")):
        row: dict[str, Any] = {"setting": setting_name}
        for metric_name in metric_columns:
            row[metric_name] = eval_payload[key][metric_name]
        row["hvg_count"] = eval_payload[key]["hvg_count"]
        row["pca_rows"] = eval_payload[key]["pca_dim"][0]
        row["pca_cols"] = eval_payload[key]["pca_dim"][1]
        metrics_rows.append(row)
    with (output_dir / "partial_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["setting", *metric_columns, "hvg_count", "pca_rows", "pca_cols"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    anchor_top = set(np.argsort(anchor_rank)[: args.nfeatures].tolist())
    candidate_top = set(np.argsort(candidate_rank)[: args.nfeatures].tolist())
    blended_top = set(np.argsort(blended_rank)[: args.nfeatures].tolist())
    metadata = {
        "dataset": args.dataset,
        "anchor_method": args.anchor_method,
        "candidate_method": args.candidate_method,
        "alpha": args.alpha,
        "nfeatures": args.nfeatures,
        "export_metadata": export_metadata,
        "overlap_blend_to_anchor": len(blended_top & anchor_top),
        "overlap_blend_to_candidate": len(blended_top & candidate_top),
        "overlap_anchor_to_candidate": len(anchor_top & candidate_top),
    }
    _write_json(output_dir / "blend_metadata.json", metadata)
    _write_json(output_dir / "compute_context.json", _compute_context(args.r_env_root.resolve(), "rank_blend_cpu_statistical_route"))
    _write_text(
        output_dir / "rank_blend_partial_summary.md",
        "\n".join(
            [
                "# benchmarkHVG Rank Blend Partial Summary",
                "",
                f"- dataset: `{args.dataset}`",
                f"- anchor: `{args.anchor_method}`",
                f"- candidate: `{args.candidate_method}`",
                f"- alpha: `{args.alpha:.2f}`",
                *_summary_metric_lines(eval_payload),
                f"- blend vs baseline HVG overlap: `{eval_payload['overlap']['adapter_vs_baseline_hvg_overlap']}` / `{args.nfeatures}`",
                f"- blend vs anchor top overlap: `{metadata['overlap_blend_to_anchor']}` / `{args.nfeatures}`",
                f"- blend vs candidate top overlap: `{metadata['overlap_blend_to_candidate']}` / `{args.nfeatures}`",
            ]
        )
        + "\n",
    )
    print(f"Wrote rank blend partial artifacts to {output_dir}")


if __name__ == "__main__":
    main()
