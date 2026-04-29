from __future__ import annotations

import argparse
import csv
import json
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

from hvg_research.baselines import normalize_log1p
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
    parser = argparse.ArgumentParser(description="Run constrained anchor repair HVG official partial experiments.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--r-env-root", type=Path, default=DEFAULT_R_ENV_ROOT)
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--anchor-method", type=str, default="adaptive_hybrid_hvg")
    parser.add_argument("--candidate-method", type=str, default="adaptive_risk_parity_safe_hvg")
    parser.add_argument("--budgets", type=str, default="25,50,100,200")
    parser.add_argument("--candidate-pool-multiplier", type=float, default=2.0)
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
    budgets = tuple(int(part.strip()) for part in args.budgets.split(",") if part.strip())
    if any(budget < 0 or budget > args.nfeatures for budget in budgets):
        raise ValueError("Budgets must be in [0, nfeatures]")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact directory: {output_dir}")
    output_dir.mkdir(parents=True)

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
    anchor_scores = _nonfinite_to_floor(np.asarray(registry[args.anchor_method](counts, None, args.nfeatures), dtype=np.float64))
    candidate_scores = _nonfinite_to_floor(np.asarray(registry[args.candidate_method](counts, None, args.nfeatures), dtype=np.float64))
    safety_scores = compute_safety_scores(counts)

    anchor_rank = _scores_to_rank(anchor_scores)
    candidate_rank = _scores_to_rank(candidate_scores)
    anchor_top = set(np.argsort(anchor_rank)[: args.nfeatures].tolist())
    pool_size = min(int(round(args.nfeatures * args.candidate_pool_multiplier)), counts.shape[1])
    candidate_pool = [idx for idx in np.argsort(candidate_rank)[:pool_size].tolist() if idx not in anchor_top]
    removable_pool = list(anchor_top)

    add_priority = sorted(candidate_pool, key=lambda idx: (-candidate_scores[idx], -safety_scores[idx], candidate_rank[idx]))
    remove_priority = sorted(removable_pool, key=lambda idx: (candidate_scores[idx], safety_scores[idx], -anchor_rank[idx]))

    all_metrics_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for budget in budgets:
        repaired_top = set(anchor_top)
        add_genes = add_priority[:budget]
        remove_genes = remove_priority[:budget]
        repaired_top.difference_update(remove_genes)
        repaired_top.update(add_genes)
        if len(repaired_top) != args.nfeatures:
            raise RuntimeError(f"Repair budget {budget} produced {len(repaired_top)} genes, expected {args.nfeatures}")

        repair_scores = np.asarray(anchor_scores, dtype=np.float64).copy()
        floor = float(np.min(repair_scores) - 2.0)
        repair_scores[:] = floor
        for idx in repaired_top:
            repair_scores[idx] = float(args.nfeatures - candidate_rank[idx]) + 0.001 * safety_scores[idx]
        repair_rank = _scores_to_rank(repair_scores)
        method_prefix = f"anchor_repair_{args.candidate_method}_budget_{budget}"
        rank_tsv = output_dir / f"{method_prefix}_full_rank.tsv"
        order = np.argsort(repair_rank)
        rank_rows = [
            {"gene": gene_names[idx], "rank": int(repair_rank[idx]), "score": float(repair_scores[idx])}
            for idx in order
        ]
        _write_tsv(rank_tsv, rank_rows, fieldnames=["gene", "rank", "score"])

        eval_json = output_dir / f"official_extra_rank_eval_budget_{budget}.json"
        eval_payload = run_official_eval(
            rscript=rscript,
            r_env=r_env,
            expr_rds=expr_rds,
            annotation_rds=annotation_rds,
            evaluation_mode=evaluation_mode,
            continuous_input=continuous_input,
            rank_tsv=rank_tsv,
            output_json=eval_json,
            nfeatures=args.nfeatures,
            method_list=args.method_list,
        )
        metric_columns = _metric_columns_for_mode(str(eval_payload["evaluation_mode"]))
        for setting_name, key in (("baseline_mix", "baseline"), (f"adapter_mix_with_{method_prefix}_rank", "adapter")):
            row: dict[str, Any] = {"budget": budget, "setting": setting_name}
            for metric_name in metric_columns:
                row[metric_name] = eval_payload[key][metric_name]
            row["hvg_count"] = eval_payload[key]["hvg_count"]
            row["pca_rows"] = eval_payload[key]["pca_dim"][0]
            row["pca_cols"] = eval_payload[key]["pca_dim"][1]
            all_metrics_rows.append(row)
        run_rows.append(
            {
                "budget": budget,
                "added_count": len(add_genes),
                "removed_count": len(remove_genes),
                "repair_vs_anchor_overlap": len(repaired_top & anchor_top),
                "repair_vs_candidate_top_overlap": len(repaired_top & set(np.argsort(candidate_rank)[: args.nfeatures].tolist())),
                "adapter_vs_baseline_overlap": eval_payload["overlap"]["adapter_vs_baseline_hvg_overlap"],
                "summary": _summary_metric_lines(eval_payload),
            }
        )

    fieldnames = sorted({key for row in all_metrics_rows for key in row})
    with (output_dir / "repair_partial_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics_rows)
    _write_json(
        output_dir / "repair_metadata.json",
        {
            "dataset": args.dataset,
            "anchor_method": args.anchor_method,
            "candidate_method": args.candidate_method,
            "budgets": budgets,
            "candidate_pool_multiplier": args.candidate_pool_multiplier,
            "nfeatures": args.nfeatures,
            "anchor_candidate_top_overlap": len(anchor_top & set(np.argsort(candidate_rank)[: args.nfeatures].tolist())),
            "runs": run_rows,
        },
    )
    _write_json(output_dir / "compute_context.json", _compute_context(args.r_env_root.resolve(), "anchor_repair_cpu_statistical_route"))
    _write_text(output_dir / "anchor_repair_partial_summary.md", render_summary(args=args, run_rows=run_rows))
    print(f"Wrote anchor repair partial artifacts to {output_dir}")


def compute_safety_scores(counts: np.ndarray) -> np.ndarray:
    x = normalize_log1p(counts)
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    dropout = np.mean(counts <= 0, axis=0)
    lib = np.log1p(np.sum(counts, axis=1))
    corr = np.zeros(counts.shape[1], dtype=np.float64)
    centered_lib = lib - np.mean(lib)
    lib_norm = np.linalg.norm(centered_lib)
    for start in range(0, x.shape[1], 512):
        stop = min(start + 512, x.shape[1])
        block = x[:, start:stop] - np.mean(x[:, start:stop], axis=0)
        denom = np.linalg.norm(block, axis=0) * max(lib_norm, 1e-8)
        corr[start:stop] = np.abs((centered_lib @ block) / np.maximum(denom, 1e-8))
    mean_rank = percentile_rank(mean)
    var_rank = percentile_rank(var)
    dropout_safe = 1.0 - percentile_rank(dropout)
    lib_safe = 1.0 - percentile_rank(corr)
    return 0.30 * mean_rank + 0.30 * var_rank + 0.20 * dropout_safe + 0.20 * lib_safe


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
    return ranks


def run_official_eval(
    *,
    rscript: Path,
    r_env: dict[str, str],
    expr_rds: Path,
    annotation_rds: Path,
    evaluation_mode: str,
    continuous_input: str,
    rank_tsv: Path,
    output_json: Path,
    nfeatures: int,
    method_list: str,
) -> dict[str, Any]:
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
        str(output_json),
        "--nfeatures",
        str(nfeatures),
        "--method-list-csv",
        method_list,
    ]
    if evaluation_mode == "continuous":
        eval_cmd.extend(["--continuous-input", continuous_input])
    eval_result = _run_command(eval_cmd, env=r_env, timeout=3_600_000)
    if not eval_result.ok or not output_json.exists():
        raise RuntimeError(f"Official eval failed: {eval_result.stderr}\n{eval_result.error}")
    return json.loads(output_json.read_text(encoding="utf-8"))


def render_summary(*, args: argparse.Namespace, run_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Constrained Anchor Repair Partial Summary",
        "",
        f"- dataset: `{args.dataset}`",
        f"- anchor: `{args.anchor_method}`",
        f"- candidate: `{args.candidate_method}`",
        f"- budgets: `{args.budgets}`",
    ]
    for row in run_rows:
        lines.append(f"- budget `{row['budget']}`: repair-anchor-overlap `{row['repair_vs_anchor_overlap']}` / `{args.nfeatures}`")
        lines.extend(row["summary"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
