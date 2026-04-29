from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
SCRIPT_ROOT = ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_benchmarkhvg_adapter_stub_partial as adapter_partial
from hvg_research.methods import build_default_method_registry
from run_crossdisciplinary_anchor_nudges_smoke import (
    _dataset_route_signals,
    _prepare_probe_ranks,
    _positive_gain,
)
from run_crossdisciplinary_taste_smoke import rank_percentile, zscore


TARGET_DATASETS = ("duo8_pbmc", "pbmc3k_multi")
CONTROL_DATASETS = ("duo4un_pbmc", "pbmc_cite")
ABLATION_DATASETS = TARGET_DATASETS + CONTROL_DATASETS
ABLATION_METHODS = (
    "anchor_energy_niche_nudge_hvg",
    "anchor_energy_niche_agreement_hvg",
    "anchor_energy_niche_intersection_hvg",
    "anchor_energy_niche_ecoheavy_hvg",
)


@dataclass
class VariantMethod:
    name: str
    fn: Callable[[np.ndarray, np.ndarray | None, int], np.ndarray]
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal partial ablation for anchor-energy-niche variants.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / f"artifacts_anchor_energy_niche_formal_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--r-env-root", type=Path, default=adapter_partial.DEFAULT_R_ENV_ROOT)
    parser.add_argument("--datasets", type=str, default=",".join(ABLATION_DATASETS))
    parser.add_argument("--nfeatures", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=31)
    parser.add_argument("--refine-epochs", type=int, default=0)
    parser.add_argument(
        "--method-list",
        type=str,
        default="mv_lognc,logmv_lognc,scran_pos,seuratv1,mean_max_nc",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _scores_to_rank(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return ranks


def _nonfinite_to_floor(scores: np.ndarray) -> np.ndarray:
    fixed = np.asarray(scores, dtype=np.float64).copy()
    finite = np.isfinite(fixed)
    if np.any(~finite):
        replacement = float(np.min(fixed[finite]) - 1.0) if np.any(finite) else 0.0
        fixed[~finite] = replacement
    return fixed


def _load_lines(path: Path) -> list[str]:
    return [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]


def _load_counts(counts_mtx: Path, gene_names_path: Path, cell_names_path: Path) -> tuple[np.ndarray, list[str]]:
    counts_raw = adapter_partial.spio.mmread(str(counts_mtx))
    if adapter_partial.sparse.issparse(counts_raw):
        counts_gene_by_cell = counts_raw.tocsr()
        counts = np.asarray(counts_gene_by_cell.transpose().toarray(), dtype=np.float32)
    else:
        counts = np.asarray(counts_raw.T, dtype=np.float32)
    gene_names = _load_lines(gene_names_path)
    cell_names = _load_lines(cell_names_path)
    if counts.shape != (len(cell_names), len(gene_names)):
        raise ValueError(
            f"Count matrix shape mismatch after transpose: got {counts.shape}, expected {(len(cell_names), len(gene_names))}"
        )
    return counts, gene_names


def _compare_counts(eval_payload: dict[str, Any]) -> tuple[int, int, int, bool, bool]:
    metric_names = adapter_partial._metric_columns_for_mode(str(eval_payload["evaluation_mode"]))
    wins = 0
    losses = 0
    ties = 0
    for metric_name in metric_names:
        delta = float(eval_payload["adapter"][metric_name] - eval_payload["baseline"][metric_name])
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    clean_win = wins > 0 and losses == 0
    mixed_tradeoff = wins > 0 and losses > 0
    return wins, losses, ties, clean_win, mixed_tradeoff


def _export_dataset(
    *,
    dataset_name: str,
    dataset_paths: dict[str, Path],
    output_dir: Path,
    rscript: Path,
    r_env: dict[str, str],
) -> dict[str, Any]:
    expr_rds = dataset_paths["expr_rds"]
    annotation_rds = dataset_paths["annotation_rds"]
    evaluation_mode = str(dataset_paths["evaluation_mode"])
    export_json = output_dir / f"{dataset_name}_export_metadata.json"
    counts_mtx = output_dir / f"{dataset_name}_counts.mtx"
    gene_names_path = output_dir / f"{dataset_name}_gene_names.tsv"
    cell_names_path = output_dir / f"{dataset_name}_cell_names.tsv"
    annotation_out = output_dir / f"{dataset_name}_annotation.tsv"
    export_cmd = [
        str(rscript),
        "--vanilla",
        str(adapter_partial.R_HELPER),
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
    export_result = adapter_partial._run_command(export_cmd, env=r_env, timeout=900_000)
    if not export_result.ok or not export_json.exists():
        raise RuntimeError(
            f"Failed to export {dataset_name}.\nstdout:\n{export_result.stdout}\n\nstderr:\n{export_result.stderr}\n\nerror:\n{export_result.error}"
        )
    payload = json.loads(export_json.read_text(encoding="utf-8"))
    payload.update(
        {
            "counts_mtx": str(counts_mtx),
            "gene_names_path": str(gene_names_path),
            "cell_names_path": str(cell_names_path),
            "annotation_out": str(annotation_out),
            "evaluation_mode": evaluation_mode,
        }
    )
    return payload


def _build_variants(
    *,
    anchor_scorer: Callable[[np.ndarray, np.ndarray | None, int], np.ndarray],
    seed: int,
) -> dict[str, VariantMethod]:
    def current_variant(counts: np.ndarray, batches: np.ndarray | None, top_k: int) -> np.ndarray:
        anchor = rank_percentile(anchor_scorer(counts, batches, top_k))
        probes = _prepare_probe_ranks(counts, batches, top_k, seed)
        route = _dataset_route_signals(counts, seed)
        gain = 0.70 * _positive_gain(probes["energy"], anchor) + 0.55 * _positive_gain(probes["eco"], anchor)
        weight = 0.10 + 0.12 * route["state_clarity"]
        score = anchor + weight * gain - 0.05 * probes["depth_penalty"] - 0.03 * probes["hub_penalty"]
        return zscore(score)

    def agreement_variant(counts: np.ndarray, batches: np.ndarray | None, top_k: int) -> np.ndarray:
        anchor = rank_percentile(anchor_scorer(counts, batches, top_k))
        probes = _prepare_probe_ranks(counts, batches, top_k, seed)
        route = _dataset_route_signals(counts, seed)
        agreement_rank = np.minimum(probes["energy"], probes["eco"])
        gain = _positive_gain(agreement_rank, anchor)
        weight = 0.08 + 0.10 * route["state_clarity"]
        score = anchor + weight * gain - 0.06 * probes["depth_penalty"] - 0.03 * probes["hub_penalty"]
        return zscore(score)

    def intersection_variant(counts: np.ndarray, batches: np.ndarray | None, top_k: int) -> np.ndarray:
        anchor = rank_percentile(anchor_scorer(counts, batches, top_k))
        probes = _prepare_probe_ranks(counts, batches, top_k, seed)
        route = _dataset_route_signals(counts, seed)
        joint = 0.5 * (probes["energy"] + probes["eco"])
        active = (probes["energy"] > anchor) & (probes["eco"] > anchor)
        gain = np.where(active, np.maximum(0.0, joint - anchor), 0.0)
        weight = 0.05 + 0.08 * route["state_clarity"]
        score = anchor + weight * gain - 0.06 * probes["depth_penalty"] - 0.03 * probes["hub_penalty"]
        return zscore(score)

    def ecoheavy_variant(counts: np.ndarray, batches: np.ndarray | None, top_k: int) -> np.ndarray:
        anchor = rank_percentile(anchor_scorer(counts, batches, top_k))
        probes = _prepare_probe_ranks(counts, batches, top_k, seed)
        route = _dataset_route_signals(counts, seed)
        gain = 0.40 * _positive_gain(probes["energy"], anchor) + 0.80 * _positive_gain(probes["eco"], anchor)
        weight = 0.08 + 0.10 * route["state_clarity"]
        score = anchor + weight * gain - 0.06 * probes["depth_penalty"] - 0.03 * probes["hub_penalty"]
        return zscore(score)

    return {
        "anchor_energy_niche_nudge_hvg": VariantMethod(
            name="anchor_energy_niche_nudge_hvg",
            fn=current_variant,
            metadata={
                "variant_family": "anchor_energy_niche_ablation",
                "variant_kind": "reference_current",
            },
        ),
        "anchor_energy_niche_agreement_hvg": VariantMethod(
            name="anchor_energy_niche_agreement_hvg",
            fn=agreement_variant,
            metadata={
                "variant_family": "anchor_energy_niche_ablation",
                "variant_kind": "agreement_min_gate",
            },
        ),
        "anchor_energy_niche_intersection_hvg": VariantMethod(
            name="anchor_energy_niche_intersection_hvg",
            fn=intersection_variant,
            metadata={
                "variant_family": "anchor_energy_niche_ablation",
                "variant_kind": "strict_intersection_gate",
            },
        ),
        "anchor_energy_niche_ecoheavy_hvg": VariantMethod(
            name="anchor_energy_niche_ecoheavy_hvg",
            fn=ecoheavy_variant,
            metadata={
                "variant_family": "anchor_energy_niche_ablation",
                "variant_kind": "eco_heavy_safer_penalty",
            },
        ),
    }


def main() -> None:
    args = parse_args()
    dataset_names = tuple(name.strip() for name in args.datasets.split(",") if name.strip())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    rscript, r_env = adapter_partial._r_runtime_env(args.r_env_root.resolve())
    if rscript is None or r_env is None:
        raise FileNotFoundError(f"Rscript not found under {args.r_env_root}")

    compute_context = adapter_partial._compute_context(
        args.r_env_root.resolve(),
        "anchor_energy_niche_formal_ablation_cpu_custom_methods",
    )
    _write_json(output_dir / "compute_context.json", compute_context)

    registry = build_default_method_registry(
        top_k=args.nfeatures,
        refine_epochs=args.refine_epochs,
        random_state=args.random_state,
        gate_model_path=None,
    )
    anchor_scorer = registry["adaptive_hybrid_hvg"]
    variants = _build_variants(anchor_scorer=anchor_scorer, seed=args.random_state)

    summary_rows: list[dict[str, Any]] = []
    dataset_export_rows: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        dataset_paths = adapter_partial.DATASET_REGISTRY[dataset_name]
        dataset_dir = output_dir / f"dataset_export__{dataset_name}"
        dataset_dir.mkdir(parents=True, exist_ok=False)
        export_payload = _export_dataset(
            dataset_name=dataset_name,
            dataset_paths=dataset_paths,
            output_dir=dataset_dir,
            rscript=rscript,
            r_env=r_env,
        )
        counts, gene_names = _load_counts(
            Path(export_payload["counts_mtx"]),
            Path(export_payload["gene_names_path"]),
            Path(export_payload["cell_names_path"]),
        )
        evaluation_mode = str(export_payload["evaluation_mode"])
        continuous_input = str(dataset_paths.get("continuous_input", "MultiomeATAC"))
        dataset_group = "target" if dataset_name in TARGET_DATASETS else "control"
        route_signals = _dataset_route_signals(counts, args.random_state)
        dataset_export_rows.append(
            {
                "dataset": dataset_name,
                "group_name": dataset_group,
                "evaluation_mode": evaluation_mode,
                "state_clarity": route_signals["state_clarity"],
                "boundary_fraction": route_signals["boundary_fraction"],
            }
        )

        for method_name, variant in variants.items():
            run_dir = output_dir / f"{method_name}__{dataset_name}"
            run_dir.mkdir(parents=True, exist_ok=False)
            raw_scores = np.asarray(variant.fn(counts, None, args.nfeatures), dtype=np.float64)
            scores = _nonfinite_to_floor(raw_scores)
            ranks = _scores_to_rank(scores)
            rank_tsv = run_dir / f"{method_name}_full_rank.tsv"
            with rank_tsv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["gene", "rank", "score"], delimiter="\t")
                writer.writeheader()
                writer.writerows(
                    {"gene": gene_names[idx], "rank": int(ranks[idx]), "score": float(scores[idx])}
                    for idx in np.argsort(ranks)
                )

            metadata = {
                **variant.metadata,
                "dataset": dataset_name,
                "group_name": dataset_group,
                "evaluation_mode": evaluation_mode,
                "state_clarity": route_signals["state_clarity"],
                "boundary_fraction": route_signals["boundary_fraction"],
                "hvg_score_min": float(np.min(scores)),
                "hvg_score_max": float(np.max(scores)),
                "hvg_score_std": float(np.std(scores)),
            }
            _write_json(run_dir / f"{method_name}_method_metadata.json", metadata)

            eval_json = run_dir / "official_extra_rank_eval.json"
            eval_cmd = [
                str(rscript),
                "--vanilla",
                str(adapter_partial.R_HELPER),
                "--mode",
                "run_adapter_eval",
                "--expr-rds",
                str(dataset_paths["expr_rds"]),
                "--annotation-rds",
                str(dataset_paths["annotation_rds"]),
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
            eval_result = adapter_partial._run_command(eval_cmd, env=r_env, timeout=3_600_000)
            if not eval_result.ok or not eval_json.exists():
                raise RuntimeError(
                    f"Failed to run adapter eval for {method_name} on {dataset_name}.\nstdout:\n{eval_result.stdout}\n\nstderr:\n{eval_result.stderr}\n\nerror:\n{eval_result.error}"
                )
            eval_payload = json.loads(eval_json.read_text(encoding="utf-8"))
            wins, losses, ties, clean_win, mixed_tradeoff = _compare_counts(eval_payload)

            row: dict[str, Any] = {
                "method": method_name,
                "dataset": dataset_name,
                "group_name": dataset_group,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "hvg_overlap_to_baseline": int(eval_payload["overlap"]["adapter_vs_baseline_hvg_overlap"]),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "clean_win": clean_win,
                "mixed_tradeoff": mixed_tradeoff,
            }
            for metric_name in adapter_partial._metric_columns_for_mode(evaluation_mode):
                row[f"delta_{metric_name}"] = float(eval_payload["adapter"][metric_name] - eval_payload["baseline"][metric_name])
            summary_rows.append(row)

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / "anchor_energy_niche_ablation_summary.csv", index=False)
    pd.DataFrame(dataset_export_rows).to_csv(output_dir / "anchor_energy_niche_dataset_route_signals.csv", index=False)

    method_rows: list[dict[str, Any]] = []
    for method_name, group in summary_frame.groupby("method"):
        target_group = group[group["group_name"] == "target"]
        control_group = group[group["group_name"] == "control"]
        method_rows.append(
            {
                "method": method_name,
                "target_dataset_count": int(len(target_group)),
                "control_dataset_count": int(len(control_group)),
                "clean_win_count": int(group["clean_win"].sum()),
                "mixed_tradeoff_count": int(group["mixed_tradeoff"].sum()),
                "total_wins": int(group["wins"].sum()),
                "total_losses": int(group["losses"].sum()),
                "target_wins": int(target_group["wins"].sum()),
                "target_losses": int(target_group["losses"].sum()),
                "control_wins": int(control_group["wins"].sum()),
                "control_losses": int(control_group["losses"].sum()),
                "mean_overlap_to_baseline": float(group["hvg_overlap_to_baseline"].mean()),
            }
        )
    method_frame = pd.DataFrame(method_rows).sort_values(
        ["target_wins", "control_losses", "clean_win_count", "mean_overlap_to_baseline"],
        ascending=[False, True, False, False],
    )
    method_frame.to_csv(output_dir / "anchor_energy_niche_ablation_method_summary.csv", index=False)

    best_method = str(method_frame.iloc[0]["method"])
    lines = [
        "# Anchor Energy-Niche Formal Ablation Decision",
        "",
        "## Scope",
        "",
        f"- Variants: {', '.join(f'`{name}`' for name in variants)}.",
        f"- Target datasets: {', '.join(f'`{name}`' for name in TARGET_DATASETS)}.",
        f"- Control datasets: {', '.join(f'`{name}`' for name in CONTROL_DATASETS)}.",
        "- Benchmark layer: official `benchmarkHVG` partial adapter route through `extra.rank`.",
        "",
        "## Method Summary",
        "",
    ]
    for row in method_frame.itertuples(index=False):
        lines.append(
            f"- `{row.method}`: clean_win={int(row.clean_win_count)}/4, mixed={int(row.mixed_tradeoff_count)}/4, "
            f"target wins/losses={int(row.target_wins)}/{int(row.target_losses)}, "
            f"control wins/losses={int(row.control_wins)}/{int(row.control_losses)}, "
            f"mean_overlap={float(row.mean_overlap_to_baseline):.1f}/{args.nfeatures}."
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- Best target-vs-control compromise in this ablation is `{best_method}`.",
        "- Continue only if the best variant improves target wins without increasing control losses in a material way.",
        "- If all variants remain mixed with symmetric payback, stop tuning and treat the line as benchmark-negative.",
        "",
    ]
    _write_text(output_dir / "anchor_energy_niche_ablation_decision.md", "\n".join(lines))


if __name__ == "__main__":
    main()
