from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

import numpy as np
from scipy import io as spio
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
SCRIPT_ROOT = ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_benchmarkhvg_adapter_stub_partial as adapter_partial
from hvg_research.methods import build_default_method_registry
from run_crossdisciplinary_anchor_nudges_smoke import score_anchor_energy_niche_nudge_hvg
from run_crossdisciplinary_taste_smoke import score_morphological_boundary_hvg


FORMAL_DATASETS = ("duo4_pbmc", "duo4un_pbmc", "duo8_pbmc", "pbmc3k_multi", "pbmc_cite")
FORMAL_METHODS = (
    "adaptive_core_consensus_hvg",
    "morphological_boundary_hvg",
    "anchor_energy_niche_nudge_hvg",
)


@dataclass
class CustomMethod:
    name: str
    fn: Callable[[np.ndarray, np.ndarray | None, int], np.ndarray]
    metadata_fn: Callable[[], dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal benchmarkHVG partial comparison for cross-disciplinary candidates.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / f"artifacts_crossdisciplinary_formal_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--r-env-root", type=Path, default=adapter_partial.DEFAULT_R_ENV_ROOT)
    parser.add_argument("--datasets", type=str, default=",".join(FORMAL_DATASETS))
    parser.add_argument("--methods", type=str, default=",".join(FORMAL_METHODS))
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


def _method_metadata_dict(method_fn: Any, method_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"method": method_name}
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


def _load_counts(counts_mtx: Path, gene_names_path: Path, cell_names_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
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
            f"Count matrix shape mismatch after transpose: got {counts.shape}, expected {(len(cell_names), len(gene_names))}"
        )
    return counts, gene_names, cell_names


def _build_custom_methods(registry: dict[str, Callable], args: argparse.Namespace) -> dict[str, CustomMethod]:
    anchor_method = registry["adaptive_hybrid_hvg"]

    def _anchor_energy(counts: np.ndarray, batches: np.ndarray | None, current_top_k: int) -> np.ndarray:
        return score_anchor_energy_niche_nudge_hvg(
            counts,
            batches,
            current_top_k,
            args.random_state,
            anchor_method,
        )

    return {
        "morphological_boundary_hvg": CustomMethod(
            name="morphological_boundary_hvg",
            fn=lambda counts, batches, current_top_k: score_morphological_boundary_hvg(
                counts,
                batches,
                current_top_k,
                args.random_state,
            ),
            metadata_fn=lambda: {
                "method_family": "crossdisciplinary_candidate",
                "candidate_theme": "mathematical_morphology_boundary",
                "implementation_source": "scripts/run_crossdisciplinary_taste_smoke.py",
                "selected_device_note": "cpu_path_with_torch_detection_only",
            },
        ),
        "anchor_energy_niche_nudge_hvg": CustomMethod(
            name="anchor_energy_niche_nudge_hvg",
            fn=_anchor_energy,
            metadata_fn=lambda: {
                "method_family": "crossdisciplinary_anchor_nudge",
                "candidate_theme": "energy_landscape_plus_ecological_niche",
                "anchor_reference": "adaptive_hybrid_hvg",
                "anchor_last_gate_metadata": getattr(anchor_method, "last_gate_metadata", {}),
                "anchor_last_gate_source": getattr(anchor_method, "last_gate_source", None),
                "implementation_source": "scripts/run_crossdisciplinary_anchor_nudges_smoke.py",
                "selected_device_note": "cpu_path_with_torch_detection_only",
            },
        ),
    }


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


def _compare_counts(eval_payload: dict[str, Any]) -> tuple[int, int, int, bool, bool]:
    evaluation_mode = str(eval_payload["evaluation_mode"])
    metric_names = adapter_partial._metric_columns_for_mode(evaluation_mode)
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


def main() -> None:
    args = parse_args()
    datasets = tuple(name.strip() for name in args.datasets.split(",") if name.strip())
    methods = tuple(name.strip() for name in args.methods.split(",") if name.strip())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    rscript, r_env = adapter_partial._r_runtime_env(args.r_env_root.resolve())
    if rscript is None or r_env is None:
        raise FileNotFoundError(f"Rscript not found under {args.r_env_root}")

    compute_context = adapter_partial._compute_context(
        args.r_env_root.resolve(),
        "cuda_visible_but_formal_crossdisciplinary_partial_uses_cpu_side_custom_scorers",
    )
    _write_json(output_dir / "compute_context.json", compute_context)

    registry = build_default_method_registry(
        top_k=args.nfeatures,
        refine_epochs=args.refine_epochs,
        random_state=args.random_state,
        gate_model_path=None,
    )
    custom_methods = _build_custom_methods(registry, args)

    summary_rows: list[dict[str, Any]] = []
    per_method_aggregate: list[dict[str, Any]] = []

    for dataset_name in datasets:
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
        counts, gene_names, _ = _load_counts(
            Path(export_payload["counts_mtx"]),
            Path(export_payload["gene_names_path"]),
            Path(export_payload["cell_names_path"]),
        )
        evaluation_mode = str(export_payload["evaluation_mode"])
        continuous_input = str(dataset_paths.get("continuous_input", "MultiomeATAC"))

        for method_name in methods:
            run_dir = output_dir / f"{method_name}__{dataset_name}"
            run_dir.mkdir(parents=True, exist_ok=False)

            if method_name in custom_methods:
                method = custom_methods[method_name]
                raw_scores = np.asarray(method.fn(counts, None, args.nfeatures), dtype=np.float64)
                method_metadata = method.metadata_fn()
            else:
                method_fn = registry[method_name]
                raw_scores = np.asarray(method_fn(counts, None, args.nfeatures), dtype=np.float64)
                method_metadata = _method_metadata_dict(method_fn, method_name)

            scores = _nonfinite_to_floor(raw_scores)
            ranks = _scores_to_rank(scores)
            rank_rows = [
                {"gene": gene_names[idx], "rank": int(ranks[idx]), "score": float(scores[idx])}
                for idx in np.argsort(ranks)
            ]
            rank_tsv = run_dir / f"{method_name}_full_rank.tsv"
            with rank_tsv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["gene", "rank", "score"], delimiter="\t")
                writer.writeheader()
                writer.writerows(rank_rows)

            method_metadata.update(
                {
                    "dataset": dataset_name,
                    "evaluation_mode": evaluation_mode,
                    "nfeatures": args.nfeatures,
                    "random_state": args.random_state,
                    "score_length": int(scores.shape[0]),
                    "score_min": float(np.min(scores)),
                    "score_max": float(np.max(scores)),
                    "score_mean": float(np.mean(scores)),
                    "score_std": float(np.std(scores)),
                }
            )
            _write_json(run_dir / f"{method_name}_method_metadata.json", method_metadata)

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

            metric_row: dict[str, Any] = {
                "method": method_name,
                "dataset": dataset_name,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "hvg_overlap_to_baseline": int(eval_payload["overlap"]["adapter_vs_baseline_hvg_overlap"]),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "clean_win": clean_win,
                "mixed_tradeoff": mixed_tradeoff,
            }
            for metric_name in adapter_partial._metric_columns_for_mode(evaluation_mode):
                metric_row[f"delta_{metric_name}"] = float(eval_payload["adapter"][metric_name] - eval_payload["baseline"][metric_name])
            summary_rows.append(metric_row)

            _write_text(
                run_dir / "comparison_summary.md",
                "\n".join(
                    [
                        f"# {method_name} on {dataset_name}",
                        "",
                        f"- evaluation_mode: `{evaluation_mode}`",
                        f"- wins/losses/ties: `{wins}` / `{losses}` / `{ties}`",
                        f"- clean_win: `{clean_win}`",
                        f"- mixed_tradeoff: `{mixed_tradeoff}`",
                        f"- hvg_overlap_to_baseline: `{metric_row['hvg_overlap_to_baseline']}` / `{args.nfeatures}`",
                    ]
                    + [
                        f"- delta_{metric_name}: `{metric_row[f'delta_{metric_name}']:.4f}`"
                        for metric_name in adapter_partial._metric_columns_for_mode(evaluation_mode)
                    ]
                    + [""]
                ),
            )

    summary_df = np.asarray(summary_rows, dtype=object)
    del summary_df
    import pandas as pd

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / "crossdisciplinary_formal_partial_summary.csv", index=False)

    for method_name, group in summary_frame.groupby("method"):
        per_method_aggregate.append(
            {
                "method": method_name,
                "dataset_count": int(len(group)),
                "clean_win_count": int(group["clean_win"].sum()),
                "mixed_tradeoff_count": int(group["mixed_tradeoff"].sum()),
                "no_positive_count": int((group["wins"] == 0).sum()),
                "total_wins": int(group["wins"].sum()),
                "total_losses": int(group["losses"].sum()),
                "mean_overlap_to_baseline": float(group["hvg_overlap_to_baseline"].mean()),
            }
        )
    method_frame = pd.DataFrame(per_method_aggregate).sort_values(
        ["clean_win_count", "total_wins", "total_losses", "mean_overlap_to_baseline"],
        ascending=[False, False, True, False],
    )
    method_frame.to_csv(output_dir / "crossdisciplinary_formal_partial_method_summary.csv", index=False)

    best_method = str(method_frame.iloc[0]["method"])
    decision_lines = [
        "# Cross-Disciplinary Formal Partial Decision",
        "",
        "## Scope",
        "",
        f"- Methods: {', '.join(f'`{name}`' for name in methods)}.",
        f"- Datasets: {', '.join(f'`{name}`' for name in datasets)}.",
        "- Benchmark layer: official `benchmarkHVG` partial adapter route through `extra.rank`.",
        "",
        "## Method Summary",
        "",
    ]
    for row in method_frame.itertuples(index=False):
        decision_lines.append(
            f"- `{row.method}`: clean_win={int(row.clean_win_count)}/{int(row.dataset_count)}, "
            f"mixed={int(row.mixed_tradeoff_count)}/{int(row.dataset_count)}, "
            f"wins/losses={int(row.total_wins)}/{int(row.total_losses)}, "
            f"mean_overlap={float(row.mean_overlap_to_baseline):.1f}/{args.nfeatures}."
        )
    decision_lines += [
        "",
        "## Decision",
        "",
        f"- Best formal partial readout in this comparison is `{best_method}`.",
        "- Promotion criterion remains strict: a candidate should show at least one clean pilot or a clearly dominant mixed-tradeoff pattern without broad payback.",
        "- If clean wins remain zero, the result is negative evidence, not a launch signal for a new mainline.",
        "",
    ]
    _write_text(output_dir / "crossdisciplinary_formal_partial_decision.md", "\n".join(decision_lines))


if __name__ == "__main__":
    main()
