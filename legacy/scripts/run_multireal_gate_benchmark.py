from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from hvg_research import (
    GateLearningConfig,
    build_default_method_registry,
    discover_scrna_input_specs,
    evaluate_real_selection,
    load_scrna_dataset,
    subsample_dataset,
    train_gate_model,
)
from hvg_research.eval import timed_call
from hvg_research.methods import (
    FRONTIER_ANCHORLESS_METHOD,
    FRONTIER_LITE_METHOD,
    canonicalize_method_names,
)


FRONTIER_FIXED_METHOD = "ablation_frontier_fixed_frontier"
FRONTIER_FULL_METHOD = "learnable_gate_bank_pairregret_permissioned_escapecert_frontier"
BANK_METHOD = "learnable_gate_bank"
MV_RESIDUAL_METHOD = "mv_residual"
DEFAULT_METHODS = (
    FRONTIER_ANCHORLESS_METHOD,
    FRONTIER_LITE_METHOD,
    FRONTIER_FIXED_METHOD,
    BANK_METHOD,
    FRONTIER_FULL_METHOD,
    MV_RESIDUAL_METHOD,
)
CANDIDATE_METHODS = (
    FRONTIER_ANCHORLESS_METHOD,
    FRONTIER_LITE_METHOD,
    FRONTIER_FIXED_METHOD,
)
POSITIVE_METRICS = (
    "ari",
    "nmi",
    "label_silhouette",
    "batch_mixing",
    "neighbor_preservation",
    "cluster_silhouette",
    "stability",
)
NEGATIVE_METRICS = ("runtime_sec",)
TIE_EPS = 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-real robust frontier benchmark.")
    parser.add_argument("--output-dir", type=str, default="artifacts_gate_learning_v_next9_multireal")
    parser.add_argument("--gate-model-path", type=str, default=None)
    parser.add_argument("--force-retrain-gate", action="store_true")
    parser.add_argument("--top-k-values", type=str, default="40,80")
    parser.add_argument("--seeds", type=str, default="7,17,27")
    parser.add_argument("--bootstrap-samples", type=int, default=5)
    parser.add_argument("--refine-epochs", type=int, default=10)
    parser.add_argument("--train-seeds", type=int, default=6)
    parser.add_argument("--candidate-random-gates", type=int, default=6)
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-genes", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    top_k_values = tuple(sorted({int(part.strip()) for part in args.top_k_values.split(",") if part.strip()}))
    seeds = tuple(sorted({int(part.strip()) for part in args.seeds.split(",") if part.strip()}))
    method_names = canonicalize_method_names([part.strip() for part in args.methods.split(",") if part.strip()])
    if not top_k_values:
        raise ValueError("Expected at least one top_k value.")
    if not seeds:
        raise ValueError("Expected at least one seed.")

    device_info = resolve_device_info()
    print(json.dumps(device_info, indent=2))
    save_json(output_dir / "device_info.json", device_info)

    input_specs = discover_scrna_input_specs(ROOT)
    if not input_specs:
        raise FileNotFoundError("No local real scRNA-seq inputs were discovered in the repository.")

    inventory_df = pd.DataFrame([spec_to_row(spec) for spec in input_specs]).sort_values(
        ["selected_for_benchmark", "dataset_name", "file_format", "dataset_id"],
        ascending=[False, True, True, True],
    )
    inventory_df.to_csv(output_dir / "dataset_inventory_all.csv", index=False)
    benchmark_specs = [spec for spec in input_specs if spec.selected_for_benchmark]
    pd.DataFrame([spec_to_row(spec) for spec in benchmark_specs]).to_csv(output_dir / "dataset_inventory_unique.csv", index=False)

    gate_summary = ensure_gate_checkpoint(
        output_dir=output_dir,
        gate_model_path=args.gate_model_path,
        force_retrain=args.force_retrain_gate,
        top_k=max(top_k_values),
        train_seeds=args.train_seeds,
        candidate_random_gates=args.candidate_random_gates,
    )
    gate_model_path = str(gate_summary["checkpoint_path"])
    save_json(output_dir / "gate_training_summary.json", gate_summary)

    all_rows: list[dict[str, float | int | str | bool]] = []
    dataset_rows: dict[str, pd.DataFrame] = {}
    for spec in benchmark_specs:
        dataset_dir = datasets_dir / spec.dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
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
            max_cells=args.max_cells,
            max_genes=args.max_genes,
            random_state=seeds[0],
        )
        dataset = subsample_dataset(
            dataset,
            max_cells=args.max_cells,
            max_genes=args.max_genes,
            random_state=seeds[0],
        )
        save_json(
            dataset_dir / "dataset_info.json",
            {
                "dataset_id": spec.dataset_id,
                "dataset_name": spec.dataset_name,
                "input_path": spec.input_path,
                "file_format": spec.file_format,
                "transpose": spec.transpose,
                "obs_path": spec.obs_path,
                "genes_path": spec.genes_path,
                "cells_path": spec.cells_path,
                "labels_col": spec.labels_col,
                "batches_col": spec.batches_col,
                "cells": int(dataset.counts.shape[0]),
                "genes": int(dataset.counts.shape[1]),
                "label_classes": None if dataset.labels is None else int(len(np.unique(dataset.labels))),
                "batch_classes": None if dataset.batches is None else int(len(np.unique(dataset.batches))),
            },
        )
        print(
            f"Dataset={spec.dataset_name} format={spec.file_format} cells={dataset.counts.shape[0]} "
            f"genes={dataset.counts.shape[1]} labels_col={spec.labels_col} batches_col={spec.batches_col}"
        )

        rows = run_dataset_benchmark(
            dataset=dataset,
            dataset_id=spec.dataset_id,
            input_spec=spec,
            method_names=method_names,
            gate_model_path=gate_model_path,
            refine_epochs=args.refine_epochs,
            top_k_values=top_k_values,
            seeds=seeds,
            bootstrap_samples=args.bootstrap_samples,
        )
        dataset_df = pd.DataFrame(rows)
        dataset_df = add_run_level_scores(dataset_df)
        dataset_rows[spec.dataset_id] = dataset_df
        all_rows.extend(dataset_df.to_dict(orient="records"))

        dataset_df.to_csv(dataset_dir / "raw_results.csv", index=False)
        dataset_summary = summarize_by_keys(dataset_df, keys=["dataset", "dataset_id", "method", "top_k"])
        dataset_summary.to_csv(dataset_dir / "method_summary.csv", index=False)
        dataset_ranking = rank_within_group(
            summarize_by_keys(dataset_df, keys=["dataset", "dataset_id", "method"]),
            group_cols=["dataset", "dataset_id"],
            rank_col="dataset_rank",
        )
        dataset_ranking.to_csv(dataset_dir / "method_ranking.csv", index=False)

    all_results = pd.DataFrame(all_rows)
    all_results.to_csv(output_dir / "multireal_raw_results.csv", index=False)

    run_rankings = all_results[
        [
            "dataset",
            "dataset_id",
            "method",
            "seed",
            "top_k",
            "overall_score",
            "overall_rank",
            "is_winner",
            *[metric for metric in POSITIVE_METRICS if metric in all_results.columns],
            "runtime_sec",
        ]
    ].copy()
    run_rankings.to_csv(output_dir / "multireal_run_rankings.csv", index=False)

    summary_by_dataset_topk = summarize_by_keys(all_results, keys=["dataset", "dataset_id", "method", "top_k"])
    summary_by_dataset_topk = rank_within_group(
        summary_by_dataset_topk,
        group_cols=["dataset", "dataset_id", "top_k"],
        rank_col="summary_rank",
    )
    summary_by_dataset_topk.to_csv(output_dir / "multireal_dataset_topk_summary.csv", index=False)

    dataset_rankings = rank_within_group(
        summarize_by_keys(all_results, keys=["dataset", "dataset_id", "method"]),
        group_cols=["dataset", "dataset_id"],
        rank_col="overall_rank",
    )
    dataset_rankings.to_csv(output_dir / "multireal_dataset_rankings.csv", index=False)

    global_summary = rank_within_group(
        summarize_by_keys(all_results, keys=["method"]),
        group_cols=[],
        rank_col="global_rank",
    )
    global_summary.to_csv(output_dir / "multireal_global_summary.csv", index=False)

    pairwise = compute_pairwise_wtl(all_results, methods=method_names)
    pairwise.to_csv(output_dir / "multireal_pairwise_win_tie_loss.csv", index=False)

    decision = build_decision_payload(
        benchmark_specs=benchmark_specs,
        global_summary=global_summary,
        pairwise=pairwise,
    )
    save_json(output_dir / "decision_summary.json", decision)
    (output_dir / "decision_report.md").write_text(
        build_decision_report(
            decision=decision,
            inventory_df=inventory_df,
            dataset_rankings=dataset_rankings,
            global_summary=global_summary,
            pairwise=pairwise,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


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


def ensure_gate_checkpoint(
    *,
    output_dir: Path,
    gate_model_path: str | None,
    force_retrain: bool,
    top_k: int,
    train_seeds: int,
    candidate_random_gates: int,
) -> dict[str, object]:
    if gate_model_path is not None:
        return {
            "checkpoint_path": gate_model_path,
            "reused_external_checkpoint": True,
        }

    checkpoint_path = output_dir / "learnable_gate.pt"
    summary_path = output_dir / "gate_training_summary.json"
    if checkpoint_path.exists() and summary_path.exists() and not force_retrain:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    config = GateLearningConfig(
        top_k=top_k,
        random_state=17,
        train_seeds=tuple(range(train_seeds)),
        candidate_random_gates=candidate_random_gates,
    )
    summary = train_gate_model(output_dir=str(output_dir), config=config)
    summary["reused_external_checkpoint"] = False
    return summary


def run_dataset_benchmark(
    *,
    dataset,
    dataset_id: str,
    input_spec,
    method_names: tuple[str, ...],
    gate_model_path: str,
    refine_epochs: int,
    top_k_values: tuple[int, ...],
    seeds: tuple[int, ...],
    bootstrap_samples: int,
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    for seed in seeds:
        for requested_top_k in top_k_values:
            current_top_k = min(requested_top_k, dataset.counts.shape[1])
            methods = build_default_method_registry(
                top_k=current_top_k,
                refine_epochs=refine_epochs,
                random_state=seed,
                gate_model_path=gate_model_path,
            )
            selected_methods = {name: methods[name] for name in method_names}
            for method_name, method_fn in selected_methods.items():
                scores, elapsed = timed_call(method_fn, dataset.counts, dataset.batches, current_top_k)
                selected = np.argsort(scores)[-current_top_k:]
                metrics = evaluate_real_selection(
                    counts=dataset.counts,
                    selected_genes=selected,
                    scorer_fn=lambda subset_counts, subset_batches, fn=method_fn, top_k=current_top_k: fn(
                        subset_counts,
                        subset_batches,
                        top_k,
                    ),
                    labels=dataset.labels,
                    batches=dataset.batches,
                    top_k=current_top_k,
                    random_state=seed,
                    n_bootstrap=bootstrap_samples,
                )
                row: dict[str, float | int | str | bool] = {
                    "dataset": dataset.name,
                    "dataset_id": dataset_id,
                    "dataset_name": input_spec.dataset_name,
                    "input_path": input_spec.input_path,
                    "file_format": input_spec.file_format,
                    "labels_col": input_spec.labels_col or "",
                    "batches_col": input_spec.batches_col or "",
                    "cells": int(dataset.counts.shape[0]),
                    "genes": int(dataset.counts.shape[1]),
                    "method": method_name,
                    "seed": int(seed),
                    "top_k_requested": int(requested_top_k),
                    "top_k": int(current_top_k),
                    "runtime_sec": float(elapsed),
                    "gate_source": str(getattr(method_fn, "last_gate_source", "") or ""),
                }
                row.update({key: float(value) for key, value in metrics.items()})
                rows.append(row)
    return rows


def add_run_level_scores(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    group_cols = ["dataset", "dataset_id", "seed", "top_k"]
    for _, group in df.groupby(group_cols, sort=False):
        scored = group.copy()
        score = np.zeros(len(scored), dtype=np.float64)
        for metric in POSITIVE_METRICS:
            if metric not in scored.columns:
                continue
            score += min_max_scale(scored[metric].to_numpy(dtype=np.float64))
        for metric in NEGATIVE_METRICS:
            if metric not in scored.columns:
                continue
            score -= 0.15 * min_max_scale(scored[metric].to_numpy(dtype=np.float64))
        scored["overall_score"] = score
        scored = scored.sort_values(["overall_score", "runtime_sec", "method"], ascending=[False, True, True]).reset_index(drop=True)
        scored["overall_rank"] = np.arange(1, len(scored) + 1)
        best_score = float(scored["overall_score"].max()) if len(scored) else float("nan")
        scored["is_winner"] = np.abs(scored["overall_score"] - best_score) <= TIE_EPS
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


def summarize_by_keys(df: pd.DataFrame, *, keys: list[str]) -> pd.DataFrame:
    tracked_metrics = [
        "ari",
        "nmi",
        "label_silhouette",
        "batch_mixing",
        "neighbor_preservation",
        "cluster_silhouette",
        "stability",
        "runtime_sec",
        "overall_score",
        "overall_rank",
    ]
    available_metrics = [metric for metric in tracked_metrics if metric in df.columns]
    rows: list[dict[str, object]] = []
    for group_key, group in df.groupby(keys, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {key: value for key, value in zip(keys, group_key, strict=False)}
        for metric in available_metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            row[metric] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=0))
        row["run_count"] = int(len(group))
        row["win_count"] = int(group["is_winner"].sum()) if "is_winner" in group.columns else 0
        row["win_rate"] = float(row["win_count"] / max(row["run_count"], 1))
        row["seed_count"] = int(group["seed"].nunique()) if "seed" in group.columns else 0
        row["top_k_count"] = int(group["top_k"].nunique()) if "top_k" in group.columns else 0
        rows.append(row)
    return pd.DataFrame(rows)


def rank_within_group(df: pd.DataFrame, *, group_cols: list[str], rank_col: str) -> pd.DataFrame:
    ranking_df = df.copy()
    if not len(ranking_df):
        return ranking_df

    frames = []
    if group_cols:
        iterator = ranking_df.groupby(group_cols, dropna=False, sort=False)
    else:
        iterator = [((), ranking_df)]

    for _, group in iterator:
        ranked = group.sort_values(
            ["overall_rank", "win_rate", "overall_score", "runtime_sec", "method"],
            ascending=[True, False, False, True, True],
        ).reset_index(drop=True)
        ranked[rank_col] = np.arange(1, len(ranked) + 1)
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True)


def compute_pairwise_wtl(df: pd.DataFrame, *, methods: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    run_keys = ["dataset", "dataset_id", "seed", "top_k"]
    for method_a, method_b in combinations(methods, 2):
        diffs: list[float] = []
        wins = 0
        ties = 0
        losses = 0
        for _, group in df.groupby(run_keys, sort=False):
            row_a = group[group["method"] == method_a]
            row_b = group[group["method"] == method_b]
            if row_a.empty or row_b.empty:
                continue
            diff = float(row_a.iloc[0]["overall_score"] - row_b.iloc[0]["overall_score"])
            diffs.append(diff)
            if abs(diff) <= TIE_EPS:
                ties += 1
            elif diff > 0:
                wins += 1
            else:
                losses += 1

        p_value = float("nan")
        non_zero = np.asarray([diff for diff in diffs if abs(diff) > TIE_EPS], dtype=np.float64)
        if len(non_zero) >= 1:
            try:
                p_value = float(wilcoxon(non_zero, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                p_value = float("nan")

        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "method_a_wins": wins,
                "ties": ties,
                "method_a_losses": losses,
                "comparisons": int(len(diffs)),
                "mean_score_diff": float(np.mean(diffs)) if diffs else float("nan"),
                "wilcoxon_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def build_decision_payload(
    *,
    benchmark_specs: list,
    global_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> dict[str, object]:
    candidate_summary = global_summary[global_summary["method"].isin(CANDIDATE_METHODS)].copy()
    candidate_summary = candidate_summary.sort_values(
        ["global_rank", "overall_rank", "win_rate", "overall_score", "runtime_sec"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)
    best_candidate = str(candidate_summary.iloc[0]["method"]) if not candidate_summary.empty else None
    best_overall = str(global_summary.sort_values("global_rank").iloc[0]["method"]) if not global_summary.empty else None

    pairwise_rows = []
    if best_candidate is not None:
        for challenger in CANDIDATE_METHODS:
            if challenger == best_candidate:
                continue
            row = select_pairwise_row(pairwise, method_a=best_candidate, method_b=challenger)
            if row is not None:
                pairwise_rows.append(row)

    unique_real_dataset_count = len(benchmark_specs)
    robust_lead = bool(
        best_candidate is not None
        and unique_real_dataset_count >= 3
        and all(int(row["wins"]) > int(row["losses"]) for row in pairwise_rows)
    )
    return {
        "unique_real_dataset_count": unique_real_dataset_count,
        "best_overall_method": best_overall,
        "best_candidate_method": best_candidate,
        "unique_mainline_supported": robust_lead,
        "candidate_pairwise_vs_alternatives": pairwise_rows,
        "data_gap": unique_real_dataset_count < 3,
    }


def build_decision_report(
    *,
    decision: dict[str, object],
    inventory_df: pd.DataFrame,
    dataset_rankings: pd.DataFrame,
    global_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> str:
    sections = [
        "# Multi-Real Frontier Decision",
        "",
        f"- Unique local real datasets available: {decision['unique_real_dataset_count']}",
        f"- Best overall method across available runs: {decision['best_overall_method']}",
        f"- Best frontier candidate: {decision['best_candidate_method']}",
        f"- Enough evidence to freeze a unique mainline: {decision['unique_mainline_supported']}",
        f"- Local real-data gap present (<3 unique datasets): {decision['data_gap']}",
        "",
        "## Dataset Inventory",
        inventory_df.round(6).to_string(index=False),
        "",
        "## Per-Dataset Rankings",
        dataset_rankings.round(6).to_string(index=False),
        "",
        "## Global Summary",
        global_summary.round(6).to_string(index=False),
        "",
        "## Pairwise Win/Tie/Loss",
        pairwise.round(6).to_string(index=False),
    ]
    return "\n".join(sections)


def select_pairwise_row(pairwise: pd.DataFrame, *, method_a: str, method_b: str) -> dict[str, object] | None:
    direct = pairwise[(pairwise["method_a"] == method_a) & (pairwise["method_b"] == method_b)]
    if not direct.empty:
        row = direct.iloc[0]
        return {
            "challenger": method_b,
            "wins": int(row["method_a_wins"]),
            "ties": int(row["ties"]),
            "losses": int(row["method_a_losses"]),
            "mean_score_diff": float(row["mean_score_diff"]),
            "wilcoxon_p": float(row["wilcoxon_p"]) if pd.notna(row["wilcoxon_p"]) else None,
        }

    reverse = pairwise[(pairwise["method_a"] == method_b) & (pairwise["method_b"] == method_a)]
    if reverse.empty:
        return None
    row = reverse.iloc[0]
    return {
        "challenger": method_b,
        "wins": int(row["method_a_losses"]),
        "ties": int(row["ties"]),
        "losses": int(row["method_a_wins"]),
        "mean_score_diff": float(-row["mean_score_diff"]),
        "wilcoxon_p": float(row["wilcoxon_p"]) if pd.notna(row["wilcoxon_p"]) else None,
    }


def min_max_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or np.allclose(values.max(), values.min()):
        return np.zeros_like(values, dtype=np.float64)
    return (values - values.min()) / (values.max() - values.min())


def spec_to_row(spec) -> dict[str, object]:
    return {
        "dataset_id": spec.dataset_id,
        "dataset_name": spec.dataset_name,
        "input_path": spec.input_path,
        "file_format": spec.file_format,
        "transpose": spec.transpose,
        "obs_path": spec.obs_path,
        "genes_path": spec.genes_path,
        "cells_path": spec.cells_path,
        "labels_col": spec.labels_col,
        "batches_col": spec.batches_col,
        "cells": spec.cell_count,
        "genes": spec.gene_count,
        "label_classes": spec.label_classes,
        "batch_classes": spec.batch_classes,
        "fingerprint": spec.fingerprint,
        "duplicate_of": spec.duplicate_of,
        "selected_for_benchmark": spec.selected_for_benchmark,
    }


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
