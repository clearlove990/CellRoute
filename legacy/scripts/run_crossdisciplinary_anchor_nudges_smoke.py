from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hvg_research import build_default_method_registry
from hvg_research.eval import evaluate_real_selection
from run_crossdisciplinary_taste_smoke import (
    DATASETS,
    BASELINE_METHODS,
    IDEA_CARDS,
    choose_device,
    depth_penalty,
    load_dataset,
    norm_counts,
    pseudo_states,
    rank_percentile,
    score_ecological_niche_partition_hvg,
    score_energy_well_curvature_hvg,
    score_mdl_state_code_hvg,
    score_morphological_boundary_hvg,
    score_ot_barycentric_displacement_hvg,
    state_embedding,
    zscore,
)


ANCHOR_IDEA_CARDS = {
    "anchor_energy_niche_nudge_hvg": {
        "taste": "Use energy-landscape curvature as a benchmark-safe correction term, but only where ecology-style niche partitioning agrees.",
        "why": "Pure creative scorers were too unstable. This line keeps the published-strength anchor and injects a mechanistic correction only on genes jointly favored by basin curvature and niche specificity.",
        "mechanism": "Anchor rank plus a gated positive nudge from energy-well and ecological scores, with depth and hub penalties.",
    },
    "anchor_mdl_energy_code_hvg": {
        "taste": "Treat anchor genes as a prior and let MDL/compression decide which disagreements are worth paying for.",
        "why": "This reframes novelty as efficient state coding instead of raw dispersion, but preserves benchmark safety by only nudging upward against the anchor.",
        "mechanism": "Anchor rank plus positive MDL and energy gains, with disagreement damped when the dataset lacks crisp unsupervised state structure.",
    },
    "anchor_boundary_transport_nudge_hvg": {
        "taste": "Keep the anchor for state interiors and reserve creativity for transition fronts and transport geometry.",
        "why": "Boundary and OT scorers were too aggressive alone, but they may still rescue genes the anchor misses in region-like or transition-heavy regimes.",
        "mechanism": "Anchor rank plus a trajectory-gated boundary/transport bonus, penalized by depth and generic prevalence.",
    },
}


def _safe_silhouette(embedding: np.ndarray, labels: np.ndarray) -> float:
    if embedding.shape[0] <= len(np.unique(labels)) or len(np.unique(labels)) < 2:
        return 0.0
    try:
        return float(silhouette_score(embedding, labels))
    except ValueError:
        return 0.0


def _dataset_route_signals(counts: np.ndarray, seed: int) -> dict[str, float]:
    x = norm_counts(counts)
    emb = state_embedding(x, seed)
    states = pseudo_states(emb, seed)
    n_neighbors = min(16, max(6, x.shape[0] // 45))
    idx = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(emb).kneighbors(return_distance=False)[:, 1:]
    disagreement = np.mean(states[idx] != states[:, None], axis=1)
    boundary_fraction = float(np.mean(disagreement > 0.25))
    state_clarity = max(_safe_silhouette(emb, states), 0.0)
    return {
        "state_clarity": float(np.clip(state_clarity, 0.0, 1.0)),
        "boundary_fraction": float(np.clip(boundary_fraction, 0.0, 1.0)),
    }


def _prepare_probe_ranks(counts: np.ndarray, batches: np.ndarray | None, top_k: int, seed: int) -> dict[str, np.ndarray]:
    x = norm_counts(counts)
    return {
        "energy": rank_percentile(score_energy_well_curvature_hvg(counts, batches, top_k, seed)),
        "eco": rank_percentile(score_ecological_niche_partition_hvg(counts, batches, top_k, seed)),
        "mdl": rank_percentile(score_mdl_state_code_hvg(counts, batches, top_k, seed)),
        "boundary": rank_percentile(score_morphological_boundary_hvg(counts, batches, top_k, seed)),
        "ot": rank_percentile(score_ot_barycentric_displacement_hvg(counts, batches, top_k, seed)),
        "depth_penalty": rank_percentile(depth_penalty(counts, x)),
        "hub_penalty": rank_percentile((x > 0).mean(axis=0) * x.mean(axis=0)),
    }


def _positive_gain(source: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, source - anchor)


def score_anchor_energy_niche_nudge_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None,
    top_k: int,
    seed: int,
    anchor_scorer: Callable[[np.ndarray, np.ndarray | None, int], np.ndarray],
) -> np.ndarray:
    anchor = rank_percentile(anchor_scorer(counts, batches, top_k))
    probes = _prepare_probe_ranks(counts, batches, top_k, seed)
    route = _dataset_route_signals(counts, seed)
    gain = 0.70 * _positive_gain(probes["energy"], anchor) + 0.55 * _positive_gain(probes["eco"], anchor)
    weight = 0.10 + 0.12 * route["state_clarity"]
    score = anchor + weight * gain - 0.05 * probes["depth_penalty"] - 0.03 * probes["hub_penalty"]
    return zscore(score)


def score_anchor_mdl_energy_code_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None,
    top_k: int,
    seed: int,
    anchor_scorer: Callable[[np.ndarray, np.ndarray | None, int], np.ndarray],
) -> np.ndarray:
    anchor = rank_percentile(anchor_scorer(counts, batches, top_k))
    probes = _prepare_probe_ranks(counts, batches, top_k, seed)
    route = _dataset_route_signals(counts, seed)
    gain = 0.75 * _positive_gain(probes["mdl"], anchor) + 0.45 * _positive_gain(probes["energy"], anchor)
    weight = 0.08 + 0.10 * route["state_clarity"]
    score = anchor + weight * gain - 0.05 * probes["depth_penalty"] - 0.02 * probes["hub_penalty"]
    return zscore(score)


def score_anchor_boundary_transport_nudge_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None,
    top_k: int,
    seed: int,
    anchor_scorer: Callable[[np.ndarray, np.ndarray | None, int], np.ndarray],
) -> np.ndarray:
    anchor = rank_percentile(anchor_scorer(counts, batches, top_k))
    probes = _prepare_probe_ranks(counts, batches, top_k, seed)
    route = _dataset_route_signals(counts, seed)
    gain = 0.65 * _positive_gain(probes["boundary"], anchor) + 0.55 * _positive_gain(probes["ot"], anchor)
    weight = 0.05 + 0.15 * route["boundary_fraction"]
    score = anchor + weight * gain - 0.05 * probes["depth_penalty"] - 0.03 * probes["hub_penalty"]
    return zscore(score)


def add_overall_score(df: pd.DataFrame) -> pd.DataFrame:
    positives = [
        col
        for col in [
            "ari",
            "nmi",
            "label_silhouette",
            "cluster_silhouette",
            "neighbor_preservation",
            "stability",
            "batch_mixing",
        ]
        if col in df.columns
    ]
    score = np.zeros(len(df), dtype=np.float64)
    for col in positives:
        values = df[col].to_numpy(dtype=np.float64)
        if np.all(np.isnan(values)):
            continue
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            score += np.nan_to_num((values - vmin) / (vmax - vmin), nan=0.0)
    if "runtime_sec" in df.columns:
        values = df["runtime_sec"].to_numpy(dtype=np.float64)
        if not np.all(np.isnan(values)):
            vmin = np.nanmin(values)
            vmax = np.nanmax(values)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                score -= 0.12 * np.nan_to_num((values - vmin) / (vmax - vmin), nan=0.0)
    df["overall_score"] = score
    return df


def evaluate_method(
    dataset,
    method_name: str,
    scorer: Callable[[np.ndarray, np.ndarray | None, int], np.ndarray],
    top_k: int,
    seed: int,
    n_bootstrap: int,
) -> tuple[dict[str, object], np.ndarray]:
    start = time.perf_counter()
    scores = scorer(dataset.counts, dataset.batches, top_k)
    elapsed = time.perf_counter() - start
    selected = np.argsort(scores)[-top_k:]
    metrics = evaluate_real_selection(
        counts=dataset.counts,
        selected_genes=selected,
        scorer_fn=lambda subset_counts, subset_batches, fn=scorer: fn(
            subset_counts,
            subset_batches,
            min(top_k, subset_counts.shape[1]),
        ),
        labels=dataset.labels,
        batches=dataset.batches,
        top_k=top_k,
        random_state=seed,
        n_bootstrap=n_bootstrap,
    )
    metrics.update({"dataset": dataset.name, "method": method_name, "runtime_sec": elapsed, "top_k": top_k})
    return metrics, scores


def top_overlap(a: np.ndarray, b: np.ndarray, top_k: int) -> tuple[float, float]:
    aa = set(np.argsort(a)[-top_k:].tolist())
    bb = set(np.argsort(b)[-top_k:].tolist())
    inter = len(aa & bb)
    union = len(aa | bb)
    return inter / max(top_k, 1), inter / max(union, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-disciplinary anchor-nudge HVG smoke.")
    parser.add_argument("--output-dir", default="artifacts_crossdisciplinary_anchor_nudges_20260425_v1")
    parser.add_argument("--datasets", nargs="+", default=["paul15", "GBM_sd"])
    parser.add_argument("--extended-datasets", nargs="+", default=["FBM_cite", "cellxgene_human_kidney_nonpt"])
    parser.add_argument("--max-cells", type=int, default=900)
    parser.add_argument("--max-genes", type=int, default=2500)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--n-bootstrap", type=int, default=2)
    parser.add_argument("--extended-top-n", type=int, default=2)
    args = parser.parse_args()

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "compute_context.json").write_text(json.dumps(choose_device(), indent=2), encoding="utf-8")
    (out / "idea_cards.json").write_text(json.dumps(ANCHOR_IDEA_CARDS, indent=2), encoding="utf-8")

    registry = build_default_method_registry(top_k=args.top_k, random_state=args.seed, refine_epochs=0)
    anchor_scorer = registry["adaptive_hybrid_hvg"]
    baseline_methods: dict[str, Callable[[np.ndarray, np.ndarray | None, int], np.ndarray]] = {
        name: registry[name] for name in BASELINE_METHODS if name in registry
    }
    candidate_methods = {
        "anchor_energy_niche_nudge_hvg": lambda counts, batches, top_k: score_anchor_energy_niche_nudge_hvg(
            counts, batches, top_k, args.seed, anchor_scorer
        ),
        "anchor_mdl_energy_code_hvg": lambda counts, batches, top_k: score_anchor_mdl_energy_code_hvg(
            counts, batches, top_k, args.seed, anchor_scorer
        ),
        "anchor_boundary_transport_nudge_hvg": lambda counts, batches, top_k: score_anchor_boundary_transport_nudge_hvg(
            counts, batches, top_k, args.seed, anchor_scorer
        ),
    }
    methods = {**baseline_methods, **candidate_methods}

    rows: list[dict[str, object]] = []
    overlaps: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        dataset = load_dataset(DATASETS[dataset_name], args.max_cells, args.max_genes, args.seed)
        top_k = min(args.top_k, dataset.counts.shape[1])
        dataset_summaries.append(
            {
                "dataset": dataset.name,
                "cells": int(dataset.counts.shape[0]),
                "genes": int(dataset.counts.shape[1]),
                "labels": dataset.labels is not None,
                "label_classes": None if dataset.labels is None else int(len(np.unique(dataset.labels))),
                "batches": dataset.batches is not None,
                "top_k": top_k,
            }
        )
        score_cache: dict[str, np.ndarray] = {}
        for method_name, scorer in methods.items():
            row, scores = evaluate_method(dataset, method_name, scorer, top_k, args.seed, args.n_bootstrap)
            rows.append(row)
            score_cache[method_name] = scores
        anchor_scores = score_cache["adaptive_hybrid_hvg"]
        for method_name in candidate_methods:
            overlap, jaccard = top_overlap(anchor_scores, score_cache[method_name], top_k)
            overlaps.append(
                {
                    "dataset": dataset.name,
                    "method": method_name,
                    "anchor_overlap": overlap,
                    "anchor_jaccard": jaccard,
                }
            )

    smoke_df = add_overall_score(pd.DataFrame(rows))
    smoke_df.to_csv(out / "smoke_results.csv", index=False)
    overlap_df = pd.DataFrame(overlaps)
    overlap_df.to_csv(out / "anchor_overlap.csv", index=False)
    (out / "dataset_summary.json").write_text(json.dumps(dataset_summaries, indent=2), encoding="utf-8")

    summary = []
    baseline_set = set(BASELINE_METHODS)
    for method_name in candidate_methods:
        ranks = []
        wins = 0
        metric_wins = 0
        delta_anchor = []
        delta_best = []
        for dataset_name, group in smoke_df.groupby("dataset"):
            ordered = group.sort_values("overall_score", ascending=False).reset_index(drop=True)
            rank = int(ordered.index[ordered["method"] == method_name][0] + 1)
            ranks.append(rank)
            cand = group[group["method"] == method_name].iloc[0]
            anchor = group[group["method"] == "adaptive_hybrid_hvg"].iloc[0]
            best_base_score = float(group[group["method"].isin(baseline_set)]["overall_score"].max())
            delta_anchor.append(float(cand["overall_score"] - anchor["overall_score"]))
            delta_best.append(float(cand["overall_score"] - best_base_score))
            if rank == 1:
                wins += 1
            for metric in ["ari", "nmi", "label_silhouette", "cluster_silhouette", "neighbor_preservation", "stability"]:
                if metric in group.columns and float(cand[metric]) >= float(group[group["method"].isin(baseline_set)][metric].max()):
                    metric_wins += 1
        ov = overlap_df[overlap_df["method"] == method_name]
        summary.append(
            {
                "method": method_name,
                "mean_rank": float(np.mean(ranks)),
                "wins": wins,
                "metric_wins": metric_wins,
                "mean_delta_vs_anchor": float(np.mean(delta_anchor)),
                "mean_delta_vs_best_baseline": float(np.mean(delta_best)),
                "max_delta_vs_best_baseline": float(np.max(delta_best)),
                "mean_anchor_overlap": float(ov["anchor_overlap"].mean()),
                "smoke_pass": bool(wins >= 1 or np.max(delta_best) > -0.03 or metric_wins >= 3 or np.mean(delta_anchor) > 0.02),
            }
        )
    summary_df = pd.DataFrame(summary).sort_values(
        ["smoke_pass", "wins", "metric_wins", "mean_delta_vs_best_baseline"],
        ascending=[False, False, False, False],
    )
    summary_df.to_csv(out / "candidate_summary.csv", index=False)

    top_candidates = summary_df[summary_df["smoke_pass"]].head(args.extended_top_n)["method"].tolist()
    ext_rows = []
    if top_candidates:
        ext_methods = {name: candidate_methods[name] for name in top_candidates}
        ext_methods.update(
            {name: baseline_methods[name] for name in ["adaptive_hybrid_hvg", "adaptive_stat_hvg", "multinomial_deviance_hvg", "variance"] if name in baseline_methods}
        )
        for dataset_name in args.extended_datasets:
            dataset = load_dataset(DATASETS[dataset_name], args.max_cells, args.max_genes, args.seed)
            top_k = min(args.top_k, dataset.counts.shape[1])
            for method_name, scorer in ext_methods.items():
                row, _ = evaluate_method(dataset, method_name, scorer, top_k, args.seed, args.n_bootstrap)
                ext_rows.append(row)
    ext_df = add_overall_score(pd.DataFrame(ext_rows)) if ext_rows else pd.DataFrame()
    if not ext_df.empty:
        ext_df.to_csv(out / "extended_dataset_results.csv", index=False)

    report = render_report(smoke_df, summary_df, ext_df, dataset_summaries, top_candidates, args)
    (out / "anchor_nudges_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved artifacts to: {out}")


def render_report(
    smoke_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    dataset_summaries: list[dict[str, object]],
    top_candidates: list[str],
    args: argparse.Namespace,
) -> str:
    lines = ["# Cross-Disciplinary Anchor-Nudge Smoke", ""]
    lines += ["## Motivation", ""]
    lines.append("The first round showed that pure cross-disciplinary single scorers had real research taste but weak benchmark safety. This second round keeps the anchor and only allows upward nudges from novel mechanisms.")
    lines.append("These lines are not consensus/risk-parity replicas: each is a one-anchor, one-idea correction with explicit interpretability.")
    lines += ["", "## Candidate Mainlines", ""]
    for method_name, card in ANCHOR_IDEA_CARDS.items():
        row = summary_df[summary_df["method"] == method_name].iloc[0]
        lines.append(f"### `{method_name}`")
        lines.append(f"- Taste: {card['taste']}")
        lines.append(f"- Why: {card['why']}")
        lines.append(f"- Mechanism: {card['mechanism']}")
        lines.append(
            f"- Smoke: pass={bool(row['smoke_pass'])}, mean_rank={row['mean_rank']:.2f}, wins={int(row['wins'])}, "
            f"metric_wins={int(row['metric_wins'])}, mean_delta_vs_anchor={row['mean_delta_vs_anchor']:.4f}, "
            f"mean_delta_vs_best_baseline={row['mean_delta_vs_best_baseline']:.4f}, anchor_overlap={row['mean_anchor_overlap']:.3f}."
        )
        lines.append("")
    lines += ["## Smoke Setup", ""]
    for item in dataset_summaries:
        lines.append(
            f"- `{item['dataset']}`: cells={item['cells']}, genes={item['genes']}, labels={item['labels']}, "
            f"label_classes={item['label_classes']}, batches={item['batches']}, top_k={item['top_k']}."
        )
    lines.append(f"- max_cells=`{args.max_cells}`, max_genes=`{args.max_genes}`, n_bootstrap=`{args.n_bootstrap}`, seed=`{args.seed}`.")
    lines += ["", "## Candidate Ranking", ""]
    cols = [
        "method",
        "mean_rank",
        "wins",
        "metric_wins",
        "mean_delta_vs_anchor",
        "mean_delta_vs_best_baseline",
        "max_delta_vs_best_baseline",
        "mean_anchor_overlap",
        "smoke_pass",
    ]
    lines.append(summary_df[cols].round(4).to_markdown(index=False))
    lines += ["", "## Per-Dataset Top Results", ""]
    for dataset_name, group in smoke_df.groupby("dataset"):
        cols2 = [
            col
            for col in [
                "method",
                "overall_score",
                "ari",
                "nmi",
                "label_silhouette",
                "cluster_silhouette",
                "neighbor_preservation",
                "stability",
                "runtime_sec",
            ]
            if col in group.columns
        ]
        lines.append(f"### `{dataset_name}`")
        lines.append(group.sort_values("overall_score", ascending=False)[cols2].head(9).round(4).to_markdown(index=False))
        lines.append("")
    lines += ["## Extended Dataset Test", ""]
    if top_candidates and not ext_df.empty:
        lines.append("Unlocked candidates: " + ", ".join(f"`{name}`" for name in top_candidates) + ".")
        for dataset_name, group in ext_df.groupby("dataset"):
            cols2 = [
                col
                for col in [
                    "method",
                    "overall_score",
                    "ari",
                    "nmi",
                    "label_silhouette",
                    "cluster_silhouette",
                    "neighbor_preservation",
                    "stability",
                    "runtime_sec",
                ]
                if col in group.columns
            ]
            lines.append(f"### `{dataset_name}`")
            lines.append(group.sort_values("overall_score", ascending=False)[cols2].round(4).to_markdown(index=False))
            lines.append("")
    else:
        lines.append("No candidate crossed the relaxed extension gate.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
