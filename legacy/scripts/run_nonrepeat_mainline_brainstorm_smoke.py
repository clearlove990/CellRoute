from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hvg_research import build_default_method_registry, load_scrna_dataset
from hvg_research.baselines import normalize_log1p
from hvg_research.eval import bootstrap_jaccard, evaluate_real_selection


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    fmt: str = "h5ad"
    labels_col: str | None = None
    batches_col: str | None = None
    obs_path: str | None = None
    genes_path: str | None = None
    cells_path: str | None = None
    transpose: bool = False


DATASETS = {
    "paul15": DatasetSpec(
        "paul15",
        ROOT / "data" / "real_inputs" / "paul15" / "matrix" / "matrix.mtx",
        fmt="mtx",
        labels_col="paul15_clusters",
        obs_path=str(ROOT / "data" / "real_inputs" / "paul15" / "obs.csv"),
        genes_path=str(ROOT / "data" / "real_inputs" / "paul15" / "matrix" / "genes.tsv"),
        cells_path=str(ROOT / "data" / "real_inputs" / "paul15" / "matrix" / "barcodes.tsv"),
        transpose=True,
    ),
    "GBM_sd": DatasetSpec(
        "GBM_sd",
        ROOT / "data" / "real_inputs" / "GBM_sd" / "matrix" / "matrix.mtx",
        fmt="mtx",
        labels_col="gbm_region_label",
        obs_path=str(ROOT / "data" / "real_inputs" / "GBM_sd" / "obs.csv"),
        genes_path=str(ROOT / "data" / "real_inputs" / "GBM_sd" / "matrix" / "genes.tsv"),
        cells_path=str(ROOT / "data" / "real_inputs" / "GBM_sd" / "matrix" / "barcodes.tsv"),
        transpose=True,
    ),
    "FBM_cite": DatasetSpec(
        "FBM_cite",
        ROOT / "data" / "real_inputs" / "FBM_cite" / "matrix" / "matrix.mtx",
        fmt="mtx",
        labels_col="adt_celltype_label",
        obs_path=str(ROOT / "data" / "real_inputs" / "FBM_cite" / "obs.csv"),
        genes_path=str(ROOT / "data" / "real_inputs" / "FBM_cite" / "matrix" / "genes.tsv"),
        cells_path=str(ROOT / "data" / "real_inputs" / "FBM_cite" / "matrix" / "barcodes.tsv"),
        transpose=True,
    ),
    "cellxgene_human_kidney_nonpt": DatasetSpec(
        "cellxgene_human_kidney_nonpt", ROOT / "data" / "real_inputs" / "cellxgene_human_kidney_nonpt" / "source.h5ad"
    ),
}

EXISTING_MAINLINES = {
    "adaptive_core_consensus_hvg",
    "adaptive_rank_aggregate_hvg",
    "adaptive_eb_shrinkage_hvg",
    "adaptive_invariant_residual_hvg",
    "adaptive_spectral_locality_hvg",
    "adaptive_stability_jackknife_hvg",
    "adaptive_risk_parity_hvg",
    "sigma_safe_core_v5_hvg",
    "sigma_safe_core_v6_hvg",
    "adaptive_hybrid_hvg",
    "holdout_risk_release_hvg",
    "learnable_gate_bank_pairregret_permissioned_escapecert_frontier_lite",
}

BASELINE_METHODS = (
    "adaptive_hybrid_hvg",
    "adaptive_stat_hvg",
    "multinomial_deviance_hvg",
    "variance",
    "mv_residual",
    "fano",
    "analytic_pearson_residual_hvg",
    "seurat_v3_like_hvg",
)

IDEA_CARDS = {
    "marker_anti_hub_hvg": {
        "mainline": "marker-contrast with anti-housekeeping hub penalty",
        "why": "Benchmark headroom often rewards cell-type separability, but classical HVG can waste slots on globally high-expression hub/housekeeping genes. A one-vs-rest pseudo-marker score tests whether simple unsupervised contrasts can improve ARI/NMI without adding another selector/routing layer.",
        "not_repeat": "It is not consensus/rank aggregation, EB shrinkage, invariance, graph locality, jackknife stability, risk parity, sigma, or learned gating; it uses cluster-contrast geometry plus an anti-hub penalty.",
        "smoke_hypothesis": "ARI/NMI and label silhouette should rise on labeled compact datasets; neighbor preservation may trade off.",
    },
    "depth_residual_quantile_hvg": {
        "mainline": "depth-residual quantile tail scorer",
        "why": "A common benchmark failure is selecting genes that track library size or detection depth rather than biology. Regressing expression on log-depth and scoring robust high residual quantiles is a cheap way to target dropout/depth-confounded datasets without batch-invariance machinery.",
        "not_repeat": "It is residualization against per-cell depth only, not count-model Pearson residual, EB shrinkage, batch invariance, or stability resampling.",
        "smoke_hypothesis": "Batch mixing/neighbor preservation should improve when depth is a shortcut; very marker-heavy datasets may lose ARI.",
    },
    "metagene_redundancy_hvg": {
        "mainline": "metagene redundancy-pruned HVG",
        "why": "Top HVG lists can be redundant, over-selecting co-expressed modules. Penalizing gene-gene correlation to a small PCA metagene basis tests whether diversity of selected genes gives benchmark gains at fixed top-k.",
        "not_repeat": "It is redundancy control at gene-set geometry level, not multi-scorer risk parity/consensus and not graph locality over cells.",
        "smoke_hypothesis": "Neighbor preservation and cluster silhouette should improve if redundancy is the bottleneck; overlap with anchor should drop moderately, not catastrophically.",
    },
    "rare_state_tail_hvg": {
        "mainline": "rare-state upper-tail enrichment scorer",
        "why": "Mean-variance methods underweight genes active in small rare populations. A high-quantile-vs-median score tests a biologically interpretable rare-cell angle that current benchmark routes do not explicitly cover.",
        "not_repeat": "It is a distribution-tail/rare-state detector, not Fano fallback, graph spectral locality, stability, or adaptive routing.",
        "smoke_hypothesis": "It should help datasets with rare labeled states, while failing clearly on homogeneous atlas controls.",
    },
    "bimodality_gate_hvg": {
        "mainline": "zero-inflated bimodality gate scorer",
        "why": "Genes separating discrete cell states often have bimodal detected expression, not just high variance. A detection-rate entropy plus positive-expression separation score tests whether state-switch genes improve clustering metrics.",
        "not_repeat": "It is a single distribution-shape scorer, not graph, invariance, rank aggregation, EB, sigma, or learned gate selection.",
        "smoke_hypothesis": "It should lift ARI/NMI on discrete-label datasets and may hurt continuous trajectory-like neighbor preservation.",
    },
}


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    denom = max(len(order) - 1, 1)
    return ranks / denom


def robust_log_norm(counts: np.ndarray) -> np.ndarray:
    return normalize_log1p(np.asarray(counts, dtype=np.float32))


def score_marker_anti_hub_hvg(counts: np.ndarray, batches: np.ndarray | None, top_k: int, random_state: int = 0) -> np.ndarray:
    del batches, top_k
    x = robust_log_norm(counts)
    n_cells = x.shape[0]
    pca_dim = min(15, x.shape[1], n_cells - 1)
    if pca_dim < 2:
        return zscore(np.var(x, axis=0))
    emb = PCA(n_components=pca_dim, random_state=random_state).fit_transform(x)
    n_clusters = min(12, max(3, int(np.sqrt(n_cells / 35.0))))
    labels = KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state).fit_predict(emb)
    global_mean = x.mean(axis=0)
    global_std = x.std(axis=0) + 1e-6
    contrasts = []
    for label in np.unique(labels):
        mask = labels == label
        if mask.sum() < 5 or (~mask).sum() < 5:
            continue
        diff = (x[mask].mean(axis=0) - x[~mask].mean(axis=0)) / global_std
        contrasts.append(diff)
    if not contrasts:
        marker = zscore(np.var(x, axis=0))
    else:
        marker = np.max(np.vstack(contrasts), axis=0)
    prevalence = (x > 0).mean(axis=0)
    hub_penalty = np.exp(-((prevalence - 0.55) / 0.33) ** 2)
    mean_penalty = rank_percentile(global_mean)
    score = zscore(marker) + 0.30 * zscore(np.var(x, axis=0)) - 0.45 * zscore(mean_penalty * hub_penalty)
    return zscore(score)


def score_depth_residual_quantile_hvg(counts: np.ndarray, batches: np.ndarray | None, top_k: int, random_state: int = 0) -> np.ndarray:
    del batches, top_k, random_state
    x = robust_log_norm(counts)
    depth = np.log1p(np.asarray(counts, dtype=np.float64).sum(axis=1))
    depth = zscore(depth)
    centered_depth = depth - depth.mean()
    centered_x = x - x.mean(axis=0, keepdims=True)
    beta = centered_depth @ centered_x / (centered_depth @ centered_depth + 1e-8)
    residual = centered_x - centered_depth[:, None] * beta[None, :]
    upper_tail = np.quantile(residual, 0.90, axis=0)
    lower_tail = np.quantile(residual, 0.10, axis=0)
    tail_spread = upper_tail - lower_tail
    depth_corr_penalty = np.abs(beta)
    score = zscore(tail_spread) + 0.25 * zscore(np.var(residual, axis=0)) - 0.60 * zscore(depth_corr_penalty)
    return zscore(score)


def score_metagene_redundancy_hvg(counts: np.ndarray, batches: np.ndarray | None, top_k: int, random_state: int = 0) -> np.ndarray:
    del batches, top_k
    x = robust_log_norm(counts)
    var_score = rank_percentile(np.var(x, axis=0))
    centered = x - x.mean(axis=0, keepdims=True)
    pca_dim = min(8, centered.shape[0] - 1, centered.shape[1])
    if pca_dim < 2:
        return zscore(var_score)
    emb = PCA(n_components=pca_dim, random_state=random_state).fit_transform(centered)
    emb = (emb - emb.mean(axis=0, keepdims=True)) / (emb.std(axis=0, keepdims=True) + 1e-6)
    gene_std = centered.std(axis=0) + 1e-6
    corr = np.abs(centered.T @ emb / max(centered.shape[0] - 1, 1) / gene_std[:, None])
    max_loading = corr.max(axis=1)
    loading_entropy = -(corr / (corr.sum(axis=1, keepdims=True) + 1e-8) * np.log(corr / (corr.sum(axis=1, keepdims=True) + 1e-8) + 1e-8)).sum(axis=1)
    redundancy_penalty = max_loading - 0.20 * loading_entropy
    score = zscore(var_score) - 0.55 * zscore(redundancy_penalty) + 0.20 * zscore(np.mean(corr, axis=1))
    return zscore(score)


def score_rare_state_tail_hvg(counts: np.ndarray, batches: np.ndarray | None, top_k: int, random_state: int = 0) -> np.ndarray:
    del batches, top_k, random_state
    x = robust_log_norm(counts)
    q95 = np.quantile(x, 0.95, axis=0)
    q75 = np.quantile(x, 0.75, axis=0)
    q50 = np.quantile(x, 0.50, axis=0)
    prevalence = (x > 0).mean(axis=0)
    rare_window = np.exp(-((prevalence - 0.12) / 0.16) ** 2)
    tail_lift = q95 - q50
    concentrated_tail = q95 - q75
    score = zscore(tail_lift) + 0.65 * zscore(concentrated_tail) + 0.45 * zscore(rare_window) - 0.20 * zscore(prevalence > 0.75)
    return zscore(score)


def score_bimodality_gate_hvg(counts: np.ndarray, batches: np.ndarray | None, top_k: int, random_state: int = 0) -> np.ndarray:
    del batches, top_k, random_state
    x = robust_log_norm(counts)
    detected = x > 0
    prevalence = detected.mean(axis=0)
    entropy = -(prevalence * np.log(prevalence + 1e-8) + (1.0 - prevalence) * np.log(1.0 - prevalence + 1e-8))
    positive_mean = np.divide((x * detected).sum(axis=0), detected.sum(axis=0) + 1e-8)
    negative_mean = np.divide((x * ~detected).sum(axis=0), (~detected).sum(axis=0) + 1e-8)
    separation = positive_mean - negative_mean
    mid_prevalence = np.exp(-((prevalence - 0.35) / 0.28) ** 2)
    score = 0.75 * zscore(separation) + 0.65 * zscore(entropy * mid_prevalence) + 0.20 * zscore(np.var(x, axis=0))
    return zscore(score)


CANDIDATE_SCORERS: dict[str, Callable[[np.ndarray, np.ndarray | None, int, int], np.ndarray]] = {
    "marker_anti_hub_hvg": score_marker_anti_hub_hvg,
    "depth_residual_quantile_hvg": score_depth_residual_quantile_hvg,
    "metagene_redundancy_hvg": score_metagene_redundancy_hvg,
    "rare_state_tail_hvg": score_rare_state_tail_hvg,
    "bimodality_gate_hvg": score_bimodality_gate_hvg,
}


def choose_device() -> dict[str, object]:
    if torch is None:
        return {"torch_available": False, "cuda_available": False, "selected_device": "cpu", "note": "torch unavailable; numpy/sklearn CPU smoke"}
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        return {
            "torch_available": True,
            "torch_version": torch.__version__,
            "cuda_available": True,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "selected_device": "cuda",
            "note": "CUDA detected; existing registry torch scorers may use GPU, smoke candidates are numpy/sklearn CPU prototypes",
        }
    return {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": False,
        "selected_device": "cpu",
        "note": "CUDA unavailable; seamless CPU fallback",
    }


def load_dataset(spec: DatasetSpec, max_cells: int, max_genes: int, seed: int):
    return load_scrna_dataset(
        str(spec.path),
        file_format=spec.fmt,
        transpose=spec.transpose,
        obs_path=spec.obs_path,
        genes_path=spec.genes_path,
        cells_path=spec.cells_path,
        labels_col=spec.labels_col,
        batches_col=spec.batches_col,
        max_cells=max_cells,
        max_genes=max_genes,
        random_state=seed,
        dataset_name=spec.name,
    )


def add_overall_score(df: pd.DataFrame) -> pd.DataFrame:
    positive_metrics = [
        metric
        for metric in ["ari", "nmi", "label_silhouette", "cluster_silhouette", "neighbor_preservation", "stability", "batch_mixing"]
        if metric in df.columns
    ]
    score = np.zeros(len(df), dtype=np.float64)
    for metric in positive_metrics:
        values = df[metric].to_numpy(dtype=np.float64)
        if np.nanmax(values) > np.nanmin(values):
            score += (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
    if "runtime_sec" in df.columns:
        values = df["runtime_sec"].to_numpy(dtype=np.float64)
        if np.nanmax(values) > np.nanmin(values):
            score -= 0.15 * (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
    df["overall_score"] = score
    return df


def evaluate_method(dataset, method_name: str, scorer: Callable, top_k: int, seed: int, n_bootstrap: int) -> dict[str, object]:
    start = time.perf_counter()
    if method_name in CANDIDATE_SCORERS:
        scores = scorer(dataset.counts, dataset.batches, top_k, seed)
        scorer_fn = lambda subset_counts, subset_batches, fn=scorer: fn(subset_counts, subset_batches, min(top_k, subset_counts.shape[1]), seed)
    else:
        scores = scorer(dataset.counts, dataset.batches, top_k)
        scorer_fn = lambda subset_counts, subset_batches, fn=scorer: fn(subset_counts, subset_batches, min(top_k, subset_counts.shape[1]))
    elapsed = time.perf_counter() - start
    selected = np.argsort(scores)[-top_k:]
    metrics = evaluate_real_selection(
        counts=dataset.counts,
        selected_genes=selected,
        scorer_fn=scorer_fn,
        labels=dataset.labels,
        batches=dataset.batches,
        top_k=top_k,
        random_state=seed,
        n_bootstrap=n_bootstrap,
    )
    metrics.update(
        {
            "dataset": dataset.name,
            "method": method_name,
            "runtime_sec": elapsed,
            "top_k": top_k,
            "selected_min_score": float(np.min(scores[selected])),
            "score_std": float(np.std(scores)),
        }
    )
    return metrics


def overlap_to_anchor(dataset, anchor_scores: np.ndarray, candidate_scores: np.ndarray, top_k: int) -> dict[str, float]:
    anchor_top = set(np.argsort(anchor_scores)[-top_k:].tolist())
    cand_top = set(np.argsort(candidate_scores)[-top_k:].tolist())
    inter = len(anchor_top & cand_top)
    union = len(anchor_top | cand_top)
    return {"anchor_overlap": inter / max(top_k, 1), "anchor_jaccard": inter / max(union, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Brainstorm non-duplicate HVG mainlines and smoke-test them.")
    parser.add_argument("--output-dir", default="artifacts_nonrepeat_mainline_brainstorm_smoke_20260424_v1")
    parser.add_argument("--datasets", nargs="+", default=["paul15", "GBM_sd"])
    parser.add_argument("--extended-datasets", nargs="+", default=["FBM_cite", "cellxgene_human_kidney_nonpt"])
    parser.add_argument("--max-cells", type=int, default=900)
    parser.add_argument("--max-genes", type=int, default=2500)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-bootstrap", type=int, default=2)
    parser.add_argument("--extended-top-n", type=int, default=2)
    args = parser.parse_args()

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    registry = build_default_method_registry(top_k=args.top_k, random_state=args.seed, refine_epochs=0)
    methods: dict[str, Callable] = {name: registry[name] for name in BASELINE_METHODS if name in registry}
    methods.update(CANDIDATE_SCORERS)

    device = choose_device()
    (out / "compute_context.json").write_text(json.dumps(device, indent=2), encoding="utf-8")
    (out / "idea_cards.json").write_text(json.dumps(IDEA_CARDS, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []

    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        dataset = load_dataset(spec, max_cells=args.max_cells, max_genes=args.max_genes, seed=args.seed)
        top_k = min(args.top_k, dataset.counts.shape[1])
        dataset_summaries.append(
            {
                "dataset": dataset.name,
                "cells": int(dataset.counts.shape[0]),
                "genes": int(dataset.counts.shape[1]),
                "labels": dataset.labels is not None,
                "label_classes": None if dataset.labels is None else int(len(np.unique(dataset.labels))),
                "batches": dataset.batches is not None,
                "batch_classes": None if dataset.batches is None else int(len(np.unique(dataset.batches))),
                "top_k": top_k,
            }
        )
        anchor_scores = registry["adaptive_hybrid_hvg"](dataset.counts, dataset.batches, top_k)
        for method_name, scorer in methods.items():
            row = evaluate_method(dataset, method_name, scorer, top_k, args.seed, args.n_bootstrap)
            rows.append(row)
            if method_name in CANDIDATE_SCORERS:
                cand_scores = scorer(dataset.counts, dataset.batches, top_k, args.seed)
                overlap_rows.append({"dataset": dataset.name, "method": method_name, **overlap_to_anchor(dataset, anchor_scores, cand_scores, top_k)})

    smoke_df = add_overall_score(pd.DataFrame(rows))
    smoke_path = out / "smoke_results.csv"
    smoke_df.to_csv(smoke_path, index=False)
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(out / "anchor_overlap.csv", index=False)
    (out / "dataset_summary.json").write_text(json.dumps(dataset_summaries, indent=2), encoding="utf-8")

    baseline_names = set(BASELINE_METHODS)
    candidate_summary = []
    for method_name in CANDIDATE_SCORERS:
        method_rows = smoke_df[smoke_df["method"] == method_name]
        wins = 0
        positive_vs_anchor = []
        positive_vs_best_baseline = []
        ranks = []
        for dataset_name, group in smoke_df.groupby("dataset"):
            ordered = group.sort_values("overall_score", ascending=False).reset_index(drop=True)
            rank = int(ordered.index[ordered["method"] == method_name][0] + 1)
            ranks.append(rank)
            cand_score = float(group[group["method"] == method_name]["overall_score"].iloc[0])
            anchor_score = float(group[group["method"] == "adaptive_hybrid_hvg"]["overall_score"].iloc[0])
            best_base = float(group[group["method"].isin(baseline_names)]["overall_score"].max())
            positive_vs_anchor.append(cand_score - anchor_score)
            positive_vs_best_baseline.append(cand_score - best_base)
            if rank == 1:
                wins += 1
        ov = overlap_df[overlap_df["method"] == method_name]
        candidate_summary.append(
            {
                "method": method_name,
                "mean_rank": float(np.mean(ranks)),
                "wins": wins,
                "mean_delta_vs_anchor": float(np.mean(positive_vs_anchor)),
                "mean_delta_vs_best_baseline": float(np.mean(positive_vs_best_baseline)),
                "max_delta_vs_best_baseline": float(np.max(positive_vs_best_baseline)),
                "mean_anchor_overlap": float(ov["anchor_overlap"].mean()) if not ov.empty else float("nan"),
                "smoke_pass": bool(wins >= 1 or np.max(positive_vs_best_baseline) > 0.15 or np.mean(positive_vs_anchor) > 0.20),
            }
        )
    summary_df = pd.DataFrame(candidate_summary).sort_values(
        ["smoke_pass", "wins", "mean_delta_vs_best_baseline", "mean_delta_vs_anchor"], ascending=[False, False, False, False]
    )
    summary_df.to_csv(out / "candidate_summary.csv", index=False)

    top_candidates = summary_df[summary_df["smoke_pass"]].head(args.extended_top_n)["method"].tolist()
    extended_rows = []
    if top_candidates:
        extended_methods = {name: CANDIDATE_SCORERS[name] for name in top_candidates}
        extended_methods.update({name: registry[name] for name in ["adaptive_hybrid_hvg", "adaptive_stat_hvg", "multinomial_deviance_hvg", "variance"] if name in registry})
        for dataset_name in args.extended_datasets:
            spec = DATASETS[dataset_name]
            dataset = load_dataset(spec, max_cells=args.max_cells, max_genes=args.max_genes, seed=args.seed)
            top_k = min(args.top_k, dataset.counts.shape[1])
            for method_name, scorer in extended_methods.items():
                extended_rows.append(evaluate_method(dataset, method_name, scorer, top_k, args.seed, args.n_bootstrap))
    if extended_rows:
        ext_df = add_overall_score(pd.DataFrame(extended_rows))
        ext_df.to_csv(out / "extended_dataset_results.csv", index=False)
    else:
        ext_df = pd.DataFrame()

    report = render_report(smoke_df, summary_df, overlap_df, ext_df, dataset_summaries, top_candidates, args)
    (out / "brainstorm_smoke_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved artifacts to: {out}")


def render_report(
    smoke_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    dataset_summaries: list[dict[str, object]],
    top_candidates: list[str],
    args: argparse.Namespace,
) -> str:
    lines = ["# Non-Repeated Mainline Brainstorm Smoke Report", ""]
    lines += ["## Non-Repeat Boundary", ""]
    lines.append("Excluded prior/current lines: " + ", ".join(sorted(EXISTING_MAINLINES)) + ".")
    lines.append("All smoke candidates below use single-scorer mechanisms not already represented by those lines.")
    lines += ["", "## Candidate Ideas", ""]
    for name, card in IDEA_CARDS.items():
        row = summary_df[summary_df["method"] == name].iloc[0].to_dict()
        lines.append(f"### `{name}`")
        lines.append(f"- Mainline: {card['mainline']}.")
        lines.append(f"- Reason: {card['why']}")
        lines.append(f"- Non-repeat check: {card['not_repeat']}")
        lines.append(f"- Smoke hypothesis: {card['smoke_hypothesis']}")
        lines.append(
            "- Smoke readout: "
            f"pass={row['smoke_pass']}, wins={int(row['wins'])}, mean_rank={row['mean_rank']:.2f}, "
            f"mean_delta_vs_anchor={row['mean_delta_vs_anchor']:.4f}, "
            f"mean_delta_vs_best_baseline={row['mean_delta_vs_best_baseline']:.4f}, "
            f"mean_anchor_overlap={row['mean_anchor_overlap']:.3f}."
        )
        lines.append("")
    lines += ["## Smoke Setup", ""]
    for item in dataset_summaries:
        lines.append(
            f"- `{item['dataset']}`: cells={item['cells']}, genes={item['genes']}, labels={item['labels']}, "
            f"label_classes={item['label_classes']}, batches={item['batches']}, top_k={item['top_k']}."
        )
    lines.append(f"- Bootstrap rounds for stability: `{args.n_bootstrap}`; max_cells=`{args.max_cells}`; max_genes=`{args.max_genes}`.")
    lines += ["", "## Smoke Ranking", ""]
    display_cols = ["method", "mean_rank", "wins", "mean_delta_vs_anchor", "mean_delta_vs_best_baseline", "max_delta_vs_best_baseline", "mean_anchor_overlap", "smoke_pass"]
    lines.append(summary_df[display_cols].round(4).to_markdown(index=False))
    lines += ["", "## Per-Dataset Top Results", ""]
    for dataset_name, group in smoke_df.groupby("dataset"):
        cols = [c for c in ["method", "overall_score", "ari", "nmi", "label_silhouette", "cluster_silhouette", "neighbor_preservation", "stability", "batch_mixing", "runtime_sec"] if c in group.columns]
        lines.append(f"### `{dataset_name}`")
        lines.append(group.sort_values("overall_score", ascending=False)[cols].head(8).round(4).to_markdown(index=False))
        lines.append("")
    if top_candidates:
        lines += ["## Extended Dataset Test", ""]
        lines.append("Unlocked by smoke-pass candidates: " + ", ".join(f"`{x}`" for x in top_candidates) + ".")
        if not ext_df.empty:
            for dataset_name, group in ext_df.groupby("dataset"):
                cols = [c for c in ["method", "overall_score", "ari", "nmi", "label_silhouette", "cluster_silhouette", "neighbor_preservation", "stability", "batch_mixing", "runtime_sec"] if c in group.columns]
                lines.append(f"### `{dataset_name}`")
                lines.append(group.sort_values("overall_score", ascending=False)[cols].round(4).to_markdown(index=False))
                lines.append("")
    else:
        lines += ["## Extended Dataset Test", "", "No candidate passed the smoke gate, so no extended dataset test was run.", ""]
    lines += ["## Recommendation", ""]
    if top_candidates:
        first = top_candidates[0]
        lines.append(f"- Highest-priority next line: `{first}` because it cleared the smoke gate without repeating existing mainlines.")
        lines.append("- Treat this as feasibility evidence only; full benchmarkHVG partial or full benchmark should be run before claiming performance.")
    else:
        lines.append("- None of the non-repeated candidates is strong enough to replace current benchmark-backed lines yet.")
        lines.append("- The safest use of these ideas is as ablation probes or as dataset-regime diagnostics, not a new mainline.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()


