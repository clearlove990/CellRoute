from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)

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
from hvg_research.eval import evaluate_real_selection


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

EXCLUDED_PRIOR_LINES = (
    "adaptive_hybrid_hvg",
    "adaptive_core_consensus_hvg",
    "adaptive_rank_aggregate_hvg",
    "adaptive_eb_shrinkage_hvg",
    "adaptive_invariant_residual_hvg",
    "adaptive_spectral_locality_hvg",
    "adaptive_stability_jackknife_hvg",
    "adaptive_risk_parity_hvg",
    "sigma_safe_core_v5_hvg",
    "sigma_safe_core_v6_hvg",
    "marker_anti_hub_hvg",
    "depth_residual_quantile_hvg",
    "metagene_redundancy_hvg",
    "rare_state_tail_hvg",
    "bimodality_gate_hvg",
)

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

LIT_MAP = {
    "benchmark_and_hvg_context": [
        "benchmarkHVG/standard benchmark framing motivates testing against clustering, bio-conservation and batch-style metrics rather than only variance ranking.",
        "scry/deviance residual work supports count-model departures as a strong baseline family.",
        "triku/DUBStep/GeneBasis-style work motivates local-neighborhood, rare-cell, and manifold-preservation feature selection, but these candidates avoid directly reusing those mechanisms.",
    ],
    "cross_disciplinary_sources": [
        "Optimal transport: preserve distributional movement between cell neighborhoods, not only local variance.",
        "Ecology: genes as niche constructors whose value is high when they partition microhabitats without becoming universal hubs.",
        "Minimum description length: useful genes compress cell-state structure with few bits, penalizing noisy high-entropy expression.",
        "Statistical physics: informative genes sharpen metastable energy wells while not merely tracking depth.",
        "Mathematical morphology: boundary genes define state interfaces and transition fronts, not just cluster interiors.",
    ],
}

IDEA_CARDS = {
    "ot_barycentric_displacement_hvg": {
        "taste": "Treat HVG selection as preserving transport geometry: genes are valuable when their scalar field explains how cell neighborhoods would move between transcriptomic barycenters.",
        "why": "Recent single-cell methods increasingly evaluate manifold and neighborhood conservation. OT gives a principled lens for distributional movement, but this smoke uses a cheap barycentric proxy rather than a full OT solver.",
        "mechanism": "Build PCA neighborhoods, compute local barycentric displacement, then score genes whose local expression gradient aligns with displacement magnitude while penalizing depth correlation.",
        "prediction": "Should help trajectory/region datasets where biological signal is a directional field rather than discrete marker variance.",
    },
    "ecological_niche_partition_hvg": {
        "taste": "Borrow ecology: a good gene is a niche-partitioning species, high in specific microhabitats but not a cosmopolitan hub.",
        "why": "Rare-cell and local-neighborhood feature selection are active themes, but ecological occupancy/evenness gives a distinct explanatory language and a concrete scorer.",
        "mechanism": "Cluster unsupervised microhabitats, score between-niche occupancy divergence and within-niche consistency, penalize overly broad occupancy.",
        "prediction": "Should improve datasets with clear cell-type niches; should not overfit continuous homogeneous panels.",
    },
    "mdl_state_code_hvg": {
        "taste": "Invert HVG: not 'most variable', but 'cheapest code for cell-state identity'.",
        "why": "MDL/compression is an elegant cross-field criterion: useful features reduce state uncertainty per expression bit. This is distinct from variance, deviance, and rank aggregation.",
        "mechanism": "Discretize each gene into expression bins, estimate mutual information with unsupervised states, subtract entropy and depth-correlation costs.",
        "prediction": "Should favor crisp, low-complexity state genes and avoid noisy high-variance genes.",
    },
    "energy_well_curvature_hvg": {
        "taste": "View cells as samples from an energy landscape; select genes that deepen metastable wells and steepen basin curvature.",
        "why": "Energy landscapes are common in developmental biology and physics, but rarely used as a simple HVG scorer. This tests whether basin-sharpening genes help clustering metrics.",
        "mechanism": "Use PCA/KMeans basins, score genes by between-basin curvature minus within-basin thermal noise and depth shortcut.",
        "prediction": "Should help discrete or branching-state datasets; may underperform on continuous gradients.",
    },
    "morphological_boundary_hvg": {
        "taste": "A gene can matter because it draws the border between states, not because it fills either state interior.",
        "why": "Mathematical morphology and image segmentation focus on boundaries. Single-cell HVG mostly ranks cluster interiors; boundary genes could capture transitions and ambiguous cell types.",
        "mechanism": "Find cells near KNN cluster boundaries, score genes enriched or high-gradient on boundary cells, with a penalty for generic high expression.",
        "prediction": "Should improve neighbor preservation/transition-like datasets if boundaries carry biology; may trade off ARI on pure discrete labels.",
    },
}


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return np.nan_to_num((values - np.nanmean(values)) / std, nan=0.0, posinf=0.0, neginf=0.0)


def rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks / max(len(order) - 1, 1)


def safe_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean()
    denom = (np.sqrt(np.sum(ac * ac, axis=0)) * np.sqrt(np.sum(bc * bc))) + 1e-8
    return np.nan_to_num((bc @ ac) / denom, nan=0.0)


def norm_counts(counts: np.ndarray) -> np.ndarray:
    return normalize_log1p(np.asarray(counts, dtype=np.float32))


def state_embedding(x: np.ndarray, random_state: int, n_components: int = 15) -> np.ndarray:
    pca_dim = min(n_components, x.shape[0] - 1, x.shape[1])
    if pca_dim < 2:
        return x[:, : max(1, min(x.shape[1], 2))]
    return PCA(n_components=pca_dim, random_state=random_state).fit_transform(x)


def pseudo_states(embedding: np.ndarray, random_state: int) -> np.ndarray:
    n_cells = embedding.shape[0]
    n_clusters = min(14, max(3, int(np.sqrt(n_cells / 30.0))))
    return KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state).fit_predict(embedding)


def depth_penalty(counts: np.ndarray, x: np.ndarray) -> np.ndarray:
    depth = np.log1p(np.asarray(counts, dtype=np.float64).sum(axis=1))
    return np.abs(safe_corr(x, depth))


def score_ot_barycentric_displacement_hvg(counts: np.ndarray, batches: np.ndarray | None, top_k: int, random_state: int) -> np.ndarray:
    del batches, top_k
    x = norm_counts(counts)
    emb = state_embedding(x, random_state)
    n_neighbors = min(21, max(6, x.shape[0] // 35))
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(emb)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    local_center = emb[idx].mean(axis=1)
    displacement = np.linalg.norm(emb - local_center, axis=1)
    states = pseudo_states(emb, random_state)
    state_centers = np.vstack([emb[states == state].mean(axis=0) for state in np.unique(states)])
    nearest_center = state_centers[np.argmin(((emb[:, None, :] - state_centers[None, :, :]) ** 2).sum(axis=2), axis=1)]
    basin_displacement = np.linalg.norm(emb - nearest_center, axis=1)
    transport_field = zscore(displacement) + 0.5 * zscore(basin_displacement)
    local_expression = x - x[idx].mean(axis=1)
    align = np.abs(safe_corr(local_expression, transport_field))
    score = zscore(align) + 0.25 * zscore(np.var(x, axis=0)) - 0.55 * zscore(depth_penalty(counts, x))
    return zscore(score)


def score_ecological_niche_partition_hvg(counts: np.ndarray, batches: np.ndarray | None, top_k: int, random_state: int) -> np.ndarray:
    del batches, top_k
    x = norm_counts(counts)
    emb = state_embedding(x, random_state)
    states = pseudo_states(emb, random_state)
    state_ids = np.unique(states)
    global_occ = (x > 0).mean(axis=0)
    occ_by_state = []
    mean_by_state = []
    for state in state_ids:
        mask = states == state
        occ_by_state.append((x[mask] > 0).mean(axis=0))
        mean_by_state.append(x[mask].mean(axis=0))
    occ = np.vstack(occ_by_state)
    means = np.vstack(mean_by_state)
    niche_specificity = occ.max(axis=0) - np.median(occ, axis=0)
    expression_partition = means.max(axis=0) - np.median(means, axis=0)
    niche_evenness_penalty = -(global_occ * np.log(global_occ + 1e-8) + (1.0 - global_occ) * np.log(1.0 - global_occ + 1e-8))
    cosmopolitan_penalty = np.exp(-((global_occ - 0.75) / 0.20) ** 2)
    score = 0.85 * zscore(niche_specificity) + 0.55 * zscore(expression_partition) - 0.25 * zscore(niche_evenness_penalty) - 0.45 * zscore(cosmopolitan_penalty)
    return zscore(score)


def score_mdl_state_code_hvg(counts: np.ndarray, batches: np.ndarray | None, top_k: int, random_state: int) -> np.ndarray:
    del batches, top_k
    x = norm_counts(counts)
    emb = state_embedding(x, random_state)
    states = pseudo_states(emb, random_state)
    state_ids, state_inv = np.unique(states, return_inverse=True)
    n_states = len(state_ids)
    quantiles = np.quantile(x, [0.33, 0.67], axis=0)
    bins = (x > quantiles[0][None, :]).astype(np.int8) + (x > quantiles[1][None, :]).astype(np.int8)
    state_probs = np.bincount(state_inv, minlength=n_states).astype(np.float64) / len(state_inv)
    h_state = -np.sum(state_probs * np.log2(state_probs + 1e-12))
    mi = np.zeros(x.shape[1], dtype=np.float64)
    entropy_gene = np.zeros(x.shape[1], dtype=np.float64)
    for gene_idx in range(x.shape[1]):
        joint = np.zeros((n_states, 3), dtype=np.float64)
        np.add.at(joint, (state_inv, bins[:, gene_idx]), 1.0)
        joint /= max(joint.sum(), 1.0)
        p_gene = joint.sum(axis=0)
        entropy_gene[gene_idx] = -np.sum(p_gene * np.log2(p_gene + 1e-12))
        denom = state_probs[:, None] * p_gene[None, :]
        valid = joint > 0
        mi[gene_idx] = np.sum(joint[valid] * np.log2(joint[valid] / (denom[valid] + 1e-12)))
    compression_gain = mi / max(h_state, 1e-8)
    score = zscore(compression_gain) - 0.32 * zscore(entropy_gene) - 0.45 * zscore(depth_penalty(counts, x)) + 0.15 * zscore(np.var(x, axis=0))
    return zscore(score)


def score_energy_well_curvature_hvg(counts: np.ndarray, batches: np.ndarray | None, top_k: int, random_state: int) -> np.ndarray:
    del batches, top_k
    x = norm_counts(counts)
    emb = state_embedding(x, random_state)
    states = pseudo_states(emb, random_state)
    global_mean = x.mean(axis=0)
    between = np.zeros(x.shape[1], dtype=np.float64)
    within = np.zeros(x.shape[1], dtype=np.float64)
    sizes = []
    for state in np.unique(states):
        mask = states == state
        sizes.append(mask.mean())
        state_mean = x[mask].mean(axis=0)
        between += mask.mean() * (state_mean - global_mean) ** 2
        within += mask.mean() * x[mask].var(axis=0)
    curvature = between / (within + 1e-4)
    basin_balance = 1.0 - np.std(sizes) / (np.mean(sizes) + 1e-8)
    score = zscore(curvature) + 0.25 * basin_balance * zscore(between) - 0.40 * zscore(depth_penalty(counts, x))
    return zscore(score)


def score_morphological_boundary_hvg(counts: np.ndarray, batches: np.ndarray | None, top_k: int, random_state: int) -> np.ndarray:
    del batches, top_k
    x = norm_counts(counts)
    emb = state_embedding(x, random_state)
    states = pseudo_states(emb, random_state)
    n_neighbors = min(16, max(6, x.shape[0] // 45))
    idx = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(emb).kneighbors(return_distance=False)[:, 1:]
    neighbor_states = states[idx]
    boundary_strength = np.mean(neighbor_states != states[:, None], axis=1)
    boundary_mask = boundary_strength >= np.quantile(boundary_strength, 0.75)
    if boundary_mask.sum() < 10:
        boundary_mask = boundary_strength > 0
    if boundary_mask.sum() < 10:
        return zscore(np.var(x, axis=0))
    interior_mask = ~boundary_mask
    if interior_mask.sum() < 10:
        return zscore(np.var(x, axis=0))
    boundary_lift = x[boundary_mask].mean(axis=0) - x[interior_mask].mean(axis=0)
    local_gradient = np.mean(np.abs(x - x[idx].mean(axis=1)), axis=0)
    boundary_corr = np.abs(safe_corr(x, boundary_strength))
    hub_penalty = rank_percentile(x.mean(axis=0)) * rank_percentile((x > 0).mean(axis=0))
    score = 0.65 * zscore(boundary_lift) + 0.65 * zscore(local_gradient) + 0.45 * zscore(boundary_corr) - 0.35 * zscore(hub_penalty)
    return zscore(score)


CANDIDATE_SCORERS: dict[str, Callable[[np.ndarray, np.ndarray | None, int, int], np.ndarray]] = {
    "ot_barycentric_displacement_hvg": score_ot_barycentric_displacement_hvg,
    "ecological_niche_partition_hvg": score_ecological_niche_partition_hvg,
    "mdl_state_code_hvg": score_mdl_state_code_hvg,
    "energy_well_curvature_hvg": score_energy_well_curvature_hvg,
    "morphological_boundary_hvg": score_morphological_boundary_hvg,
}


def choose_device() -> dict[str, object]:
    if torch is None:
        return {"torch_available": False, "cuda_available": False, "selected_device": "cpu", "note": "torch unavailable; numpy/sklearn smoke"}
    if torch.cuda.is_available():
        return {
            "torch_available": True,
            "torch_version": torch.__version__,
            "cuda_available": True,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "selected_device": "cuda_if_registry_uses_torch",
            "note": "CUDA available; repo torch scorers may use GPU, candidate prototypes use CPU numpy/sklearn for smoke reproducibility",
        }
    return {"torch_available": True, "torch_version": torch.__version__, "cuda_available": False, "selected_device": "cpu", "note": "CPU fallback"}


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
    positives = [c for c in ["ari", "nmi", "label_silhouette", "cluster_silhouette", "neighbor_preservation", "stability", "batch_mixing"] if c in df.columns]
    score = np.zeros(len(df), dtype=np.float64)
    for col in positives:
        v = df[col].to_numpy(dtype=np.float64)
        if np.all(np.isnan(v)):
            continue
        vmin = np.nanmin(v)
        vmax = np.nanmax(v)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            score += np.nan_to_num((v - vmin) / (vmax - vmin), nan=0.0)
    if "runtime_sec" in df.columns:
        v = df["runtime_sec"].to_numpy(dtype=np.float64)
        if not np.all(np.isnan(v)):
            vmin = np.nanmin(v)
            vmax = np.nanmax(v)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                score -= 0.12 * np.nan_to_num((v - vmin) / (vmax - vmin), nan=0.0)
    df["overall_score"] = score
    return df


def evaluate_method(dataset, method_name: str, scorer: Callable, top_k: int, seed: int, n_bootstrap: int) -> tuple[dict[str, object], np.ndarray]:
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
    metrics.update({"dataset": dataset.name, "method": method_name, "runtime_sec": elapsed, "top_k": top_k})
    return metrics, scores


def top_overlap(a: np.ndarray, b: np.ndarray, top_k: int) -> tuple[float, float]:
    aa = set(np.argsort(a)[-top_k:].tolist())
    bb = set(np.argsort(b)[-top_k:].tolist())
    inter = len(aa & bb)
    union = len(aa | bb)
    return inter / max(top_k, 1), inter / max(union, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-disciplinary novel HVG mainline brainstorm smoke.")
    parser.add_argument("--output-dir", default="artifacts_crossdisciplinary_taste_smoke_20260424_v1")
    parser.add_argument("--datasets", nargs="+", default=["paul15", "GBM_sd"])
    parser.add_argument("--extended-datasets", nargs="+", default=["FBM_cite", "cellxgene_human_kidney_nonpt"])
    parser.add_argument("--max-cells", type=int, default=900)
    parser.add_argument("--max-genes", type=int, default=2500)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--n-bootstrap", type=int, default=2)
    parser.add_argument("--extended-top-n", type=int, default=2)
    args = parser.parse_args()

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "compute_context.json").write_text(json.dumps(choose_device(), indent=2), encoding="utf-8")
    (out / "literature_and_crossdisciplinary_map.json").write_text(json.dumps(LIT_MAP, indent=2), encoding="utf-8")
    (out / "idea_cards.json").write_text(json.dumps(IDEA_CARDS, indent=2), encoding="utf-8")

    registry = build_default_method_registry(top_k=args.top_k, random_state=args.seed, refine_epochs=0)
    methods: dict[str, Callable] = {name: registry[name] for name in BASELINE_METHODS if name in registry}
    methods.update(CANDIDATE_SCORERS)

    rows = []
    overlaps = []
    dataset_summaries = []
    for dataset_name in args.datasets:
        dataset = load_dataset(DATASETS[dataset_name], args.max_cells, args.max_genes, args.seed)
        top_k = min(args.top_k, dataset.counts.shape[1])
        dataset_summaries.append({
            "dataset": dataset.name,
            "cells": int(dataset.counts.shape[0]),
            "genes": int(dataset.counts.shape[1]),
            "labels": dataset.labels is not None,
            "label_classes": None if dataset.labels is None else int(len(np.unique(dataset.labels))),
            "batches": dataset.batches is not None,
            "top_k": top_k,
        })
        score_cache = {}
        for method_name, scorer in methods.items():
            row, scores = evaluate_method(dataset, method_name, scorer, top_k, args.seed, args.n_bootstrap)
            rows.append(row)
            score_cache[method_name] = scores
        anchor = score_cache["adaptive_hybrid_hvg"]
        for method_name in CANDIDATE_SCORERS:
            overlap, jaccard = top_overlap(anchor, score_cache[method_name], top_k)
            overlaps.append({"dataset": dataset.name, "method": method_name, "anchor_overlap": overlap, "anchor_jaccard": jaccard})

    smoke_df = add_overall_score(pd.DataFrame(rows))
    smoke_df.to_csv(out / "smoke_results.csv", index=False)
    overlap_df = pd.DataFrame(overlaps)
    overlap_df.to_csv(out / "anchor_overlap.csv", index=False)
    (out / "dataset_summary.json").write_text(json.dumps(dataset_summaries, indent=2), encoding="utf-8")

    summary = []
    baseline_set = set(BASELINE_METHODS)
    for method_name in CANDIDATE_SCORERS:
        ranks = []
        deltas_anchor = []
        deltas_best = []
        wins = 0
        metric_wins = 0
        for dataset_name, group in smoke_df.groupby("dataset"):
            ordered = group.sort_values("overall_score", ascending=False).reset_index(drop=True)
            rank = int(ordered.index[ordered["method"] == method_name][0] + 1)
            ranks.append(rank)
            cand = group[group["method"] == method_name].iloc[0]
            anchor = group[group["method"] == "adaptive_hybrid_hvg"].iloc[0]
            best_base_score = float(group[group["method"].isin(baseline_set)]["overall_score"].max())
            deltas_anchor.append(float(cand["overall_score"] - anchor["overall_score"]))
            deltas_best.append(float(cand["overall_score"] - best_base_score))
            if rank == 1:
                wins += 1
            for metric in ["ari", "nmi", "label_silhouette", "cluster_silhouette", "neighbor_preservation", "stability"]:
                if metric in group.columns and float(cand[metric]) >= float(group[group["method"].isin(baseline_set)][metric].max()):
                    metric_wins += 1
        ov = overlap_df[overlap_df["method"] == method_name]
        summary.append({
            "method": method_name,
            "mean_rank": float(np.mean(ranks)),
            "wins": wins,
            "metric_wins": metric_wins,
            "mean_delta_vs_anchor": float(np.mean(deltas_anchor)),
            "mean_delta_vs_best_baseline": float(np.mean(deltas_best)),
            "max_delta_vs_best_baseline": float(np.max(deltas_best)),
            "mean_anchor_overlap": float(ov["anchor_overlap"].mean()),
            "smoke_pass": bool(wins >= 1 or np.max(deltas_best) > 0.10 or metric_wins >= 3 or np.mean(deltas_anchor) > 0.15),
        })
    summary_df = pd.DataFrame(summary).sort_values(
        ["smoke_pass", "wins", "metric_wins", "mean_delta_vs_best_baseline"], ascending=[False, False, False, False]
    )
    summary_df.to_csv(out / "candidate_summary.csv", index=False)

    top_candidates = summary_df[summary_df["smoke_pass"]].head(args.extended_top_n)["method"].tolist()
    ext_rows = []
    if top_candidates:
        ext_methods = {name: CANDIDATE_SCORERS[name] for name in top_candidates}
        ext_methods.update({name: registry[name] for name in ["adaptive_hybrid_hvg", "adaptive_stat_hvg", "multinomial_deviance_hvg", "variance"] if name in registry})
        for dataset_name in args.extended_datasets:
            dataset = load_dataset(DATASETS[dataset_name], args.max_cells, args.max_genes, args.seed)
            top_k = min(args.top_k, dataset.counts.shape[1])
            for method_name, scorer in ext_methods.items():
                row, _ = evaluate_method(dataset, method_name, scorer, top_k, args.seed, args.n_bootstrap)
                ext_rows.append(row)
    ext_df = add_overall_score(pd.DataFrame(ext_rows)) if ext_rows else pd.DataFrame()
    if not ext_df.empty:
        ext_df.to_csv(out / "extended_dataset_results.csv", index=False)

    report = render_report(smoke_df, summary_df, overlap_df, ext_df, dataset_summaries, top_candidates, args)
    (out / "crossdisciplinary_brainstorm_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved artifacts to: {out}")


def render_report(smoke_df, summary_df, overlap_df, ext_df, dataset_summaries, top_candidates, args) -> str:
    lines = ["# Cross-Disciplinary HVG Mainline Brainstorm Smoke", ""]
    lines += ["## Research Scan", ""]
    lines.append("Recent HVG/feature-selection work suggests strong baselines from deviance/count residuals, local-neighborhood methods, rare-cell-aware selection, and manifold-preserving gene panels. The new candidates deliberately avoid repeating the repo's existing adaptive routing, consensus, EB shrinkage, invariance, graph/spectral locality, stability, risk-parity, sigma, and previous single-scorer probes.")
    lines.append("Cross-disciplinary sources used here: optimal transport, ecology, MDL/compression, statistical-physics energy landscapes, and mathematical morphology.")
    lines += ["", "## Excluded Prior Lines", ""]
    lines.append(", ".join(f"`{x}`" for x in EXCLUDED_PRIOR_LINES) + ".")
    lines += ["", "## Candidate Mainlines", ""]
    for name, card in IDEA_CARDS.items():
        row = summary_df[summary_df["method"] == name].iloc[0]
        lines.append(f"### `{name}`")
        lines.append(f"- Taste: {card['taste']}")
        lines.append(f"- Why: {card['why']}")
        lines.append(f"- Mechanism: {card['mechanism']}")
        lines.append(f"- Prediction: {card['prediction']}")
        lines.append(
            f"- Smoke: pass={bool(row['smoke_pass'])}, mean_rank={row['mean_rank']:.2f}, wins={int(row['wins'])}, "
            f"metric_wins={int(row['metric_wins'])}, mean_delta_vs_anchor={row['mean_delta_vs_anchor']:.4f}, "
            f"mean_delta_vs_best_baseline={row['mean_delta_vs_best_baseline']:.4f}, anchor_overlap={row['mean_anchor_overlap']:.3f}."
        )
        lines.append("")
    lines += ["## Smoke Setup", ""]
    for item in dataset_summaries:
        lines.append(f"- `{item['dataset']}`: cells={item['cells']}, genes={item['genes']}, labels={item['labels']}, label_classes={item['label_classes']}, batches={item['batches']}, top_k={item['top_k']}.")
    lines.append(f"- max_cells=`{args.max_cells}`, max_genes=`{args.max_genes}`, n_bootstrap=`{args.n_bootstrap}`, seed=`{args.seed}`.")
    lines += ["", "## Candidate Ranking", ""]
    cols = ["method", "mean_rank", "wins", "metric_wins", "mean_delta_vs_anchor", "mean_delta_vs_best_baseline", "max_delta_vs_best_baseline", "mean_anchor_overlap", "smoke_pass"]
    lines.append(summary_df[cols].round(4).to_markdown(index=False))
    lines += ["", "## Per-Dataset Top Results", ""]
    for dataset_name, group in smoke_df.groupby("dataset"):
        cols2 = [c for c in ["method", "overall_score", "ari", "nmi", "label_silhouette", "cluster_silhouette", "neighbor_preservation", "stability", "runtime_sec"] if c in group.columns]
        lines.append(f"### `{dataset_name}`")
        lines.append(group.sort_values("overall_score", ascending=False)[cols2].head(9).round(4).to_markdown(index=False))
        lines.append("")
    lines += ["## Extended Dataset Test", ""]
    if top_candidates and not ext_df.empty:
        lines.append("Unlocked candidates: " + ", ".join(f"`{x}`" for x in top_candidates) + ".")
        for dataset_name, group in ext_df.groupby("dataset"):
            cols2 = [c for c in ["method", "overall_score", "ari", "nmi", "label_silhouette", "cluster_silhouette", "neighbor_preservation", "stability", "runtime_sec"] if c in group.columns]
            lines.append(f"### `{dataset_name}`")
            lines.append(group.sort_values("overall_score", ascending=False)[cols2].round(4).to_markdown(index=False))
            lines.append("")
    else:
        lines.append("No candidate crossed the extension gate; this is evidence against immediate large-dataset escalation.")
    lines += ["## Recommendation", ""]
    if top_candidates:
        lines.append(f"- Primary follow-up: `{top_candidates[0]}` because it cleared the smoke gate and has the strongest distinct mechanism.")
        lines.append("- Next test should be a benchmarkHVG partial adapter run, not more local metric tuning.")
    else:
        best = summary_df.iloc[0]["method"]
        lines.append(f"- No new mainline is currently strong enough for promotion; the best research-taste probe is `{best}` but it remains below benchmark-backed baselines.")
        lines.append("- The useful scientific result is negative: cross-disciplinary single scorers are interesting diagnostics, but the benchmark still favors count/deviance/statistical anchors.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
