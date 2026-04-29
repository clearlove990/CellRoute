from __future__ import annotations

import time
from typing import Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .baselines import normalize_log1p


def evaluate_selection(
    counts: np.ndarray,
    selected_genes: np.ndarray,
    cell_types: np.ndarray,
    informative_genes: np.ndarray,
    batch_genes: np.ndarray,
    scorer_fn: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
    batches: np.ndarray | None = None,
    n_bootstrap: int = 5,
    top_k: int | None = None,
    random_state: int = 0,
) -> dict:
    if top_k is None:
        top_k = len(selected_genes)

    x = normalize_log1p(counts)[:, selected_genes]
    pca_dim = min(20, x.shape[1], x.shape[0] - 1)
    if pca_dim < 2:
        pca_dim = 2
    embedding = PCA(n_components=pca_dim, random_state=random_state).fit_transform(x)
    pred = KMeans(n_clusters=len(np.unique(cell_types)), n_init=5, random_state=random_state).fit_predict(embedding)

    informative_set = set(informative_genes.tolist())
    batch_set = set(batch_genes.tolist())
    selected_list = selected_genes.tolist()

    recall = np.mean([gene in informative_set for gene in selected_list])
    batch_fraction = np.mean([gene in batch_set for gene in selected_list])

    stability = (
        bootstrap_jaccard(
            counts=counts,
            batches=batches,
            scorer_fn=scorer_fn,
            top_k=top_k,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        if n_bootstrap >= 2
        else float("nan")
    )

    return {
        "ari": adjusted_rand_score(cell_types, pred),
        "nmi": normalized_mutual_info_score(cell_types, pred),
        "informative_recall": recall,
        "batch_gene_fraction": batch_fraction,
        "stability": stability,
    }


def bootstrap_jaccard(
    counts: np.ndarray,
    scorer_fn: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
    batches: np.ndarray | None,
    top_k: int,
    n_bootstrap: int = 5,
    random_state: int = 0,
) -> float:
    rng = np.random.default_rng(random_state)
    selections = []

    for _ in range(n_bootstrap):
        idx = rng.choice(counts.shape[0], size=counts.shape[0], replace=True)
        sampled_batches = None if batches is None else batches[idx]
        scores = scorer_fn(counts[idx], sampled_batches)
        selected = np.argsort(scores)[-top_k:]
        selections.append(set(selected.tolist()))

    pairwise = []
    for i in range(len(selections)):
        for j in range(i + 1, len(selections)):
            inter = len(selections[i] & selections[j])
            union = len(selections[i] | selections[j])
            pairwise.append(inter / max(union, 1))
    return float(np.mean(pairwise)) if pairwise else 0.0


def timed_call(fn: Callable, *args, **kwargs) -> tuple[object, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def evaluate_real_selection(
    counts: np.ndarray,
    selected_genes: np.ndarray,
    scorer_fn: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
    *,
    labels: np.ndarray | None = None,
    batches: np.ndarray | None = None,
    top_k: int | None = None,
    random_state: int = 0,
    n_bootstrap: int = 5,
) -> dict:
    if top_k is None:
        top_k = len(selected_genes)

    x_all = normalize_log1p(counts)
    x_sel = x_all[:, selected_genes]

    full_embedding = _pca_embedding(x_all, random_state=random_state)
    selected_embedding = _pca_embedding(x_sel, random_state=random_state)

    n_clusters = len(np.unique(labels)) if labels is not None else min(8, max(3, int(np.sqrt(counts.shape[0] / 40.0))))
    pred = KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state).fit_predict(selected_embedding)

    metrics = {
        "cluster_silhouette": _silhouette_safe(selected_embedding, pred),
        "neighbor_preservation": neighbor_preservation(
            reference_embedding=full_embedding,
            test_embedding=selected_embedding,
            n_neighbors=min(15, max(5, counts.shape[0] // 30)),
        ),
        "stability": (
            bootstrap_jaccard(
                counts=counts,
                scorer_fn=scorer_fn,
                batches=batches,
                top_k=top_k,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            if n_bootstrap >= 2
            else float("nan")
        ),
    }

    if labels is not None:
        labels = np.asarray(labels)
        metrics["ari"] = adjusted_rand_score(labels, pred)
        metrics["nmi"] = normalized_mutual_info_score(labels, pred)
        metrics["label_silhouette"] = _silhouette_safe(selected_embedding, labels)

    if batches is not None:
        batches = np.asarray(batches)
        batch_sil = _silhouette_safe(selected_embedding, batches)
        metrics["batch_silhouette"] = batch_sil
        metrics["batch_mixing"] = 1.0 - max(batch_sil, 0.0)

    return metrics


def neighbor_preservation(
    *,
    reference_embedding: np.ndarray,
    test_embedding: np.ndarray,
    n_neighbors: int = 15,
) -> float:
    if reference_embedding.shape[0] <= 2:
        return 0.0

    n_neighbors = min(n_neighbors, reference_embedding.shape[0] - 1)
    ref_idx = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(reference_embedding).kneighbors(return_distance=False)[:, 1:]
    test_idx = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(test_embedding).kneighbors(return_distance=False)[:, 1:]

    overlap = []
    for ref_row, test_row in zip(ref_idx, test_idx, strict=False):
        ref_set = set(ref_row.tolist())
        test_set = set(test_row.tolist())
        overlap.append(len(ref_set & test_set) / max(len(ref_set | test_set), 1))
    return float(np.mean(overlap))


def _pca_embedding(x: np.ndarray, random_state: int) -> np.ndarray:
    pca_dim = min(20, x.shape[1], x.shape[0] - 1)
    if pca_dim < 2:
        pca_dim = 2
    return PCA(n_components=pca_dim, random_state=random_state).fit_transform(x)


def _silhouette_safe(embedding: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or embedding.shape[0] <= len(unique):
        return 0.0
    try:
        return float(silhouette_score(embedding, labels))
    except ValueError:
        return 0.0
