from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticSCRNAData:
    counts: np.ndarray
    cell_types: np.ndarray
    batches: np.ndarray
    informative_genes: np.ndarray
    batch_genes: np.ndarray
    scenario: str


def _sample_gamma_poisson(mean_matrix: np.ndarray, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    theta_2d = theta[None, :]
    gamma_rate = rng.gamma(shape=theta_2d, scale=mean_matrix / np.maximum(theta_2d, 1e-6))
    return rng.poisson(gamma_rate).astype(np.int32)


def generate_synthetic_scrna(
    scenario: str,
    n_cells: int = 800,
    n_genes: int = 2000,
    n_cell_types: int = 4,
    n_batches: int = 2,
    random_state: int = 0,
) -> SyntheticSCRNAData:
    rng = np.random.default_rng(random_state)

    cell_types = rng.choice(n_cell_types, size=n_cells, p=_cell_type_probs(n_cell_types))
    batches = rng.integers(0, n_batches, size=n_cells)
    pseudotime = rng.uniform(0.0, 1.0, size=n_cells)

    base_log_mu = rng.normal(loc=-1.5, scale=0.7, size=n_genes)
    gene_dispersion = rng.uniform(4.0, 12.0, size=n_genes)

    informative_target = max(120, n_genes // 10)
    batch_target = max(60, n_genes // 20)
    if informative_target + batch_target > n_genes:
        informative_gene_count = max(10, int(0.55 * n_genes))
        batch_gene_count = max(5, int(0.15 * n_genes))
    else:
        informative_gene_count = informative_target
        batch_gene_count = batch_target

    informative_gene_count = min(informative_gene_count, n_genes)
    batch_gene_count = min(batch_gene_count, max(n_genes - informative_gene_count, 0))
    marker_gene_count = informative_gene_count // 2
    trajectory_gene_count = informative_gene_count - marker_gene_count

    marker_genes = np.arange(marker_gene_count)
    trajectory_genes = np.arange(marker_gene_count, informative_gene_count)
    batch_genes = np.arange(informative_gene_count, informative_gene_count + batch_gene_count)

    log_mu = np.repeat(base_log_mu[None, :], n_cells, axis=0)

    library_scale = rng.lognormal(mean=0.0, sigma=0.35, size=n_cells)
    log_mu += np.log(np.maximum(library_scale[:, None], 1e-6))

    marker_strength = rng.uniform(0.9, 2.2, size=marker_gene_count)
    rare_type = n_cell_types - 1
    for gene_id, effect in zip(marker_genes, marker_strength, strict=False):
        target_type = gene_id % n_cell_types
        if target_type == rare_type:
            effect *= 1.2
        log_mu[cell_types == target_type, gene_id] += effect

    if scenario in {"trajectory", "batch_shift"}:
        slopes = rng.normal(loc=1.6, scale=0.4, size=trajectory_gene_count)
        phase = rng.uniform(0.0, np.pi, size=trajectory_gene_count)
        smooth_signal = (
            pseudotime[:, None] * slopes[None, :]
            + 0.5 * np.sin(2.0 * np.pi * pseudotime[:, None] + phase[None, :])
        )
        log_mu[:, trajectory_genes] += smooth_signal
    else:
        for gene_id in trajectory_genes:
            target_type = (gene_id + 1) % n_cell_types
            log_mu[cell_types == target_type, gene_id] += rng.uniform(0.6, 1.2)

    batch_effect = rng.uniform(1.0, 2.0, size=batch_gene_count)
    if scenario == "batch_shift":
        batch_effect *= 1.5
    for batch_id in range(n_batches):
        direction = 1.0 if batch_id % 2 == 0 else -0.8
        batch_rows = np.where(batches == batch_id)[0]
        log_mu[np.ix_(batch_rows, batch_genes)] += direction * batch_effect[None, :]

    if scenario == "discrete":
        noise_scale = 0.18
    elif scenario == "trajectory":
        noise_scale = 0.14
    else:
        noise_scale = 0.22
    log_mu += rng.normal(loc=0.0, scale=noise_scale, size=log_mu.shape)

    mean_matrix = np.exp(np.clip(log_mu, -6.0, 6.0))
    counts = _sample_gamma_poisson(mean_matrix=mean_matrix, theta=gene_dispersion, rng=rng)

    informative_genes = np.concatenate([marker_genes, trajectory_genes]).astype(np.int64)
    return SyntheticSCRNAData(
        counts=counts,
        cell_types=cell_types.astype(np.int64),
        batches=batches.astype(np.int64),
        informative_genes=informative_genes,
        batch_genes=batch_genes.astype(np.int64),
        scenario=scenario,
    )


def _cell_type_probs(n_cell_types: int) -> np.ndarray:
    probs = np.ones(n_cell_types, dtype=np.float64)
    if n_cell_types >= 4:
        probs[-1] = 0.08
        probs[:-1] = (1.0 - probs[-1]) / (n_cell_types - 1)
    else:
        probs /= probs.sum()
    return probs
