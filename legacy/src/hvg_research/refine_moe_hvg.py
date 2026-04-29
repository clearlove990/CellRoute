from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .baselines import normalize_log1p


EXPERT_NAMES = ("cluster", "trajectory", "batch_robust", "rare", "structure", "general")


def _zscore(x: np.ndarray) -> np.ndarray:
    std = np.std(x)
    if std < 1e-8:
        return np.zeros_like(x, dtype=np.float64)
    return (x - np.mean(x)) / std


def _digitize_gene(expr: np.ndarray, bins: int = 5) -> np.ndarray:
    if np.allclose(expr, expr[0]):
        return np.zeros_like(expr, dtype=np.int64)
    quantiles = np.quantile(expr, np.linspace(0.0, 1.0, bins + 1))
    quantiles[0] -= 1e-6
    quantiles[-1] += 1e-6
    return np.digitize(expr, quantiles[1:-1], right=False)


def _silhouette_safe(embedding: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or embedding.shape[0] < len(unique) + 2:
        return 0.0
    try:
        return float(silhouette_score(embedding, labels))
    except ValueError:
        return 0.0


def _choose_knn_chunk_size(*, n_genes: int, n_neighbors: int, target_mb: int = 192) -> int:
    bytes_per_row = max(n_genes * n_neighbors * 8, 1)
    target_bytes = target_mb * 1024 * 1024
    return max(8, min(512, target_bytes // bytes_per_row))


def _chunked_knn_smoothing(
    *,
    x: np.ndarray,
    knn_idx: np.ndarray,
    gene_centered: np.ndarray,
    gene_std: np.ndarray,
    var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells, n_genes = x.shape
    n_neighbors = max(knn_idx.shape[1], 1)
    chunk_size = _choose_knn_chunk_size(n_genes=n_genes, n_neighbors=n_neighbors)

    smoothed_sum = np.zeros(n_genes, dtype=np.float64)
    smoothed_sq_sum = np.zeros(n_genes, dtype=np.float64)
    cross_sum = np.zeros(n_genes, dtype=np.float64)

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        idx_block = knn_idx[start:end]
        gathered = np.take(x, idx_block.reshape(-1), axis=0)
        gathered = gathered.reshape(end - start, n_neighbors, n_genes)
        smoothed_block = gathered.mean(axis=1)
        smoothed_sum += smoothed_block.sum(axis=0)
        smoothed_sq_sum += np.square(smoothed_block).sum(axis=0)
        cross_sum += (gene_centered[start:end] * smoothed_block).sum(axis=0)

    smoothed_mean = smoothed_sum / max(n_cells, 1)
    smoothed_var = np.clip(smoothed_sq_sum / max(n_cells, 1) - smoothed_mean**2, a_min=0.0, a_max=None)
    smooth_persistence = smoothed_var / np.maximum(var, 1e-6)
    smoothed_std = np.sqrt(smoothed_var)
    local_consistency = (cross_sum / max(n_cells, 1)) / ((gene_std + 1e-6) * (smoothed_std + 1e-6))
    return smooth_persistence, np.clip(local_consistency, -1.0, 1.0)


def _refine_surprise_scale(dataset_stats: dict[str, float]) -> float:
    batch_strength = float(dataset_stats.get("batch_strength", 0.0))
    trajectory_strength = float(dataset_stats.get("trajectory_strength", 0.0))
    cluster_strength = float(dataset_stats.get("cluster_strength", 0.0))
    batch_shift_risk = (
        max(batch_strength - 0.26, 0.0)
        * max(trajectory_strength - 0.68, 0.0)
        / max(cluster_strength + 0.10, 0.10)
    )
    return float(np.clip(1.0 - 24.0 * batch_shift_risk, 0.0, 1.0))


class MaskedDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, bottleneck_dim: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


@dataclass
class RefineMoEHVGSelector:
    top_k: int = 200
    candidate_multiplier: int = 4
    refine_epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 1e-3
    random_state: int = 0
    mode: str = "full"
    gate_model_path: str | None = None
    verbose: bool = False
    last_gate: np.ndarray | None = field(init=False, default=None, repr=False)
    last_gate_source: str | None = field(init=False, default=None, repr=False)
    last_dataset_stats: dict[str, float] | None = field(init=False, default=None, repr=False)
    last_gate_metadata: dict[str, float | str] | None = field(init=False, default=None, repr=False)

    def score_genes(
        self,
        counts: np.ndarray,
        batches: np.ndarray | None = None,
    ) -> np.ndarray:
        x, base_features, dataset_stats, expert_scores = self.prepare_context(counts=counts, batches=batches)
        if self.mode == "learnable_gate_bank":
            resolved_batches = np.zeros(counts.shape[0], dtype=np.int64) if batches is None else np.asarray(batches)
            gate, gate_source, gate_metadata = self._learnable_dataset_gate_bank(
                x=x,
                counts=counts,
                batches=resolved_batches,
                base_features=base_features,
                dataset_stats=dataset_stats,
                expert_scores=expert_scores,
                reliability_aware=False,
                decision_variant="bank",
            )
        elif self.mode == "learnable_gate_bank_reliable_refine":
            resolved_batches = np.zeros(counts.shape[0], dtype=np.int64) if batches is None else np.asarray(batches)
            gate, gate_source, gate_metadata = self._learnable_dataset_gate_bank(
                x=x,
                counts=counts,
                batches=resolved_batches,
                base_features=base_features,
                dataset_stats=dataset_stats,
                expert_scores=expert_scores,
                reliability_aware=True,
                decision_variant="reliable_refine",
            )
        elif self.mode in {
            "learnable_gate_bank_curr_refine",
            "learnable_gate_bank_curr_no_risk",
            "learnable_gate_bank_curr_no_regret",
            "learnable_gate_bank_curr_no_refine_policy",
            "learnable_gate_bank_curr_utility_only",
            "learnable_gate_bank_pairregret",
            "ablation_pairregret_no_risk",
            "ablation_pairregret_no_regret",
            "ablation_pairregret_no_pairwise_term",
            "ablation_pairregret_utility_only",
            "ablation_pairregret_no_conservative_routing",
            "ablation_pairregret_refine_on",
            "learnable_gate_bank_pairregret_calibrated",
            "ablation_pairregret_cal_no_regret",
            "ablation_pairregret_cal_no_consistency",
            "ablation_pairregret_cal_no_route_constraint",
            "ablation_pairregret_cal_utility_only",
            "ablation_pairregret_cal_no_conservative_routing",
            "ablation_pairregret_cal_refine_on",
            "learnable_gate_bank_pairregret_permissioned",
            "ablation_pairperm_no_permission_head",
            "ablation_pairperm_fixed_budget",
            "ablation_pairperm_no_anchor_adaptation",
            "ablation_pairperm_no_permission_value_decoupling",
            "ablation_pairperm_no_regret_aux",
            "ablation_pairperm_permission_only",
            "ablation_pairperm_refine_on",
            "learnable_gate_bank_pairregret_permissioned_escapecert",
            "ablation_escapecert_no_anchor_escape_head",
            "ablation_escapecert_no_set_supervision",
            "ablation_escapecert_no_candidate_admissibility",
            "ablation_escapecert_fixed_budget",
            "ablation_escapecert_no_escape_uncertainty",
            "ablation_escapecert_no_regret_aux",
            "ablation_escapecert_anchor_escape_only",
            "ablation_escapecert_admissibility_only",
            "learnable_gate_bank_pairregret_permissioned_escapecert_frontier",
            "ablation_frontier_no_teacher_distill",
            "ablation_frontier_no_frontier_head",
            "ablation_frontier_no_frontier_uncertainty",
            "ablation_frontier_fixed_frontier",
            "ablation_frontier_no_anchor_escape_head",
            "ablation_frontier_no_regret_aux",
            "ablation_frontier_teacher_only",
            "ablation_frontier_no_teacher_frontier_only",
        }:
            resolved_batches = np.zeros(counts.shape[0], dtype=np.int64) if batches is None else np.asarray(batches)
            gate, gate_source, gate_metadata = self._learnable_dataset_gate_bank(
                x=x,
                counts=counts,
                batches=resolved_batches,
                base_features=base_features,
                dataset_stats=dataset_stats,
                expert_scores=expert_scores,
                reliability_aware=True,
                decision_variant=self.mode,
            )
        else:
            gate, gate_source = self.resolve_gate(dataset_stats)
            gate_metadata = {}

        self.last_gate = gate.copy()
        self.last_gate_source = gate_source
        self.last_dataset_stats = dict(dataset_stats)
        self.last_gate_metadata = dict(gate_metadata)

        pairregret_no_refine_modes = {
            "learnable_gate_bank_pairregret",
            "ablation_pairregret_no_risk",
            "ablation_pairregret_no_regret",
            "ablation_pairregret_no_pairwise_term",
            "ablation_pairregret_utility_only",
            "ablation_pairregret_no_conservative_routing",
            "learnable_gate_bank_pairregret_calibrated",
            "ablation_pairregret_cal_no_regret",
            "ablation_pairregret_cal_no_consistency",
            "ablation_pairregret_cal_no_route_constraint",
            "ablation_pairregret_cal_utility_only",
            "ablation_pairregret_cal_no_conservative_routing",
            "learnable_gate_bank_pairregret_permissioned",
            "ablation_pairperm_no_permission_head",
            "ablation_pairperm_fixed_budget",
            "ablation_pairperm_no_anchor_adaptation",
            "ablation_pairperm_no_permission_value_decoupling",
            "ablation_pairperm_no_regret_aux",
            "ablation_pairperm_permission_only",
            "learnable_gate_bank_pairregret_permissioned_escapecert",
            "ablation_escapecert_no_anchor_escape_head",
            "ablation_escapecert_no_set_supervision",
            "ablation_escapecert_no_candidate_admissibility",
            "ablation_escapecert_fixed_budget",
            "ablation_escapecert_no_escape_uncertainty",
            "ablation_escapecert_no_regret_aux",
            "ablation_escapecert_anchor_escape_only",
            "ablation_escapecert_admissibility_only",
            "learnable_gate_bank_pairregret_permissioned_escapecert_frontier",
            "ablation_frontier_no_teacher_distill",
            "ablation_frontier_no_frontier_head",
            "ablation_frontier_no_frontier_uncertainty",
            "ablation_frontier_fixed_frontier",
            "ablation_frontier_no_anchor_escape_head",
            "ablation_frontier_no_regret_aux",
            "ablation_frontier_teacher_only",
            "ablation_frontier_no_teacher_frontier_only",
        }
        refine_metadata_modes = {
            "learnable_gate_bank_reliable_refine",
            "learnable_gate_bank_curr_refine",
            "learnable_gate_bank_curr_no_risk",
            "learnable_gate_bank_curr_no_regret",
            "learnable_gate_bank_curr_no_refine_policy",
            "learnable_gate_bank_curr_utility_only",
            "ablation_pairregret_refine_on",
            "ablation_pairregret_cal_refine_on",
            "ablation_pairperm_refine_on",
        }
        return self.score_with_context(
            x=x,
            base_features=base_features,
            dataset_stats=dataset_stats,
            expert_scores=expert_scores,
            gate=gate,
            apply_refine=self.mode != "no_refine" and self.mode not in pairregret_no_refine_modes,
            refine_metadata=gate_metadata
            if self.mode in refine_metadata_modes
            else None,
        )

    def score_genes_with_fixed_gate(
        self,
        counts: np.ndarray,
        batches: np.ndarray | None,
        gate: np.ndarray,
        *,
        apply_refine: bool = False,
    ) -> np.ndarray:
        x, base_features, dataset_stats, expert_scores = self.prepare_context(counts=counts, batches=batches)
        return self.score_with_context(
            x=x,
            base_features=base_features,
            dataset_stats=dataset_stats,
            expert_scores=expert_scores,
            gate=np.asarray(gate, dtype=np.float64),
            apply_refine=apply_refine,
        )

    def prepare_context(
        self,
        counts: np.ndarray,
        batches: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
        x = normalize_log1p(counts)
        if batches is None:
            batches = np.zeros(counts.shape[0], dtype=np.int64)
        batches = np.asarray(batches)

        base_features, dataset_stats = self._compute_features(x=x, counts=counts, batches=batches)
        expert_scores = self._expert_scores(base_features)
        return x, base_features, dataset_stats, expert_scores

    def resolve_gate(self, dataset_stats: dict[str, float]) -> tuple[np.ndarray, str]:
        n_experts = len(EXPERT_NAMES)
        uniform_gate = np.ones(n_experts, dtype=np.float64) / n_experts

        if self.mode == "no_moe":
            return uniform_gate, "uniform"

        if self.mode == "learnable_gate":
            return self._learnable_dataset_gate(dataset_stats), "learned"

        routed_gate = self._dataset_gate(dataset_stats)
        gate = 0.40 * routed_gate + 0.60 * uniform_gate
        return gate, "heuristic_blend"

    def score_with_context(
        self,
        *,
        x: np.ndarray,
        base_features: dict[str, np.ndarray],
        dataset_stats: dict[str, float],
        expert_scores: dict[str, np.ndarray],
        gate: np.ndarray,
        apply_refine: bool = True,
        refine_metadata: dict[str, float] | None = None,
    ) -> np.ndarray:
        gate = np.asarray(gate, dtype=np.float64)
        gate = np.clip(gate, 1e-8, None)
        gate = gate / gate.sum()

        combined = np.zeros(x.shape[1], dtype=np.float64)
        for weight, expert_name in zip(gate, EXPERT_NAMES, strict=False):
            combined += weight * expert_scores[expert_name]
        combined -= (0.20 + 0.40 * dataset_stats["batch_strength"]) * base_features["batch_mi"]

        candidate_k = min(self.top_k * self.candidate_multiplier, x.shape[1])
        candidate_idx = np.argsort(combined)[-candidate_k:]
        final_score = combined.copy()

        refine_scale = _refine_surprise_scale(dataset_stats)
        if not apply_refine or candidate_idx.size == 0 or refine_scale <= 1e-6:
            return final_score

        reliability = 1.0
        transfer_risk = 0.0
        refine_support = 1.0
        boundary_batch_risk = 0.0
        policy_refine_intensity: float | None = None
        if refine_metadata is not None:
            reliability = float(np.clip(refine_metadata.get("bank_reliability", 1.0), 0.0, 1.0))
            transfer_risk = float(np.clip(refine_metadata.get("bank_transfer_risk", 0.0), 0.0, 1.0))
            refine_support = float(np.clip(refine_metadata.get("bank_refine_support", 1.0), 0.0, 1.0))
            boundary_batch_risk = float(np.clip(refine_metadata.get("bank_boundary_batch_risk", 0.0), 0.0, 1.0))
            if "policy_refine_intensity" in refine_metadata:
                policy_refine_intensity = float(np.clip(refine_metadata.get("policy_refine_intensity", 0.0), 0.0, 1.0))

        if policy_refine_intensity is not None:
            safe_refine_scale = refine_scale * policy_refine_intensity
        else:
            safe_refine_scale = refine_scale * (
                0.30 + 0.70 * reliability * refine_support * (1.0 - 0.55 * transfer_risk)
            )
        if safe_refine_scale <= 1e-6:
            return final_score

        surprise = self._masked_surprise_refinement(x[:, candidate_idx])
        surprise = _zscore(surprise)
        surprise -= (0.35 + 0.35 * dataset_stats["batch_strength"]) * base_features["batch_mi"][candidate_idx]

        ranked_candidates = candidate_idx[np.argsort(combined[candidate_idx])]
        boundary_width = int((1.0 + 0.6 * reliability) * self.top_k)
        boundary_start = max(0, ranked_candidates.size - boundary_width)
        boundary_idx = ranked_candidates[boundary_start:]
        boundary_mask = np.isin(candidate_idx, boundary_idx)

        safe_support = (
            0.36 * base_features["local_consistency"][candidate_idx]
            + 0.26 * base_features["cluster_contrast"][candidate_idx]
            + 0.18 * base_features["smooth_persistence"][candidate_idx]
            + 0.12 * base_features["residual"][candidate_idx]
            - (0.28 + 0.22 * boundary_batch_risk) * base_features["batch_mi"][candidate_idx]
        )
        safe_source = safe_support[boundary_mask] if np.any(boundary_mask) else safe_support
        safe_threshold = np.quantile(safe_source, max(0.20, 0.55 - 0.30 * reliability))
        batch_threshold = np.quantile(
            base_features["batch_mi"][candidate_idx],
            max(0.20, 0.55 - 0.25 * reliability * (1.0 - 0.5 * transfer_risk)),
        )

        safe_mask = boundary_mask.copy()
        safe_mask &= base_features["batch_mi"][candidate_idx] <= batch_threshold
        safe_mask &= safe_support >= safe_threshold

        final_score[candidate_idx] += safe_refine_scale * 0.04 * surprise
        final_score[candidate_idx[safe_mask]] += safe_refine_scale * (0.08 + 0.04 * reliability) * surprise[safe_mask]
        return final_score

    def select(
        self,
        counts: np.ndarray,
        batches: np.ndarray | None = None,
    ) -> np.ndarray:
        scores = self.score_genes(counts=counts, batches=batches)
        return np.argsort(scores)[-self.top_k:]

    def _compute_features(
        self,
        *,
        x: np.ndarray,
        counts: np.ndarray,
        batches: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        n_cells, n_genes = x.shape

        pca_dim = min(20, n_genes, n_cells - 1)
        if pca_dim < 2:
            pca_dim = 2
        pca_model = PCA(n_components=pca_dim, random_state=self.random_state)
        embedding = pca_model.fit_transform(x)
        n_clusters = min(6, max(3, int(np.sqrt(n_cells / 40.0))))
        pseudo_labels = KMeans(n_clusters=n_clusters, n_init=5, random_state=self.random_state).fit_predict(embedding)

        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        dispersion = var / np.maximum(mean, 1e-6)
        residual = _mean_var_residual_from_log_expr(x)

        cluster_mi = np.zeros(n_genes, dtype=np.float64)
        batch_mi = np.zeros(n_genes, dtype=np.float64)
        rare_tail = np.zeros(n_genes, dtype=np.float64)

        cluster_sizes = np.bincount(pseudo_labels)
        rare_cluster = int(np.argmin(cluster_sizes))
        rare_mask = pseudo_labels == rare_cluster

        for gene_id in range(n_genes):
            bins = _digitize_gene(x[:, gene_id], bins=5)
            cluster_mi[gene_id] = mutual_info_score(bins, pseudo_labels)
            batch_mi[gene_id] = mutual_info_score(bins, batches)
            rare_tail[gene_id] = x[rare_mask, gene_id].mean() - x[~rare_mask, gene_id].mean()

        pc1 = embedding[:, 0]
        pc1_centered = pc1 - pc1.mean()
        pc1_std = np.std(pc1_centered) + 1e-6
        gene_centered = x - x.mean(axis=0, keepdims=True)
        gene_std = x.std(axis=0)
        corr_pc1 = np.abs((pc1_centered[:, None] * gene_centered).mean(axis=0) / (pc1_std * (gene_std + 1e-6)))

        n_neighbors = min(15, max(5, n_cells // 30))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embedding[:, : min(10, embedding.shape[1])])
        knn_idx = nbrs.kneighbors(return_distance=False)
        smooth_persistence, local_consistency = _chunked_knn_smoothing(
            x=x,
            knn_idx=knn_idx,
            gene_centered=gene_centered,
            gene_std=gene_std,
            var=var,
        )

        cluster_means = []
        for cluster_id in range(n_clusters):
            cluster_means.append(x[pseudo_labels == cluster_id].mean(axis=0))
        cluster_mean_matrix = np.vstack(cluster_means)
        cluster_contrast = np.var(cluster_mean_matrix, axis=0) / np.maximum(var, 1e-6)

        batch_sil = _silhouette_safe(embedding, batches)
        cluster_sil = _silhouette_safe(embedding, pseudo_labels)
        explained = pca_model.explained_variance_ratio_[: min(5, len(pca_model.explained_variance_ratio_))]
        explained_sum = max(explained.sum(), 1e-6)
        explained_norm = explained / explained_sum
        trajectory_strength = float(explained[0] / max(explained[:2].sum(), 1e-6))
        rare_fraction = float(cluster_sizes.min() / cluster_sizes.sum())

        library = counts.sum(axis=1).astype(np.float64)
        library_mean = max(library.mean(), 1.0)
        library_cv = float(library.std() / library_mean)
        pc_entropy = float(-(explained_norm * np.log(explained_norm + 1e-8)).sum() / np.log(len(explained_norm) + 1e-8))
        batch_count_norm = float((len(np.unique(batches)) - 1) / max(len(np.unique(batches)), 1))

        features = {
            "dispersion": _zscore(np.log1p(dispersion)),
            "residual": _zscore(residual),
            "cluster_mi": _zscore(cluster_mi),
            "batch_mi": _zscore(batch_mi),
            "rare_tail": _zscore(rare_tail),
            "corr_pc1": _zscore(corr_pc1),
            "smooth_persistence": _zscore(smooth_persistence),
            "local_consistency": _zscore(np.maximum(local_consistency, 0.0)),
            "cluster_contrast": _zscore(cluster_contrast),
        }

        dataset_stats = {
            "batch_strength": max(batch_sil, 0.0),
            "cluster_strength": max(cluster_sil, 0.0),
            "trajectory_strength": max(trajectory_strength, 0.0),
            "rare_fraction": rare_fraction,
            "dropout_rate": float(np.mean(x < 1e-3)),
            "library_cv": library_cv,
            "pc_entropy": pc_entropy,
            "n_cells_log": float(np.log1p(n_cells)),
            "n_genes_log": float(np.log1p(n_genes)),
            "batch_count_norm": batch_count_norm,
        }
        return features, dataset_stats

    def _expert_scores(self, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        cluster = 0.35 * features["dispersion"] + 0.40 * features["cluster_mi"] + 0.25 * features["rare_tail"]
        trajectory = 0.35 * features["corr_pc1"] + 0.35 * features["smooth_persistence"] + 0.30 * features["residual"]
        batch_robust = 0.50 * features["residual"] + 0.25 * features["cluster_mi"] - 0.45 * features["batch_mi"]
        rare = 0.50 * features["rare_tail"] + 0.30 * features["cluster_mi"] + 0.20 * features["residual"]
        structure = (
            0.30 * features["cluster_contrast"]
            + 0.25 * features["local_consistency"]
            + 0.20 * features["smooth_persistence"]
            + 0.15 * features["cluster_mi"]
            + 0.10 * features["residual"]
            - 0.25 * features["batch_mi"]
        )
        general = 0.65 * features["residual"] + 0.35 * features["dispersion"]
        return {
            "cluster": cluster,
            "trajectory": trajectory,
            "batch_robust": batch_robust,
            "rare": rare,
            "structure": structure,
            "general": general,
        }

    def _dataset_gate(self, dataset_stats: dict[str, float]) -> np.ndarray:
        cluster_strength = dataset_stats["cluster_strength"]
        trajectory_strength = dataset_stats["trajectory_strength"]
        batch_strength = dataset_stats["batch_strength"]
        rare_bonus = max(0.18 - dataset_stats["rare_fraction"], 0.0) * 6.0

        logits = np.array(
            [
                1.2 * cluster_strength - 0.4 * trajectory_strength,
                1.5 * trajectory_strength - 0.3 * cluster_strength,
                1.8 * batch_strength,
                1.4 * rare_bonus,
                0.9 * cluster_strength + 1.0 * trajectory_strength - 0.15 * batch_strength + 0.2,
                0.45,
            ],
            dtype=np.float64,
        )
        return softmax(logits)

    def _learnable_dataset_gate(self, dataset_stats: dict[str, float]) -> np.ndarray:
        if self.gate_model_path is None:
            raise ValueError("mode='learnable_gate' requires gate_model_path to be set.")

        gate_path = Path(self.gate_model_path)
        if not gate_path.exists():
            raise FileNotFoundError(f"Learnable gate checkpoint not found: {gate_path}")

        from .gate_learning import predict_gate_weights

        heuristic_gate = 0.40 * self._dataset_gate(dataset_stats) + 0.60 * (
            np.ones(len(EXPERT_NAMES), dtype=np.float64) / len(EXPERT_NAMES)
        )
        return predict_gate_weights(
            dataset_stats=dataset_stats,
            checkpoint_path=str(gate_path),
            heuristic_gate=heuristic_gate,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def _learnable_dataset_gate_bank(
        self,
        *,
        x: np.ndarray,
        counts: np.ndarray,
        batches: np.ndarray,
        base_features: dict[str, np.ndarray],
        dataset_stats: dict[str, float],
        expert_scores: dict[str, np.ndarray],
        reliability_aware: bool,
        decision_variant: str = "bank",
    ) -> tuple[np.ndarray, str, dict[str, float]]:
        if self.gate_model_path is None:
            raise ValueError("mode='learnable_gate_bank' requires gate_model_path to be set.")

        gate_path = Path(self.gate_model_path)
        if not gate_path.exists():
            raise FileNotFoundError(f"Learnable gate checkpoint not found: {gate_path}")

        from .gate_learning import (
            select_counterfactual_gate_bank_weights,
            select_gate_bank_weights,
            select_stage1_anchored_pairwise_gate_bank_weights,
            select_stage1_anchored_pairwise_gate_bank_weights_calibrated,
            select_stage1_anchored_pairwise_gate_bank_weights_escapecert,
            select_stage1_anchored_pairwise_gate_bank_weights_frontier,
            select_stage1_anchored_pairwise_gate_bank_weights_permissioned,
        )

        heuristic_gate = 0.40 * self._dataset_gate(dataset_stats) + 0.60 * (
            np.ones(len(EXPERT_NAMES), dtype=np.float64) / len(EXPERT_NAMES)
        )
        if decision_variant in {
            "learnable_gate_bank_curr_refine",
            "learnable_gate_bank_curr_no_risk",
            "learnable_gate_bank_curr_no_regret",
            "learnable_gate_bank_curr_no_refine_policy",
            "learnable_gate_bank_curr_utility_only",
        }:
            variant_map = {
                "learnable_gate_bank_curr_refine": "full",
                "learnable_gate_bank_curr_no_risk": "no_risk",
                "learnable_gate_bank_curr_no_regret": "no_regret",
                "learnable_gate_bank_curr_no_refine_policy": "no_refine_policy",
                "learnable_gate_bank_curr_utility_only": "utility_only",
            }
            return select_counterfactual_gate_bank_weights(
                x=x,
                counts=counts,
                batches=batches,
                dataset_stats=dataset_stats,
                base_features=base_features,
                expert_scores=expert_scores,
                checkpoint_path=str(gate_path),
                heuristic_gate=heuristic_gate,
                selector=self,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                variant=variant_map[decision_variant],
            )
        if decision_variant in {
            "learnable_gate_bank_pairregret",
            "ablation_pairregret_no_risk",
            "ablation_pairregret_no_regret",
            "ablation_pairregret_no_pairwise_term",
            "ablation_pairregret_utility_only",
            "ablation_pairregret_no_conservative_routing",
            "ablation_pairregret_refine_on",
        }:
            variant_map = {
                "learnable_gate_bank_pairregret": "full",
                "ablation_pairregret_no_risk": "no_risk",
                "ablation_pairregret_no_regret": "no_regret",
                "ablation_pairregret_no_pairwise_term": "no_pairwise_term",
                "ablation_pairregret_utility_only": "utility_only",
                "ablation_pairregret_no_conservative_routing": "no_conservative_routing",
                "ablation_pairregret_refine_on": "refine_on",
            }
            return select_stage1_anchored_pairwise_gate_bank_weights(
                x=x,
                counts=counts,
                batches=batches,
                dataset_stats=dataset_stats,
                base_features=base_features,
                expert_scores=expert_scores,
                checkpoint_path=str(gate_path),
                heuristic_gate=heuristic_gate,
                selector=self,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                variant=variant_map[decision_variant],
            )
        if decision_variant in {
            "learnable_gate_bank_pairregret_calibrated",
            "ablation_pairregret_cal_no_regret",
            "ablation_pairregret_cal_no_consistency",
            "ablation_pairregret_cal_no_route_constraint",
            "ablation_pairregret_cal_utility_only",
            "ablation_pairregret_cal_no_conservative_routing",
            "ablation_pairregret_cal_refine_on",
        }:
            variant_map = {
                "learnable_gate_bank_pairregret_calibrated": "full",
                "ablation_pairregret_cal_no_regret": "no_regret_calibration",
                "ablation_pairregret_cal_no_consistency": "no_consistency_calibration",
                "ablation_pairregret_cal_no_route_constraint": "no_route_constraint",
                "ablation_pairregret_cal_utility_only": "utility_only",
                "ablation_pairregret_cal_no_conservative_routing": "no_conservative_routing",
                "ablation_pairregret_cal_refine_on": "refine_on",
            }
            return select_stage1_anchored_pairwise_gate_bank_weights_calibrated(
                x=x,
                counts=counts,
                batches=batches,
                dataset_stats=dataset_stats,
                base_features=base_features,
                expert_scores=expert_scores,
                checkpoint_path=str(gate_path),
                heuristic_gate=heuristic_gate,
                selector=self,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                variant=variant_map[decision_variant],
            )
        if decision_variant in {
            "learnable_gate_bank_pairregret_permissioned",
            "ablation_pairperm_no_permission_head",
            "ablation_pairperm_fixed_budget",
            "ablation_pairperm_no_anchor_adaptation",
            "ablation_pairperm_no_permission_value_decoupling",
            "ablation_pairperm_no_regret_aux",
            "ablation_pairperm_permission_only",
            "ablation_pairperm_refine_on",
        }:
            variant_map = {
                "learnable_gate_bank_pairregret_permissioned": "full",
                "ablation_pairperm_no_permission_head": "no_permission_head",
                "ablation_pairperm_fixed_budget": "fixed_budget",
                "ablation_pairperm_no_anchor_adaptation": "no_anchor_adaptation",
                "ablation_pairperm_no_permission_value_decoupling": "no_permission_value_decoupling",
                "ablation_pairperm_no_regret_aux": "no_regret_aux",
                "ablation_pairperm_permission_only": "permission_only",
                "ablation_pairperm_refine_on": "refine_on",
            }
            return select_stage1_anchored_pairwise_gate_bank_weights_permissioned(
                x=x,
                counts=counts,
                batches=batches,
                dataset_stats=dataset_stats,
                base_features=base_features,
                expert_scores=expert_scores,
                checkpoint_path=str(gate_path),
                heuristic_gate=heuristic_gate,
                selector=self,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                variant=variant_map[decision_variant],
            )
        if decision_variant in {
            "learnable_gate_bank_pairregret_permissioned_escapecert",
            "ablation_escapecert_no_anchor_escape_head",
            "ablation_escapecert_no_set_supervision",
            "ablation_escapecert_no_candidate_admissibility",
            "ablation_escapecert_fixed_budget",
            "ablation_escapecert_no_escape_uncertainty",
            "ablation_escapecert_no_regret_aux",
            "ablation_escapecert_anchor_escape_only",
            "ablation_escapecert_admissibility_only",
        }:
            variant_map = {
                "learnable_gate_bank_pairregret_permissioned_escapecert": "full",
                "ablation_escapecert_no_anchor_escape_head": "no_anchor_escape_head",
                "ablation_escapecert_no_set_supervision": "no_set_supervision",
                "ablation_escapecert_no_candidate_admissibility": "no_candidate_admissibility",
                "ablation_escapecert_fixed_budget": "fixed_budget",
                "ablation_escapecert_no_escape_uncertainty": "no_escape_uncertainty",
                "ablation_escapecert_no_regret_aux": "no_regret_aux",
                "ablation_escapecert_anchor_escape_only": "anchor_escape_only",
                "ablation_escapecert_admissibility_only": "admissibility_only",
            }
            return select_stage1_anchored_pairwise_gate_bank_weights_escapecert(
                x=x,
                counts=counts,
                batches=batches,
                dataset_stats=dataset_stats,
                base_features=base_features,
                expert_scores=expert_scores,
                checkpoint_path=str(gate_path),
                heuristic_gate=heuristic_gate,
                selector=self,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                variant=variant_map[decision_variant],
            )
        if decision_variant in {
            "learnable_gate_bank_pairregret_permissioned_escapecert_frontier",
            "ablation_frontier_no_teacher_distill",
            "ablation_frontier_no_frontier_head",
            "ablation_frontier_no_frontier_uncertainty",
            "ablation_frontier_fixed_frontier",
            "ablation_frontier_no_anchor_escape_head",
            "ablation_frontier_no_regret_aux",
            "ablation_frontier_teacher_only",
            "ablation_frontier_no_teacher_frontier_only",
        }:
            variant_map = {
                "learnable_gate_bank_pairregret_permissioned_escapecert_frontier": "full",
                "ablation_frontier_no_teacher_distill": "no_teacher_distill",
                "ablation_frontier_no_frontier_head": "no_frontier_head",
                "ablation_frontier_no_frontier_uncertainty": "no_frontier_uncertainty",
                "ablation_frontier_fixed_frontier": "fixed_frontier",
                "ablation_frontier_no_anchor_escape_head": "no_anchor_escape_head",
                "ablation_frontier_no_regret_aux": "no_regret_aux",
                "ablation_frontier_teacher_only": "teacher_only",
                "ablation_frontier_no_teacher_frontier_only": "no_teacher_frontier_only",
            }
            return select_stage1_anchored_pairwise_gate_bank_weights_frontier(
                x=x,
                counts=counts,
                batches=batches,
                dataset_stats=dataset_stats,
                base_features=base_features,
                expert_scores=expert_scores,
                checkpoint_path=str(gate_path),
                heuristic_gate=heuristic_gate,
                selector=self,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                variant=variant_map[decision_variant],
            )
        return select_gate_bank_weights(
            x=x,
            counts=counts,
            batches=batches,
            dataset_stats=dataset_stats,
            base_features=base_features,
            expert_scores=expert_scores,
            checkpoint_path=str(gate_path),
            heuristic_gate=heuristic_gate,
            selector=self,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            reliability_aware=reliability_aware,
        )

    def _masked_surprise_refinement(self, candidate_x: np.ndarray) -> np.ndarray:
        torch.manual_seed(self.random_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x_tensor = torch.from_numpy(candidate_x.astype(np.float32))
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True, drop_last=False)

        model = MaskedDenoisingAutoencoder(input_dim=candidate_x.shape[1])
        if torch.cuda.is_available():
            model = model.to(device)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        else:
            model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        for _ in range(self.refine_epochs):
            model.train()
            for (batch_x,) in loader:
                batch_x = batch_x.to(device, non_blocking=torch.cuda.is_available())
                mask = torch.rand_like(batch_x) < 0.15
                masked_input = batch_x.masked_fill(mask, 0.0)
                pred = model(masked_input)
                loss_matrix = (pred - batch_x) ** 2
                masked_loss = loss_matrix[mask]
                if masked_loss.numel() == 0:
                    loss = loss_matrix.mean()
                else:
                    loss = masked_loss.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        model.eval()
        eval_loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=False, drop_last=False)
        error_sum = torch.zeros(candidate_x.shape[1], device=device)
        count_sum = torch.zeros(candidate_x.shape[1], device=device)

        with torch.no_grad():
            for (batch_x,) in eval_loader:
                batch_x = batch_x.to(device, non_blocking=torch.cuda.is_available())
                mask = torch.rand_like(batch_x) < 0.20
                masked_input = batch_x.masked_fill(mask, 0.0)
                pred = model(masked_input)
                loss_matrix = (pred - batch_x) ** 2
                error_sum += (loss_matrix * mask).sum(dim=0)
                count_sum += mask.sum(dim=0)

        surprise = error_sum / torch.clamp(count_sum, min=1.0)
        return surprise.detach().cpu().numpy()


def _mean_var_residual_from_log_expr(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    log_mean = np.log1p(mean)
    log_var = np.log1p(var)
    coeff = np.polyfit(log_mean, log_var, deg=2)
    expected = np.polyval(coeff, log_mean)
    return _zscore(log_var - expected)
