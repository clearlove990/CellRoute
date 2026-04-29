from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from .baselines import (
    normalize_log1p,
    score_analytic_pearson_residual_hvg,
    score_multinomial_deviance_hvg,
)


@dataclass(frozen=True)
class AdaptiveStatProfile:
    n_cells: int
    n_genes: int
    batch_classes: int
    dropout_rate: float
    library_cv: float
    cluster_strength: float
    batch_strength: float
    trajectory_strength: float
    pc_entropy: float
    rare_fraction: float


@dataclass(frozen=True)
class AdaptiveStatDecision:
    route_name: str
    variance_weight: float
    mv_residual_weight: float
    fano_weight: float


@dataclass(frozen=True)
class AdaptiveHybridDecision:
    route_name: str
    route_target: str
    fallback_target: str | None = None


def score_adaptive_stat_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, AdaptiveStatDecision]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    decision = resolve_adaptive_stat_decision(profile)
    blended = score_adaptive_stat_hvg_with_decision(counts, decision)
    return blended, profile, decision


def score_adaptive_stat_hvg_with_decision(
    counts: np.ndarray,
    decision: AdaptiveStatDecision,
) -> np.ndarray:
    x = normalize_log1p(counts)
    variance_scores, mv_scores, fano_scores = _compute_base_scores_from_normalized(x)
    weights = np.asarray(
        [
            decision.variance_weight,
            decision.mv_residual_weight,
            decision.fano_weight,
        ],
        dtype=np.float64,
    )
    blended = (
        weights[0] * variance_scores
        + weights[1] * mv_scores
        + weights[2] * fano_scores
    )
    return blended


def compute_adaptive_stat_profile(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    random_state: int = 0,
    max_profile_cells: int = 1500,
    max_profile_genes: int = 3000,
) -> AdaptiveStatProfile:
    n_cells, n_genes = counts.shape
    rng = np.random.default_rng(random_state)
    counts_eval = counts

    if n_cells > max_profile_cells:
        cell_idx = np.sort(rng.choice(n_cells, size=max_profile_cells, replace=False))
        counts_eval = counts[cell_idx]
        batches_eval = None if batches is None else np.asarray(batches)[cell_idx]
    else:
        counts_eval = counts
        batches_eval = None if batches is None else np.asarray(batches)

    if counts_eval.shape[1] > max_profile_genes:
        gene_var = np.var(counts_eval.astype(np.float32, copy=False), axis=0)
        gene_idx = np.argsort(gene_var)[-max_profile_genes:]
        counts_eval = counts_eval[:, np.sort(gene_idx)]

    x_eval = _normalize_log1p_fast(counts_eval)
    if x_eval.shape[0] < 2 or x_eval.shape[1] < 2:
        library = counts.sum(axis=1).astype(np.float64)
        library_cv = float(library.std() / max(library.mean(), 1.0))
        dropout_rate = float(np.mean(counts <= 0))
        return AdaptiveStatProfile(
            n_cells=int(n_cells),
            n_genes=int(n_genes),
            batch_classes=0 if batches_eval is None else int(len(np.unique(batches_eval))),
            dropout_rate=dropout_rate,
            library_cv=library_cv,
            cluster_strength=0.0,
            batch_strength=0.0,
            trajectory_strength=0.0,
            pc_entropy=0.0,
            rare_fraction=1.0,
        )

    pca_dim = min(10, x_eval.shape[0], x_eval.shape[1])
    pca_model = PCA(n_components=pca_dim, svd_solver="randomized", random_state=random_state)
    embedding = pca_model.fit_transform(x_eval)

    n_clusters = min(6, max(2, int(np.sqrt(len(x_eval) / 40.0))))
    n_clusters = min(n_clusters, x_eval.shape[0] - 1)
    if n_clusters < 2:
        n_clusters = 2
    pseudo_labels = KMeans(n_clusters=n_clusters, n_init=3, random_state=random_state).fit_predict(embedding)

    cluster_strength = max(_silhouette_safe(embedding, pseudo_labels), 0.0)
    batch_strength = 0.0
    batch_classes = 0
    if batches_eval is not None:
        batch_classes = int(len(np.unique(batches_eval)))
        if batch_classes >= 2:
            batch_strength = max(_silhouette_safe(embedding, batches_eval), 0.0)

    explained = pca_model.explained_variance_ratio_[: min(5, len(pca_model.explained_variance_ratio_))]
    explained_sum = max(float(explained.sum()), 1e-8)
    explained_norm = explained / explained_sum
    pc_entropy = float(-(explained_norm * np.log(explained_norm + 1e-8)).sum() / np.log(len(explained_norm) + 1e-8))
    trajectory_strength = float(explained[0] / max(float(explained[:2].sum()), 1e-8))

    cluster_sizes = np.bincount(pseudo_labels)
    rare_fraction = float(cluster_sizes.min() / max(cluster_sizes.sum(), 1))
    library = counts.sum(axis=1).astype(np.float64)
    library_cv = float(library.std() / max(library.mean(), 1.0))
    dropout_rate = float(np.mean(counts <= 0))

    return AdaptiveStatProfile(
        n_cells=int(n_cells),
        n_genes=int(n_genes),
        batch_classes=int(batch_classes),
        dropout_rate=dropout_rate,
        library_cv=library_cv,
        cluster_strength=cluster_strength,
        batch_strength=batch_strength,
        trajectory_strength=max(trajectory_strength, 0.0),
        pc_entropy=pc_entropy,
        rare_fraction=rare_fraction,
    )


def resolve_adaptive_stat_decision(profile: AdaptiveStatProfile) -> AdaptiveStatDecision:
    route_name = "mv_default"
    weights = np.asarray([0.10, 0.85, 0.05], dtype=np.float64)

    if (
        profile.dropout_rate > 0.875
        and profile.library_cv < 0.55
        and profile.trajectory_strength > 0.72
    ):
        route_name = "fano_noisy_trajectory"
        weights = np.asarray([0.05, 0.10, 0.85], dtype=np.float64)
    elif (
        profile.batch_classes >= 8
        and profile.library_cv > 1.0
        and profile.trajectory_strength < 0.60
    ):
        route_name = "fano_batch_heterogeneous"
        weights = np.asarray([0.10, 0.15, 0.75], dtype=np.float64)
    elif (
        profile.n_cells >= 10000
        and (
            profile.n_genes <= 12000
            or profile.pc_entropy > 0.90
            or profile.cluster_strength < 0.26
        )
    ):
        route_name = "variance_large_atlas"
        weights = np.asarray([0.70, 0.20, 0.10], dtype=np.float64)
    elif profile.n_cells >= 7000 and profile.n_genes < 20000:
        route_name = "variance_mid_large_panel"
        weights = np.asarray([0.80, 0.15, 0.05], dtype=np.float64)
    elif profile.n_genes >= 20000 and profile.cluster_strength >= 0.28:
        route_name = "mv_high_gene"
        weights = np.asarray([0.10, 0.85, 0.05], dtype=np.float64)
    elif profile.pc_entropy > 0.92 and profile.cluster_strength < 0.30:
        route_name = "variance_entropy"
        weights = np.asarray([0.75, 0.20, 0.05], dtype=np.float64)

    weights = weights / max(float(weights.sum()), 1e-8)
    return AdaptiveStatDecision(
        route_name=route_name,
        variance_weight=float(weights[0]),
        mv_residual_weight=float(weights[1]),
        fano_weight=float(weights[2]),
    )


def resolve_adaptive_hybrid_decision(profile: AdaptiveStatProfile) -> AdaptiveHybridDecision:
    if (
        profile.dropout_rate >= 0.87
        and profile.trajectory_strength >= 0.74
        and profile.library_cv <= 0.60
    ):
        return AdaptiveHybridDecision(
            route_name="fano_hard_trajectory_escape",
            route_target="fano",
        )

    if (
        profile.batch_classes >= 8
        and profile.library_cv >= 0.95
        and profile.n_genes >= 25000
        and profile.trajectory_strength <= 0.65
    ):
        return AdaptiveHybridDecision(
            route_name="frontier_batch_heterogeneous_escape",
            route_target="frontier_lite",
            fallback_target="fano",
        )

    stat_decision = resolve_adaptive_stat_decision(profile)
    return AdaptiveHybridDecision(
        route_name=f"adaptive_stat::{stat_decision.route_name}",
        route_target="adaptive_stat_blend",
    )


def score_adaptive_core_consensus_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
    include_agreement_bonus: bool = True,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    score_stack, rank_stack = _compute_anchor_core_rank_scores(counts)
    weights = _compute_anchor_core_signals(profile)

    variance_rank = rank_stack["variance"]
    mv_rank = rank_stack["mv_residual"]
    fano_rank = rank_stack["fano"]
    deviance_rank = rank_stack["multinomial_deviance_hvg"]

    classical_core = 0.45 * mv_rank + 0.35 * variance_rank + 0.20 * fano_rank
    count_bridge = 0.60 * deviance_rank + 0.25 * mv_rank + 0.15 * fano_rank
    atlas_prior = 0.75 * variance_rank + 0.25 * mv_rank
    bridge_mix = float(np.clip(
        0.10
        + 0.18 * weights["heterogeneity_signal"]
        + 0.10 * weights["trajectory_signal"]
        - 0.14 * weights["atlas_guard"],
        0.04,
        0.28,
    ))

    score = (1.0 - bridge_mix) * classical_core + bridge_mix * count_bridge
    score += 0.06 * weights["trajectory_signal"] * (fano_rank - classical_core)
    score += 0.08 * weights["atlas_guard"] * (atlas_prior - count_bridge)

    agreement_weight = 0.08 if include_agreement_bonus else 0.0
    if include_agreement_bonus:
        agreement_bonus = _zscore(-np.std(score_stack, axis=0))
        score += agreement_weight * agreement_bonus

    metadata: dict[str, float | str] = {
        "variant": "adaptive_core_consensus_hvg",
        "consensus_family": "score_level_count_bridge",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "anchor_core_consensus_v1",
        "route_name": "fixed_consensus",
        "route_target": "single_score",
        "bridge_mix": bridge_mix,
        "agreement_bonus_weight": agreement_weight,
        **weights,
    }
    return _zscore(score), profile, metadata


def score_adaptive_rank_aggregate_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
    include_agreement_bonus: bool = True,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    score_stack, rank_stack = _compute_anchor_core_rank_scores(counts)
    weights = _compute_anchor_core_signals(profile)

    sorted_ranks = np.sort(score_stack, axis=0)
    trimmed_rank = np.mean(sorted_ranks[1:3], axis=0)
    agreement_bonus = _zscore(-np.std(score_stack, axis=0))

    variance_rank = rank_stack["variance"]
    fano_rank = rank_stack["fano"]
    deviance_rank = rank_stack["multinomial_deviance_hvg"]

    score = trimmed_rank
    agreement_weight = 0.10 if include_agreement_bonus else 0.0
    score += agreement_weight * agreement_bonus
    score += 0.10 * weights["trajectory_signal"] * (fano_rank - trimmed_rank)
    score += 0.08 * weights["heterogeneity_signal"] * (deviance_rank - trimmed_rank)
    score += 0.10 * weights["atlas_guard"] * (variance_rank - trimmed_rank)

    metadata: dict[str, float | str] = {
        "variant": "adaptive_rank_aggregate_hvg",
        "consensus_family": "rank_trimmed_mean",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "anchor_core_rank_consensus_v1",
        "route_name": "trimmed_rank_aggregate",
        "route_target": "single_score",
        "agreement_bonus_weight": agreement_weight,
        **weights,
    }
    return _zscore(score), profile, metadata


def score_adaptive_eb_shrinkage_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, stack = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    variance_rank = rank_map["variance"]
    mv_rank = rank_map["mv_residual"]
    fano_rank = rank_map["fano"]
    deviance_rank = rank_map["multinomial_deviance_hvg"]
    pearson_rank = rank_map["analytic_pearson_residual_hvg"]

    classical_core = 0.42 * mv_rank + 0.28 * variance_rank + 0.18 * pearson_rank + 0.12 * fano_rank
    count_bridge = 0.40 * deviance_rank + 0.22 * pearson_rank + 0.22 * mv_rank + 0.16 * fano_rank
    trajectory_bridge = 0.42 * fano_rank + 0.22 * deviance_rank + 0.20 * pearson_rank + 0.16 * mv_rank
    atlas_prior = 0.66 * variance_rank + 0.24 * mv_rank + 0.10 * pearson_rank

    target = (
        (1.0 - weights["heterogeneity_signal"]) * classical_core
        + weights["heterogeneity_signal"] * count_bridge
    )
    target += 0.18 * weights["trajectory_signal"] * (trajectory_bridge - target)
    target += 0.10 * weights["atlas_guard"] * (atlas_prior - target)

    disagreement = np.std(stack, axis=0)
    disagreement_sq = disagreement * disagreement
    tau_sq = max(float(np.percentile(disagreement_sq, 60.0)), 1e-6)
    shrink_factor = tau_sq / (tau_sq + disagreement_sq + 1e-8)

    prior = (1.0 - weights["atlas_guard"]) * classical_core + weights["atlas_guard"] * atlas_prior
    score = prior + shrink_factor * (target - prior)
    score += 0.08 * _zscore(-disagreement)

    metadata: dict[str, float | str] = {
        "variant": "adaptive_eb_shrinkage_hvg",
        "consensus_family": "empirical_bayes_shrinkage",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "anchor_core_broad_restart_v1",
        "route_name": "empirical_bayes_shrinkage",
        "route_target": "single_score",
        "tau_sq": tau_sq,
        "mean_shrink_factor": float(np.mean(shrink_factor)),
        **weights,
    }
    return _zscore(score), profile, metadata


def score_adaptive_invariant_residual_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_invariant_residual_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="adaptive_invariant_residual_hvg",
        allow_pseudo_environments=True,
        atlas_guard_strength=0.08,
        extra_atlas_safe_strength=0.0,
        worst_group_weight=0.46,
        mean_group_weight=0.20,
        mv_group_weight=0.18,
        global_deviance_weight=0.10,
        pearson_weight=0.06,
        invariant_penalty_weight=0.15,
    )


def score_real_batch_only_invariant_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_invariant_residual_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="real_batch_only_invariant",
        allow_pseudo_environments=False,
        atlas_guard_strength=0.08,
        extra_atlas_safe_strength=0.0,
        worst_group_weight=0.46,
        mean_group_weight=0.20,
        mv_group_weight=0.18,
        global_deviance_weight=0.10,
        pearson_weight=0.06,
        invariant_penalty_weight=0.15,
    )


def score_stronger_atlas_guard_invariant_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_invariant_residual_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="stronger_atlas_guard_invariant",
        allow_pseudo_environments=True,
        atlas_guard_strength=0.18,
        extra_atlas_safe_strength=0.12,
        worst_group_weight=0.46,
        mean_group_weight=0.20,
        mv_group_weight=0.18,
        global_deviance_weight=0.10,
        pearson_weight=0.06,
        invariant_penalty_weight=0.15,
    )


def score_reduced_deviance_aggression_invariant_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_invariant_residual_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="reduced_deviance_aggression_invariant",
        allow_pseudo_environments=True,
        atlas_guard_strength=0.10,
        extra_atlas_safe_strength=0.04,
        worst_group_weight=0.30,
        mean_group_weight=0.12,
        mv_group_weight=0.26,
        global_deviance_weight=0.18,
        pearson_weight=0.14,
        invariant_penalty_weight=0.12,
    )


def score_invariant_topology_hvg_v0(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
    current_top_k: int | None = None,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_invariant_topology_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        current_top_k=current_top_k,
        variant_name="invariant_topology_hvg_v0",
        invariant_mix_weight=0.0,
        use_diversity_rerank=False,
        atlas_guard_strength=0.10,
        topology_mix_weight=0.44,
        diversity_penalty_strength=0.0,
    )


def score_invariant_topology_hvg_v1(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
    current_top_k: int | None = None,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_invariant_topology_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        current_top_k=current_top_k,
        variant_name="invariant_topology_hvg_v1",
        invariant_mix_weight=0.34,
        use_diversity_rerank=False,
        atlas_guard_strength=0.10,
        topology_mix_weight=0.44,
        diversity_penalty_strength=0.0,
    )


def score_invariant_topology_hvg_v2(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
    current_top_k: int | None = None,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_invariant_topology_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        current_top_k=current_top_k,
        variant_name="invariant_topology_hvg_v2",
        invariant_mix_weight=0.34,
        use_diversity_rerank=True,
        atlas_guard_strength=0.10,
        topology_mix_weight=0.44,
        diversity_penalty_strength=0.18,
    )


def score_invariant_topology_hvg_v0_guarded(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
    current_top_k: int | None = None,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_invariant_topology_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        current_top_k=current_top_k,
        variant_name="invariant_topology_hvg_v0_guarded",
        invariant_mix_weight=0.0,
        use_diversity_rerank=False,
        atlas_guard_strength=0.20,
        topology_mix_weight=0.34,
        diversity_penalty_strength=0.0,
    )


def score_invariant_topology_hvg_v1_guarded(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
    current_top_k: int | None = None,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_invariant_topology_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        current_top_k=current_top_k,
        variant_name="invariant_topology_hvg_v1_guarded",
        invariant_mix_weight=0.28,
        use_diversity_rerank=False,
        atlas_guard_strength=0.18,
        topology_mix_weight=0.38,
        diversity_penalty_strength=0.0,
    )


def score_invariant_topology_hvg_v2_softdiv(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
    current_top_k: int | None = None,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_invariant_topology_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        current_top_k=current_top_k,
        variant_name="invariant_topology_hvg_v2_softdiv",
        invariant_mix_weight=0.28,
        use_diversity_rerank=True,
        atlas_guard_strength=0.18,
        topology_mix_weight=0.38,
        diversity_penalty_strength=0.08,
    )


def score_invariant_topology_hvg_v3_safetygated(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
    current_top_k: int | None = None,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)
    topology_score, topology_meta = _compute_topology_locality_signal(
        counts=counts,
        random_state=random_state,
    )

    count_core = (
        0.30 * rank_map["mv_residual"]
        + 0.24 * rank_map["multinomial_deviance_hvg"]
        + 0.18 * rank_map["analytic_pearson_residual_hvg"]
        + 0.16 * rank_map["fano"]
        + 0.12 * rank_map["variance"]
    )
    topology_bridge = (
        0.62 * topology_score
        + 0.16 * rank_map["fano"]
        + 0.12 * rank_map["multinomial_deviance_hvg"]
        + 0.10 * rank_map["analytic_pearson_residual_hvg"]
    )
    topology_mix_weight = 0.40
    score = (1.0 - topology_mix_weight) * count_core + topology_mix_weight * topology_bridge
    score += 0.06 * weights["heterogeneity_signal"] * (rank_map["multinomial_deviance_hvg"] - score)
    score += 0.08 * weights["trajectory_signal"] * (rank_map["fano"] - score)
    score += 0.12 * weights["atlas_guard"] * ((0.68 * rank_map["variance"] + 0.32 * rank_map["mv_residual"]) - score)

    stat_decision = resolve_adaptive_stat_decision(profile)
    safe_stat = _rank_percentile(score_adaptive_stat_hvg_with_decision(counts, stat_decision))
    control_risk_signal = _compute_control_risk_signal(profile)
    safety_gate_strength = 0.88
    score = score + safety_gate_strength * control_risk_signal * (safe_stat - score)

    metadata: dict[str, float | str] = {
        "variant": "invariant_topology_hvg_v3_safetygated",
        "consensus_family": "invariant_topology_with_profile_safety_gate",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "invariant_topology_bank_v2",
        "route_name": "invariant_topology_hvg_v3_safetygated",
        "route_target": "single_score",
        "current_top_k": float(0 if current_top_k is None else current_top_k),
        "topology_mix_weight": topology_mix_weight,
        "control_risk_signal": control_risk_signal,
        "safety_gate_strength": safety_gate_strength,
        "safety_fallback_route": stat_decision.route_name,
        **topology_meta,
        **weights,
    }
    return _zscore(score), profile, metadata


def score_invariant_topology_hvg_v4_hardgate(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
    current_top_k: int | None = None,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    base_score, profile, base_metadata = _score_invariant_topology_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        current_top_k=current_top_k,
        variant_name="invariant_topology_hvg_v4_hardgate_base",
        invariant_mix_weight=0.0,
        use_diversity_rerank=False,
        atlas_guard_strength=0.10,
        topology_mix_weight=0.44,
        diversity_penalty_strength=0.0,
    )
    control_risk_signal = _compute_control_risk_signal(profile)
    hard_gate_threshold = 0.40
    stat_decision = resolve_adaptive_stat_decision(profile)
    used_hard_gate = float(control_risk_signal >= hard_gate_threshold)

    if used_hard_gate:
        score = _zscore(score_adaptive_stat_hvg_with_decision(counts, stat_decision))
    else:
        score = base_score

    metadata: dict[str, float | str] = {
        **base_metadata,
        "variant": "invariant_topology_hvg_v4_hardgate",
        "consensus_family": "invariant_topology_with_profile_hard_gate",
        "selector_bank_name": "invariant_topology_bank_v3",
        "route_name": "invariant_topology_hvg_v4_hardgate",
        "control_risk_signal": control_risk_signal,
        "hard_gate_threshold": hard_gate_threshold,
        "used_hard_gate": used_hard_gate,
        "safety_fallback_route": stat_decision.route_name,
    }
    return score, profile, metadata


def _score_invariant_topology_variant(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None,
    random_state: int,
    current_top_k: int | None,
    variant_name: str,
    invariant_mix_weight: float,
    use_diversity_rerank: bool,
    atlas_guard_strength: float,
    topology_mix_weight: float,
    diversity_penalty_strength: float,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    topology_score, topology_meta = _compute_topology_locality_signal(
        counts=counts,
        random_state=random_state,
    )
    invariant_support, invariant_meta = _compute_invariant_support_signal(
        counts=counts,
        batches=batches,
        random_state=random_state,
        allow_pseudo_environments=True,
    )

    count_core = (
        0.30 * rank_map["mv_residual"]
        + 0.24 * rank_map["multinomial_deviance_hvg"]
        + 0.18 * rank_map["analytic_pearson_residual_hvg"]
        + 0.16 * rank_map["fano"]
        + 0.12 * rank_map["variance"]
    )
    topology_bridge = (
        0.62 * topology_score
        + 0.16 * rank_map["fano"]
        + 0.12 * rank_map["multinomial_deviance_hvg"]
        + 0.10 * rank_map["analytic_pearson_residual_hvg"]
    )
    anchor_safe = 0.68 * rank_map["variance"] + 0.32 * rank_map["mv_residual"]

    topology_mix_weight = float(np.clip(topology_mix_weight, 0.0, 1.0))
    score = (1.0 - topology_mix_weight) * count_core + topology_mix_weight * topology_bridge
    if invariant_mix_weight > 0.0:
        score = (1.0 - invariant_mix_weight) * score + invariant_mix_weight * invariant_support
    score += 0.06 * weights["heterogeneity_signal"] * (rank_map["multinomial_deviance_hvg"] - score)
    score += 0.08 * weights["trajectory_signal"] * (rank_map["fano"] - score)
    score += atlas_guard_strength * weights["atlas_guard"] * (anchor_safe - score)
    score = _zscore(score)

    diversity_meta: dict[str, float | str] = {
        "diversity_pool_size": 0.0,
        "diversity_selected_k": 0.0,
        "diversity_sampled_cells": 0.0,
        "diversity_penalty_strength": diversity_penalty_strength,
        "diversity_device": "none",
    }
    if use_diversity_rerank and current_top_k is not None:
        score, diversity_meta = _apply_diversity_rerank(
            counts=counts,
            base_score=score,
            current_top_k=current_top_k,
            random_state=random_state,
            penalty_strength=diversity_penalty_strength,
        )

    metadata: dict[str, float | str] = {
        "variant": variant_name,
        "consensus_family": "invariant_topology_panel",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "invariant_topology_bank_v1",
        "route_name": variant_name,
        "route_target": "single_score",
        "current_top_k": float(0 if current_top_k is None else current_top_k),
        "invariant_mix_weight": invariant_mix_weight,
        "atlas_guard_strength": atlas_guard_strength,
        "topology_mix_weight": topology_mix_weight,
        **topology_meta,
        **invariant_meta,
        **diversity_meta,
        **weights,
    }
    return score, profile, metadata


def _score_invariant_residual_variant(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None,
    random_state: int,
    variant_name: str,
    allow_pseudo_environments: bool,
    atlas_guard_strength: float,
    extra_atlas_safe_strength: float,
    worst_group_weight: float,
    mean_group_weight: float,
    mv_group_weight: float,
    global_deviance_weight: float,
    pearson_weight: float,
    invariant_penalty_weight: float,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    sample_idx, env_labels, env_source = _build_invariant_environment_labels(
        counts=counts,
        batches=batches,
        random_state=random_state,
        max_cells=2400,
        target_envs=4,
        allow_pseudo_fallback=allow_pseudo_environments,
    )
    counts_sample = counts[sample_idx]
    unique_envs = [label for label in np.unique(env_labels) if np.sum(env_labels == label) >= 64]

    env_deviance_ranks: list[np.ndarray] = []
    env_mv_ranks: list[np.ndarray] = []
    for label in unique_envs:
        env_counts = counts_sample[env_labels == label]
        if env_counts.shape[0] < 64:
            continue
        env_deviance_ranks.append(_rank_percentile(score_multinomial_deviance_hvg(np.asarray(env_counts, dtype=np.float32))))
        env_mv_ranks.append(_rank_percentile(score_mean_var_residual_like_counts(env_counts)))

    if len(env_deviance_ranks) >= 2:
        env_deviance = np.vstack(env_deviance_ranks)
        env_mv = np.vstack(env_mv_ranks)
        worst_group = np.quantile(env_deviance, 0.25, axis=0)
        mean_group = np.mean(env_deviance, axis=0)
        mv_group = np.mean(env_mv, axis=0)
        invariant_penalty = np.std(env_deviance, axis=0)
    else:
        worst_group = rank_map["multinomial_deviance_hvg"]
        mean_group = rank_map["multinomial_deviance_hvg"]
        mv_group = rank_map["mv_residual"]
        invariant_penalty = np.zeros_like(worst_group, dtype=np.float64)

    score = (
        worst_group_weight * worst_group
        + mean_group_weight * mean_group
        + mv_group_weight * mv_group
        + global_deviance_weight * rank_map["multinomial_deviance_hvg"]
        + pearson_weight * rank_map["analytic_pearson_residual_hvg"]
    )
    score -= invariant_penalty_weight * invariant_penalty
    score += 0.08 * weights["trajectory_signal"] * (rank_map["fano"] - score)
    score += atlas_guard_strength * weights["atlas_guard"] * (rank_map["variance"] - score)
    if extra_atlas_safe_strength > 0.0:
        anchor_safe = 0.65 * rank_map["variance"] + 0.35 * rank_map["mv_residual"]
        score += extra_atlas_safe_strength * weights["atlas_guard"] * (anchor_safe - score)

    metadata: dict[str, float | str] = {
        "variant": variant_name,
        "consensus_family": "pseudo_environment_invariance",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "anchor_core_broad_restart_v1",
        "route_name": variant_name,
        "route_target": "single_score",
        "environment_count": float(len(unique_envs)),
        "environment_source": env_source,
        "sampled_cells": float(len(sample_idx)),
        "allow_pseudo_environments": float(allow_pseudo_environments),
        "atlas_guard_strength": atlas_guard_strength,
        "extra_atlas_safe_strength": extra_atlas_safe_strength,
        "worst_group_weight": worst_group_weight,
        "mean_group_weight": mean_group_weight,
        "mv_group_weight": mv_group_weight,
        "global_deviance_weight": global_deviance_weight,
        "pearson_weight": pearson_weight,
        "invariant_penalty_weight": invariant_penalty_weight,
        **weights,
    }
    return _zscore(score), profile, metadata


def _compute_topology_locality_signal(
    *,
    counts: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, dict[str, float | str]]:
    sample_idx = _sample_cell_indices(counts.shape[0], max_cells=2200, random_state=random_state)
    counts_sample = counts[sample_idx]
    x_sample = _normalize_log1p_fast(counts_sample)

    if x_sample.shape[0] < 32 or x_sample.shape[1] < 32:
        _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
        fallback = (
            0.45 * rank_map["fano"]
            + 0.25 * rank_map["multinomial_deviance_hvg"]
            + 0.20 * rank_map["analytic_pearson_residual_hvg"]
            + 0.10 * rank_map["variance"]
        )
        return _zscore(fallback), {
            "topology_mode": "fallback_rank_bridge",
            "topology_sampled_cells": float(len(sample_idx)),
            "topology_graph_genes": 0.0,
            "topology_neighbors": 0.0,
        }

    graph_gene_count = min(3000, x_sample.shape[1])
    gene_var = np.var(x_sample, axis=0, dtype=np.float64)
    gene_idx = np.argsort(gene_var)[-graph_gene_count:]
    graph_input = x_sample[:, np.sort(gene_idx)]
    pca_dim = min(30, graph_input.shape[0] - 1, graph_input.shape[1])
    embedding = PCA(n_components=pca_dim, svd_solver="randomized", random_state=random_state).fit_transform(graph_input)
    n_neighbors = min(15, max(4, embedding.shape[0] - 1))
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn.fit(embedding)
    neighbor_idx = knn.kneighbors(return_distance=False)[:, 1:]
    adjacency = np.zeros((embedding.shape[0], embedding.shape[0]), dtype=np.float32)
    adjacency[np.arange(embedding.shape[0])[:, None], neighbor_idx] = 1.0 / max(float(n_neighbors), 1.0)
    locality = _compute_graph_locality_scores(
        x=x_sample,
        adjacency=adjacency,
        chunk_size=512,
    )
    return _zscore(locality), {
        "topology_mode": "graph_locality",
        "topology_sampled_cells": float(len(sample_idx)),
        "topology_graph_genes": float(graph_gene_count),
        "topology_neighbors": float(n_neighbors),
    }


def _compute_invariant_support_signal(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None,
    random_state: int,
    allow_pseudo_environments: bool,
) -> tuple[np.ndarray, dict[str, float | str]]:
    _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
    sample_idx, env_labels, env_source = _build_invariant_environment_labels(
        counts=counts,
        batches=batches,
        random_state=random_state,
        max_cells=2400,
        target_envs=4,
        allow_pseudo_fallback=allow_pseudo_environments,
    )
    counts_sample = counts[sample_idx]
    unique_envs = [label for label in np.unique(env_labels) if np.sum(env_labels == label) >= 64]

    env_deviance_ranks: list[np.ndarray] = []
    env_mv_ranks: list[np.ndarray] = []
    for label in unique_envs:
        env_counts = counts_sample[env_labels == label]
        if env_counts.shape[0] < 64:
            continue
        env_deviance_ranks.append(_rank_percentile(score_multinomial_deviance_hvg(np.asarray(env_counts, dtype=np.float32))))
        env_mv_ranks.append(_rank_percentile(score_mean_var_residual_like_counts(env_counts)))

    if len(env_deviance_ranks) >= 2:
        env_deviance = np.vstack(env_deviance_ranks)
        env_mv = np.vstack(env_mv_ranks)
        worst_group = np.quantile(env_deviance, 0.25, axis=0)
        mean_group = np.mean(env_deviance, axis=0)
        mv_group = np.mean(env_mv, axis=0)
        invariant_penalty = np.std(env_deviance, axis=0)
    else:
        worst_group = rank_map["multinomial_deviance_hvg"]
        mean_group = rank_map["multinomial_deviance_hvg"]
        mv_group = rank_map["mv_residual"]
        invariant_penalty = np.zeros_like(worst_group, dtype=np.float64)

    score = (
        0.44 * worst_group
        + 0.18 * mean_group
        + 0.18 * mv_group
        + 0.12 * rank_map["multinomial_deviance_hvg"]
        + 0.08 * rank_map["analytic_pearson_residual_hvg"]
        - 0.14 * invariant_penalty
    )
    return _zscore(score), {
        "environment_count": float(len(unique_envs)),
        "environment_source": env_source,
        "invariant_sampled_cells": float(len(sample_idx)),
        "allow_pseudo_environments": float(allow_pseudo_environments),
    }


def score_adaptive_stability_jackknife_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    sample_idx, env_labels = _build_pseudo_environment_labels(
        counts=counts,
        batches=batches,
        random_state=random_state,
        max_cells=2400,
        target_envs=4,
    )
    split_indices = _build_stability_splits(
        n_cells=len(sample_idx),
        env_labels=env_labels,
        n_splits=3,
        random_state=random_state,
    )
    counts_sample = counts[sample_idx]

    split_scores: list[np.ndarray] = []
    for heldout_split in split_indices:
        keep_mask = np.ones(len(sample_idx), dtype=bool)
        keep_mask[heldout_split] = False
        split_counts = counts_sample[keep_mask]
        if split_counts.shape[0] < 96:
            continue
        _, split_rank_map, _ = _compute_anchor_core_signal_maps(split_counts)
        split_score = (
            0.32 * split_rank_map["mv_residual"]
            + 0.22 * split_rank_map["variance"]
            + 0.22 * split_rank_map["multinomial_deviance_hvg"]
            + 0.14 * split_rank_map["analytic_pearson_residual_hvg"]
            + 0.10 * split_rank_map["fano"]
        )
        split_scores.append(split_score)

    if split_scores:
        split_stack = np.vstack(split_scores)
        stable_mean = np.mean(split_stack, axis=0)
        stable_std = np.std(split_stack, axis=0)
    else:
        stable_mean = (
            0.32 * rank_map["mv_residual"]
            + 0.22 * rank_map["variance"]
            + 0.22 * rank_map["multinomial_deviance_hvg"]
            + 0.14 * rank_map["analytic_pearson_residual_hvg"]
            + 0.10 * rank_map["fano"]
        )
        stable_std = np.zeros_like(stable_mean, dtype=np.float64)

    full_core = (
        0.34 * rank_map["mv_residual"]
        + 0.22 * rank_map["variance"]
        + 0.20 * rank_map["multinomial_deviance_hvg"]
        + 0.14 * rank_map["analytic_pearson_residual_hvg"]
        + 0.10 * rank_map["fano"]
    )
    score = 0.72 * full_core + 0.28 * (stable_mean - 0.45 * stable_std)
    score += 0.06 * weights["heterogeneity_signal"] * (rank_map["multinomial_deviance_hvg"] - score)
    score += 0.05 * weights["trajectory_signal"] * (rank_map["fano"] - score)

    metadata: dict[str, float | str] = {
        "variant": "adaptive_stability_jackknife_hvg",
        "consensus_family": "jackknife_stability",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "anchor_core_broad_restart_v1",
        "route_name": "stability_jackknife",
        "route_target": "single_score",
        "sampled_cells": float(len(sample_idx)),
        "split_count": float(len(split_scores)),
        **weights,
    }
    return _zscore(score), profile, metadata


def score_adaptive_spectral_locality_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    sample_idx = _sample_cell_indices(counts.shape[0], max_cells=2200, random_state=random_state)
    counts_sample = counts[sample_idx]
    x_sample = _normalize_log1p_fast(counts_sample)

    if x_sample.shape[0] < 32 or x_sample.shape[1] < 32:
        spectral_score = (
            0.45 * rank_map["fano"]
            + 0.25 * rank_map["multinomial_deviance_hvg"]
            + 0.20 * rank_map["analytic_pearson_residual_hvg"]
            + 0.10 * rank_map["variance"]
        )
    else:
        graph_gene_count = min(3000, x_sample.shape[1])
        gene_var = np.var(x_sample, axis=0, dtype=np.float64)
        gene_idx = np.argsort(gene_var)[-graph_gene_count:]
        graph_input = x_sample[:, np.sort(gene_idx)]
        pca_dim = min(30, graph_input.shape[0] - 1, graph_input.shape[1])
        embedding = PCA(n_components=pca_dim, svd_solver="randomized", random_state=random_state).fit_transform(graph_input)
        n_neighbors = min(15, max(4, embedding.shape[0] - 1))
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        knn.fit(embedding)
        neighbor_idx = knn.kneighbors(return_distance=False)[:, 1:]
        adjacency = np.zeros((embedding.shape[0], embedding.shape[0]), dtype=np.float32)
        adjacency[np.arange(embedding.shape[0])[:, None], neighbor_idx] = 1.0 / max(float(n_neighbors), 1.0)
        locality = _compute_graph_locality_scores(
            x=x_sample,
            adjacency=adjacency,
            chunk_size=512,
        )
        spectral_score = _zscore(locality)

    score = 0.54 * spectral_score + 0.18 * rank_map["fano"] + 0.14 * rank_map["multinomial_deviance_hvg"] + 0.14 * rank_map["analytic_pearson_residual_hvg"]
    score += 0.08 * weights["atlas_guard"] * (rank_map["variance"] - score)

    metadata: dict[str, float | str] = {
        "variant": "adaptive_spectral_locality_hvg",
        "consensus_family": "graph_spectral_locality",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "anchor_core_broad_restart_v1",
        "route_name": "spectral_locality",
        "route_target": "single_score",
        "sampled_cells": float(len(sample_idx)),
        **weights,
    }
    return _zscore(score), profile, metadata


def score_adaptive_risk_parity_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, stack = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    covariance = np.cov(stack) + 1e-4 * np.eye(stack.shape[0], dtype=np.float64)
    inv_cov = np.linalg.pinv(covariance)
    risk_weights = inv_cov @ np.ones(stack.shape[0], dtype=np.float64)
    risk_weights = risk_weights / max(float(risk_weights.sum()), 1e-8)

    weighted_return = risk_weights @ stack
    sorted_stack = np.sort(stack, axis=0)
    balanced_support = np.mean(sorted_stack[2:], axis=0)
    downside_support = np.mean(sorted_stack[:2], axis=0)
    concentration = np.std(stack, axis=0)

    score = 0.46 * weighted_return + 0.34 * balanced_support + 0.12 * downside_support - 0.16 * concentration
    score += 0.06 * weights["heterogeneity_signal"] * (rank_map["multinomial_deviance_hvg"] - score)
    score += 0.05 * weights["trajectory_signal"] * (rank_map["fano"] - score)
    score += 0.07 * weights["atlas_guard"] * (rank_map["variance"] - score)

    metadata: dict[str, float | str] = {
        "variant": "adaptive_risk_parity_hvg",
        "consensus_family": "risk_parity_multi_objective",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "anchor_core_broad_restart_v1",
        "route_name": "risk_parity",
        "route_target": "single_score",
        "risk_weight_variance": float(risk_weights[0]),
        "risk_weight_mv": float(risk_weights[1]),
        "risk_weight_fano": float(risk_weights[2]),
        "risk_weight_deviance": float(risk_weights[3]),
        "risk_weight_pearson": float(risk_weights[4]),
        **weights,
    }
    return _zscore(score), profile, metadata


def score_adaptive_risk_parity_safe_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    risk_score, profile, risk_metadata = score_adaptive_risk_parity_hvg(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    anchor_safe = 0.58 * rank_map["variance"] + 0.32 * rank_map["mv_residual"] + 0.10 * rank_map["fano"]
    deviance_guard = 0.52 * rank_map["multinomial_deviance_hvg"] + 0.48 * anchor_safe
    safe_prior = (1.0 - weights["atlas_guard"]) * deviance_guard + weights["atlas_guard"] * anchor_safe
    pull_to_safe = float(np.clip(0.20 + 0.18 * weights["atlas_guard"], 0.20, 0.38))
    score = (1.0 - pull_to_safe) * _rank_percentile(risk_score) + pull_to_safe * safe_prior
    score += 0.04 * weights["heterogeneity_signal"] * (rank_map["multinomial_deviance_hvg"] - score)
    score += 0.10 * weights["atlas_guard"] * (anchor_safe - score)

    metadata: dict[str, float | str] = {
        **risk_metadata,
        "variant": "adaptive_risk_parity_safe_hvg",
        "consensus_family": "risk_parity_multi_objective_safe",
        "selector_bank_name": "anchor_core_broad_restart_v2",
        "route_name": "risk_parity_safe",
        "safe_pull_weight": pull_to_safe,
        "safe_prior_family": "variance_mv_fano_guarded_deviance",
    }
    return _zscore(score), profile, metadata


def score_adaptive_risk_parity_ultrasafe_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    risk_score, profile, risk_metadata = score_adaptive_risk_parity_hvg(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    anchor_safe = 0.64 * rank_map["variance"] + 0.28 * rank_map["mv_residual"] + 0.08 * rank_map["fano"]
    biological_nudge = 0.54 * rank_map["multinomial_deviance_hvg"] + 0.46 * rank_map["analytic_pearson_residual_hvg"]
    risk_rank = _rank_percentile(risk_score)
    nudge_weight = float(np.clip(0.10 + 0.06 * weights["heterogeneity_signal"], 0.10, 0.16))
    score = (1.0 - nudge_weight) * anchor_safe + nudge_weight * (0.55 * risk_rank + 0.45 * biological_nudge)
    score += 0.04 * weights["atlas_guard"] * (rank_map["variance"] - score)

    metadata: dict[str, float | str] = {
        **risk_metadata,
        "variant": "adaptive_risk_parity_ultrasafe_hvg",
        "consensus_family": "risk_parity_ultrasafe_anchor_nudge",
        "selector_bank_name": "anchor_core_broad_restart_v2",
        "route_name": "risk_parity_ultrasafe",
        "nudge_weight": nudge_weight,
        "safe_prior_family": "variance_mv_fano_anchor_with_small_risk_nudge",
    }
    return _zscore(score), profile, metadata


def score_sigma_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_sigma_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="sigma_hvg",
        use_invariance=True,
        use_graph=True,
        use_stability=True,
        use_shrink=True,
        env_mode="mixed",
        use_deviance=True,
        use_pearson=True,
    )


def score_sigma_no_invariance_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_sigma_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="sigma_no_invariance_hvg",
        use_invariance=False,
        use_graph=True,
        use_stability=True,
        use_shrink=True,
        env_mode="mixed",
        use_deviance=True,
        use_pearson=True,
    )


def score_sigma_no_graph_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_sigma_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="sigma_no_graph_hvg",
        use_invariance=True,
        use_graph=False,
        use_stability=True,
        use_shrink=True,
        env_mode="mixed",
        use_deviance=True,
        use_pearson=True,
    )


def score_sigma_no_stability_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_sigma_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="sigma_no_stability_hvg",
        use_invariance=True,
        use_graph=True,
        use_stability=False,
        use_shrink=True,
        env_mode="mixed",
        use_deviance=True,
        use_pearson=True,
    )


def score_sigma_no_shrink_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_sigma_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="sigma_no_shrink_hvg",
        use_invariance=True,
        use_graph=True,
        use_stability=True,
        use_shrink=False,
        env_mode="mixed",
        use_deviance=True,
        use_pearson=True,
    )


def score_sigma_real_batch_only_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_sigma_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="sigma_real_batch_only_hvg",
        use_invariance=True,
        use_graph=True,
        use_stability=True,
        use_shrink=True,
        env_mode="real_batch_only",
        use_deviance=True,
        use_pearson=True,
    )


def score_sigma_pseudo_env_only_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_sigma_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="sigma_pseudo_env_only_hvg",
        use_invariance=True,
        use_graph=True,
        use_stability=True,
        use_shrink=True,
        env_mode="pseudo_only",
        use_deviance=True,
        use_pearson=True,
    )


def score_sigma_safe_core_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_sigma_safe_core_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="sigma_safe_core_hvg",
        use_real_batch_invariance=True,
        use_stability=True,
        use_deviance=True,
        use_pearson=True,
        shrink_strength=1.85,
    )


def score_sigma_safe_core_no_invariance_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_sigma_safe_core_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="sigma_safe_core_no_invariance_hvg",
        use_real_batch_invariance=False,
        use_stability=True,
        use_deviance=True,
        use_pearson=True,
        shrink_strength=1.85,
    )


def score_sigma_safe_core_no_stability_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    return _score_sigma_safe_core_variant(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variant_name="sigma_safe_core_no_stability_hvg",
        use_real_batch_invariance=True,
        use_stability=False,
        use_deviance=True,
        use_pearson=True,
        shrink_strength=1.85,
    )


def score_sigma_safe_core_v3_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    variance_rank = rank_map["variance"]
    mv_rank = rank_map["mv_residual"]
    fano_rank = rank_map["fano"]
    deviance_rank = rank_map["multinomial_deviance_hvg"]
    pearson_rank = rank_map["analytic_pearson_residual_hvg"]

    adaptive_decision = resolve_adaptive_stat_decision(profile)
    adaptive_anchor_core = score_adaptive_stat_hvg_with_decision(counts, adaptive_decision)
    adaptive_anchor_rank = _rank_percentile(adaptive_anchor_core)
    atlas_prior = (
        0.50 * adaptive_anchor_rank
        + 0.24 * variance_rank
        + 0.16 * mv_rank
        + 0.10 * pearson_rank
    )
    count_bridge = 0.42 * mv_rank + 0.30 * deviance_rank + 0.18 * pearson_rank + 0.10 * fano_rank
    env_component, env_uncertainty, env_count, env_source = _score_sigma_invariance_component(
        counts=counts,
        batches=batches,
        random_state=random_state,
        mv_rank=mv_rank,
        deviance_rank=deviance_rank,
        pearson_rank=pearson_rank,
        atlas_prior=atlas_prior,
        heterogeneity_signal=weights["heterogeneity_signal"],
        atlas_guard=weights["atlas_guard"],
        env_mode="real_batch_only",
    )
    stability_component, stability_uncertainty, split_count = _score_sigma_stability_component(
        counts=counts,
        batches=batches,
        random_state=random_state,
        variance_rank=variance_rank,
        mv_rank=mv_rank,
        fano_rank=fano_rank,
        deviance_rank=deviance_rank,
        pearson_rank=pearson_rank,
    )

    donor_budget = float(
        np.clip(
            0.02
            + 0.18 * _scale_unit(profile.batch_classes, 3.0, 10.0)
            + 0.16 * _scale_unit(profile.batch_strength, 0.0, 0.10)
            + 0.08 * weights["heterogeneity_signal"]
            - 0.20 * weights["atlas_guard"]
            - 0.08 * _scale_unit(profile.trajectory_strength, 0.68, 0.84),
            0.0,
            0.20,
        )
    )

    positive_support = (
        0.50 * np.maximum(deviance_rank - atlas_prior, 0.0)
        + 0.30 * np.maximum(pearson_rank - atlas_prior, 0.0)
        + 0.20 * np.maximum(mv_rank - atlas_prior, 0.0)
    )
    agreement = _zscore(
        -np.std(
            np.vstack(
                [
                    count_bridge,
                    env_component,
                    stability_component,
                    adaptive_anchor_rank,
                ]
            ),
            axis=0,
        )
    )
    admissibility_raw = 0.65 * _zscore(positive_support) + 0.35 * agreement
    admissibility = np.clip((_rank_percentile(admissibility_raw) - 0.55) / 0.30, 0.0, 1.0)

    delta = (
        0.62 * (count_bridge - atlas_prior)
        + 0.23 * (env_component - atlas_prior)
        + 0.15 * (stability_component - atlas_prior)
    )
    gated_delta = admissibility * (
        np.maximum(delta, 0.0) + 0.10 * np.minimum(delta, 0.0)
    )

    uncertainty = np.mean(
        np.vstack(
            [
                _rank_percentile(np.std(np.vstack([count_bridge, env_component, stability_component]), axis=0)),
                _rank_percentile(env_uncertainty),
                _rank_percentile(stability_uncertainty),
                np.full_like(delta, weights["atlas_guard"], dtype=np.float64),
            ]
        ),
        axis=0,
    )
    uncertainty_sq = np.square(uncertainty)
    tau_sq = max(float(np.percentile(uncertainty_sq, 40.0)), 1e-6)
    shrink_factor = tau_sq / (tau_sq + 2.40 * uncertainty_sq + 1e-8)

    score = atlas_prior + shrink_factor * donor_budget * gated_delta
    score += 0.02 * _zscore(admissibility_raw)

    metadata: dict[str, float | str] = {
        "variant": "sigma_safe_core_v3_hvg",
        "consensus_family": "sigma_safe_core_selective_update",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "sigma_safe_core_v3",
        "route_name": "sigma_safe_core_v3",
        "route_target": "single_score",
        "sigma_base_weight": float(1.0 - donor_budget),
        "sigma_count_bridge_weight": donor_budget * 0.62,
        "sigma_inv_weight": donor_budget * 0.23,
        "sigma_graph_weight": 0.0,
        "sigma_stability_weight": donor_budget * 0.15,
        "sigma_mean_uncertainty": float(np.mean(uncertainty)),
        "sigma_mean_shrink_factor": float(np.mean(shrink_factor)),
        "sigma_tau_sq": tau_sq,
        "sigma_shrink_strength": 2.40,
        "sigma_environment_count": env_count,
        "sigma_environment_source": env_source,
        "sigma_split_count": split_count,
        "sigma_use_invariance": 1.0,
        "sigma_use_graph": 0.0,
        "sigma_use_stability": 1.0,
        "sigma_use_shrink": 1.0,
        "sigma_use_deviance": 1.0,
        "sigma_use_pearson": 1.0,
        "sigma_donor_budget": donor_budget,
        "sigma_mean_admissibility": float(np.mean(admissibility)),
        "adaptive_stat_route_name": adaptive_decision.route_name,
        **weights,
    }
    return _zscore(score), profile, metadata


def score_sigma_safe_core_v4_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    mv_rank = rank_map["mv_residual"]
    fano_rank = rank_map["fano"]
    deviance_rank = rank_map["multinomial_deviance_hvg"]
    pearson_rank = rank_map["analytic_pearson_residual_hvg"]

    adaptive_decision = resolve_adaptive_stat_decision(profile)
    adaptive_anchor_core = score_adaptive_stat_hvg_with_decision(counts, adaptive_decision)
    safe_prior = _rank_percentile(adaptive_anchor_core)
    count_bridge = 0.34 * mv_rank + 0.32 * deviance_rank + 0.22 * pearson_rank + 0.12 * fano_rank

    batch_gate = _scale_unit(profile.batch_classes, 2.0, 6.0)
    donor_budget = float(
        np.clip(
            0.16 * batch_gate
            + 0.06 * weights["heterogeneity_signal"]
            - 0.08 * weights["atlas_guard"],
            0.0,
            0.18,
        )
    )

    env_component = count_bridge.copy()
    env_uncertainty = np.zeros_like(count_bridge, dtype=np.float64)
    env_count = 0.0
    env_source = "disabled"
    stability_component = count_bridge.copy()
    stability_uncertainty = np.zeros_like(count_bridge, dtype=np.float64)
    split_count = 0.0
    if donor_budget > 1e-8:
        env_component, env_uncertainty, env_count, env_source = _score_sigma_invariance_component(
            counts=counts,
            batches=batches,
            random_state=random_state,
            mv_rank=mv_rank,
            deviance_rank=deviance_rank,
            pearson_rank=pearson_rank,
            atlas_prior=safe_prior,
            heterogeneity_signal=weights["heterogeneity_signal"],
            atlas_guard=weights["atlas_guard"],
            env_mode="real_batch_only",
        )
        stability_component, stability_uncertainty, split_count = _score_sigma_stability_component(
            counts=counts,
            batches=batches,
            random_state=random_state,
            variance_rank=rank_map["variance"],
            mv_rank=mv_rank,
            fano_rank=fano_rank,
            deviance_rank=deviance_rank,
            pearson_rank=pearson_rank,
        )

    target = 0.68 * count_bridge + 0.20 * env_component + 0.12 * stability_component
    positive_support = (
        0.50 * np.maximum(deviance_rank - safe_prior, 0.0)
        + 0.30 * np.maximum(pearson_rank - safe_prior, 0.0)
        + 0.20 * np.maximum(mv_rank - safe_prior, 0.0)
    )
    agreement = _zscore(
        -np.std(
            np.vstack([count_bridge, env_component, stability_component, safe_prior]),
            axis=0,
        )
    )
    admissibility_raw = 0.70 * _zscore(positive_support) + 0.30 * agreement
    admissibility = np.clip((_rank_percentile(admissibility_raw) - 0.60) / 0.25, 0.0, 1.0)
    delta = target - safe_prior
    gated_delta = admissibility * (np.maximum(delta, 0.0) + 0.04 * np.minimum(delta, 0.0))

    uncertainty = np.mean(
        np.vstack(
            [
                _rank_percentile(np.std(np.vstack([count_bridge, env_component, stability_component]), axis=0)),
                _rank_percentile(env_uncertainty),
                _rank_percentile(stability_uncertainty),
                np.full_like(delta, weights["atlas_guard"], dtype=np.float64),
            ]
        ),
        axis=0,
    )
    uncertainty_sq = np.square(uncertainty)
    tau_sq = max(float(np.percentile(uncertainty_sq, 35.0)), 1e-6)
    shrink_factor = tau_sq / (tau_sq + 3.20 * uncertainty_sq + 1e-8)

    score = safe_prior + shrink_factor * donor_budget * gated_delta
    score += donor_budget * 0.01 * _zscore(admissibility_raw)

    metadata: dict[str, float | str] = {
        "variant": "sigma_safe_core_v4_hvg",
        "consensus_family": "sigma_batch_gated_selective_update",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "sigma_safe_core_v4",
        "route_name": "sigma_safe_core_v4",
        "route_target": "single_score",
        "sigma_base_weight": float(1.0 - donor_budget),
        "sigma_count_bridge_weight": donor_budget * 0.68,
        "sigma_inv_weight": donor_budget * 0.20,
        "sigma_graph_weight": 0.0,
        "sigma_stability_weight": donor_budget * 0.12,
        "sigma_mean_uncertainty": float(np.mean(uncertainty)),
        "sigma_mean_shrink_factor": float(np.mean(shrink_factor)),
        "sigma_tau_sq": tau_sq,
        "sigma_shrink_strength": 3.20,
        "sigma_environment_count": env_count,
        "sigma_environment_source": env_source,
        "sigma_split_count": split_count,
        "sigma_use_invariance": float(donor_budget > 1e-8),
        "sigma_use_graph": 0.0,
        "sigma_use_stability": float(donor_budget > 1e-8),
        "sigma_use_shrink": 1.0,
        "sigma_use_deviance": 1.0,
        "sigma_use_pearson": 1.0,
        "sigma_donor_budget": donor_budget,
        "sigma_batch_gate": batch_gate,
        "sigma_mean_admissibility": float(np.mean(admissibility)),
        "adaptive_stat_route_name": adaptive_decision.route_name,
        **weights,
    }
    return _zscore(score), profile, metadata


def score_sigma_safe_core_v5_hvg(
    counts: np.ndarray,
    batches: np.ndarray | None = None,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    variance_rank = rank_map["variance"]
    mv_rank = rank_map["mv_residual"]
    fano_rank = rank_map["fano"]
    deviance_rank = rank_map["multinomial_deviance_hvg"]
    pearson_rank = rank_map["analytic_pearson_residual_hvg"]

    adaptive_decision = resolve_adaptive_stat_decision(profile)
    adaptive_anchor_core = score_adaptive_stat_hvg_with_decision(counts, adaptive_decision)
    safe_prior = _rank_percentile(adaptive_anchor_core)
    count_bridge = 0.44 * mv_rank + 0.28 * deviance_rank + 0.18 * pearson_rank + 0.10 * fano_rank

    batch_gate = _scale_unit(profile.batch_classes, 2.0, 6.0)
    donor_budget = float(
        np.clip(
            0.30 * batch_gate
            + 0.08 * weights["heterogeneity_signal"]
            - 0.08 * weights["atlas_guard"],
            0.0,
            0.34,
        )
    )

    env_component = count_bridge.copy()
    env_uncertainty = np.zeros_like(count_bridge, dtype=np.float64)
    env_count = 0.0
    env_source = "disabled"
    stability_component = count_bridge.copy()
    stability_uncertainty = np.zeros_like(count_bridge, dtype=np.float64)
    split_count = 0.0
    if donor_budget > 1e-8:
        env_component, env_uncertainty, env_count, env_source = _score_sigma_invariance_component(
            counts=counts,
            batches=batches,
            random_state=random_state,
            mv_rank=mv_rank,
            deviance_rank=deviance_rank,
            pearson_rank=pearson_rank,
            atlas_prior=safe_prior,
            heterogeneity_signal=weights["heterogeneity_signal"],
            atlas_guard=weights["atlas_guard"],
            env_mode="real_batch_only",
        )
        stability_component, stability_uncertainty, split_count = _score_sigma_stability_component(
            counts=counts,
            batches=batches,
            random_state=random_state,
            variance_rank=variance_rank,
            mv_rank=mv_rank,
            fano_rank=fano_rank,
            deviance_rank=deviance_rank,
            pearson_rank=pearson_rank,
        )

    target = 0.56 * count_bridge + 0.28 * env_component + 0.16 * stability_component
    disagreement = np.std(
        np.vstack([safe_prior, count_bridge, env_component, stability_component]),
        axis=0,
    )
    uncertainty = np.mean(
        np.vstack(
            [
                _rank_percentile(disagreement),
                _rank_percentile(env_uncertainty),
                _rank_percentile(stability_uncertainty),
                np.full_like(disagreement, weights["atlas_guard"], dtype=np.float64),
            ]
        ),
        axis=0,
    )
    uncertainty_sq = np.square(uncertainty)
    tau_sq = max(float(np.percentile(uncertainty_sq, 45.0)), 1e-6)
    shrink_factor = tau_sq / (tau_sq + 1.35 * uncertainty_sq + 1e-8)

    score = safe_prior + donor_budget * shrink_factor * (target - safe_prior)
    score += donor_budget * 0.025 * _zscore(-disagreement)

    metadata: dict[str, float | str] = {
        "variant": "sigma_safe_core_v5_hvg",
        "consensus_family": "sigma_batch_gated_stronger_update",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "sigma_safe_core_v5",
        "route_name": "sigma_safe_core_v5",
        "route_target": "single_score",
        "sigma_base_weight": float(1.0 - donor_budget),
        "sigma_count_bridge_weight": donor_budget * 0.56,
        "sigma_inv_weight": donor_budget * 0.28,
        "sigma_graph_weight": 0.0,
        "sigma_stability_weight": donor_budget * 0.16,
        "sigma_mean_uncertainty": float(np.mean(uncertainty)),
        "sigma_mean_shrink_factor": float(np.mean(shrink_factor)),
        "sigma_tau_sq": tau_sq,
        "sigma_shrink_strength": 1.35,
        "sigma_environment_count": env_count,
        "sigma_environment_source": env_source,
        "sigma_split_count": split_count,
        "sigma_use_invariance": float(donor_budget > 1e-8),
        "sigma_use_graph": 0.0,
        "sigma_use_stability": float(donor_budget > 1e-8),
        "sigma_use_shrink": 1.0,
        "sigma_use_deviance": 1.0,
        "sigma_use_pearson": 1.0,
        "sigma_donor_budget": donor_budget,
        "sigma_batch_gate": batch_gate,
        "adaptive_stat_route_name": adaptive_decision.route_name,
        **weights,
    }
    return _zscore(score), profile, metadata


def _score_sigma_variant(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None,
    random_state: int,
    variant_name: str,
    use_invariance: bool,
    use_graph: bool,
    use_stability: bool,
    use_shrink: bool,
    env_mode: str,
    use_deviance: bool,
    use_pearson: bool,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, stack = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    variance_rank = rank_map["variance"]
    mv_rank = rank_map["mv_residual"]
    fano_rank = rank_map["fano"]
    deviance_rank = rank_map["multinomial_deviance_hvg"]
    pearson_rank = rank_map["analytic_pearson_residual_hvg"]

    deviance_core = deviance_rank if use_deviance else mv_rank
    pearson_core = pearson_rank if use_pearson else mv_rank
    atlas_prior = 0.52 * variance_rank + 0.24 * mv_rank + 0.14 * pearson_core + 0.10 * fano_rank
    base_component = (
        0.30 * mv_rank
        + 0.28 * deviance_core
        + 0.18 * pearson_core
        + 0.14 * variance_rank
        + 0.10 * fano_rank
    )

    invariance_component = base_component.copy()
    environment_count = 0.0
    environment_source = "disabled"
    invariance_uncertainty = np.zeros_like(base_component, dtype=np.float64)
    if use_invariance:
        invariance_component, invariance_uncertainty, environment_count, environment_source = _score_sigma_invariance_component(
            counts=counts,
            batches=batches,
            random_state=random_state,
            mv_rank=mv_rank,
            deviance_rank=deviance_core,
            pearson_rank=pearson_core,
            atlas_prior=atlas_prior,
            heterogeneity_signal=weights["heterogeneity_signal"],
            atlas_guard=weights["atlas_guard"],
            env_mode=env_mode,
        )

    graph_component = base_component.copy()
    graph_uncertainty = np.zeros_like(base_component, dtype=np.float64)
    graph_sampled_cells = 0.0
    if use_graph:
        graph_component, graph_uncertainty, graph_sampled_cells = _score_sigma_graph_component(
            counts=counts,
            random_state=random_state,
            variance_rank=variance_rank,
            fano_rank=fano_rank,
            deviance_rank=deviance_core,
            pearson_rank=pearson_core,
            atlas_prior=atlas_prior,
            atlas_guard=weights["atlas_guard"],
        )

    stability_component = base_component.copy()
    stability_uncertainty = np.zeros_like(base_component, dtype=np.float64)
    split_count = 0.0
    if use_stability:
        stability_component, stability_uncertainty, split_count = _score_sigma_stability_component(
            counts=counts,
            batches=batches,
            random_state=random_state,
            variance_rank=variance_rank,
            mv_rank=mv_rank,
            fano_rank=fano_rank,
            deviance_rank=deviance_core,
            pearson_rank=pearson_core,
        )

    component_stack = [base_component]
    raw_weights = [0.50 + 0.22 * weights["atlas_guard"]]
    if use_invariance:
        component_stack.append(invariance_component)
        raw_weights.append(0.12 + 0.28 * weights["heterogeneity_signal"])
    if use_graph:
        component_stack.append(graph_component)
        raw_weights.append(0.08 + 0.12 * weights["heterogeneity_signal"] + 0.18 * weights["trajectory_signal"])
    if use_stability:
        component_stack.append(stability_component)
        raw_weights.append(0.12 + 0.10 * weights["heterogeneity_signal"] + 0.10 * weights["trajectory_signal"])
    component_matrix = np.vstack(component_stack)
    normalized_component_weights = np.asarray(raw_weights, dtype=np.float64)
    normalized_component_weights = normalized_component_weights / max(normalized_component_weights.sum(), 1e-8)
    target = np.zeros_like(base_component, dtype=np.float64)
    for component_weight, component in zip(normalized_component_weights, component_stack, strict=False):
        target += float(component_weight) * component

    disagreement = np.std(component_matrix, axis=0)
    uncertainty_terms = [_rank_percentile(disagreement)]
    if use_invariance:
        uncertainty_terms.append(_rank_percentile(invariance_uncertainty))
    if use_graph:
        uncertainty_terms.append(_rank_percentile(graph_uncertainty))
    if use_stability:
        uncertainty_terms.append(_rank_percentile(stability_uncertainty))
    uncertainty = np.mean(np.vstack(uncertainty_terms), axis=0)
    uncertainty_sq = np.square(uncertainty)
    tau_sq = max(float(np.percentile(uncertainty_sq, 55.0)), 1e-6)
    shrink_factor = tau_sq / (tau_sq + uncertainty_sq + 1e-8)
    safe_target = target + 0.06 * _zscore(-disagreement)
    if use_shrink:
        score = atlas_prior + shrink_factor * (safe_target - atlas_prior)
    else:
        score = safe_target
        shrink_factor = np.ones_like(score, dtype=np.float64)
    score += 0.04 * weights["trajectory_signal"] * (fano_rank - atlas_prior)
    score += 0.05 * weights["heterogeneity_signal"] * (deviance_core - atlas_prior)

    metadata: dict[str, float | str] = {
        "variant": variant_name,
        "consensus_family": "robust_invariant_graph_shrinkage",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "sigma_anchor_bank_v1",
        "route_name": "sigma_single_score",
        "route_target": "single_score",
        "sigma_base_weight": float(normalized_component_weights[0]),
        "sigma_inv_weight": float(normalized_component_weights[1]) if use_invariance else 0.0,
        "sigma_graph_weight": float(normalized_component_weights[1 + int(use_invariance)]) if use_graph else 0.0,
        "sigma_stability_weight": float(normalized_component_weights[-1]) if use_stability else 0.0,
        "sigma_mean_uncertainty": float(np.mean(uncertainty)),
        "sigma_mean_shrink_factor": float(np.mean(shrink_factor)),
        "sigma_tau_sq": tau_sq,
        "sigma_environment_count": environment_count,
        "sigma_environment_source": environment_source,
        "sigma_split_count": split_count,
        "sigma_graph_sampled_cells": graph_sampled_cells,
        "sigma_use_invariance": float(use_invariance),
        "sigma_use_graph": float(use_graph),
        "sigma_use_stability": float(use_stability),
        "sigma_use_shrink": float(use_shrink),
        "sigma_use_deviance": float(use_deviance),
        "sigma_use_pearson": float(use_pearson),
        **weights,
    }
    return _zscore(score), profile, metadata


def score_sigma_safe_core_v6_hvg(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None,
    anchor_scores: np.ndarray,
    profile: AdaptiveStatProfile,
    random_state: int = 0,
) -> tuple[np.ndarray, dict[str, float | str]]:
    _, rank_map, _ = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    mv_rank = rank_map["mv_residual"]
    deviance_rank = rank_map["multinomial_deviance_hvg"]
    pearson_rank = rank_map["analytic_pearson_residual_hvg"]
    anchor_rank = _rank_percentile(anchor_scores)

    batch_gate = _scale_unit(profile.batch_classes, 3.0, 10.0)
    batch_strength_gate = _scale_unit(profile.batch_strength, 0.015, 0.06)
    donor_budget = float(
        np.clip(
            0.20 * batch_gate
            + 0.12 * batch_strength_gate
            + 0.04 * weights["heterogeneity_signal"]
            - 0.06 * weights["atlas_guard"],
            0.0,
            0.24,
        )
    )

    metadata: dict[str, float | str] = {
        "variant": "sigma_safe_core_v6_hvg",
        "consensus_family": "sigma_anchor_reuse_real_batch_upgrade",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "sigma_safe_core_v6",
        "route_name": "sigma_safe_core_v6",
        "route_target": "single_score",
        "sigma_anchor_reuse": 1.0,
        "sigma_base_weight": float(1.0 - donor_budget),
        "sigma_count_bridge_weight": donor_budget * 0.60,
        "sigma_inv_weight": donor_budget * 0.40,
        "sigma_graph_weight": 0.0,
        "sigma_stability_weight": 0.0,
        "sigma_mean_uncertainty": 0.0,
        "sigma_mean_shrink_factor": 1.0,
        "sigma_tau_sq": 0.0,
        "sigma_shrink_strength": 2.10,
        "sigma_environment_count": 0.0,
        "sigma_environment_source": "anchor_reuse",
        "sigma_split_count": 0.0,
        "sigma_use_invariance": 0.0,
        "sigma_use_graph": 0.0,
        "sigma_use_stability": 0.0,
        "sigma_use_shrink": 1.0,
        "sigma_use_deviance": 1.0,
        "sigma_use_pearson": 1.0,
        "sigma_donor_budget": donor_budget,
        "sigma_batch_gate": batch_gate,
        "sigma_batch_strength_gate": batch_strength_gate,
        "sigma_mean_admissibility": 0.0,
        **weights,
    }
    if donor_budget <= 1e-8 or batches is None or profile.batch_classes < 3:
        metadata["route_name"] = "sigma_safe_core_v6::anchor_reuse"
        return np.asarray(anchor_scores, dtype=np.float64), metadata

    count_bridge = 0.50 * mv_rank + 0.32 * deviance_rank + 0.18 * pearson_rank
    env_component, env_uncertainty, env_count, env_source = _score_sigma_invariance_component(
        counts=counts,
        batches=batches,
        random_state=random_state,
        mv_rank=mv_rank,
        deviance_rank=deviance_rank,
        pearson_rank=pearson_rank,
        atlas_prior=anchor_rank,
        heterogeneity_signal=weights["heterogeneity_signal"],
        atlas_guard=weights["atlas_guard"],
        env_mode="real_batch_only",
    )
    metadata["sigma_environment_count"] = env_count
    metadata["sigma_environment_source"] = env_source

    if env_source != "real_batch" or env_count < 3:
        metadata["route_name"] = "sigma_safe_core_v6::anchor_reuse"
        return np.asarray(anchor_scores, dtype=np.float64), metadata

    target = 0.60 * count_bridge + 0.40 * env_component
    agreement = _zscore(-np.std(np.vstack([anchor_rank, count_bridge, env_component]), axis=0))
    positive_support = (
        0.55 * np.maximum(deviance_rank - anchor_rank, 0.0)
        + 0.30 * np.maximum(pearson_rank - anchor_rank, 0.0)
        + 0.15 * np.maximum(mv_rank - anchor_rank, 0.0)
    )
    admissibility_raw = 0.65 * _zscore(positive_support) + 0.35 * agreement
    admissibility = np.clip((_rank_percentile(admissibility_raw) - 0.55) / 0.25, 0.0, 1.0)
    delta = target - anchor_rank
    gated_delta = admissibility * (np.maximum(delta, 0.0) + 0.02 * np.minimum(delta, 0.0))

    disagreement = np.std(np.vstack([anchor_rank, count_bridge, env_component]), axis=0)
    uncertainty = np.mean(
        np.vstack(
            [
                _rank_percentile(disagreement),
                _rank_percentile(env_uncertainty),
                np.full_like(disagreement, weights["atlas_guard"], dtype=np.float64),
            ]
        ),
        axis=0,
    )
    uncertainty_sq = np.square(uncertainty)
    tau_sq = max(float(np.percentile(uncertainty_sq, 40.0)), 1e-6)
    shrink_factor = tau_sq / (tau_sq + 2.10 * uncertainty_sq + 1e-8)

    score = anchor_rank + donor_budget * shrink_factor * gated_delta
    score += donor_budget * 0.015 * _zscore(admissibility_raw)

    metadata["sigma_use_invariance"] = 1.0
    metadata["sigma_mean_uncertainty"] = float(np.mean(uncertainty))
    metadata["sigma_mean_shrink_factor"] = float(np.mean(shrink_factor))
    metadata["sigma_tau_sq"] = tau_sq
    metadata["sigma_mean_admissibility"] = float(np.mean(admissibility))
    return _zscore(score), metadata


def _score_sigma_safe_core_variant(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None,
    random_state: int,
    variant_name: str,
    use_real_batch_invariance: bool,
    use_stability: bool,
    use_deviance: bool,
    use_pearson: bool,
    shrink_strength: float,
) -> tuple[np.ndarray, AdaptiveStatProfile, dict[str, float | str]]:
    profile = compute_adaptive_stat_profile(
        counts=counts,
        batches=batches,
        random_state=random_state,
    )
    _, rank_map, stack = _compute_anchor_core_signal_maps(counts)
    weights = _compute_anchor_core_signals(profile)

    variance_rank = rank_map["variance"]
    mv_rank = rank_map["mv_residual"]
    fano_rank = rank_map["fano"]
    deviance_rank = rank_map["multinomial_deviance_hvg"] if use_deviance else mv_rank
    pearson_rank = rank_map["analytic_pearson_residual_hvg"] if use_pearson else mv_rank

    adaptive_decision = resolve_adaptive_stat_decision(profile)
    adaptive_anchor_core = score_adaptive_stat_hvg_with_decision(counts, adaptive_decision)
    adaptive_anchor_rank = _rank_percentile(adaptive_anchor_core)

    atlas_prior = (
        0.46 * adaptive_anchor_rank
        + 0.28 * variance_rank
        + 0.16 * mv_rank
        + 0.10 * pearson_rank
    )
    count_bridge = 0.44 * mv_rank + 0.28 * deviance_rank + 0.18 * pearson_rank + 0.10 * fano_rank
    bridge_cap = float(
        np.clip(
            0.06 + 0.12 * weights["heterogeneity_signal"] + 0.08 * weights["trajectory_signal"] - 0.10 * weights["atlas_guard"],
            0.02,
            0.18,
        )
    )
    target = atlas_prior + bridge_cap * (count_bridge - atlas_prior)

    env_component = target.copy()
    env_uncertainty = np.zeros_like(target, dtype=np.float64)
    env_count = 0.0
    env_source = "disabled"
    if use_real_batch_invariance:
        env_component, env_uncertainty, env_count, env_source = _score_sigma_invariance_component(
            counts=counts,
            batches=batches,
            random_state=random_state,
            mv_rank=mv_rank,
            deviance_rank=deviance_rank,
            pearson_rank=pearson_rank,
            atlas_prior=atlas_prior,
            heterogeneity_signal=weights["heterogeneity_signal"],
            atlas_guard=weights["atlas_guard"],
            env_mode="real_batch_only",
        )
        env_step = 0.06 + 0.12 * weights["heterogeneity_signal"]
        target += env_step * (env_component - target)

    stability_component = target.copy()
    stability_uncertainty = np.zeros_like(target, dtype=np.float64)
    split_count = 0.0
    if use_stability:
        stability_component, stability_uncertainty, split_count = _score_sigma_stability_component(
            counts=counts,
            batches=batches,
            random_state=random_state,
            variance_rank=variance_rank,
            mv_rank=mv_rank,
            fano_rank=fano_rank,
            deviance_rank=deviance_rank,
            pearson_rank=pearson_rank,
        )
        stability_step = 0.05 + 0.06 * weights["trajectory_signal"]
        target += stability_step * (stability_component - target)

    disagreement = np.std(
        np.vstack(
            [
                atlas_prior,
                count_bridge,
                env_component,
                stability_component,
                adaptive_anchor_rank,
            ]
        ),
        axis=0,
    )
    uncertainty = np.mean(
        np.vstack(
            [
                _rank_percentile(disagreement),
                _rank_percentile(env_uncertainty),
                _rank_percentile(stability_uncertainty),
                np.full_like(disagreement, weights["atlas_guard"], dtype=np.float64),
            ]
        ),
        axis=0,
    )
    uncertainty_sq = np.square(uncertainty)
    tau_sq = max(float(np.percentile(uncertainty_sq, 45.0)), 1e-6)
    shrink_factor = tau_sq / (tau_sq + shrink_strength * uncertainty_sq + 1e-8)

    score = atlas_prior + shrink_factor * (target - atlas_prior)
    score += 0.03 * _zscore(-disagreement)

    metadata: dict[str, float | str] = {
        "variant": variant_name,
        "consensus_family": "sigma_safe_core",
        "selector_policy": "single_scorer_fixed_fusion",
        "selector_stage": "anchor_core_upgrade",
        "selector_bank_name": "sigma_safe_core_v1",
        "route_name": "sigma_safe_core",
        "route_target": "single_score",
        "sigma_base_weight": 1.0 - bridge_cap,
        "sigma_count_bridge_weight": bridge_cap,
        "sigma_inv_weight": float(0.0 if not use_real_batch_invariance else 0.06 + 0.12 * weights["heterogeneity_signal"]),
        "sigma_graph_weight": 0.0,
        "sigma_stability_weight": float(0.0 if not use_stability else 0.05 + 0.06 * weights["trajectory_signal"]),
        "sigma_mean_uncertainty": float(np.mean(uncertainty)),
        "sigma_mean_shrink_factor": float(np.mean(shrink_factor)),
        "sigma_tau_sq": tau_sq,
        "sigma_shrink_strength": shrink_strength,
        "sigma_environment_count": env_count,
        "sigma_environment_source": env_source,
        "sigma_split_count": split_count,
        "sigma_use_invariance": float(use_real_batch_invariance),
        "sigma_use_graph": 0.0,
        "sigma_use_stability": float(use_stability),
        "sigma_use_shrink": 1.0,
        "sigma_use_deviance": float(use_deviance),
        "sigma_use_pearson": float(use_pearson),
        "adaptive_stat_route_name": adaptive_decision.route_name,
        **weights,
    }
    return _zscore(score), profile, metadata


def _score_sigma_invariance_component(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None,
    random_state: int,
    mv_rank: np.ndarray,
    deviance_rank: np.ndarray,
    pearson_rank: np.ndarray,
    atlas_prior: np.ndarray,
    heterogeneity_signal: float,
    atlas_guard: float,
    env_mode: str,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    if env_mode == "pseudo_only":
        sample_idx, env_labels = _build_pseudo_environment_labels(
            counts=counts,
            batches=None,
            random_state=random_state,
            max_cells=2400,
            target_envs=4,
        )
        env_source = "pseudo_only"
    elif env_mode == "real_batch_only":
        sample_idx, env_labels, env_source = _build_invariant_environment_labels(
            counts=counts,
            batches=batches,
            random_state=random_state,
            max_cells=2400,
            target_envs=4,
            allow_pseudo_fallback=False,
        )
    else:
        sample_idx, env_labels, env_source = _build_invariant_environment_labels(
            counts=counts,
            batches=batches,
            random_state=random_state,
            max_cells=2400,
            target_envs=4,
            allow_pseudo_fallback=True,
        )

    counts_sample = counts[sample_idx]
    unique_envs = [label for label in np.unique(env_labels) if np.sum(env_labels == label) >= 64]
    env_scores: list[np.ndarray] = []
    for label in unique_envs:
        env_counts = counts_sample[env_labels == label]
        if env_counts.shape[0] < 64:
            continue
        env_mv = _rank_percentile(score_mean_var_residual_like_counts(env_counts))
        env_dev = _rank_percentile(score_multinomial_deviance_hvg(np.asarray(env_counts, dtype=np.float32)))
        env_pearson = _rank_percentile(score_analytic_pearson_residual_hvg(np.asarray(env_counts, dtype=np.float32)))
        env_scores.append(0.48 * env_dev + 0.28 * env_pearson + 0.24 * env_mv)

    if len(env_scores) >= 2:
        env_matrix = np.vstack(env_scores)
        worst_group = np.quantile(env_matrix, 0.25, axis=0)
        mean_group = np.mean(env_matrix, axis=0)
        env_std = np.std(env_matrix, axis=0)
    else:
        worst_group = deviance_rank
        mean_group = 0.60 * deviance_rank + 0.25 * pearson_rank + 0.15 * mv_rank
        env_std = np.zeros_like(worst_group, dtype=np.float64)
        env_source = "global_fallback" if env_source == "global_fallback" else env_source

    component = 0.44 * worst_group + 0.28 * mean_group + 0.16 * pearson_rank + 0.12 * mv_rank
    component -= (0.14 + 0.10 * heterogeneity_signal) * env_std
    component += 0.06 * heterogeneity_signal * (deviance_rank - atlas_prior)
    component += 0.06 * atlas_guard * (atlas_prior - component)
    return component, env_std, float(len(unique_envs)), env_source


def _score_sigma_graph_component(
    *,
    counts: np.ndarray,
    random_state: int,
    variance_rank: np.ndarray,
    fano_rank: np.ndarray,
    deviance_rank: np.ndarray,
    pearson_rank: np.ndarray,
    atlas_prior: np.ndarray,
    atlas_guard: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    sample_idx = _sample_cell_indices(counts.shape[0], max_cells=2200, random_state=random_state)
    counts_sample = counts[sample_idx]
    x_sample = _normalize_log1p_fast(counts_sample)
    if x_sample.shape[0] < 32 or x_sample.shape[1] < 32:
        fallback = 0.45 * fano_rank + 0.25 * deviance_rank + 0.20 * pearson_rank + 0.10 * variance_rank
        return fallback, np.zeros_like(fallback, dtype=np.float64), float(len(sample_idx))

    graph_gene_count = min(3000, x_sample.shape[1])
    gene_var = np.var(x_sample, axis=0, dtype=np.float64)
    gene_idx = np.argsort(gene_var)[-graph_gene_count:]
    graph_input = x_sample[:, np.sort(gene_idx)]
    pca_dim = min(30, graph_input.shape[0] - 1, graph_input.shape[1])
    if pca_dim < 2:
        fallback = 0.45 * fano_rank + 0.25 * deviance_rank + 0.20 * pearson_rank + 0.10 * variance_rank
        return fallback, np.zeros_like(fallback, dtype=np.float64), float(len(sample_idx))

    embedding = PCA(n_components=pca_dim, svd_solver="randomized", random_state=random_state).fit_transform(graph_input)
    n_neighbors = min(15, max(4, embedding.shape[0] - 1))
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn.fit(embedding)
    neighbor_idx = knn.kneighbors(return_distance=False)[:, 1:]
    adjacency = np.zeros((embedding.shape[0], embedding.shape[0]), dtype=np.float32)
    adjacency[np.arange(embedding.shape[0])[:, None], neighbor_idx] = 1.0 / max(float(n_neighbors), 1.0)
    locality = _zscore(_compute_graph_locality_scores(x=x_sample, adjacency=adjacency, chunk_size=512))
    component = 0.42 * locality + 0.24 * fano_rank + 0.18 * deviance_rank + 0.16 * pearson_rank
    graph_uncertainty = _rank_percentile(np.maximum(locality - atlas_prior, 0.0))
    component += 0.08 * atlas_guard * (atlas_prior - component)
    return component, graph_uncertainty, float(len(sample_idx))


def _score_sigma_stability_component(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None,
    random_state: int,
    variance_rank: np.ndarray,
    mv_rank: np.ndarray,
    fano_rank: np.ndarray,
    deviance_rank: np.ndarray,
    pearson_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    sample_idx, env_labels = _build_pseudo_environment_labels(
        counts=counts,
        batches=batches,
        random_state=random_state,
        max_cells=2400,
        target_envs=4,
    )
    split_indices = _build_stability_splits(
        n_cells=len(sample_idx),
        env_labels=env_labels,
        n_splits=3,
        random_state=random_state,
    )
    counts_sample = counts[sample_idx]

    split_scores: list[np.ndarray] = []
    for heldout_split in split_indices:
        keep_mask = np.ones(len(sample_idx), dtype=bool)
        keep_mask[heldout_split] = False
        split_counts = counts_sample[keep_mask]
        if split_counts.shape[0] < 96:
            continue
        _, split_rank_map, _ = _compute_anchor_core_signal_maps(split_counts)
        split_scores.append(
            0.30 * split_rank_map["mv_residual"]
            + 0.25 * split_rank_map["multinomial_deviance_hvg"]
            + 0.18 * split_rank_map["analytic_pearson_residual_hvg"]
            + 0.17 * split_rank_map["variance"]
            + 0.10 * split_rank_map["fano"]
        )

    full_core = 0.30 * mv_rank + 0.25 * deviance_rank + 0.18 * pearson_rank + 0.17 * variance_rank + 0.10 * fano_rank
    if split_scores:
        split_stack = np.vstack(split_scores)
        stable_mean = np.mean(split_stack, axis=0)
        stable_std = np.std(split_stack, axis=0)
        component = 0.70 * full_core + 0.30 * (stable_mean - 0.40 * stable_std)
        return component, stable_std, float(len(split_scores))
    return full_core, np.zeros_like(full_core, dtype=np.float64), 0.0


def profile_to_dict(profile: AdaptiveStatProfile, decision: AdaptiveStatDecision) -> dict[str, float | str]:
    del decision
    return {key: float(value) for key, value in asdict(profile).items()}


def _silhouette_safe(embedding: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or embedding.shape[0] <= len(unique):
        return 0.0
    try:
        return float(silhouette_score(embedding, labels))
    except ValueError:
        return 0.0


def _compute_base_scores_from_normalized(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    variance_scores = _zscore(var)
    fano_scores = _zscore(var / np.maximum(mean, 1e-6))

    log_mean = np.log1p(mean)
    log_var = np.log1p(var)
    coeff = np.polyfit(log_mean, log_var, deg=2)
    expected = np.polyval(coeff, log_mean)
    mv_scores = _zscore(log_var - expected)
    return variance_scores, mv_scores, fano_scores


def _compute_anchor_core_signal_maps(
    counts: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    x = normalize_log1p(counts)
    variance_scores, mv_scores, fano_scores = _compute_base_scores_from_normalized(x)
    counts_f32 = np.asarray(counts, dtype=np.float32)
    deviance_scores = score_multinomial_deviance_hvg(counts_f32)
    pearson_scores = score_analytic_pearson_residual_hvg(counts_f32)
    score_map = {
        "variance": variance_scores,
        "mv_residual": mv_scores,
        "fano": fano_scores,
        "multinomial_deviance_hvg": deviance_scores,
        "analytic_pearson_residual_hvg": pearson_scores,
    }
    rank_map = {name: _rank_percentile(values) for name, values in score_map.items()}
    stack = np.vstack(
        [
            rank_map["variance"],
            rank_map["mv_residual"],
            rank_map["fano"],
            rank_map["multinomial_deviance_hvg"],
            rank_map["analytic_pearson_residual_hvg"],
        ]
    )
    return score_map, rank_map, stack


def _compute_anchor_core_rank_scores(counts: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    _, rank_map_full, _ = _compute_anchor_core_signal_maps(counts)
    rank_map = {
        "variance": rank_map_full["variance"],
        "mv_residual": rank_map_full["mv_residual"],
        "fano": rank_map_full["fano"],
        "multinomial_deviance_hvg": rank_map_full["multinomial_deviance_hvg"],
    }
    stacked = np.vstack(
        [
            rank_map["variance"],
            rank_map["mv_residual"],
            rank_map["fano"],
            rank_map["multinomial_deviance_hvg"],
        ]
    )
    return stacked, rank_map


def _sample_cell_indices(n_cells: int, *, max_cells: int, random_state: int) -> np.ndarray:
    if n_cells <= max_cells:
        return np.arange(n_cells, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_cells, size=max_cells, replace=False)).astype(np.int64, copy=False)


def _build_pseudo_environment_labels(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None,
    random_state: int,
    max_cells: int,
    target_envs: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_idx = _sample_cell_indices(counts.shape[0], max_cells=max_cells, random_state=random_state)
    if batches is not None:
        batch_sample = np.asarray(batches)[sample_idx]
        _, counts_per_batch = np.unique(batch_sample, return_counts=True)
        if counts_per_batch.size >= 2 and int(np.sum(counts_per_batch >= 32)) >= 2:
            _, inverse = np.unique(batch_sample, return_inverse=True)
            return sample_idx, inverse.astype(np.int32, copy=False)

    counts_sample = counts[sample_idx]
    x_sample = _normalize_log1p_fast(counts_sample)
    if x_sample.shape[0] < 8 or x_sample.shape[1] < 8:
        env_labels = np.arange(x_sample.shape[0], dtype=np.int32) % max(target_envs, 2)
        return sample_idx, env_labels

    graph_gene_count = min(2000, x_sample.shape[1])
    gene_var = np.var(x_sample, axis=0, dtype=np.float64)
    gene_idx = np.argsort(gene_var)[-graph_gene_count:]
    graph_input = x_sample[:, np.sort(gene_idx)]
    pca_dim = min(15, graph_input.shape[0] - 1, graph_input.shape[1])
    if pca_dim < 2:
        env_labels = np.arange(x_sample.shape[0], dtype=np.int32) % max(target_envs, 2)
        return sample_idx, env_labels
    embedding = PCA(n_components=pca_dim, svd_solver="randomized", random_state=random_state).fit_transform(graph_input)
    n_envs = min(target_envs, max(2, int(np.sqrt(len(sample_idx) / 120.0)) + 1))
    env_labels = KMeans(n_clusters=n_envs, n_init=5, random_state=random_state).fit_predict(embedding)
    return sample_idx, env_labels.astype(np.int32, copy=False)


def _build_invariant_environment_labels(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None,
    random_state: int,
    max_cells: int,
    target_envs: int,
    allow_pseudo_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    sample_idx = _sample_cell_indices(counts.shape[0], max_cells=max_cells, random_state=random_state)
    if batches is not None:
        batch_sample = np.asarray(batches)[sample_idx]
        _, inverse = np.unique(batch_sample, return_inverse=True)
        counts_per_batch = np.bincount(inverse)
        if counts_per_batch.size >= 2 and int(np.sum(counts_per_batch >= 32)) >= 2:
            return sample_idx, inverse.astype(np.int32, copy=False), "real_batch"

    if allow_pseudo_fallback:
        _, env_labels = _build_pseudo_environment_labels(
            counts=counts,
            batches=None,
            random_state=random_state,
            max_cells=max_cells,
            target_envs=target_envs,
        )
        return sample_idx, env_labels.astype(np.int32, copy=False), "pseudo_kmeans"

    # Return a single environment so the caller cleanly falls back to the non-invariant global score.
    return sample_idx, np.zeros(len(sample_idx), dtype=np.int32), "global_fallback"


def _build_stability_splits(
    *,
    n_cells: int,
    env_labels: np.ndarray,
    n_splits: int,
    random_state: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(random_state)
    splits: list[list[int]] = [[] for _ in range(max(n_splits, 2))]
    for label in np.unique(env_labels):
        label_idx = np.flatnonzero(env_labels == label).astype(np.int64, copy=False)
        rng.shuffle(label_idx)
        for offset, cell_idx in enumerate(label_idx.tolist()):
            splits[offset % len(splits)].append(int(cell_idx))
    realized = [np.asarray(part, dtype=np.int64) for part in splits if part]
    if not realized:
        all_idx = np.arange(n_cells, dtype=np.int64)
        rng.shuffle(all_idx)
        return [all_idx[offset::n_splits] for offset in range(n_splits)]
    return realized


def _compute_graph_locality_scores(
    *,
    x: np.ndarray,
    adjacency: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    n_genes = x.shape[1]
    scores = np.zeros(n_genes, dtype=np.float64)
    use_cuda = torch.cuda.is_available() and adjacency.shape[0] <= 2400

    if use_cuda:
        device = torch.device("cuda")
        adjacency_tensor = torch.from_numpy(np.asarray(adjacency, dtype=np.float32)).to(device)
        for start in range(0, n_genes, chunk_size):
            end = min(start + chunk_size, n_genes)
            chunk = torch.from_numpy(np.asarray(x[:, start:end], dtype=np.float32)).to(device)
            chunk = chunk - chunk.mean(dim=0, keepdim=True)
            chunk = chunk / chunk.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            smooth = adjacency_tensor @ chunk
            locality = (chunk * smooth).mean(dim=0)
            residual = (chunk - smooth).pow(2).mean(dim=0)
            scores[start:end] = (locality - 0.18 * residual).detach().cpu().numpy()
        return scores

    adjacency_f32 = np.asarray(adjacency, dtype=np.float32)
    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        chunk = np.asarray(x[:, start:end], dtype=np.float32)
        chunk = chunk - np.mean(chunk, axis=0, keepdims=True, dtype=np.float32)
        chunk = chunk / np.maximum(np.std(chunk, axis=0, keepdims=True, dtype=np.float32), np.float32(1e-6))
        smooth = adjacency_f32 @ chunk
        locality = np.mean(chunk * smooth, axis=0, dtype=np.float64)
        residual = np.mean(np.square(chunk - smooth), axis=0, dtype=np.float64)
        scores[start:end] = locality - 0.18 * residual
    return scores


def _apply_diversity_rerank(
    *,
    counts: np.ndarray,
    base_score: np.ndarray,
    current_top_k: int,
    random_state: int,
    penalty_strength: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    n_genes = int(len(base_score))
    target_k = max(1, min(int(current_top_k), n_genes))
    if target_k < 2 or n_genes < 8:
        return np.asarray(base_score, dtype=np.float64), {
            "diversity_pool_size": 0.0,
            "diversity_selected_k": float(target_k),
            "diversity_sampled_cells": 0.0,
            "diversity_penalty_strength": penalty_strength,
            "diversity_device": "none",
        }

    candidate_pool = min(n_genes, max(target_k * 4, target_k + 64), 1200)
    candidate_idx = np.argsort(base_score)[-candidate_pool:]
    sample_idx = _sample_cell_indices(counts.shape[0], max_cells=1800, random_state=random_state)
    x_sample = _normalize_log1p_fast(counts[sample_idx])[:, candidate_idx]
    if x_sample.shape[0] < 16 or x_sample.shape[1] < target_k:
        return np.asarray(base_score, dtype=np.float64), {
            "diversity_pool_size": float(candidate_pool),
            "diversity_selected_k": float(target_k),
            "diversity_sampled_cells": float(len(sample_idx)),
            "diversity_penalty_strength": penalty_strength,
            "diversity_device": "fallback_skip",
        }

    abs_corr, corr_device = _compute_abs_correlation_matrix(x_sample)
    np.fill_diagonal(abs_corr, 0.0)
    candidate_base = _rank_percentile(base_score[candidate_idx])
    selected_local = _greedy_diversity_selection(
        base_values=candidate_base,
        abs_corr=abs_corr,
        target_k=target_k,
        penalty_strength=penalty_strength,
    )
    selected_idx = candidate_idx[selected_local]

    final_score = _rank_percentile(base_score) - 2.0
    selected_values = np.linspace(2.0, 1.0, num=target_k, endpoint=False, dtype=np.float64)
    final_score[selected_idx] = selected_values
    return final_score, {
        "diversity_pool_size": float(candidate_pool),
        "diversity_selected_k": float(target_k),
        "diversity_sampled_cells": float(len(sample_idx)),
        "diversity_penalty_strength": penalty_strength,
        "diversity_device": corr_device,
    }


def _compute_abs_correlation_matrix(x: np.ndarray) -> tuple[np.ndarray, str]:
    x_f32 = np.asarray(x, dtype=np.float32)
    use_cuda = torch.cuda.is_available() and x_f32.shape[0] <= 2200 and x_f32.shape[1] <= 1400
    if use_cuda:
        device = torch.device("cuda")
        chunk = torch.from_numpy(x_f32).to(device)
        chunk = chunk - chunk.mean(dim=0, keepdim=True)
        chunk = chunk / chunk.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        corr = (chunk.transpose(0, 1) @ chunk) / max(chunk.shape[0] - 1, 1)
        return corr.abs().clamp_max(1.0).detach().cpu().numpy(), "cuda"

    centered = x_f32 - np.mean(x_f32, axis=0, keepdims=True, dtype=np.float32)
    scale = np.maximum(np.std(centered, axis=0, keepdims=True, dtype=np.float32), np.float32(1e-6))
    normalized = centered / scale
    corr = (normalized.T @ normalized) / max(normalized.shape[0] - 1, 1)
    return np.abs(corr, dtype=np.float32).astype(np.float64, copy=False), "cpu"


def _greedy_diversity_selection(
    *,
    base_values: np.ndarray,
    abs_corr: np.ndarray,
    target_k: int,
    penalty_strength: float,
) -> np.ndarray:
    pool_size = int(len(base_values))
    if pool_size <= target_k:
        return np.arange(pool_size, dtype=np.int64)

    selected: list[int] = []
    available = np.ones(pool_size, dtype=bool)
    max_corr = np.zeros(pool_size, dtype=np.float64)
    for _ in range(target_k):
        utility = np.asarray(base_values, dtype=np.float64) - penalty_strength * max_corr
        utility[~available] = -np.inf
        next_idx = int(np.argmax(utility))
        if not np.isfinite(utility[next_idx]):
            break
        selected.append(next_idx)
        available[next_idx] = False
        max_corr = np.maximum(max_corr, abs_corr[next_idx])
    if len(selected) < target_k:
        remainder = np.flatnonzero(available)
        if remainder.size > 0:
            fill = remainder[np.argsort(base_values[remainder])[::-1][: (target_k - len(selected))]]
            selected.extend(fill.tolist())
    return np.asarray(selected[:target_k], dtype=np.int64)


def score_mean_var_residual_like_counts(counts: np.ndarray) -> np.ndarray:
    x = normalize_log1p(counts)
    _, mv_scores, _ = _compute_base_scores_from_normalized(x)
    return mv_scores


def _compute_anchor_core_signals(profile: AdaptiveStatProfile) -> dict[str, float]:
    atlas_guard = float(np.clip(
        0.40 * _scale_unit(profile.n_cells, 7000.0, 16000.0)
        + 0.30 * _scale_unit(profile.pc_entropy, 0.82, 0.96)
        + 0.20 * _scale_unit(0.34 - profile.cluster_strength, 0.0, 0.12)
        + 0.10 * _scale_unit(0.12 - profile.batch_strength, 0.0, 0.12),
        0.0,
        1.0,
    ))
    heterogeneity_signal = float(np.clip(
        0.35 * _scale_unit(profile.batch_classes, 2.0, 12.0)
        + 0.20 * _scale_unit(profile.batch_strength, 0.0, 0.10)
        + 0.20 * _scale_unit(profile.cluster_strength, 0.22, 0.42)
        + 0.15 * _scale_unit(profile.library_cv, 0.50, 1.30)
        + 0.10 * _scale_unit(0.08 - profile.rare_fraction, 0.0, 0.08),
        0.0,
        1.0,
    ))
    trajectory_signal = float(np.clip(
        0.45 * _scale_unit(profile.dropout_rate, 0.82, 0.92)
        + 0.35 * _scale_unit(profile.trajectory_strength, 0.60, 0.78)
        + 0.20 * _scale_unit(0.60 - profile.library_cv, 0.0, 0.30),
        0.0,
        1.0,
    ))
    return {
        "atlas_guard": atlas_guard,
        "heterogeneity_signal": heterogeneity_signal,
        "trajectory_signal": trajectory_signal,
    }


def _compute_control_risk_signal(profile: AdaptiveStatProfile) -> float:
    if profile.batch_classes >= 2:
        return 0.0
    return float(
        np.clip(
            0.45 * _scale_unit(profile.pc_entropy, 0.88, 0.95)
            + 0.25 * _scale_unit(profile.library_cv, 0.75, 1.10)
            + 0.20 * _scale_unit(0.34 - profile.cluster_strength, 0.0, 0.08)
            + 0.10 * _scale_unit(0.84 - profile.dropout_rate, 0.0, 0.08),
            0.0,
            1.0,
        )
    )


def _zscore(x: np.ndarray) -> np.ndarray:
    std = np.std(x)
    if std < 1e-8:
        return np.zeros_like(x, dtype=np.float64)
    return (x - np.mean(x)) / std


def _rank_percentile(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = (np.arange(values.size, dtype=np.float64) + 0.5) / max(float(values.size), 1.0)
    return ranks


def _scale_unit(value: float | int, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return float(np.clip((float(value) - lower) / (upper - lower), 0.0, 1.0))


def _normalize_log1p_fast(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    counts_f32 = np.asarray(counts, dtype=np.float32)
    library = counts_f32.sum(axis=1, keepdims=True, dtype=np.float32)
    library = np.maximum(library, np.float32(1.0))
    normalized = counts_f32 / library * np.float32(target_sum)
    return np.log1p(normalized, dtype=np.float32)
