from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .adaptive_stat import (
    compute_adaptive_stat_profile,
    profile_to_dict,
    resolve_adaptive_hybrid_decision,
    resolve_adaptive_stat_decision,
    score_invariant_topology_hvg_v0,
    score_invariant_topology_hvg_v1,
    score_invariant_topology_hvg_v2,
    score_invariant_topology_hvg_v0_guarded,
    score_invariant_topology_hvg_v1_guarded,
    score_invariant_topology_hvg_v2_softdiv,
    score_invariant_topology_hvg_v3_safetygated,
    score_invariant_topology_hvg_v4_hardgate,
    score_adaptive_eb_shrinkage_hvg,
    score_adaptive_invariant_residual_hvg,
    score_adaptive_risk_parity_hvg,
    score_adaptive_risk_parity_safe_hvg,
    score_adaptive_risk_parity_ultrasafe_hvg,
    score_adaptive_spectral_locality_hvg,
    score_adaptive_stability_jackknife_hvg,
    score_real_batch_only_invariant_hvg,
    score_reduced_deviance_aggression_invariant_hvg,
    score_adaptive_core_consensus_hvg,
    score_adaptive_rank_aggregate_hvg,
    score_adaptive_stat_hvg,
    score_adaptive_stat_hvg_with_decision,
    score_sigma_hvg,
    score_sigma_no_graph_hvg,
    score_sigma_no_invariance_hvg,
    score_sigma_no_shrink_hvg,
    score_sigma_no_stability_hvg,
    score_sigma_pseudo_env_only_hvg,
    score_sigma_real_batch_only_hvg,
    score_sigma_safe_core_hvg,
    score_sigma_safe_core_no_invariance_hvg,
    score_sigma_safe_core_no_stability_hvg,
    score_sigma_safe_core_v3_hvg,
    score_sigma_safe_core_v4_hvg,
    score_sigma_safe_core_v5_hvg,
    score_sigma_safe_core_v6_hvg,
    score_stronger_atlas_guard_invariant_hvg,
)
from .baselines import (
    score_analytic_pearson_residual_hvg,
    score_fano,
    score_mean_var_residual,
    score_multinomial_deviance_hvg,
    score_seurat_v3_like_hvg,
    score_variance,
)
from .official_baselines import (
    score_scanpy_cell_ranger_hvg,
    score_scanpy_seurat_v3_hvg,
    score_scran_model_gene_var_hvg,
    score_seurat_r_vst_hvg,
    score_triku_hvg,
)
from .holdout_selector import RiskControlledSelectorPolicy
from .refine_moe_hvg import RefineMoEHVGSelector


MethodFn = Callable[[np.ndarray, np.ndarray | None, int], np.ndarray]

FRONTIER_LITE_METHOD = "learnable_gate_bank_pairregret_permissioned_escapecert_frontier_lite"
FRONTIER_LITE_INTERNAL_MODE = "ablation_frontier_no_teacher_frontier_only"
FRONTIER_ANCHORLESS_METHOD = "learnable_gate_bank_pairregret_permissioned_escapecert_frontier_anchorless"
FRONTIER_ANCHORLESS_INTERNAL_MODE = "ablation_frontier_no_anchor_escape_head"
HOLDOUT_RISK_SELECTOR_METHOD = "holdout_risk_release_hvg"
METHOD_ALIASES = {
    FRONTIER_LITE_INTERNAL_MODE: FRONTIER_LITE_METHOD,
    FRONTIER_ANCHORLESS_INTERNAL_MODE: FRONTIER_ANCHORLESS_METHOD,
}


def canonicalize_method_name(method_name: str) -> str:
    return METHOD_ALIASES.get(method_name, method_name)


def canonicalize_method_names(method_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    canonical_names: list[str] = []
    seen: set[str] = set()
    for method_name in method_names:
        canonical_name = canonicalize_method_name(str(method_name))
        if canonical_name in seen:
            continue
        seen.add(canonical_name)
        canonical_names.append(canonical_name)
    return tuple(canonical_names)


def _attach_empty_metadata(method_fn: MethodFn) -> MethodFn:
    setattr(method_fn, "last_gate_metadata", {})
    setattr(method_fn, "last_gate_source", None)
    setattr(method_fn, "last_dataset_stats", {})
    setattr(method_fn, "last_gate", None)
    return method_fn


def _score_method(
    scorer: Callable[[np.ndarray], np.ndarray],
    *,
    static_metadata: dict[str, float | str] | None = None,
) -> MethodFn:
    def run(counts: np.ndarray, batches: np.ndarray | None, current_top_k: int) -> np.ndarray:
        del batches, current_top_k
        run.last_gate_metadata = dict(static_metadata or {})
        run.last_gate_source = None
        run.last_dataset_stats = {}
        run.last_gate = None
        return scorer(counts)

    return _attach_empty_metadata(run)


def _batch_aware_score_method(
    scorer: Callable[[np.ndarray, np.ndarray | None, int], tuple[np.ndarray, dict[str, object]]],
    *,
    static_metadata: dict[str, float | str] | None = None,
) -> MethodFn:
    def run(counts: np.ndarray, batches: np.ndarray | None, current_top_k: int) -> np.ndarray:
        scores, metadata = scorer(counts, batches, current_top_k)
        run.last_gate_metadata = {
            **dict(static_metadata or {}),
            **{key: value for key, value in metadata.items()},
        }
        run.last_gate_source = None
        run.last_dataset_stats = {}
        run.last_gate = None
        return scores

    return _attach_empty_metadata(run)


def _selector_method(
    *,
    mode: str,
    refine_epochs: int,
    random_state: int,
    gate_model_path: str | None,
) -> MethodFn:
    def run(counts: np.ndarray, batches: np.ndarray | None, current_top_k: int) -> np.ndarray:
        selector = RefineMoEHVGSelector(
            top_k=current_top_k,
            refine_epochs=refine_epochs,
            random_state=random_state,
            mode=mode,
            gate_model_path=gate_model_path,
        )
        scores = selector.score_genes(counts, batches)
        run.last_gate_metadata = dict(selector.last_gate_metadata or {})
        run.last_gate_source = selector.last_gate_source
        run.last_dataset_stats = dict(selector.last_dataset_stats or {})
        run.last_gate = None if selector.last_gate is None else np.asarray(selector.last_gate, dtype=np.float64).copy()
        return scores

    return _attach_empty_metadata(run)


def _adaptive_stat_method(
    *,
    random_state: int,
) -> MethodFn:
    def run(counts: np.ndarray, batches: np.ndarray | None, current_top_k: int) -> np.ndarray:
        del current_top_k
        scores, profile, decision = score_adaptive_stat_hvg(
            counts,
            batches,
            random_state=random_state,
        )
        run.last_gate_source = decision.route_name
        run.last_dataset_stats = profile_to_dict(profile, decision)
        run.last_gate = np.asarray(
            [
                decision.variance_weight,
                decision.mv_residual_weight,
                decision.fano_weight,
            ],
            dtype=np.float64,
        )
        run.last_gate_metadata = {
            "route_name": decision.route_name,
            "variance_weight": float(decision.variance_weight),
            "mv_residual_weight": float(decision.mv_residual_weight),
            "fano_weight": float(decision.fano_weight),
            "selector_policy": "profile_conditioned_weighted_blend",
            "selector_stage": "adaptive_stat",
            "selector_bank_name": "classical_stat_bank_v1",
            "route_target": "adaptive_stat_blend",
            "fallback_target": "",
            "resolved_method": "adaptive_stat_hvg",
            "route_category": "default",
        }
        return scores

    return _attach_empty_metadata(run)


def _adaptive_hybrid_method(
    *,
    refine_epochs: int,
    random_state: int,
    gate_model_path: str | None,
) -> MethodFn:
    frontier_checkpoint_available = gate_model_path is not None and Path(gate_model_path).exists()

    def _compute_scores(
        *,
        counts: np.ndarray,
        batches: np.ndarray | None,
        current_top_k: int,
        profile,
    ) -> tuple[np.ndarray, dict[str, float | str], str, np.ndarray | None, dict[str, float | str]]:
        hybrid_decision = resolve_adaptive_hybrid_decision(profile)
        stat_decision = resolve_adaptive_stat_decision(profile)
        dataset_stats = profile_to_dict(profile, stat_decision)
        metadata: dict[str, float | str] = {
            "route_name": hybrid_decision.route_name,
            "route_target": hybrid_decision.route_target,
            "fallback_target": "" if hybrid_decision.fallback_target is None else hybrid_decision.fallback_target,
            "frontier_checkpoint_available": float(frontier_checkpoint_available),
            "selector_policy": "profile_conditioned_routing",
            "selector_stage": "adaptive_hybrid",
            "selector_bank_name": "published_expert_bank_selector_v1",
            "route_category": (
                "escape" if hybrid_decision.route_target == "frontier_lite" else "fallback" if hybrid_decision.route_target == "fano" else "default"
            ),
        }

        if hybrid_decision.route_target == "fano":
            metadata["resolved_method"] = "fano"
            metadata["used_fallback"] = 0.0
            return (
                score_fano(counts),
                metadata,
                hybrid_decision.route_name,
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                dataset_stats,
            )

        if hybrid_decision.route_target == "frontier_lite":
            if frontier_checkpoint_available:
                try:
                    selector = RefineMoEHVGSelector(
                        top_k=current_top_k,
                        refine_epochs=refine_epochs,
                        random_state=random_state,
                        mode=FRONTIER_LITE_INTERNAL_MODE,
                        gate_model_path=gate_model_path,
                    )
                    scores = selector.score_genes(counts, batches)
                    metadata["resolved_method"] = FRONTIER_LITE_METHOD
                    metadata["used_fallback"] = 0.0
                    metadata["frontier_internal_mode"] = FRONTIER_LITE_INTERNAL_MODE
                    for key, value in dict(selector.last_gate_metadata or {}).items():
                        metadata[f"frontier_{key}"] = value
                    dataset_stats = {
                        **dataset_stats,
                        **dict(selector.last_dataset_stats or {}),
                    }
                    return scores, metadata, hybrid_decision.route_name, None, dataset_stats
                except Exception as exc:
                    metadata["frontier_error_type"] = type(exc).__name__
                    metadata["frontier_error_message"] = str(exc)

            metadata["resolved_method"] = "fano"
            metadata["used_fallback"] = 1.0
            return (
                score_fano(counts),
                metadata,
                hybrid_decision.route_name,
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                dataset_stats,
            )

        metadata["resolved_method"] = "adaptive_stat_hvg"
        metadata["used_fallback"] = 0.0
        metadata["adaptive_stat_route_name"] = stat_decision.route_name
        metadata["variance_weight"] = float(stat_decision.variance_weight)
        metadata["mv_residual_weight"] = float(stat_decision.mv_residual_weight)
        metadata["fano_weight"] = float(stat_decision.fano_weight)
        gate = np.asarray(
            [
                stat_decision.variance_weight,
                stat_decision.mv_residual_weight,
                stat_decision.fano_weight,
            ],
            dtype=np.float64,
        )
        return (
            score_adaptive_stat_hvg_with_decision(counts, stat_decision),
            metadata,
            hybrid_decision.route_name,
            gate,
            dataset_stats,
        )

    def run(counts: np.ndarray, batches: np.ndarray | None, current_top_k: int) -> np.ndarray:
        profile = compute_adaptive_stat_profile(
            counts=counts,
            batches=batches,
            random_state=random_state,
        )
        scores, metadata, gate_source, gate, dataset_stats = _compute_scores(
            counts=counts,
            batches=batches,
            current_top_k=current_top_k,
            profile=profile,
        )
        run.last_gate_source = gate_source
        run.last_dataset_stats = dataset_stats
        run.last_gate = None if gate is None else np.asarray(gate, dtype=np.float64).copy()
        run.last_gate_metadata = metadata
        return scores

    return _attach_empty_metadata(run)


def _adaptive_hybrid_sigma_safe_core_v6_method(
    *,
    refine_epochs: int,
    random_state: int,
    gate_model_path: str | None,
) -> MethodFn:
    frontier_checkpoint_available = gate_model_path is not None and Path(gate_model_path).exists()

    def _compute_anchor_scores(
        *,
        counts: np.ndarray,
        batches: np.ndarray | None,
        current_top_k: int,
        profile,
    ) -> tuple[np.ndarray, dict[str, float | str], str, np.ndarray | None, dict[str, float | str]]:
        hybrid_decision = resolve_adaptive_hybrid_decision(profile)
        stat_decision = resolve_adaptive_stat_decision(profile)
        dataset_stats = profile_to_dict(profile, stat_decision)
        metadata: dict[str, float | str] = {
            "route_name": hybrid_decision.route_name,
            "route_target": hybrid_decision.route_target,
            "fallback_target": "" if hybrid_decision.fallback_target is None else hybrid_decision.fallback_target,
            "frontier_checkpoint_available": float(frontier_checkpoint_available),
            "selector_policy": "profile_conditioned_routing",
            "selector_stage": "adaptive_hybrid",
            "selector_bank_name": "published_expert_bank_selector_v1",
            "route_category": (
                "escape" if hybrid_decision.route_target == "frontier_lite" else "fallback" if hybrid_decision.route_target == "fano" else "default"
            ),
        }

        if hybrid_decision.route_target == "fano":
            metadata["resolved_method"] = "fano"
            metadata["used_fallback"] = 0.0
            return (
                score_fano(counts),
                metadata,
                hybrid_decision.route_name,
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                dataset_stats,
            )

        if hybrid_decision.route_target == "frontier_lite":
            if frontier_checkpoint_available:
                try:
                    selector = RefineMoEHVGSelector(
                        top_k=current_top_k,
                        refine_epochs=refine_epochs,
                        random_state=random_state,
                        mode=FRONTIER_LITE_INTERNAL_MODE,
                        gate_model_path=gate_model_path,
                    )
                    scores = selector.score_genes(counts, batches)
                    metadata["resolved_method"] = FRONTIER_LITE_METHOD
                    metadata["used_fallback"] = 0.0
                    metadata["frontier_internal_mode"] = FRONTIER_LITE_INTERNAL_MODE
                    for key, value in dict(selector.last_gate_metadata or {}).items():
                        metadata[f"frontier_{key}"] = value
                    dataset_stats = {
                        **dataset_stats,
                        **dict(selector.last_dataset_stats or {}),
                    }
                    return scores, metadata, hybrid_decision.route_name, None, dataset_stats
                except Exception as exc:
                    metadata["frontier_error_type"] = type(exc).__name__
                    metadata["frontier_error_message"] = str(exc)

            metadata["resolved_method"] = "fano"
            metadata["used_fallback"] = 1.0
            return (
                score_fano(counts),
                metadata,
                hybrid_decision.route_name,
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                dataset_stats,
            )

        metadata["resolved_method"] = "adaptive_stat_hvg"
        metadata["used_fallback"] = 0.0
        metadata["adaptive_stat_route_name"] = stat_decision.route_name
        metadata["variance_weight"] = float(stat_decision.variance_weight)
        metadata["mv_residual_weight"] = float(stat_decision.mv_residual_weight)
        metadata["fano_weight"] = float(stat_decision.fano_weight)
        gate = np.asarray(
            [
                stat_decision.variance_weight,
                stat_decision.mv_residual_weight,
                stat_decision.fano_weight,
            ],
            dtype=np.float64,
        )
        return (
            score_adaptive_stat_hvg_with_decision(counts, stat_decision),
            metadata,
            hybrid_decision.route_name,
            gate,
            dataset_stats,
        )

    def run(counts: np.ndarray, batches: np.ndarray | None, current_top_k: int) -> np.ndarray:
        profile = compute_adaptive_stat_profile(
            counts=counts,
            batches=batches,
            random_state=random_state,
        )
        anchor_scores, anchor_metadata, anchor_gate_source, anchor_gate, dataset_stats = _compute_anchor_scores(
            counts=counts,
            batches=batches,
            current_top_k=current_top_k,
            profile=profile,
        )
        upgrade_scores, upgrade_metadata = score_sigma_safe_core_v6_hvg(
            counts=counts,
            batches=batches,
            anchor_scores=anchor_scores,
            profile=profile,
            random_state=random_state,
        )
        metadata = {
            **anchor_metadata,
            **upgrade_metadata,
            "selector_policy": "single_scorer_fixed_fusion",
            "selector_stage": "anchor_core_upgrade",
            "selector_bank_name": "sigma_safe_core_v6",
            "route_target": "single_score",
            "resolved_method": "sigma_safe_core_v6_hvg",
            "anchor_gate_source": anchor_gate_source,
            "anchor_resolved_method": anchor_metadata.get("resolved_method", ""),
        }
        run.last_gate_source = str(metadata.get("route_name", "sigma_safe_core_v6"))
        run.last_dataset_stats = dataset_stats
        run.last_gate = None
        run.last_gate_metadata = metadata
        if str(metadata.get("route_name", "")).endswith("anchor_reuse"):
            return np.asarray(anchor_scores, dtype=np.float64)
        return upgrade_scores

    return _attach_empty_metadata(run)


def _adaptive_core_method(
    *,
    kind: str,
    random_state: int,
) -> MethodFn:
    def run(counts: np.ndarray, batches: np.ndarray | None, current_top_k: int) -> np.ndarray:
        del current_top_k
        if kind == "adaptive_core_consensus_hvg":
            scores, profile, metadata = score_adaptive_core_consensus_hvg(
                counts,
                batches,
                random_state=random_state,
                include_agreement_bonus=True,
            )
        elif kind == "adaptive_core_consensus_no_agreement_hvg":
            scores, profile, metadata = score_adaptive_core_consensus_hvg(
                counts,
                batches,
                random_state=random_state,
                include_agreement_bonus=False,
            )
            metadata = {
                **metadata,
                "variant": "adaptive_core_consensus_no_agreement_hvg",
                "consensus_family": "score_level_count_bridge_ablation",
            }
        elif kind == "adaptive_rank_aggregate_hvg":
            scores, profile, metadata = score_adaptive_rank_aggregate_hvg(
                counts,
                batches,
                random_state=random_state,
                include_agreement_bonus=True,
            )
        elif kind == "adaptive_rank_aggregate_no_agreement_hvg":
            scores, profile, metadata = score_adaptive_rank_aggregate_hvg(
                counts,
                batches,
                random_state=random_state,
                include_agreement_bonus=False,
            )
            metadata = {
                **metadata,
                "variant": "adaptive_rank_aggregate_no_agreement_hvg",
                "consensus_family": "rank_trimmed_mean_ablation",
            }
        elif kind == "adaptive_eb_shrinkage_hvg":
            scores, profile, metadata = score_adaptive_eb_shrinkage_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "adaptive_invariant_residual_hvg":
            scores, profile, metadata = score_adaptive_invariant_residual_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "real_batch_only_invariant":
            scores, profile, metadata = score_real_batch_only_invariant_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "stronger_atlas_guard_invariant":
            scores, profile, metadata = score_stronger_atlas_guard_invariant_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "reduced_deviance_aggression_invariant":
            scores, profile, metadata = score_reduced_deviance_aggression_invariant_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "adaptive_stability_jackknife_hvg":
            scores, profile, metadata = score_adaptive_stability_jackknife_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "adaptive_spectral_locality_hvg":
            scores, profile, metadata = score_adaptive_spectral_locality_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "adaptive_risk_parity_hvg":
            scores, profile, metadata = score_adaptive_risk_parity_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "adaptive_risk_parity_safe_hvg":
            scores, profile, metadata = score_adaptive_risk_parity_safe_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "adaptive_risk_parity_ultrasafe_hvg":
            scores, profile, metadata = score_adaptive_risk_parity_ultrasafe_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_hvg":
            scores, profile, metadata = score_sigma_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_no_invariance_hvg":
            scores, profile, metadata = score_sigma_no_invariance_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_no_graph_hvg":
            scores, profile, metadata = score_sigma_no_graph_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_no_stability_hvg":
            scores, profile, metadata = score_sigma_no_stability_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_no_shrink_hvg":
            scores, profile, metadata = score_sigma_no_shrink_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_real_batch_only_hvg":
            scores, profile, metadata = score_sigma_real_batch_only_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_pseudo_env_only_hvg":
            scores, profile, metadata = score_sigma_pseudo_env_only_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_safe_core_hvg":
            scores, profile, metadata = score_sigma_safe_core_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_safe_core_no_invariance_hvg":
            scores, profile, metadata = score_sigma_safe_core_no_invariance_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_safe_core_no_stability_hvg":
            scores, profile, metadata = score_sigma_safe_core_no_stability_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_safe_core_v3_hvg":
            scores, profile, metadata = score_sigma_safe_core_v3_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_safe_core_v4_hvg":
            scores, profile, metadata = score_sigma_safe_core_v4_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        elif kind == "sigma_safe_core_v5_hvg":
            scores, profile, metadata = score_sigma_safe_core_v5_hvg(
                counts,
                batches,
                random_state=random_state,
            )
        else:
            raise KeyError(f"Unsupported adaptive core method kind: {kind}")

        run.last_gate_source = str(metadata.get("route_name", kind))
        run.last_dataset_stats = profile_to_dict(profile, resolve_adaptive_stat_decision(profile))
        run.last_gate = None
        run.last_gate_metadata = {
            **metadata,
            "method_family": "anchor_core_upgrade",
            "resolved_method": kind,
        }
        return scores

    return _attach_empty_metadata(run)


def _adaptive_topology_method(
    *,
    kind: str,
    random_state: int,
) -> MethodFn:
    def run(counts: np.ndarray, batches: np.ndarray | None, current_top_k: int) -> np.ndarray:
        if kind == "invariant_topology_hvg_v0":
            scores, profile, metadata = score_invariant_topology_hvg_v0(
                counts,
                batches,
                random_state=random_state,
                current_top_k=current_top_k,
            )
        elif kind == "invariant_topology_hvg_v1":
            scores, profile, metadata = score_invariant_topology_hvg_v1(
                counts,
                batches,
                random_state=random_state,
                current_top_k=current_top_k,
            )
        elif kind == "invariant_topology_hvg_v2":
            scores, profile, metadata = score_invariant_topology_hvg_v2(
                counts,
                batches,
                random_state=random_state,
                current_top_k=current_top_k,
            )
        elif kind == "invariant_topology_hvg_v0_guarded":
            scores, profile, metadata = score_invariant_topology_hvg_v0_guarded(
                counts,
                batches,
                random_state=random_state,
                current_top_k=current_top_k,
            )
        elif kind == "invariant_topology_hvg_v1_guarded":
            scores, profile, metadata = score_invariant_topology_hvg_v1_guarded(
                counts,
                batches,
                random_state=random_state,
                current_top_k=current_top_k,
            )
        elif kind == "invariant_topology_hvg_v2_softdiv":
            scores, profile, metadata = score_invariant_topology_hvg_v2_softdiv(
                counts,
                batches,
                random_state=random_state,
                current_top_k=current_top_k,
            )
        elif kind == "invariant_topology_hvg_v3_safetygated":
            scores, profile, metadata = score_invariant_topology_hvg_v3_safetygated(
                counts,
                batches,
                random_state=random_state,
                current_top_k=current_top_k,
            )
        elif kind == "invariant_topology_hvg_v4_hardgate":
            scores, profile, metadata = score_invariant_topology_hvg_v4_hardgate(
                counts,
                batches,
                random_state=random_state,
                current_top_k=current_top_k,
            )
        else:
            raise KeyError(f"Unsupported invariant topology method kind: {kind}")

        run.last_gate_source = str(metadata.get("route_name", kind))
        run.last_dataset_stats = profile_to_dict(profile, resolve_adaptive_stat_decision(profile))
        run.last_gate = None
        run.last_gate_metadata = {
            **metadata,
            "method_family": "anchor_core_upgrade",
            "resolved_method": kind,
        }
        return scores

    return _attach_empty_metadata(run)


def _holdout_selector_method(
    *,
    random_state: int,
    policy_path: str,
    base_registry: dict[str, MethodFn],
) -> MethodFn:
    loaded_policy: RiskControlledSelectorPolicy | None = None

    def run(counts: np.ndarray, batches: np.ndarray | None, current_top_k: int) -> np.ndarray:
        nonlocal loaded_policy
        if loaded_policy is None:
            loaded_policy = RiskControlledSelectorPolicy.load_json(policy_path)
        profile = compute_adaptive_stat_profile(
            counts=counts,
            batches=batches,
            random_state=random_state,
        )
        profile_dict = profile_to_dict(profile, resolve_adaptive_stat_decision(profile))
        decision = loaded_policy.predict(profile_dict)
        selected_method = decision.selected_method
        if selected_method == HOLDOUT_RISK_SELECTOR_METHOD:
            raise RuntimeError("Holdout selector resolved to itself, which is not allowed.")
        if selected_method not in base_registry:
            raise KeyError(f"Resolved expert '{selected_method}' is not available in the method registry.")

        base_method = base_registry[selected_method]
        scores = base_method(counts, batches, current_top_k)
        run.last_gate_source = decision.decision_reason
        run.last_dataset_stats = profile_dict
        run.last_gate = None
        metadata = decision.to_metadata()
        metadata["route_name"] = "holdout_risk_release"
        metadata["route_target"] = decision.proposed_method
        metadata["fallback_target"] = decision.safe_anchor
        metadata["route_category"] = "abstain" if decision.abstained else "override"
        metadata["selector_device"] = loaded_policy.device_name
        selected_metadata = getattr(base_method, "last_gate_metadata", None)
        if isinstance(selected_metadata, dict):
            for key, value in selected_metadata.items():
                metadata[f"resolved_{key}"] = value
        run.last_gate_metadata = metadata
        return scores

    return _attach_empty_metadata(run)


def build_default_method_registry(
    *,
    top_k: int,
    refine_epochs: int,
    random_state: int,
    gate_model_path: str | None = None,
    holdout_selector_policy_path: str | None = None,
    official_baseline_strict: bool = False,
) -> dict[str, MethodFn]:
    del top_k
    registry: dict[str, MethodFn] = {
        "variance": _score_method(
            score_variance,
            static_metadata={
                "method_family": "base_scorer",
                "expert_group": "classical_stat",
                "expert_bank_name": "classical_stat_bank_v1",
            },
        ),
        "fano": _score_method(
            score_fano,
            static_metadata={
                "method_family": "base_scorer",
                "expert_group": "classical_stat",
                "expert_bank_name": "classical_stat_bank_v1",
            },
        ),
        "mv_residual": _score_method(
            score_mean_var_residual,
            static_metadata={
                "method_family": "base_scorer",
                "expert_group": "classical_stat",
                "expert_bank_name": "classical_stat_bank_v1",
            },
        ),
        "analytic_pearson_residual_hvg": _score_method(
            score_analytic_pearson_residual_hvg,
            static_metadata={
                "method_family": "base_scorer",
                "expert_group": "published_expert",
                "expert_bank_name": "published_expert_bank_v1",
            },
        ),
        "seurat_v3_like_hvg": _score_method(
            score_seurat_v3_like_hvg,
            static_metadata={
                "method_family": "base_scorer",
                "expert_group": "published_expert",
                "expert_bank_name": "published_expert_bank_v1",
            },
        ),
        "multinomial_deviance_hvg": _score_method(
            score_multinomial_deviance_hvg,
            static_metadata={
                "method_family": "base_scorer",
                "expert_group": "published_expert",
                "expert_bank_name": "published_expert_bank_v1",
                "implementation_source": "paper_reproduction_multinomial_binomial_deviance",
                "reproduction_tier": "strict_reproduced_from_paper_or_release",
            },
        ),
        "scanpy_seurat_v3_hvg": _batch_aware_score_method(
            lambda counts, batches, current_top_k: score_scanpy_seurat_v3_hvg(
                counts,
                batches,
                top_k=current_top_k,
                allow_fallback=not official_baseline_strict,
            ),
            static_metadata={
                "method_family": "base_scorer",
                "expert_group": "published_expert",
                "expert_bank_name": "official_scanpy_expert_bank_v1",
                "official_implementation": "scanpy",
            },
        ),
        "scanpy_cell_ranger_hvg": _batch_aware_score_method(
            lambda counts, batches, current_top_k: score_scanpy_cell_ranger_hvg(
                counts,
                batches,
                top_k=current_top_k,
                allow_fallback=not official_baseline_strict,
            ),
            static_metadata={
                "method_family": "base_scorer",
                "expert_group": "published_expert",
                "expert_bank_name": "official_scanpy_expert_bank_v1",
                "official_implementation": "scanpy",
            },
        ),
        "triku_hvg": _batch_aware_score_method(
            lambda counts, batches, current_top_k: score_triku_hvg(
                counts,
                batches,
                top_k=current_top_k,
                allow_fallback=not official_baseline_strict,
            ),
            static_metadata={
                "method_family": "base_scorer",
                "expert_group": "published_expert",
                "expert_bank_name": "official_triku_expert_bank_v1",
                "official_implementation": "triku",
            },
        ),
        "seurat_r_vst_hvg": _batch_aware_score_method(
            lambda counts, batches, current_top_k: score_seurat_r_vst_hvg(
                counts,
                batches,
                top_k=current_top_k,
                allow_fallback=not official_baseline_strict,
            ),
            static_metadata={
                "method_family": "base_scorer",
                "expert_group": "published_expert",
                "expert_bank_name": "official_seurat_expert_bank_v1",
                "official_implementation": "Seurat/R",
            },
        ),
        "scran_model_gene_var_hvg": _batch_aware_score_method(
            lambda counts, batches, current_top_k: score_scran_model_gene_var_hvg(
                counts,
                batches,
                top_k=current_top_k,
                allow_fallback=not official_baseline_strict,
            ),
            static_metadata={
                "method_family": "base_scorer",
                "expert_group": "published_expert",
                "expert_bank_name": "official_scran_expert_bank_v1",
                "official_implementation": "scran",
            },
        ),
        "adaptive_stat_hvg": _adaptive_stat_method(random_state=random_state),
        "adaptive_core_consensus_hvg": _adaptive_core_method(
            kind="adaptive_core_consensus_hvg",
            random_state=random_state,
        ),
        "adaptive_rank_aggregate_hvg": _adaptive_core_method(
            kind="adaptive_rank_aggregate_hvg",
            random_state=random_state,
        ),
        "adaptive_eb_shrinkage_hvg": _adaptive_core_method(
            kind="adaptive_eb_shrinkage_hvg",
            random_state=random_state,
        ),
        "adaptive_invariant_residual_hvg": _adaptive_core_method(
            kind="adaptive_invariant_residual_hvg",
            random_state=random_state,
        ),
        "real_batch_only_invariant": _adaptive_core_method(
            kind="real_batch_only_invariant",
            random_state=random_state,
        ),
        "stronger_atlas_guard_invariant": _adaptive_core_method(
            kind="stronger_atlas_guard_invariant",
            random_state=random_state,
        ),
        "reduced_deviance_aggression_invariant": _adaptive_core_method(
            kind="reduced_deviance_aggression_invariant",
            random_state=random_state,
        ),
        "adaptive_stability_jackknife_hvg": _adaptive_core_method(
            kind="adaptive_stability_jackknife_hvg",
            random_state=random_state,
        ),
        "adaptive_spectral_locality_hvg": _adaptive_core_method(
            kind="adaptive_spectral_locality_hvg",
            random_state=random_state,
        ),
        "adaptive_risk_parity_hvg": _adaptive_core_method(
            kind="adaptive_risk_parity_hvg",
            random_state=random_state,
        ),
        "adaptive_risk_parity_safe_hvg": _adaptive_core_method(
            kind="adaptive_risk_parity_safe_hvg",
            random_state=random_state,
        ),
        "adaptive_risk_parity_ultrasafe_hvg": _adaptive_core_method(
            kind="adaptive_risk_parity_ultrasafe_hvg",
            random_state=random_state,
        ),
        "invariant_topology_hvg_v0": _adaptive_topology_method(
            kind="invariant_topology_hvg_v0",
            random_state=random_state,
        ),
        "invariant_topology_hvg_v1": _adaptive_topology_method(
            kind="invariant_topology_hvg_v1",
            random_state=random_state,
        ),
        "invariant_topology_hvg_v2": _adaptive_topology_method(
            kind="invariant_topology_hvg_v2",
            random_state=random_state,
        ),
        "invariant_topology_hvg_v0_guarded": _adaptive_topology_method(
            kind="invariant_topology_hvg_v0_guarded",
            random_state=random_state,
        ),
        "invariant_topology_hvg_v1_guarded": _adaptive_topology_method(
            kind="invariant_topology_hvg_v1_guarded",
            random_state=random_state,
        ),
        "invariant_topology_hvg_v2_softdiv": _adaptive_topology_method(
            kind="invariant_topology_hvg_v2_softdiv",
            random_state=random_state,
        ),
        "invariant_topology_hvg_v3_safetygated": _adaptive_topology_method(
            kind="invariant_topology_hvg_v3_safetygated",
            random_state=random_state,
        ),
        "invariant_topology_hvg_v4_hardgate": _adaptive_topology_method(
            kind="invariant_topology_hvg_v4_hardgate",
            random_state=random_state,
        ),
        "sigma_hvg": _adaptive_core_method(
            kind="sigma_hvg",
            random_state=random_state,
        ),
        "sigma_no_invariance_hvg": _adaptive_core_method(
            kind="sigma_no_invariance_hvg",
            random_state=random_state,
        ),
        "sigma_no_graph_hvg": _adaptive_core_method(
            kind="sigma_no_graph_hvg",
            random_state=random_state,
        ),
        "sigma_no_stability_hvg": _adaptive_core_method(
            kind="sigma_no_stability_hvg",
            random_state=random_state,
        ),
        "sigma_no_shrink_hvg": _adaptive_core_method(
            kind="sigma_no_shrink_hvg",
            random_state=random_state,
        ),
        "sigma_real_batch_only_hvg": _adaptive_core_method(
            kind="sigma_real_batch_only_hvg",
            random_state=random_state,
        ),
        "sigma_pseudo_env_only_hvg": _adaptive_core_method(
            kind="sigma_pseudo_env_only_hvg",
            random_state=random_state,
        ),
        "sigma_safe_core_hvg": _adaptive_core_method(
            kind="sigma_safe_core_hvg",
            random_state=random_state,
        ),
        "sigma_safe_core_no_invariance_hvg": _adaptive_core_method(
            kind="sigma_safe_core_no_invariance_hvg",
            random_state=random_state,
        ),
        "sigma_safe_core_no_stability_hvg": _adaptive_core_method(
            kind="sigma_safe_core_no_stability_hvg",
            random_state=random_state,
        ),
        "sigma_safe_core_v3_hvg": _adaptive_core_method(
            kind="sigma_safe_core_v3_hvg",
            random_state=random_state,
        ),
        "sigma_safe_core_v4_hvg": _adaptive_core_method(
            kind="sigma_safe_core_v4_hvg",
            random_state=random_state,
        ),
        "sigma_safe_core_v5_hvg": _adaptive_core_method(
            kind="sigma_safe_core_v5_hvg",
            random_state=random_state,
        ),
        "sigma_safe_core_v6_hvg": _adaptive_hybrid_sigma_safe_core_v6_method(
            refine_epochs=refine_epochs,
            random_state=random_state,
            gate_model_path=gate_model_path,
        ),
        "adaptive_rank_aggregate_no_agreement_hvg": _adaptive_core_method(
            kind="adaptive_rank_aggregate_no_agreement_hvg",
            random_state=random_state,
        ),
        "adaptive_core_consensus_no_agreement_hvg": _adaptive_core_method(
            kind="adaptive_core_consensus_no_agreement_hvg",
            random_state=random_state,
        ),
        "adaptive_hybrid_hvg": _adaptive_hybrid_method(
            refine_epochs=refine_epochs,
            random_state=random_state,
            gate_model_path=gate_model_path,
        ),
        "refine_moe_hvg": _selector_method(
            mode="full",
            refine_epochs=refine_epochs,
            random_state=random_state,
            gate_model_path=None,
        ),
        "ablation_no_moe": _selector_method(
            mode="no_moe",
            refine_epochs=refine_epochs,
            random_state=random_state,
            gate_model_path=None,
        ),
        "ablation_no_refine": _selector_method(
            mode="no_refine",
            refine_epochs=refine_epochs,
            random_state=random_state,
            gate_model_path=None,
        ),
    }
    if gate_model_path is None:
        if holdout_selector_policy_path is not None:
            base_registry = dict(registry)
            registry[HOLDOUT_RISK_SELECTOR_METHOD] = _holdout_selector_method(
                random_state=random_state,
                policy_path=holdout_selector_policy_path,
                base_registry=base_registry,
            )
        return registry

    mode_map = {
        "learnable_gate": "learnable_gate",
        "learnable_gate_bank": "learnable_gate_bank",
        "learnable_gate_bank_reliable_refine": "learnable_gate_bank_reliable_refine",
        "learnable_gate_bank_curr_refine": "learnable_gate_bank_curr_refine",
        "ablation_curr_no_risk": "learnable_gate_bank_curr_no_risk",
        "ablation_curr_no_regret": "learnable_gate_bank_curr_no_regret",
        "ablation_curr_no_refine_policy": "learnable_gate_bank_curr_no_refine_policy",
        "ablation_curr_utility_only": "learnable_gate_bank_curr_utility_only",
        "learnable_gate_bank_pairregret": "learnable_gate_bank_pairregret",
        "ablation_pairregret_no_risk": "ablation_pairregret_no_risk",
        "ablation_pairregret_no_regret": "ablation_pairregret_no_regret",
        "ablation_pairregret_no_pairwise_term": "ablation_pairregret_no_pairwise_term",
        "ablation_pairregret_utility_only": "ablation_pairregret_utility_only",
        "ablation_pairregret_no_conservative_routing": "ablation_pairregret_no_conservative_routing",
        "ablation_pairregret_refine_on": "ablation_pairregret_refine_on",
        "learnable_gate_bank_pairregret_calibrated": "learnable_gate_bank_pairregret_calibrated",
        "ablation_pairregret_cal_no_regret": "ablation_pairregret_cal_no_regret",
        "ablation_pairregret_cal_no_consistency": "ablation_pairregret_cal_no_consistency",
        "ablation_pairregret_cal_no_route_constraint": "ablation_pairregret_cal_no_route_constraint",
        "ablation_pairregret_cal_utility_only": "ablation_pairregret_cal_utility_only",
        "ablation_pairregret_cal_no_conservative_routing": "ablation_pairregret_cal_no_conservative_routing",
        "ablation_pairregret_cal_refine_on": "ablation_pairregret_cal_refine_on",
        "learnable_gate_bank_pairregret_permissioned": "learnable_gate_bank_pairregret_permissioned",
        "ablation_pairperm_no_permission_head": "ablation_pairperm_no_permission_head",
        "ablation_pairperm_fixed_budget": "ablation_pairperm_fixed_budget",
        "ablation_pairperm_no_anchor_adaptation": "ablation_pairperm_no_anchor_adaptation",
        "ablation_pairperm_no_permission_value_decoupling": "ablation_pairperm_no_permission_value_decoupling",
        "ablation_pairperm_no_regret_aux": "ablation_pairperm_no_regret_aux",
        "ablation_pairperm_permission_only": "ablation_pairperm_permission_only",
        "ablation_pairperm_refine_on": "ablation_pairperm_refine_on",
        "learnable_gate_bank_pairregret_permissioned_escapecert": "learnable_gate_bank_pairregret_permissioned_escapecert",
        "ablation_escapecert_no_anchor_escape_head": "ablation_escapecert_no_anchor_escape_head",
        "ablation_escapecert_no_set_supervision": "ablation_escapecert_no_set_supervision",
        "ablation_escapecert_no_candidate_admissibility": "ablation_escapecert_no_candidate_admissibility",
        "ablation_escapecert_fixed_budget": "ablation_escapecert_fixed_budget",
        "ablation_escapecert_no_escape_uncertainty": "ablation_escapecert_no_escape_uncertainty",
        "ablation_escapecert_no_regret_aux": "ablation_escapecert_no_regret_aux",
        "ablation_escapecert_anchor_escape_only": "ablation_escapecert_anchor_escape_only",
        "ablation_escapecert_admissibility_only": "ablation_escapecert_admissibility_only",
        "learnable_gate_bank_pairregret_permissioned_escapecert_frontier": "learnable_gate_bank_pairregret_permissioned_escapecert_frontier",
        "ablation_frontier_no_teacher_distill": "ablation_frontier_no_teacher_distill",
        "ablation_frontier_no_frontier_head": "ablation_frontier_no_frontier_head",
        "ablation_frontier_no_frontier_uncertainty": "ablation_frontier_no_frontier_uncertainty",
        "ablation_frontier_fixed_frontier": "ablation_frontier_fixed_frontier",
        FRONTIER_ANCHORLESS_METHOD: FRONTIER_ANCHORLESS_INTERNAL_MODE,
        "ablation_frontier_no_regret_aux": "ablation_frontier_no_regret_aux",
        "ablation_frontier_teacher_only": "ablation_frontier_teacher_only",
        FRONTIER_LITE_METHOD: FRONTIER_LITE_INTERNAL_MODE,
    }
    registry.update(
        {
            method_name: _selector_method(
                mode=mode,
                refine_epochs=refine_epochs,
                random_state=random_state,
                gate_model_path=gate_model_path,
            )
            for method_name, mode in mode_map.items()
        }
    )
    if holdout_selector_policy_path is not None:
        base_registry = dict(registry)
        registry[HOLDOUT_RISK_SELECTOR_METHOD] = _holdout_selector_method(
            random_state=random_state,
            policy_path=holdout_selector_policy_path,
            base_registry=base_registry,
        )
    return registry
