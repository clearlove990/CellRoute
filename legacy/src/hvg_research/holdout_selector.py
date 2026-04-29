from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DEFAULT_PROFILE_FEATURES = (
    "n_cells",
    "n_genes",
    "batch_classes",
    "dropout_rate",
    "library_cv",
    "cluster_strength",
    "batch_strength",
    "trajectory_strength",
    "pc_entropy",
    "rare_fraction",
)

DEFAULT_PUBLISHED_EXPERT_BANK = (
    "analytic_pearson_residual_hvg",
    "multinomial_deviance_hvg",
    "scanpy_cell_ranger_hvg",
    "scanpy_seurat_v3_hvg",
    "scran_model_gene_var_hvg",
    "seurat_r_vst_hvg",
    "seurat_v3_like_hvg",
    "triku_hvg",
)


def choose_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class RiskControlledSelectorConfig:
    safe_anchor: str = "adaptive_hybrid_hvg"
    candidate_methods: tuple[str, ...] = DEFAULT_PUBLISHED_EXPERT_BANK
    feature_names: tuple[str, ...] = DEFAULT_PROFILE_FEATURES
    k_neighbors: int = 5
    utility_uncertainty_scale: float = 0.60
    biology_uncertainty_scale: float = 0.45
    biology_weight: float = 0.35
    official_reliability_floor: float = 0.70
    official_penalty_scale: float = 0.20
    biology_guardrail: float = -0.015
    threshold_grid: tuple[float, ...] = (
        -0.25,
        -0.20,
        -0.15,
        -0.10,
        -0.05,
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
    )


@dataclass(frozen=True)
class SelectorDecision:
    selected_method: str
    proposed_method: str
    safe_anchor: str
    abstained: bool
    decision_reason: str
    route_threshold: float
    route_score: float
    confidence: float
    predicted_utility_delta: float
    predicted_utility_uncertainty: float
    utility_lcb: float
    predicted_biology_delta: float
    predicted_biology_uncertainty: float
    biology_lcb: float
    official_reliability: float
    neighbor_names: tuple[str, ...] = field(default_factory=tuple)
    neighbor_weights: tuple[float, ...] = field(default_factory=tuple)

    def to_metadata(self) -> dict[str, float | str]:
        payload = {
            "selector_policy": "holdout_risk_controlled_release",
            "selector_stage": "holdout_selector",
            "selector_bank_name": "published_official_expert_bank_v1",
            "safe_anchor": self.safe_anchor,
            "proposed_method": self.proposed_method,
            "resolved_method": self.selected_method,
            "abstained": float(self.abstained),
            "decision_reason": self.decision_reason,
            "route_threshold": float(self.route_threshold),
            "route_score": float(self.route_score),
            "confidence": float(self.confidence),
            "predicted_utility_delta": float(self.predicted_utility_delta),
            "predicted_utility_uncertainty": float(self.predicted_utility_uncertainty),
            "utility_lcb": float(self.utility_lcb),
            "predicted_biology_delta": float(self.predicted_biology_delta),
            "predicted_biology_uncertainty": float(self.predicted_biology_uncertainty),
            "biology_lcb": float(self.biology_lcb),
            "official_reliability": float(self.official_reliability),
            "neighbor_names": ",".join(self.neighbor_names),
            "neighbor_weights": ",".join(f"{value:.6f}" for value in self.neighbor_weights),
        }
        return payload


@dataclass
class RiskControlledSelectorPolicy:
    config: RiskControlledSelectorConfig
    feature_medians: dict[str, float]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    feature_table: dict[str, dict[str, float]]
    overall_table: dict[str, dict[str, float]]
    biology_table: dict[str, dict[str, float]]
    official_reliability: dict[str, float]
    excluded_methods: dict[str, str]
    route_threshold: float
    device_name: str = "cpu"

    @classmethod
    def fit_from_tables(
        cls,
        *,
        feature_df: pd.DataFrame,
        overall_df: pd.DataFrame,
        biology_df: pd.DataFrame,
        official_audit_df: pd.DataFrame | None = None,
        config: RiskControlledSelectorConfig | None = None,
    ) -> RiskControlledSelectorPolicy:
        cfg = RiskControlledSelectorConfig() if config is None else config
        clean_feature_df = _prepare_feature_df(feature_df=feature_df, feature_names=cfg.feature_names)
        overall_df = overall_df.copy()
        biology_df = biology_df.copy()
        if cfg.safe_anchor not in overall_df.columns:
            raise KeyError(f"Safe anchor '{cfg.safe_anchor}' missing from overall-score table.")
        if cfg.safe_anchor not in biology_df.columns:
            raise KeyError(f"Safe anchor '{cfg.safe_anchor}' missing from biology table.")

        reliability_map, excluded_methods = _resolve_method_reliability(
            candidate_methods=cfg.candidate_methods,
            overall_df=overall_df,
            official_audit_df=official_audit_df,
            reliability_floor=cfg.official_reliability_floor,
        )
        kept_methods = tuple(method for method in cfg.candidate_methods if method in reliability_map)
        if not kept_methods:
            raise ValueError("No admissible candidate methods remained after official/fallback filtering.")

        feature_medians = {
            column: float(clean_feature_df[column].median()) for column in clean_feature_df.columns
        }
        filled_features = clean_feature_df.fillna(feature_medians)
        feature_means = {column: float(filled_features[column].mean()) for column in filled_features.columns}
        feature_stds = {}
        standardized = filled_features.copy()
        for column in standardized.columns:
            std_value = float(standardized[column].std(ddof=0))
            if not np.isfinite(std_value) or std_value < 1e-8:
                std_value = 1.0
            feature_stds[column] = std_value
            standardized[column] = (standardized[column] - feature_means[column]) / std_value

        kept_columns = [cfg.safe_anchor, *kept_methods]
        trimmed_overall = overall_df.loc[standardized.index, kept_columns].copy()
        trimmed_biology = biology_df.loc[standardized.index, kept_columns].copy()

        policy = cls(
            config=RiskControlledSelectorConfig(
                safe_anchor=cfg.safe_anchor,
                candidate_methods=kept_methods,
                feature_names=tuple(standardized.columns.tolist()),
                k_neighbors=cfg.k_neighbors,
                utility_uncertainty_scale=cfg.utility_uncertainty_scale,
                biology_uncertainty_scale=cfg.biology_uncertainty_scale,
                biology_weight=cfg.biology_weight,
                official_reliability_floor=cfg.official_reliability_floor,
                official_penalty_scale=cfg.official_penalty_scale,
                biology_guardrail=cfg.biology_guardrail,
                threshold_grid=cfg.threshold_grid,
            ),
            feature_medians=feature_medians,
            feature_means=feature_means,
            feature_stds=feature_stds,
            feature_table=_df_to_nested_dict(standardized),
            overall_table=_df_to_nested_dict(trimmed_overall),
            biology_table=_df_to_nested_dict(trimmed_biology),
            official_reliability=reliability_map,
            excluded_methods=excluded_methods,
            route_threshold=0.0,
            device_name=str(choose_torch_device()),
        )
        policy.route_threshold = policy._calibrate_route_threshold()
        return policy

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return tuple(self.feature_table)

    def predict(self, profile: dict[str, float]) -> SelectorDecision:
        standardized_profile = self._standardize_profile(profile)
        return self._predict_internal(
            standardized_profile=standardized_profile,
            available_datasets=self.dataset_names,
            route_threshold=self.route_threshold,
        )

    def save_json(self, path: str | Path) -> None:
        target_path = Path(path)
        payload = {
            "config": asdict(self.config),
            "feature_medians": self.feature_medians,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "feature_table": self.feature_table,
            "overall_table": self.overall_table,
            "biology_table": self.biology_table,
            "official_reliability": self.official_reliability,
            "excluded_methods": self.excluded_methods,
            "route_threshold": float(self.route_threshold),
            "device_name": self.device_name,
        }
        target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> RiskControlledSelectorPolicy:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config_raw = dict(payload["config"])
        config = RiskControlledSelectorConfig(
            safe_anchor=str(config_raw["safe_anchor"]),
            candidate_methods=tuple(config_raw["candidate_methods"]),
            feature_names=tuple(config_raw["feature_names"]),
            k_neighbors=int(config_raw["k_neighbors"]),
            utility_uncertainty_scale=float(config_raw["utility_uncertainty_scale"]),
            biology_uncertainty_scale=float(config_raw["biology_uncertainty_scale"]),
            biology_weight=float(config_raw["biology_weight"]),
            official_reliability_floor=float(config_raw["official_reliability_floor"]),
            official_penalty_scale=float(config_raw["official_penalty_scale"]),
            biology_guardrail=float(config_raw["biology_guardrail"]),
            threshold_grid=tuple(float(value) for value in config_raw["threshold_grid"]),
        )
        return cls(
            config=config,
            feature_medians=_as_float_dict(payload["feature_medians"]),
            feature_means=_as_float_dict(payload["feature_means"]),
            feature_stds=_as_float_dict(payload["feature_stds"]),
            feature_table=_nested_float_dict(payload["feature_table"]),
            overall_table=_nested_float_dict(payload["overall_table"]),
            biology_table=_nested_float_dict(payload["biology_table"]),
            official_reliability=_as_float_dict(payload["official_reliability"]),
            excluded_methods={str(key): str(value) for key, value in payload.get("excluded_methods", {}).items()},
            route_threshold=float(payload["route_threshold"]),
            device_name=str(payload.get("device_name", "cpu")),
        )

    def _calibrate_route_threshold(self) -> float:
        thresholds = tuple(float(value) for value in self.config.threshold_grid)
        threshold_stats: list[tuple[float, float, float, float]] = []
        for threshold in thresholds:
            selected_scores: list[float] = []
            utility_deltas: list[float] = []
            biology_deltas: list[float] = []
            for dataset_name in self.dataset_names:
                available = tuple(name for name in self.dataset_names if name != dataset_name)
                decision = self._predict_internal(
                    standardized_profile=self.feature_table[dataset_name],
                    available_datasets=available,
                    route_threshold=threshold,
                )
                selected_method = decision.selected_method
                selected_scores.append(self.overall_table[dataset_name][selected_method])
                utility_deltas.append(
                    self.overall_table[dataset_name][selected_method]
                    - self.overall_table[dataset_name][self.config.safe_anchor]
                )
                biology_deltas.append(
                    self.biology_table[dataset_name][selected_method]
                    - self.biology_table[dataset_name][self.config.safe_anchor]
                )
            threshold_stats.append(
                (
                    float(threshold),
                    float(np.mean(selected_scores)),
                    float(np.mean(utility_deltas)),
                    float(np.mean(biology_deltas)),
                )
            )

        feasible = [
            item for item in threshold_stats if item[3] >= self.config.biology_guardrail
        ]
        candidates = feasible if feasible else threshold_stats
        best_threshold, _, _, _ = max(
            candidates,
            key=lambda item: (item[1], item[2], item[3], -abs(item[0])),
        )
        return float(best_threshold)

    def _predict_internal(
        self,
        *,
        standardized_profile: dict[str, float],
        available_datasets: tuple[str, ...],
        route_threshold: float,
    ) -> SelectorDecision:
        if not available_datasets:
            raise ValueError("Selector policy requires at least one training dataset.")
        distances = []
        for dataset_name in available_datasets:
            feature_row = self.feature_table[dataset_name]
            vector = np.asarray(
                [feature_row[name] - standardized_profile[name] for name in self.config.feature_names],
                dtype=np.float64,
            )
            distances.append((dataset_name, float(np.sqrt(np.square(vector).sum()))))
        distances.sort(key=lambda item: item[1])
        neighbors = distances[: min(self.config.k_neighbors, len(distances))]
        neighbor_names = tuple(name for name, _ in neighbors)
        inverse_distance = np.asarray(
            [1.0 / (distance + 1e-6) for _, distance in neighbors],
            dtype=np.float64,
        )
        weight_sum = max(float(inverse_distance.sum()), 1e-8)
        neighbor_weights = tuple(float(value / weight_sum) for value in inverse_distance)

        best_method = self.config.safe_anchor
        best_route_score = -float("inf")
        best_stats = None
        for method_name in self.config.candidate_methods:
            utility_delta = np.asarray(
                [
                    self.overall_table[dataset_name][method_name]
                    - self.overall_table[dataset_name][self.config.safe_anchor]
                    for dataset_name in neighbor_names
                ],
                dtype=np.float64,
            )
            biology_delta = np.asarray(
                [
                    self.biology_table[dataset_name][method_name]
                    - self.biology_table[dataset_name][self.config.safe_anchor]
                    for dataset_name in neighbor_names
                ],
                dtype=np.float64,
            )
            utility_mu, utility_std = _weighted_stats(utility_delta, neighbor_weights)
            biology_mu, biology_std = _weighted_stats(biology_delta, neighbor_weights)
            utility_lcb = utility_mu - self.config.utility_uncertainty_scale * utility_std
            biology_lcb = biology_mu - self.config.biology_uncertainty_scale * biology_std
            reliability = float(self.official_reliability.get(method_name, 1.0))
            reliability_penalty = self.config.official_penalty_scale * max(0.0, 1.0 - reliability)
            route_score = utility_lcb + self.config.biology_weight * biology_lcb - reliability_penalty
            if route_score > best_route_score:
                best_route_score = float(route_score)
                best_method = method_name
                best_stats = (
                    float(utility_mu),
                    float(utility_std),
                    float(utility_lcb),
                    float(biology_mu),
                    float(biology_std),
                    float(biology_lcb),
                    reliability,
                )

        if best_stats is None:
            raise RuntimeError("Selector policy did not evaluate any candidate methods.")
        (
            utility_mu,
            utility_std,
            utility_lcb,
            biology_mu,
            biology_std,
            biology_lcb,
            reliability,
        ) = best_stats

        abstained = bool(best_route_score <= route_threshold)
        if abstained:
            selected_method = self.config.safe_anchor
            decision_reason = "abstain_to_safe_anchor"
        else:
            selected_method = best_method
            decision_reason = "route_to_candidate"

        uncertainty_mass = utility_std + self.config.biology_weight * biology_std
        confidence_logit = best_route_score - route_threshold - 0.50 * uncertainty_mass
        confidence = float(reliability * (1.0 / (1.0 + np.exp(-6.0 * confidence_logit))))
        return SelectorDecision(
            selected_method=selected_method,
            proposed_method=best_method,
            safe_anchor=self.config.safe_anchor,
            abstained=abstained,
            decision_reason=decision_reason,
            route_threshold=float(route_threshold),
            route_score=float(best_route_score),
            confidence=confidence,
            predicted_utility_delta=utility_mu,
            predicted_utility_uncertainty=utility_std,
            utility_lcb=utility_lcb,
            predicted_biology_delta=biology_mu,
            predicted_biology_uncertainty=biology_std,
            biology_lcb=biology_lcb,
            official_reliability=reliability,
            neighbor_names=neighbor_names,
            neighbor_weights=neighbor_weights,
        )

    def _standardize_profile(self, profile: dict[str, float]) -> dict[str, float]:
        values = {}
        for feature_name in self.config.feature_names:
            raw_value = profile.get(feature_name, np.nan)
            if not np.isfinite(raw_value):
                raw_value = self.feature_medians[feature_name]
            values[feature_name] = float(
                (float(raw_value) - self.feature_means[feature_name]) / self.feature_stds[feature_name]
            )
        return values


def _prepare_feature_df(*, feature_df: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    resolved_columns: dict[str, str] = {}
    missing = []
    for feature_name in feature_names:
        if feature_name in feature_df.columns:
            resolved_columns[feature_name] = feature_name
            continue
        prefixed_name = f"stat_{feature_name}"
        if prefixed_name in feature_df.columns:
            resolved_columns[feature_name] = prefixed_name
            continue
        missing.append(feature_name)
    if missing:
        raise KeyError(f"Missing required selector profile features: {missing}")
    clean = pd.DataFrame(index=feature_df.index.astype(str))
    for feature_name in feature_names:
        clean[feature_name] = feature_df[resolved_columns[feature_name]]
    clean.index = feature_df.index.astype(str)
    return clean.sort_index()


def _resolve_method_reliability(
    *,
    candidate_methods: tuple[str, ...],
    overall_df: pd.DataFrame,
    official_audit_df: pd.DataFrame | None,
    reliability_floor: float,
) -> tuple[dict[str, float], dict[str, str]]:
    reliability_map: dict[str, float] = {}
    excluded_methods: dict[str, str] = {}
    audit_lookup = {}
    if official_audit_df is not None and not official_audit_df.empty and "method" in official_audit_df.columns:
        audit_lookup = official_audit_df.set_index("method", drop=False).to_dict(orient="index")

    for method_name in candidate_methods:
        if method_name not in overall_df.columns:
            excluded_methods[method_name] = "missing_from_benchmark"
            continue
        audit_row = audit_lookup.get(method_name)
        if audit_row is None:
            reliability_map[method_name] = 1.0
            continue
        if "selector_admissible" in audit_row and not bool(audit_row["selector_admissible"]):
            excluded_methods[method_name] = str(
                audit_row.get("selector_exclusion_reason", "excluded_by_audit")
            )
            continue
        explicit_reliability = audit_row.get("selector_reliability", np.nan)
        if np.isfinite(explicit_reliability):
            reliability = float(explicit_reliability)
            if reliability < reliability_floor:
                excluded_methods[method_name] = (
                    f"selector_reliability={reliability:.3f} below reliability floor"
                )
                continue
            reliability_map[method_name] = reliability
            continue
        fallback_rate = audit_row.get("official_fallback_rate", np.nan)
        if not np.isfinite(fallback_rate):
            reliability_map[method_name] = 1.0
            continue
        reliability = float(1.0 - float(fallback_rate))
        if reliability < reliability_floor:
            excluded_methods[method_name] = (
                f"official_fallback_rate={float(fallback_rate):.3f} below reliability floor"
            )
            continue
        reliability_map[method_name] = reliability
    return reliability_map, excluded_methods


def _df_to_nested_dict(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for index_value, row in df.iterrows():
        payload[str(index_value)] = {
            str(column): float(row[column]) for column in df.columns
        }
    return payload


def _nested_float_dict(payload: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        str(outer_key): {str(inner_key): float(inner_value) for inner_key, inner_value in inner.items()}
        for outer_key, inner in payload.items()
    }


def _as_float_dict(payload: dict[str, float]) -> dict[str, float]:
    return {str(key): float(value) for key, value in payload.items()}


def _weighted_stats(values: np.ndarray, weights: tuple[float, ...]) -> tuple[float, float]:
    weight_array = np.asarray(weights, dtype=np.float64)
    mean_value = float(np.dot(weight_array, values))
    variance = float(np.dot(weight_array, np.square(values - mean_value)))
    return mean_value, float(np.sqrt(max(variance, 0.0)))
