from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .baselines import normalize_log1p
from .eval import neighbor_preservation
from .refine_moe_hvg import EXPERT_NAMES, RefineMoEHVGSelector
from .synthetic import generate_synthetic_scrna


DEFAULT_GATE_FEATURE_KEYS = (
    "cluster_strength",
    "trajectory_strength",
    "batch_strength",
    "rare_fraction",
    "dropout_rate",
    "library_cv",
    "pc_entropy",
    "n_cells_log",
    "n_genes_log",
    "batch_count_norm",
)

BANK_BOOST_EXPERTS = ("batch_robust", "general", "cluster", "rare", "structure")
BANK_PROXY_FEATURE_KEYS = (
    "proxy_pseudo_silhouette",
    "proxy_cluster_silhouette",
    "proxy_batch_mixing",
    "proxy_neighbor_preservation",
    "proxy_structure_support",
    "proxy_transfer_risk",
    "proxy_safe_refine_support",
    "selected_batch_mi_mean",
    "selected_cluster_mi_mean",
    "selected_local_consistency_mean",
    "selected_residual_mean",
    "selected_dispersion_mean",
    "boundary_batch_mi_mean",
    "boundary_local_consistency_mean",
    "boundary_residual_mean",
    "score_margin",
    "gate_entropy",
    "dist_to_heuristic",
    "dist_to_stage1",
    "prototype_distance",
)

_RUNTIME_CACHE: dict[tuple[str, str], dict[str, object]] = {}


def choose_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GateLearningConfig:
    top_k: int = 200
    random_state: int = 0
    train_scenarios: tuple[str, ...] = ("discrete", "trajectory", "batch_shift")
    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    cell_options: tuple[int, ...] = (600, 900, 1200)
    gene_options: tuple[int, ...] = (1200, 1800, 2400)
    candidate_random_gates: int = 6
    reward_temperature: float = 0.12
    train_with_refine: bool = False
    gate_type: str = "mlp"
    hidden_dim: int = 32
    dropout: float = 0.10
    train_epochs: int = 180
    learning_rate: float = 5e-3
    weight_decay: float = 2e-4
    batch_size: int = 16
    val_fraction: float = 0.25
    entropy_bonus: float = 0.01
    patience: int = 30
    residual_scale: float = 0.70
    preference_weight: float = 0.30
    anchor_weight: float = 0.08
    preference_margin: float = 0.02
    floor_weight: float = 1.10
    floor_ratio: float = 0.92
    floor_margin_ari: float = 0.03
    floor_margin_nmi: float = 0.04
    floor_margin_structure: float = 0.05
    tradeoff_penalty_weight: float = 1.35
    baseline_ce_weight: float = 0.18
    baseline_blend_scale: float = 0.10
    prototype_k: int = 2
    prototype_blend: float = 0.65
    heuristic_fallback: float = 0.18
    safety_floor_strength: float = 0.18
    bank_prototype_candidates: int = 3
    bank_boost_scale: float = 0.58
    bank_hidden_dim: int = 32
    bank_dropout: float = 0.05
    bank_train_epochs: int = 140
    bank_learning_rate: float = 4e-3
    bank_weight_decay: float = 1e-4
    bank_temperature: float = 0.18
    bank_mix_temperature: float = 0.12
    bank_patience: int = 25
    bank_uncertainty_views: int = 3
    bank_subsample_ratio: float = 0.78
    bank_reliability_blend: float = 0.72
    bank_transfer_penalty: float = 0.45
    bank_min_reliability: float = 0.12
    curr_hidden_dim: int = 40
    curr_dropout: float = 0.05
    curr_train_epochs: int = 160
    curr_learning_rate: float = 3e-3
    curr_weight_decay: float = 1e-4
    curr_patience: int = 28
    curr_policy_temperature: float = 0.18
    curr_ranking_weight: float = 0.85
    curr_regression_weight: float = 1.0
    curr_refine_weight: float = 0.55
    curr_risk_weight: float = 1.15
    curr_regret_weight: float = 0.90
    curr_conservative_temperature: float = 0.30
    curr_uncertainty_weight: float = 0.55
    curr_refine_temperature: float = 0.22
    curr_refine_conservative_weight: float = 0.55
    pairwise_hidden_dim: int = 48
    pairwise_dropout: float = 0.05
    pairwise_train_epochs: int = 180
    pairwise_learning_rate: float = 3e-3
    pairwise_weight_decay: float = 1e-4
    pairwise_patience: int = 28
    pairwise_regression_weight: float = 1.0
    pairwise_counterfactual_weight: float = 0.85
    pairwise_margin_weight: float = 0.40
    pairwise_risk_weight: float = 1.10
    pairwise_regret_weight: float = 0.85
    pairwise_margin_scale: float = 0.20
    pairwise_route_threshold: float = 0.05
    pairwise_safe_utility_floor: float = 0.03
    pairwise_safe_utility_temperature: float = 0.08
    pairwise_safe_risk_budget: float = 0.24
    pairwise_safe_risk_temperature: float = 0.10
    pairwise_consistency_threshold: float = 0.62
    pairwise_consistency_margin_scale: float = 0.18
    pairwise_consistency_std_scale: float = 0.14
    pairwise_min_regret_weight: float = 0.05
    pairwise_constraint_floor: float = 0.50
    pairperm_hidden_dim: int = 56
    pairperm_dropout: float = 0.05
    pairperm_train_epochs: int = 220
    pairperm_learning_rate: float = 3e-3
    pairperm_weight_decay: float = 1e-4
    pairperm_patience: int = 32
    pairperm_value_weight: float = 1.0
    pairperm_permission_weight: float = 1.20
    pairperm_budget_weight: float = 0.75
    pairperm_regret_weight: float = 0.80
    pairperm_permission_threshold: float = 0.50
    pairperm_value_scale: float = 0.75
    pairperm_risk_budget_floor: float = 0.10
    pairperm_risk_budget_ceiling: float = 0.52
    pairperm_consistency_budget_floor: float = 0.24
    pairperm_consistency_budget_ceiling: float = 0.84
    pairperm_risk_budget_temperature: float = 0.08
    pairperm_consistency_budget_temperature: float = 0.08
    escapecert_hidden_dim: int = 64
    escapecert_dropout: float = 0.05
    escapecert_train_epochs: int = 260
    escapecert_learning_rate: float = 3e-3
    escapecert_weight_decay: float = 1e-4
    escapecert_patience: int = 36
    escapecert_value_weight: float = 1.0
    escapecert_anchor_weight: float = 1.15
    escapecert_admissibility_weight: float = 1.0
    escapecert_budget_weight: float = 0.70
    escapecert_regret_weight: float = 0.80
    escapecert_uncertainty_weight: float = 0.30
    escapecert_set_consistency_weight: float = 0.55
    escapecert_cardinality_weight: float = 0.22
    escapecert_topm: int = 3
    escapecert_anchor_temperature: float = 0.20
    escapecert_admissibility_threshold: float = 0.45
    frontier_hidden_dim: int = 72
    frontier_dropout: float = 0.05
    frontier_train_epochs: int = 300
    frontier_learning_rate: float = 3e-3
    frontier_weight_decay: float = 1e-4
    frontier_patience: int = 40
    frontier_value_weight: float = 1.0
    frontier_anchor_weight: float = 1.15
    frontier_teacher_weight: float = 1.0
    frontier_accept_weight: float = 1.10
    frontier_budget_weight: float = 0.25
    frontier_regret_weight: float = 0.80
    frontier_uncertainty_weight: float = 0.35
    frontier_coverage_weight: float = 0.55
    frontier_false_release_weight: float = 0.38
    frontier_missed_escape_weight: float = 0.82
    frontier_teacher_topm: int = 4
    frontier_teacher_temperature: float = 0.18
    frontier_margin_temperature: float = 0.20
    frontier_accept_threshold: float = 0.45


class LearnableDatasetGate(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        gate_type: str = "mlp",
        hidden_dim: int = 32,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if gate_type == "linear":
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GateBankScorer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CounterfactualTriFactorPolicy(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 40,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.regret_head = nn.Linear(hidden_dim, 1)
        self.refine_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        utility = torch.sigmoid(self.utility_head(hidden)).squeeze(-1)
        risk = torch.sigmoid(self.risk_head(hidden)).squeeze(-1)
        regret = torch.sigmoid(self.regret_head(hidden)).squeeze(-1)
        refine_delta = torch.tanh(self.refine_head(hidden)).squeeze(-1)
        return utility, risk, regret, refine_delta


class Stage1AnchoredPairwiseRouter(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 48,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.regret_head = nn.Linear(hidden_dim, 1)
        self.margin_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        utility = torch.sigmoid(self.utility_head(hidden)).squeeze(-1)
        risk = torch.sigmoid(self.risk_head(hidden)).squeeze(-1)
        regret = torch.sigmoid(self.regret_head(hidden)).squeeze(-1)
        margin = torch.tanh(self.margin_head(hidden)).squeeze(-1)
        return utility, risk, regret, margin


class PermissionedStage1AnchoredPairwiseRouter(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 56,
        dropout: float = 0.05,
        risk_budget_floor: float = 0.10,
        risk_budget_ceiling: float = 0.52,
        consistency_budget_floor: float = 0.24,
        consistency_budget_ceiling: float = 0.84,
    ) -> None:
        super().__init__()
        self.risk_budget_floor = float(risk_budget_floor)
        self.risk_budget_ceiling = float(risk_budget_ceiling)
        self.consistency_budget_floor = float(consistency_budget_floor)
        self.consistency_budget_ceiling = float(consistency_budget_ceiling)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.regret_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.permission_head = nn.Linear(hidden_dim, 1)
        self.risk_budget_head = nn.Linear(hidden_dim, 1)
        self.consistency_budget_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        utility = torch.sigmoid(self.utility_head(hidden)).squeeze(-1)
        risk = torch.sigmoid(self.risk_head(hidden)).squeeze(-1)
        regret = torch.sigmoid(self.regret_head(hidden)).squeeze(-1)
        value = torch.tanh(self.value_head(hidden)).squeeze(-1)
        permission_logit = self.permission_head(hidden).squeeze(-1)
        risk_budget = self.risk_budget_floor + (
            self.risk_budget_ceiling - self.risk_budget_floor
        ) * torch.sigmoid(self.risk_budget_head(hidden)).squeeze(-1)
        consistency_budget = self.consistency_budget_floor + (
            self.consistency_budget_ceiling - self.consistency_budget_floor
        ) * torch.sigmoid(self.consistency_budget_head(hidden)).squeeze(-1)
        return (
            utility,
            risk,
            regret,
            value,
            permission_logit,
            risk_budget,
            consistency_budget,
        )


class EscapeCertStage1AnchoredPairwiseRouter(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.05,
        risk_budget_floor: float = 0.10,
        risk_budget_ceiling: float = 0.52,
        consistency_budget_floor: float = 0.24,
        consistency_budget_ceiling: float = 0.84,
    ) -> None:
        super().__init__()
        self.risk_budget_floor = float(risk_budget_floor)
        self.risk_budget_ceiling = float(risk_budget_ceiling)
        self.consistency_budget_floor = float(consistency_budget_floor)
        self.consistency_budget_ceiling = float(consistency_budget_ceiling)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.regret_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.anchor_escape_head = nn.Linear(hidden_dim, 1)
        self.admissibility_head = nn.Linear(hidden_dim, 1)
        self.risk_budget_head = nn.Linear(hidden_dim, 1)
        self.consistency_budget_head = nn.Linear(hidden_dim, 1)
        self.escape_uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = self.backbone(x)
        utility = torch.sigmoid(self.utility_head(hidden)).squeeze(-1)
        risk = torch.sigmoid(self.risk_head(hidden)).squeeze(-1)
        regret = torch.sigmoid(self.regret_head(hidden)).squeeze(-1)
        value = torch.tanh(self.value_head(hidden)).squeeze(-1)
        anchor_escape_logit = self.anchor_escape_head(hidden).squeeze(-1)
        admissibility_logit = self.admissibility_head(hidden).squeeze(-1)
        risk_budget = self.risk_budget_floor + (
            self.risk_budget_ceiling - self.risk_budget_floor
        ) * torch.sigmoid(self.risk_budget_head(hidden)).squeeze(-1)
        consistency_budget = self.consistency_budget_floor + (
            self.consistency_budget_ceiling - self.consistency_budget_floor
        ) * torch.sigmoid(self.consistency_budget_head(hidden)).squeeze(-1)
        escape_uncertainty = torch.sigmoid(self.escape_uncertainty_head(hidden)).squeeze(-1)
        return (
            utility,
            risk,
            regret,
            value,
            anchor_escape_logit,
            admissibility_logit,
            risk_budget,
            consistency_budget,
            escape_uncertainty,
        )


class FrontierStage1AnchoredPairwiseRouter(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 72,
        dropout: float = 0.05,
        risk_budget_floor: float = 0.10,
        risk_budget_ceiling: float = 0.52,
        consistency_budget_floor: float = 0.24,
        consistency_budget_ceiling: float = 0.84,
    ) -> None:
        super().__init__()
        self.risk_budget_floor = float(risk_budget_floor)
        self.risk_budget_ceiling = float(risk_budget_ceiling)
        self.consistency_budget_floor = float(consistency_budget_floor)
        self.consistency_budget_ceiling = float(consistency_budget_ceiling)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.regret_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.anchor_escape_head = nn.Linear(hidden_dim, 1)
        self.teacher_support_head = nn.Linear(hidden_dim, 1)
        self.frontier_head = nn.Linear(hidden_dim, 1)
        self.risk_budget_head = nn.Linear(hidden_dim, 1)
        self.consistency_budget_head = nn.Linear(hidden_dim, 1)
        self.frontier_uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = self.backbone(x)
        utility = torch.sigmoid(self.utility_head(hidden)).squeeze(-1)
        risk = torch.sigmoid(self.risk_head(hidden)).squeeze(-1)
        regret = torch.sigmoid(self.regret_head(hidden)).squeeze(-1)
        value = torch.tanh(self.value_head(hidden)).squeeze(-1)
        anchor_escape_logit = self.anchor_escape_head(hidden).squeeze(-1)
        teacher_support_logit = self.teacher_support_head(hidden).squeeze(-1)
        frontier_logit = self.frontier_head(hidden).squeeze(-1)
        risk_budget = self.risk_budget_floor + (
            self.risk_budget_ceiling - self.risk_budget_floor
        ) * torch.sigmoid(self.risk_budget_head(hidden)).squeeze(-1)
        consistency_budget = self.consistency_budget_floor + (
            self.consistency_budget_ceiling - self.consistency_budget_floor
        ) * torch.sigmoid(self.consistency_budget_head(hidden)).squeeze(-1)
        frontier_uncertainty = torch.sigmoid(self.frontier_uncertainty_head(hidden)).squeeze(-1)
        return (
            utility,
            risk,
            regret,
            value,
            anchor_escape_logit,
            teacher_support_logit,
            frontier_logit,
            risk_budget,
            consistency_budget,
            frontier_uncertainty,
        )


def dataset_stats_to_vector(
    dataset_stats: dict[str, float],
    feature_keys: tuple[str, ...] = DEFAULT_GATE_FEATURE_KEYS,
) -> np.ndarray:
    return np.asarray([float(dataset_stats.get(key, 0.0)) for key in feature_keys], dtype=np.float32)


def downstream_reward(metrics: dict[str, float]) -> float:
    structure_score = structure_preservation_score(metrics)
    reward = (
        0.38 * float(metrics.get("ari", 0.0))
        + 0.28 * float(metrics.get("nmi", 0.0))
        + 0.20 * structure_score
        + 0.08 * float(metrics.get("batch_mixing", 0.0))
        + 0.06 * float(metrics.get("stability", 0.0))
    )
    return float(reward)


def structure_preservation_score(metrics: dict[str, float]) -> float:
    label_sil = 0.5 * (float(metrics.get("label_silhouette", 0.0)) + 1.0)
    cluster_sil = 0.5 * (float(metrics.get("cluster_silhouette", 0.0)) + 1.0)
    neighbor = float(metrics.get("neighbor_preservation", 0.0))
    return float(0.38 * label_sil + 0.27 * cluster_sil + 0.35 * neighbor)


def build_gate_candidates(
    *,
    heuristic_gate: np.ndarray,
    n_random: int,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray]:
    candidates: list[np.ndarray] = []
    names: list[str] = []

    uniform = np.ones(len(EXPERT_NAMES), dtype=np.float64) / len(EXPERT_NAMES)
    heuristic = _normalize_gate_np(heuristic_gate)

    def add(name: str, gate: np.ndarray) -> None:
        names.append(name)
        candidates.append(_normalize_gate_np(gate))

    add("uniform", uniform)
    add("heuristic", heuristic)
    add("heuristic_blend", 0.65 * heuristic + 0.35 * uniform)

    for expert_idx, expert_name in enumerate(EXPERT_NAMES):
        one_hot = np.zeros(len(EXPERT_NAMES), dtype=np.float64)
        one_hot[expert_idx] = 1.0
        add(f"expert_{expert_name}", one_hot)

    for expert_idx, expert_name in enumerate(EXPERT_NAMES):
        boosted = 0.60 * np.eye(len(EXPERT_NAMES), dtype=np.float64)[expert_idx] + 0.40 * heuristic
        add(f"boosted_{expert_name}", boosted)

    for random_idx in range(n_random):
        add(f"random_{random_idx:02d}", rng.dirichlet(alpha=0.70 * np.ones(len(EXPERT_NAMES), dtype=np.float64)))

    return names, np.stack(candidates, axis=0)


def gate_bank_candidate_names(*, prototype_candidates: int) -> tuple[str, ...]:
    names = ["heuristic", "stage1_residual", "stage1_blended", "prototype_mean"]
    names.extend(f"prototype_{idx + 1}" for idx in range(prototype_candidates))
    names.extend(f"boosted_{expert}" for expert in BANK_BOOST_EXPERTS)
    return tuple(names)


def _gate_entropy_np(gate: np.ndarray) -> float:
    gate = _normalize_gate_np(gate)
    return float(-(gate * np.log(gate + 1e-8)).sum())


def _sigmoid_np(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _pairwise_positive_score(
    *,
    raw_utility_gain: float,
    config: GateLearningConfig,
) -> float:
    centered = (raw_utility_gain - config.pairwise_safe_utility_floor) / max(
        config.pairwise_safe_utility_temperature,
        1e-6,
    )
    return float(np.clip(_sigmoid_np(centered), 0.0, 1.0))


def _pairwise_safety_score(
    *,
    failure_risk: float,
    config: GateLearningConfig,
) -> float:
    centered = (config.pairwise_safe_risk_budget - failure_risk) / max(
        config.pairwise_safe_risk_temperature,
        1e-6,
    )
    return float(np.clip(_sigmoid_np(centered), 0.0, 1.0))


def _pairwise_consistency_score(
    *,
    win_rate: float,
    margin_mean: float,
    margin_std: float,
    config: GateLearningConfig,
) -> float:
    clipped_win_rate = float(np.clip(win_rate, 0.0, 1.0))
    win_score = _sigmoid_np((clipped_win_rate - 0.5) / 0.12)
    margin_score = _sigmoid_np(margin_mean / max(config.pairwise_consistency_margin_scale, 1e-6))
    positive_margin = max(margin_mean, 0.0)
    stability_score = float(
        np.clip(
            (positive_margin + config.pairwise_consistency_margin_scale)
            / (
                positive_margin
                + max(margin_std, 0.0)
                + config.pairwise_consistency_margin_scale
            ),
            0.0,
            1.0,
        )
    )
    return float(
        np.clip(
            0.45 * win_score
            + 0.35 * margin_score
            + 0.20 * stability_score,
            0.0,
            1.0,
        )
    )


def _pairwise_route_constraint_score(
    *,
    raw_utility_gain: float,
    failure_risk: float,
    route_consistency: float,
    config: GateLearningConfig,
) -> float:
    positive_score = _pairwise_positive_score(raw_utility_gain=raw_utility_gain, config=config)
    safety_score = _pairwise_safety_score(failure_risk=failure_risk, config=config)
    route_consistency = float(np.clip(route_consistency, 0.0, 1.0))
    return float(
        np.clip(
            positive_score * np.sqrt(max(safety_score, 1e-8) * max(route_consistency, 1e-8)),
            0.0,
            1.0,
        )
    )


def _pairwise_safe_positive_mask(
    *,
    raw_utility_gain: float,
    failure_risk: float,
    fallback_regret_raw: float,
    route_consistency: float,
    route_margin_raw: float,
    config: GateLearningConfig,
) -> float:
    route_constraint_score = _pairwise_route_constraint_score(
        raw_utility_gain=raw_utility_gain,
        failure_risk=failure_risk,
        route_consistency=route_consistency,
        config=config,
    )
    passes = (
        raw_utility_gain > 0.0
        and fallback_regret_raw > 0.0
        and route_margin_raw > 0.0
        and route_constraint_score >= config.pairwise_constraint_floor
    )
    return 1.0 if passes else 0.0


def _build_pairwise_calibration_targets(
    *,
    raw_utility_gain: float,
    failure_risk: float,
    fallback_regret_raw: float,
    route_margin_raw: float,
    route_consistency: float,
    config: GateLearningConfig,
) -> dict[str, float]:
    route_win_raw = float(route_margin_raw > 0.0)
    route_consistency = float(np.clip(route_consistency, 0.0, 1.0))
    positive_score = _pairwise_positive_score(raw_utility_gain=raw_utility_gain, config=config)
    safety_score = _pairwise_safety_score(failure_risk=failure_risk, config=config)
    route_constraint_score = _pairwise_route_constraint_score(
        raw_utility_gain=raw_utility_gain,
        failure_risk=failure_risk,
        route_consistency=route_consistency,
        config=config,
    )
    safe_positive_mask = _pairwise_safe_positive_mask(
        raw_utility_gain=raw_utility_gain,
        failure_risk=failure_risk,
        fallback_regret_raw=fallback_regret_raw,
        route_consistency=route_consistency,
        route_margin_raw=route_margin_raw,
        config=config,
    )
    safe_gain_support = float(np.sqrt(max(positive_score, 1e-8) * max(safety_score, 1e-8)))
    safe_utility_gain = float(np.clip(raw_utility_gain * safe_gain_support, 0.0, 1.0))
    fallback_regret_calibrated = float(
        np.clip(fallback_regret_raw * route_constraint_score, 0.0, 1.0)
    )
    route_margin_calibrated = float(
        np.tanh(route_margin_raw / max(config.pairwise_margin_scale, 1e-6)) * route_constraint_score
    )
    route_permission_target = float(np.clip(route_win_raw * route_constraint_score, 0.0, 1.0))
    route_win_safe = float(route_permission_target >= config.pairwise_constraint_floor and safe_positive_mask > 0.5)
    regret_supervision_weight = float(
        np.clip(
            config.pairwise_min_regret_weight
            + (1.0 - config.pairwise_min_regret_weight) * np.power(route_constraint_score, 1.5),
            0.0,
            1.0,
        )
    )
    final_route_score = float(
        safe_utility_gain
        + config.pairwise_regret_weight * fallback_regret_calibrated
        - config.pairwise_risk_weight * failure_risk
        + config.pairwise_margin_weight * route_margin_calibrated
    )
    return {
        "pairwise_raw_utility_gain": float(raw_utility_gain),
        "pairwise_safe_utility_gain": safe_utility_gain,
        "pairwise_failure_risk": float(failure_risk),
        "pairwise_fallback_regret_raw": float(fallback_regret_raw),
        "pairwise_fallback_regret_calibrated": fallback_regret_calibrated,
        "pairwise_route_margin_raw": float(route_margin_raw),
        "pairwise_route_margin_calibrated": route_margin_calibrated,
        "pairwise_route_win_raw": route_win_raw,
        "pairwise_route_win_safe": route_win_safe,
        "pairwise_route_permission_target": route_permission_target,
        "pairwise_route_consistency": route_consistency,
        "pairwise_safe_positive_mask": safe_positive_mask,
        "pairwise_regret_supervision_weight": regret_supervision_weight,
        "pairwise_route_constraint_score": route_constraint_score,
        "pairwise_final_route_score": final_route_score,
    }


def _apply_pairwise_calibration_variant(
    *,
    calibrated: dict[str, float],
    raw_utility_gain: float,
    failure_risk: float,
    fallback_regret_raw: float,
    route_margin_raw: float,
    variant: str,
    config: GateLearningConfig,
) -> dict[str, float]:
    adjusted = dict(calibrated)
    raw_margin_target = float(np.tanh(route_margin_raw / max(config.pairwise_margin_scale, 1e-6)))

    if variant == "no_regret_calibration":
        adjusted["pairwise_fallback_regret_calibrated"] = float(fallback_regret_raw)
        adjusted["pairwise_regret_supervision_weight"] = 1.0 if fallback_regret_raw > 0.0 else config.pairwise_min_regret_weight
    elif variant == "no_route_constraint":
        route_win_raw = float(route_margin_raw > 0.0)
        adjusted["pairwise_safe_utility_gain"] = float(raw_utility_gain)
        adjusted["pairwise_fallback_regret_calibrated"] = float(fallback_regret_raw)
        adjusted["pairwise_route_margin_calibrated"] = raw_margin_target
        adjusted["pairwise_route_permission_target"] = route_win_raw
        adjusted["pairwise_route_win_safe"] = route_win_raw
        adjusted["pairwise_safe_positive_mask"] = float(route_win_raw > 0.5 and raw_utility_gain > config.pairwise_safe_utility_floor)
        adjusted["pairwise_regret_supervision_weight"] = (
            1.0 if fallback_regret_raw > 0.0 else config.pairwise_min_regret_weight
        )
        adjusted["pairwise_route_constraint_score"] = 1.0
    elif variant == "utility_only":
        adjusted["pairwise_failure_risk"] = 0.0
        adjusted["pairwise_fallback_regret_calibrated"] = 0.0
        adjusted["pairwise_regret_supervision_weight"] = config.pairwise_min_regret_weight

    adjusted["pairwise_final_route_score"] = float(
        adjusted["pairwise_safe_utility_gain"]
        + config.pairwise_regret_weight * adjusted["pairwise_fallback_regret_calibrated"]
        - config.pairwise_risk_weight * adjusted["pairwise_failure_risk"]
        + config.pairwise_margin_weight * adjusted["pairwise_route_margin_calibrated"]
    )
    return adjusted


def _pairperm_anchor_strength(
    *,
    anchor_features: dict[str, float],
) -> float:
    structure_support = _sigmoid_np(2.2 * float(anchor_features.get("proxy_structure_support", 0.0)))
    refine_support = _sigmoid_np(1.4 * float(anchor_features.get("proxy_safe_refine_support", 0.0)))
    transfer_safety = 1.0 - _sigmoid_np(2.2 * float(anchor_features.get("proxy_transfer_risk", 0.0)) - 0.5)
    margin_support = _sigmoid_np(4.0 * float(anchor_features.get("score_margin", 0.0)))
    boundary_safety = 1.0 - _sigmoid_np(1.8 * float(anchor_features.get("boundary_batch_mi_mean", 0.0)))
    return float(
        np.clip(
            0.30 * structure_support
            + 0.24 * refine_support
            + 0.18 * transfer_safety
            + 0.16 * margin_support
            + 0.12 * boundary_safety,
            0.0,
            1.0,
        )
    )


def _pairperm_candidate_escape_support(
    *,
    safe_utility_gain: float,
    fallback_regret_calibrated: float,
    failure_risk: float,
    route_consistency: float,
    route_margin_raw: float,
    safe_positive_mask: float,
    config: GateLearningConfig,
) -> float:
    utility_support = _pairwise_positive_score(raw_utility_gain=safe_utility_gain, config=config)
    regret_support = _sigmoid_np((fallback_regret_calibrated - 0.02) / 0.06)
    margin_support = _sigmoid_np(route_margin_raw / max(config.pairwise_margin_scale, 1e-6))
    safety_support = _sigmoid_np(
        (config.pairwise_safe_risk_budget - failure_risk) / max(config.pairwise_safe_risk_temperature, 1e-6)
    )
    return float(
        np.clip(
            0.26 * utility_support
            + 0.20 * regret_support
            + 0.18 * route_consistency
            + 0.16 * margin_support
            + 0.12 * safety_support
            + 0.08 * safe_positive_mask,
            0.0,
            1.0,
        )
    )


def _pairperm_route_value_target(
    *,
    safe_utility_gain: float,
    fallback_regret_calibrated: float,
    failure_risk: float,
    route_margin_calibrated: float,
    config: GateLearningConfig,
    use_regret_aux: bool = True,
) -> tuple[float, float]:
    regret_term = fallback_regret_calibrated if use_regret_aux else 0.0
    raw_value = (
        safe_utility_gain
        + config.pairwise_regret_weight * regret_term
        - 0.65 * failure_risk
        + 0.20 * route_margin_calibrated
    )
    value_target = float(np.tanh(raw_value / max(config.pairperm_value_scale, 1e-6)))
    return raw_value, value_target


def _pairperm_anchor_context(
    *,
    candidate_summaries: list[dict[str, float]],
    anchor_features: dict[str, float],
) -> dict[str, float]:
    anchor_strength = _pairperm_anchor_strength(anchor_features=anchor_features)
    non_stage1 = [summary for summary in candidate_summaries if summary.get("candidate_is_stage1", 0.0) < 0.5]
    if not non_stage1:
        return {
            "anchor_strength": anchor_strength,
            "anchor_escape_pressure": 0.0,
            "anchor_fragility": 1.0 - anchor_strength,
            "anchor_routability": 0.0,
        }

    escape_support = np.asarray([summary["escape_support"] for summary in non_stage1], dtype=np.float64)
    route_consistency = np.asarray([summary["route_consistency"] for summary in non_stage1], dtype=np.float64)
    value_target = np.asarray([summary["route_value_target"] for summary in non_stage1], dtype=np.float64)
    safe_utility = np.asarray([summary["safe_utility_gain"] for summary in non_stage1], dtype=np.float64)

    best_escape = float(np.max(escape_support))
    mean_escape = float(np.mean(escape_support))
    safe_share = float(np.mean(escape_support >= 0.50))
    best_consistency = float(np.max(route_consistency))
    best_value = float(np.max(value_target))
    best_safe_utility = float(np.max(safe_utility))

    escape_pressure = float(
        np.clip(
            0.36 * best_escape
            + 0.18 * mean_escape
            + 0.16 * safe_share
            + 0.15 * best_consistency
            + 0.15 * (1.0 - anchor_strength),
            0.0,
            1.0,
        )
    )
    anchor_fragility = float(
        np.clip(
            0.48 * escape_pressure
            + 0.18 * _sigmoid_np(2.0 * float(anchor_features.get("proxy_transfer_risk", 0.0)))
            + 0.16 * (1.0 - anchor_strength)
            + 0.10 * _sigmoid_np(1.6 * float(anchor_features.get("boundary_batch_mi_mean", 0.0)))
            + 0.08 * (1.0 - _sigmoid_np(4.0 * float(anchor_features.get("score_margin", 0.0)))),
            0.0,
            1.0,
        )
    )
    anchor_routability = float(
        np.clip(
            0.34 * best_escape
            + 0.20 * best_consistency
            + 0.18 * _sigmoid_np(6.0 * best_safe_utility - 1.0)
            + 0.16 * _sigmoid_np(3.5 * best_value)
            + 0.12 * (1.0 - anchor_strength),
            0.0,
            1.0,
        )
    )
    return {
        "anchor_strength": anchor_strength,
        "anchor_escape_pressure": escape_pressure,
        "anchor_fragility": anchor_fragility,
        "anchor_routability": anchor_routability,
    }


def _pairperm_budget_targets(
    *,
    anchor_context: dict[str, float],
    dataset_stats: dict[str, float],
    config: GateLearningConfig,
    variant: str,
) -> tuple[float, float]:
    base_risk_budget = float(config.pairwise_safe_risk_budget)
    base_consistency_budget = float(config.pairwise_consistency_threshold)

    if variant == "fixed_budget":
        risk_budget = base_risk_budget
        consistency_budget = base_consistency_budget
    elif variant == "no_anchor_adaptation":
        dataset_pressure = float(
            np.clip(
                0.45 * float(dataset_stats.get("batch_strength", 0.0))
                + 0.30 * float(dataset_stats.get("trajectory_strength", 0.0))
                + 0.25 * (1.0 - float(dataset_stats.get("cluster_strength", 0.0))),
                0.0,
                1.0,
            )
        )
        risk_budget = base_risk_budget + 0.14 * dataset_pressure - 0.04 * float(dataset_stats.get("cluster_strength", 0.0))
        consistency_budget = (
            base_consistency_budget
            - 0.14 * dataset_pressure
            + 0.05 * float(dataset_stats.get("cluster_strength", 0.0))
        )
    else:
        risk_budget = (
            base_risk_budget
            + 0.16 * anchor_context["anchor_escape_pressure"]
            + 0.12 * anchor_context["anchor_fragility"]
            + 0.08 * anchor_context["anchor_routability"]
            - 0.11 * anchor_context["anchor_strength"]
        )
        consistency_budget = (
            base_consistency_budget
            + 0.10 * anchor_context["anchor_strength"]
            - 0.17 * anchor_context["anchor_escape_pressure"]
            - 0.10 * anchor_context["anchor_fragility"]
            - 0.05 * anchor_context["anchor_routability"]
        )

    risk_budget = float(
        np.clip(
            risk_budget,
            config.pairperm_risk_budget_floor,
            config.pairperm_risk_budget_ceiling,
        )
    )
    consistency_budget = float(
        np.clip(
            consistency_budget,
            config.pairperm_consistency_budget_floor,
            config.pairperm_consistency_budget_ceiling,
        )
    )
    return risk_budget, consistency_budget


def _pairperm_budget_score(
    *,
    failure_risk: float,
    route_consistency: float,
    adaptive_risk_budget: float,
    adaptive_consistency_budget: float,
    config: GateLearningConfig,
) -> dict[str, float]:
    risk_pass = _sigmoid_np(
        (adaptive_risk_budget - failure_risk) / max(config.pairperm_risk_budget_temperature, 1e-6)
    )
    consistency_pass = _sigmoid_np(
        (route_consistency - adaptive_consistency_budget) / max(config.pairperm_consistency_budget_temperature, 1e-6)
    )
    permission_budget_score = float(np.clip(0.54 * risk_pass + 0.46 * consistency_pass, 0.0, 1.0))
    permission_margin = float(
        0.5
        * (
            (adaptive_risk_budget - failure_risk) / max(config.pairperm_risk_budget_temperature, 1e-6)
            + (route_consistency - adaptive_consistency_budget)
            / max(config.pairperm_consistency_budget_temperature, 1e-6)
        )
    )
    permission_confidence = float(
        np.clip(
            0.55 * abs(risk_pass - 0.5) * 2.0 + 0.45 * abs(consistency_pass - 0.5) * 2.0,
            0.0,
            1.0,
        )
    )
    return {
        "risk_pass": float(risk_pass),
        "consistency_pass": float(consistency_pass),
        "permission_budget_score": permission_budget_score,
        "permission_margin": permission_margin,
        "permission_confidence": permission_confidence,
    }


def _pairperm_block_reason(
    *,
    candidate_name: str,
    safe_positive_mask: float,
    failure_risk: float,
    route_consistency: float,
    adaptive_risk_budget: float,
    adaptive_consistency_budget: float,
    anchor_escape_pressure: float,
    permission_calibrated: float,
    config: GateLearningConfig,
) -> str:
    if candidate_name == "stage1_blended":
        return "anchor_stage1"
    if safe_positive_mask <= 0.0:
        return "not_safe_positive"
    if failure_risk > adaptive_risk_budget and route_consistency < adaptive_consistency_budget:
        return "risk_and_consistency_budget"
    if failure_risk > adaptive_risk_budget:
        return "risk_budget"
    if route_consistency < adaptive_consistency_budget:
        return "consistency_budget"
    if anchor_escape_pressure < 0.35 and permission_calibrated < config.pairperm_permission_threshold:
        return "anchor_not_fragile_enough"
    if permission_calibrated < config.pairperm_permission_threshold:
        return "permission_low"
    return "allowed"


def _escapecert_variant_flags(variant: str) -> dict[str, bool]:
    return {
        "use_anchor_head": variant not in {"no_anchor_escape_head", "admissibility_only"},
        "use_set_supervision": variant != "no_set_supervision",
        "use_admissibility_head": variant not in {"no_candidate_admissibility", "anchor_escape_only"},
        "use_uncertainty": variant != "no_escape_uncertainty",
        "use_regret_aux": variant != "no_regret_aux",
        "fixed_budget": variant == "fixed_budget",
    }


def _escapecert_candidate_membership_oracle(
    *,
    safe_utility_gain: float,
    fallback_regret_calibrated: float,
    failure_risk: float,
    route_consistency: float,
    route_value_target: float,
    route_constraint_score: float,
    escape_support: float,
    config: GateLearningConfig,
    use_regret_aux: bool,
) -> float:
    utility_support = _pairwise_positive_score(raw_utility_gain=safe_utility_gain, config=config)
    value_support = _sigmoid_np(3.8 * route_value_target)
    regret_support = (
        _sigmoid_np((fallback_regret_calibrated - 0.015) / 0.05)
        if use_regret_aux
        else _sigmoid_np(3.0 * route_value_target - 0.2)
    )
    risk_safety = _sigmoid_np((0.34 - failure_risk) / 0.08)
    consistency_support = _sigmoid_np((route_consistency - 0.52) / 0.08)
    constraint_support = _sigmoid_np((route_constraint_score - 0.34) / 0.10)
    return float(
        np.clip(
            0.22 * utility_support
            + 0.22 * value_support
            + 0.16 * regret_support
            + 0.16 * risk_safety
            + 0.14 * consistency_support
            + 0.06 * constraint_support
            + 0.04 * escape_support,
            0.0,
            1.0,
        )
    )


def _escapecert_anchor_targets(
    *,
    candidate_summaries: list[dict[str, float | str]],
    anchor_features: dict[str, float],
    config: GateLearningConfig,
    use_set_supervision: bool,
) -> dict[str, object]:
    anchor_strength = _pairperm_anchor_strength(anchor_features=anchor_features)
    non_stage1 = [summary for summary in candidate_summaries if float(summary.get("candidate_is_stage1", 0.0)) < 0.5]
    if not non_stage1:
        return {
            "anchor_strength": anchor_strength,
            "anchor_escape_target": 0.0,
            "anchor_escape_soft_oracle_gain": 0.0,
            "anchor_escape_safe_candidate_count": 0.0,
            "anchor_escape_topm_mass": 0.0,
            "anchor_escape_counterfactual_margin": 0.0,
            "anchor_escape_uncertainty_target": 1.0 - anchor_strength,
            "allowed_set_target_mass": 0.0,
            "permission_set_consistency_target": 0.0,
            "counterfactual_escape_support": 0.0,
            "top_candidate_names": tuple(),
        }

    admissibility_oracle = np.asarray(
        [float(summary["admissibility_oracle"]) for summary in non_stage1],
        dtype=np.float64,
    )
    escape_support = np.asarray([float(summary["escape_support"]) for summary in non_stage1], dtype=np.float64)
    route_value = np.asarray([float(summary["route_value_target"]) for summary in non_stage1], dtype=np.float64)
    regret = np.asarray([float(summary["fallback_regret_calibrated"]) for summary in non_stage1], dtype=np.float64)
    failure_risk = np.asarray([float(summary["failure_risk"]) for summary in non_stage1], dtype=np.float64)
    route_consistency = np.asarray([float(summary["route_consistency"]) for summary in non_stage1], dtype=np.float64)
    legacy_permission = np.asarray(
        [float(summary.get("legacy_permission_target", 0.0)) for summary in non_stage1],
        dtype=np.float64,
    )

    combined_support = (
        0.46 * admissibility_oracle
        + 0.24 * np.clip(route_value, 0.0, 1.0)
        + 0.16 * escape_support
        + 0.14 * np.clip(regret, 0.0, 1.0)
    )
    order = np.argsort(combined_support)[::-1]
    topm = max(1, min(int(config.escapecert_topm), len(order)))
    top_idx = order[:topm]
    top_weights = _softmax_np(4.0 * combined_support[top_idx])
    safe_candidate_count = float(np.sum(admissibility_oracle >= 0.50))
    topm_mass = float(np.sum(top_weights * admissibility_oracle[top_idx]))
    soft_oracle_gain = float(np.sum(top_weights * np.clip(route_value[top_idx], 0.0, 1.0)))
    counterfactual_support = float(np.sum(top_weights * escape_support[top_idx]))
    counterfactual_margin = float(
        np.sum(top_weights * route_value[top_idx])
        + 0.35 * np.sum(top_weights * np.clip(regret[top_idx], 0.0, 1.0))
        - 0.25 * np.sum(top_weights * failure_risk[top_idx])
    )
    mean_consistency = float(np.sum(top_weights * route_consistency[top_idx]))
    mean_safe_share = float(np.mean(admissibility_oracle >= 0.50))

    if use_set_supervision:
        anchor_escape_target = float(
            np.clip(
                0.28 * topm_mass
                + 0.22 * _sigmoid_np(counterfactual_margin / 0.12)
                + 0.16 * mean_safe_share
                + 0.16 * mean_consistency
                + 0.10 * counterfactual_support
                + 0.08 * (1.0 - anchor_strength),
                0.0,
                1.0,
            )
        )
    else:
        anchor_escape_target = float(np.clip(np.max(legacy_permission), 0.0, 1.0))
        soft_oracle_gain = float(np.max(np.clip(route_value, 0.0, 1.0)))
        counterfactual_support = float(np.max(legacy_permission))
        topm_mass = float(np.max(admissibility_oracle))
        safe_candidate_count = float(np.sum(legacy_permission >= 0.50))
        counterfactual_margin = float(np.max(route_value))

    allowed_set_target_mass = float(np.clip(np.mean(admissibility_oracle), 0.0, 1.0))
    uncertainty_target = float(
        np.clip(
            (1.0 - np.clip(abs(counterfactual_margin) / 0.35, 0.0, 1.0))
            * (0.60 + 0.40 * (1.0 - topm_mass)),
            0.0,
            1.0,
        )
    )
    permission_set_consistency_target = float(
        np.clip(
            0.65 * anchor_escape_target + 0.35 * topm_mass,
            0.0,
            1.0,
        )
    )
    top_candidate_names = tuple(str(non_stage1[int(idx)]["candidate_name"]) for idx in top_idx)
    return {
        "anchor_strength": anchor_strength,
        "anchor_escape_target": anchor_escape_target,
        "anchor_escape_soft_oracle_gain": soft_oracle_gain,
        "anchor_escape_safe_candidate_count": safe_candidate_count,
        "anchor_escape_topm_mass": topm_mass,
        "anchor_escape_counterfactual_margin": counterfactual_margin,
        "anchor_escape_uncertainty_target": uncertainty_target,
        "allowed_set_target_mass": allowed_set_target_mass,
        "permission_set_consistency_target": permission_set_consistency_target,
        "counterfactual_escape_support": counterfactual_support,
        "top_candidate_names": top_candidate_names,
    }


def _escapecert_block_reason(
    *,
    candidate_name: str,
    anchor_escape_calibrated: float,
    candidate_admissibility_calibrated: float,
    candidate_budget_score: float,
    anchor_threshold: float,
    admissibility_threshold: float,
) -> str:
    if candidate_name == "stage1_blended":
        return "anchor_stage1"
    if anchor_escape_calibrated >= anchor_threshold and candidate_admissibility_calibrated >= admissibility_threshold:
        return "allowed"
    if anchor_escape_calibrated < anchor_threshold:
        return "anchor_escape_low"
    if candidate_budget_score < 0.25:
        return "escape_budget_low"
    if candidate_admissibility_calibrated < admissibility_threshold:
        return "not_in_admissible_set"
    return "allowed"


def _frontier_variant_flags(variant: str) -> dict[str, bool]:
    return {
        "use_anchor_head": variant != "no_anchor_escape_head",
        "use_teacher_head": variant != "no_teacher_frontier_only",
        "use_teacher_distill": variant not in {"no_teacher_distill", "no_teacher_frontier_only"},
        "use_frontier_head": variant not in {"no_frontier_head", "teacher_only"},
        "use_frontier_uncertainty": variant != "no_frontier_uncertainty",
        "fixed_frontier": variant == "fixed_frontier",
        "use_regret_aux": variant != "no_regret_aux",
        "teacher_only": variant == "teacher_only",
    }


def _frontier_teacher_membership_oracle(
    *,
    safe_utility_gain: float,
    fallback_regret_calibrated: float,
    failure_risk: float,
    route_consistency: float,
    route_value_target: float,
    route_constraint_score: float,
    escape_support: float,
    config: GateLearningConfig,
    use_regret_aux: bool,
) -> float:
    utility_support = _pairwise_positive_score(raw_utility_gain=safe_utility_gain, config=config)
    value_support = _sigmoid_np(3.4 * route_value_target + 0.10)
    regret_support = (
        _sigmoid_np((fallback_regret_calibrated - 0.010) / 0.06)
        if use_regret_aux
        else _sigmoid_np(2.8 * route_value_target - 0.10)
    )
    permissive_risk = _sigmoid_np((0.44 - failure_risk) / 0.12)
    consistency_support = _sigmoid_np((route_consistency - 0.44) / 0.10)
    constraint_support = _sigmoid_np((route_constraint_score - 0.22) / 0.12)
    return float(
        np.clip(
            0.24 * utility_support
            + 0.22 * value_support
            + 0.16 * regret_support
            + 0.14 * permissive_risk
            + 0.12 * consistency_support
            + 0.08 * constraint_support
            + 0.04 * escape_support,
            0.0,
            1.0,
        )
    )


def _frontier_anchor_targets(
    *,
    candidate_summaries: list[dict[str, float | str]],
    anchor_features: dict[str, float],
    config: GateLearningConfig,
) -> dict[str, object]:
    anchor_strength = _pairperm_anchor_strength(anchor_features=anchor_features)
    non_stage1 = [summary for summary in candidate_summaries if float(summary.get("candidate_is_stage1", 0.0)) < 0.5]
    if not non_stage1:
        return {
            "anchor_strength": anchor_strength,
            "anchor_escape_target": 0.0,
            "anchor_escape_soft_oracle_gain": 0.0,
            "anchor_escape_safe_candidate_count": 0.0,
            "anchor_escape_topm_mass": 0.0,
            "anchor_escape_counterfactual_margin": 0.0,
            "anchor_escape_uncertainty_target": 1.0 - anchor_strength,
            "teacher_set_mass": 0.0,
            "teacher_topm_mass": 0.0,
            "frontier_target_mass": 0.0,
            "frontier_coverage_target": 0.0,
            "frontier_consistency_target": 0.0,
            "counterfactual_escape_support": 0.0,
            "top_candidate_names": tuple(),
        }

    teacher_oracle = np.asarray([float(summary["teacher_oracle"]) for summary in non_stage1], dtype=np.float64)
    frontier_risk_reserve = np.asarray(
        [float(summary["frontier_risk_reserve"]) for summary in non_stage1],
        dtype=np.float64,
    )
    escape_support = np.asarray([float(summary["escape_support"]) for summary in non_stage1], dtype=np.float64)
    route_value = np.asarray([float(summary["route_value_target"]) for summary in non_stage1], dtype=np.float64)
    regret = np.asarray([float(summary["fallback_regret_calibrated"]) for summary in non_stage1], dtype=np.float64)
    failure_risk = np.asarray([float(summary["failure_risk"]) for summary in non_stage1], dtype=np.float64)
    route_consistency = np.asarray([float(summary["route_consistency"]) for summary in non_stage1], dtype=np.float64)

    teacher_priority = (
        0.52 * teacher_oracle
        + 0.18 * np.clip(route_value, 0.0, 1.0)
        + 0.14 * np.clip(regret, 0.0, 1.0)
        + 0.10 * frontier_risk_reserve
        + 0.06 * escape_support
    )
    order = np.argsort(teacher_priority)[::-1]
    topm = max(1, min(int(config.frontier_teacher_topm), len(order)))
    top_idx = order[:topm]
    scaled_teacher = teacher_priority[top_idx] / max(config.frontier_teacher_temperature, 1e-6)
    top_weights = _softmax_np(scaled_teacher)

    teacher_set_mass = float(np.clip(np.mean(teacher_oracle), 0.0, 1.0))
    teacher_topm_mass = float(np.sum(top_weights * teacher_oracle[top_idx]))
    frontier_target_mass = float(
        np.clip(
            0.58 * teacher_topm_mass + 0.28 * np.sum(top_weights * frontier_risk_reserve[top_idx]) + 0.14 * teacher_set_mass,
            0.0,
            1.0,
        )
    )
    soft_oracle_gain = float(np.sum(top_weights * np.clip(route_value[top_idx], 0.0, 1.0)))
    counterfactual_support = float(np.sum(top_weights * escape_support[top_idx]))
    counterfactual_margin = float(
        np.sum(top_weights * route_value[top_idx])
        + 0.28 * np.sum(top_weights * np.clip(regret[top_idx], 0.0, 1.0))
        - 0.18 * np.sum(top_weights * failure_risk[top_idx])
    )
    mean_consistency = float(np.sum(top_weights * route_consistency[top_idx]))
    safe_candidate_count = float(np.sum((teacher_oracle >= 0.52) & (frontier_risk_reserve >= 0.38)))
    frontier_coverage_target = float(
        np.clip(
            0.52 * frontier_target_mass + 0.28 * mean_consistency + 0.20 * counterfactual_support,
            0.0,
            1.0,
        )
    )
    frontier_consistency_target = float(
        np.clip(
            0.48 * teacher_topm_mass + 0.32 * frontier_target_mass + 0.20 * mean_consistency,
            0.0,
            1.0,
        )
    )
    anchor_escape_target = float(
        np.clip(
            0.24 * teacher_topm_mass
            + 0.20 * _sigmoid_np(counterfactual_margin / 0.10)
            + 0.16 * teacher_set_mass
            + 0.14 * mean_consistency
            + 0.12 * np.sum(top_weights * frontier_risk_reserve[top_idx])
            + 0.08 * counterfactual_support
            + 0.06 * (1.0 - anchor_strength),
            0.0,
            1.0,
        )
    )
    uncertainty_target = float(
        np.clip(
            (1.0 - np.clip(abs(counterfactual_margin) / 0.45, 0.0, 1.0))
            * (0.55 + 0.45 * (1.0 - frontier_target_mass)),
            0.0,
            1.0,
        )
    )
    top_candidate_names = tuple(str(non_stage1[int(idx)]["candidate_name"]) for idx in top_idx)
    return {
        "anchor_strength": anchor_strength,
        "anchor_escape_target": anchor_escape_target,
        "anchor_escape_soft_oracle_gain": soft_oracle_gain,
        "anchor_escape_safe_candidate_count": safe_candidate_count,
        "anchor_escape_topm_mass": teacher_topm_mass,
        "anchor_escape_counterfactual_margin": counterfactual_margin,
        "anchor_escape_uncertainty_target": uncertainty_target,
        "teacher_set_mass": teacher_set_mass,
        "teacher_topm_mass": teacher_topm_mass,
        "frontier_target_mass": frontier_target_mass,
        "frontier_coverage_target": frontier_coverage_target,
        "frontier_consistency_target": frontier_consistency_target,
        "counterfactual_escape_support": counterfactual_support,
        "top_candidate_names": top_candidate_names,
    }


def _frontier_block_reason(
    *,
    candidate_name: str,
    anchor_escape_calibrated: float,
    teacher_set_prob: float,
    frontier_accept_prob: float,
    frontier_risk_reserve: float,
    anchor_threshold: float,
    frontier_threshold: float,
) -> str:
    if candidate_name == "stage1_blended":
        return "anchor_stage1"
    if anchor_escape_calibrated < anchor_threshold:
        return "anchor_escape_low"
    if teacher_set_prob < 0.40 and frontier_accept_prob < frontier_threshold:
        return "outside_teacher_set"
    if frontier_risk_reserve < 0.25 and frontier_accept_prob < frontier_threshold:
        return "frontier_risk_reject"
    if frontier_accept_prob < frontier_threshold:
        return "frontier_boundary_reject"
    return "allowed"


def _prepare_bank_proxy_context(
    *,
    x: np.ndarray,
    batches: np.ndarray | None,
    random_state: int,
) -> dict[str, object]:
    full_embedding = _pca_embedding(x, random_state=random_state)
    n_clusters = min(6, max(3, int(np.sqrt(x.shape[0] / 40.0))))
    pseudo_labels = KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state).fit_predict(full_embedding)
    return {
        "full_embedding": full_embedding,
        "pseudo_labels": pseudo_labels.astype(np.int64),
        "n_neighbors": min(15, max(5, x.shape[0] // 30)),
        "batches": None if batches is None else np.asarray(batches),
    }


def _gate_bank_candidate_features(
    *,
    candidate_name: str,
    gate: np.ndarray,
    score_vector: np.ndarray,
    selected_idx: np.ndarray,
    boundary_idx: np.ndarray,
    dataset_stats: dict[str, float],
    base_features: dict[str, np.ndarray],
    x: np.ndarray,
    proxy_context: dict[str, object],
    heuristic_gate: np.ndarray,
    stage1_gate: np.ndarray,
    prototype_distance: float,
    bank_names: tuple[str, ...],
    random_state: int,
) -> dict[str, float]:
    x_sel = x[:, selected_idx]
    selected_embedding = _pca_embedding(x_sel, random_state=random_state)
    n_clusters = len(np.unique(np.asarray(proxy_context["pseudo_labels"])))
    pred = KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state).fit_predict(selected_embedding)

    batches = proxy_context["batches"]
    batch_mixing = 0.5
    if batches is not None:
        batch_sil = _silhouette_safe_local(selected_embedding, np.asarray(batches))
        batch_mixing = 1.0 - max(batch_sil, 0.0)

    top_scores = np.asarray(score_vector[selected_idx], dtype=np.float64)
    next_pool = np.setdiff1d(boundary_idx, selected_idx, assume_unique=False)
    if next_pool.size == 0:
        next_scores = np.asarray([np.median(score_vector)], dtype=np.float64)
    else:
        next_scores = np.asarray(score_vector[next_pool], dtype=np.float64)

    features: dict[str, float] = {
        "proxy_pseudo_silhouette": _silhouette_safe_local(
            selected_embedding,
            np.asarray(proxy_context["pseudo_labels"]),
        ),
        "proxy_cluster_silhouette": _silhouette_safe_local(selected_embedding, pred),
        "proxy_batch_mixing": float(batch_mixing),
        "proxy_neighbor_preservation": neighbor_preservation(
            reference_embedding=np.asarray(proxy_context["full_embedding"]),
            test_embedding=selected_embedding,
            n_neighbors=int(proxy_context["n_neighbors"]),
        ),
        "selected_batch_mi_mean": float(np.mean(base_features["batch_mi"][selected_idx])),
        "selected_cluster_mi_mean": float(np.mean(base_features["cluster_mi"][selected_idx])),
        "selected_local_consistency_mean": float(np.mean(base_features["local_consistency"][selected_idx])),
        "selected_residual_mean": float(np.mean(base_features["residual"][selected_idx])),
        "selected_dispersion_mean": float(np.mean(base_features["dispersion"][selected_idx])),
        "boundary_batch_mi_mean": float(np.mean(base_features["batch_mi"][boundary_idx])),
        "boundary_local_consistency_mean": float(np.mean(base_features["local_consistency"][boundary_idx])),
        "boundary_residual_mean": float(np.mean(base_features["residual"][boundary_idx])),
        "score_margin": float(np.mean(top_scores) - np.mean(next_scores)),
        "gate_entropy": _gate_entropy_np(gate),
        "dist_to_heuristic": float(np.linalg.norm(_normalize_gate_np(gate) - _normalize_gate_np(heuristic_gate))),
        "dist_to_stage1": float(np.linalg.norm(_normalize_gate_np(gate) - _normalize_gate_np(stage1_gate))),
        "prototype_distance": float(prototype_distance),
    }
    features["proxy_structure_support"] = float(
        0.42 * features["proxy_neighbor_preservation"]
        + 0.18 * features["proxy_pseudo_silhouette"]
        + 0.15 * features["proxy_cluster_silhouette"]
        + 0.22 * features["selected_local_consistency_mean"]
        - 0.18 * features["selected_batch_mi_mean"]
        - 0.10 * features["boundary_batch_mi_mean"]
    )
    features["proxy_transfer_risk"] = float(
        0.42 * features["selected_batch_mi_mean"]
        + 0.25 * features["boundary_batch_mi_mean"]
        + 0.18 * features["dist_to_stage1"]
        + 0.14 * features["prototype_distance"]
        - 0.30 * features["proxy_neighbor_preservation"]
        - 0.24 * features["selected_local_consistency_mean"]
        - 0.18 * features["proxy_pseudo_silhouette"]
    )
    features["proxy_safe_refine_support"] = float(
        0.34 * features["selected_local_consistency_mean"]
        + 0.24 * features["boundary_local_consistency_mean"]
        + 0.18 * features["selected_residual_mean"]
        + 0.12 * features["score_margin"]
        + 0.10 * features["proxy_neighbor_preservation"]
        - 0.28 * features["boundary_batch_mi_mean"]
        - 0.16 * features["selected_batch_mi_mean"]
    )

    features.update({f"stat_{key}": float(dataset_stats[key]) for key in DEFAULT_GATE_FEATURE_KEYS})
    features.update({f"gate_{expert}": float(weight) for expert, weight in zip(EXPERT_NAMES, gate, strict=False)})
    for bank_name in bank_names:
        features[f"bank_id_{bank_name}"] = 1.0 if candidate_name == bank_name else 0.0
    return features


def _pairwise_feature_columns(*, prototype_candidates: int) -> list[str]:
    bank_names = gate_bank_candidate_names(prototype_candidates=prototype_candidates)
    columns = [f"stat_{key}" for key in DEFAULT_GATE_FEATURE_KEYS]
    for expert in EXPERT_NAMES:
        columns.extend(
            [
                f"candidate_gate_{expert}",
                f"anchor_gate_{expert}",
                f"delta_gate_{expert}",
                f"abs_delta_gate_{expert}",
            ]
        )
    for key in BANK_PROXY_FEATURE_KEYS:
        columns.extend(
            [
                f"candidate_{key}",
                f"anchor_{key}",
                f"delta_{key}",
                f"abs_delta_{key}",
            ]
        )
    columns.extend(
        [
            "candidate_is_stage1",
            "anchor_is_stage1",
            "candidate_anchor_gate_distance",
            "candidate_anchor_proxy_l1",
            "candidate_anchor_score_margin_delta",
            "candidate_anchor_transfer_delta",
            "candidate_anchor_structure_delta",
            "candidate_anchor_refine_delta",
        ]
    )
    columns.extend(f"candidate_bank_id_{name}" for name in bank_names)
    return columns


def _build_stage1_pairwise_feature_dict(
    *,
    candidate_features: dict[str, float],
    anchor_features: dict[str, float],
    candidate_name: str,
    bank_names: tuple[str, ...],
) -> dict[str, float]:
    features: dict[str, float] = {}
    for key in DEFAULT_GATE_FEATURE_KEYS:
        features[f"stat_{key}"] = float(candidate_features[f"stat_{key}"])

    gate_delta_sq = 0.0
    proxy_delta_l1 = 0.0
    for expert in EXPERT_NAMES:
        candidate_value = float(candidate_features[f"gate_{expert}"])
        anchor_value = float(anchor_features[f"gate_{expert}"])
        delta = candidate_value - anchor_value
        features[f"candidate_gate_{expert}"] = candidate_value
        features[f"anchor_gate_{expert}"] = anchor_value
        features[f"delta_gate_{expert}"] = delta
        features[f"abs_delta_gate_{expert}"] = abs(delta)
        gate_delta_sq += delta * delta

    for key in BANK_PROXY_FEATURE_KEYS:
        candidate_value = float(candidate_features[key])
        anchor_value = float(anchor_features[key])
        delta = candidate_value - anchor_value
        features[f"candidate_{key}"] = candidate_value
        features[f"anchor_{key}"] = anchor_value
        features[f"delta_{key}"] = delta
        features[f"abs_delta_{key}"] = abs(delta)
        proxy_delta_l1 += abs(delta)

    features["candidate_is_stage1"] = 1.0 if candidate_name == "stage1_blended" else 0.0
    features["anchor_is_stage1"] = 1.0
    features["candidate_anchor_gate_distance"] = float(np.sqrt(gate_delta_sq))
    features["candidate_anchor_proxy_l1"] = float(proxy_delta_l1 / max(len(BANK_PROXY_FEATURE_KEYS), 1))
    features["candidate_anchor_score_margin_delta"] = float(
        candidate_features["score_margin"] - anchor_features["score_margin"]
    )
    features["candidate_anchor_transfer_delta"] = float(
        candidate_features["proxy_transfer_risk"] - anchor_features["proxy_transfer_risk"]
    )
    features["candidate_anchor_structure_delta"] = float(
        candidate_features["proxy_structure_support"] - anchor_features["proxy_structure_support"]
    )
    features["candidate_anchor_refine_delta"] = float(
        candidate_features["proxy_safe_refine_support"] - anchor_features["proxy_safe_refine_support"]
    )
    for bank_name in bank_names:
        features[f"candidate_bank_id_{bank_name}"] = 1.0 if candidate_name == bank_name else 0.0
    return features


def _stage1_relative_failure_risk(
    *,
    candidate_row: dict[str, float | int | str],
    anchor_row: dict[str, float | int | str],
) -> float:
    ari_drop = max(0.0, float(anchor_row["route_ari"]) - float(candidate_row["route_ari"]))
    nmi_drop = max(0.0, float(anchor_row["route_nmi"]) - float(candidate_row["route_nmi"]))
    structure_drop = max(
        0.0,
        float(anchor_row["route_structure_score"]) - float(candidate_row["route_structure_score"]),
    )
    neighbor_drop = max(
        0.0,
        float(anchor_row["route_neighbor_preservation"]) - float(candidate_row["route_neighbor_preservation"]),
    )
    batch_drop = max(
        0.0,
        float(anchor_row["route_batch_mixing"]) - float(candidate_row["route_batch_mixing"]),
    )
    transfer_proxy = _sigmoid_np(float(candidate_row["proxy_transfer_risk"]))
    boundary_proxy = _sigmoid_np(float(candidate_row["boundary_batch_mi_mean"]))
    adjusted_drop = max(
        0.0,
        float(anchor_row["route_adjusted_reward"]) - float(candidate_row["route_adjusted_reward"]),
    )
    risk = (
        1.35 * ari_drop
        + 1.15 * nmi_drop
        + 0.95 * structure_drop
        + 0.35 * neighbor_drop
        + 0.20 * batch_drop
        + 0.18 * transfer_proxy
        + 0.10 * boundary_proxy
        + 0.35 * adjusted_drop
    )
    return float(np.clip(risk, 0.0, 1.0))


def _build_stage1_pairwise_training_frame(
    *,
    bank_df: pd.DataFrame,
    config: GateLearningConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    bank_names = gate_bank_candidate_names(prototype_candidates=config.bank_prototype_candidates)
    for dataset_id, group in bank_df.groupby("dataset_id", sort=True):
        records = group.to_dict(orient="records")
        anchor_row = next(record for record in records if str(record["candidate_name"]) == "stage1_blended")
        anchor_features = {
            key: float(anchor_row[key])
            for key in _bank_feature_columns(prototype_candidates=config.bank_prototype_candidates)
        }
        for record in records:
            candidate_name = str(record["candidate_name"])
            candidate_features = {
                key: float(record[key])
                for key in _bank_feature_columns(prototype_candidates=config.bank_prototype_candidates)
            }
            utility_gain = float(
                np.clip(
                    max(0.0, float(record["route_reward"]) - float(anchor_row["route_reward"])),
                    0.0,
                    1.0,
                )
            )
            fallback_regret = float(
                np.clip(
                    max(0.0, float(record["route_adjusted_reward"]) - float(anchor_row["route_adjusted_reward"])),
                    0.0,
                    1.0,
                )
            )
            failure_risk = 0.0 if candidate_name == "stage1_blended" else _stage1_relative_failure_risk(
                candidate_row=record,
                anchor_row=anchor_row,
            )
            route_margin_raw = float(record["route_adjusted_reward"]) - float(anchor_row["route_adjusted_reward"])
            route_margin_target = float(
                np.tanh(route_margin_raw / max(config.pairwise_margin_scale, 1e-6))
            )
            route_win_target = float(route_margin_raw > 0.0)
            sample_weight = float(
                np.clip(
                    0.15
                    + abs(route_margin_raw)
                    * (1.0 + float(record["stat_batch_strength"]) + 0.25 * float(record["stat_trajectory_strength"]))
                    + 0.50 * utility_gain
                    + 0.35 * fallback_regret,
                    0.10,
                    3.0,
                )
            )
            row: dict[str, float | int | str] = {
                "dataset_id": int(dataset_id),
                "scenario": str(record["scenario"]),
                "seed": int(record["seed"]),
                "candidate_name": candidate_name,
                "anchor_name": "stage1_blended",
                "pairwise_utility_gain_target": utility_gain,
                "pairwise_failure_risk_target": failure_risk,
                "pairwise_fallback_regret_target": fallback_regret,
                "pairwise_route_margin_raw_target": route_margin_raw,
                "pairwise_route_margin_target": route_margin_target,
                "pairwise_route_win_target": route_win_target,
                "pairwise_sample_weight": sample_weight,
                "pairwise_stage1_route_adjusted": float(anchor_row["route_adjusted_reward"]),
                "pairwise_candidate_route_adjusted": float(record["route_adjusted_reward"]),
            }
            row.update(
                _build_stage1_pairwise_feature_dict(
                    candidate_features=candidate_features,
                    anchor_features=anchor_features,
                    candidate_name=candidate_name,
                    bank_names=bank_names,
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _build_stage1_pairwise_calibrated_training_frame(
    *,
    bank_df: pd.DataFrame,
    config: GateLearningConfig,
    variant: str = "full",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    bank_names = gate_bank_candidate_names(prototype_candidates=config.bank_prototype_candidates)
    for dataset_id, group in bank_df.groupby("dataset_id", sort=True):
        records = group.to_dict(orient="records")
        anchor_row = next(record for record in records if str(record["candidate_name"]) == "stage1_blended")
        anchor_features = {
            key: float(anchor_row[key])
            for key in _bank_feature_columns(prototype_candidates=config.bank_prototype_candidates)
        }
        for record in records:
            candidate_name = str(record["candidate_name"])
            candidate_features = {
                key: float(record[key])
                for key in _bank_feature_columns(prototype_candidates=config.bank_prototype_candidates)
            }
            raw_utility_gain = float(
                np.clip(
                    max(0.0, float(record["route_reward"]) - float(anchor_row["route_reward"])),
                    0.0,
                    1.0,
                )
            )
            fallback_regret_raw = float(
                np.clip(
                    max(0.0, float(record["route_adjusted_reward"]) - float(anchor_row["route_adjusted_reward"])),
                    0.0,
                    1.0,
                )
            )
            failure_risk = 0.0 if candidate_name == "stage1_blended" else _stage1_relative_failure_risk(
                candidate_row=record,
                anchor_row=anchor_row,
            )
            route_margin_raw = float(record["route_adjusted_reward"]) - float(anchor_row["route_adjusted_reward"])
            route_consistency = float(np.clip(record.get("route_view_consistency", 1.0), 0.0, 1.0))
            if variant == "no_consistency_calibration":
                route_consistency = 1.0
            calibrated = _build_pairwise_calibration_targets(
                raw_utility_gain=raw_utility_gain,
                failure_risk=failure_risk,
                fallback_regret_raw=fallback_regret_raw,
                route_margin_raw=route_margin_raw,
                route_consistency=route_consistency,
                config=config,
            )
            calibrated = _apply_pairwise_calibration_variant(
                calibrated=calibrated,
                raw_utility_gain=raw_utility_gain,
                failure_risk=failure_risk,
                fallback_regret_raw=fallback_regret_raw,
                route_margin_raw=route_margin_raw,
                variant=variant,
                config=config,
            )
            utility_win = float(raw_utility_gain > config.pairwise_safe_utility_floor)
            safety_pass = float(failure_risk <= config.pairwise_safe_risk_budget)
            consistency_pass = float(route_consistency >= config.pairwise_consistency_threshold)
            sample_weight = float(
                np.clip(
                    0.10
                    + abs(calibrated["pairwise_final_route_score"])
                    + 0.65 * calibrated["pairwise_safe_utility_gain"]
                    + 0.45 * calibrated["pairwise_route_permission_target"]
                    + 0.35 * calibrated["pairwise_route_constraint_score"]
                    + 0.25 * calibrated["pairwise_regret_supervision_weight"],
                    0.10,
                    3.0,
                )
            )
            row: dict[str, float | int | str] = {
                "dataset_id": int(dataset_id),
                "scenario": str(record["scenario"]),
                "seed": int(record["seed"]),
                "candidate_name": candidate_name,
                "anchor_name": "stage1_blended",
                "pairwise_utility_gain_target": calibrated["pairwise_safe_utility_gain"],
                "pairwise_failure_risk_target": failure_risk,
                "pairwise_fallback_regret_target": calibrated["pairwise_fallback_regret_calibrated"],
                "pairwise_route_margin_raw_target": route_margin_raw,
                "pairwise_route_margin_target": calibrated["pairwise_route_margin_calibrated"],
                "pairwise_route_win_target": calibrated["pairwise_route_permission_target"],
                "pairwise_sample_weight": sample_weight,
                "pairwise_stage1_route_adjusted": float(anchor_row["route_adjusted_reward"]),
                "pairwise_candidate_route_adjusted": float(record["route_adjusted_reward"]),
                "pairwise_utility_win": utility_win,
                "pairwise_safety_pass": safety_pass,
                "pairwise_consistency_pass": consistency_pass,
                "pairwise_route_permission_target": calibrated["pairwise_route_permission_target"],
                "pairwise_routed_away_from_stage1": calibrated["pairwise_route_win_safe"],
                "pairwise_anchor_selection_outcome": 1.0 - calibrated["pairwise_route_win_safe"],
            }
            row.update(calibrated)
            row.update(
                _build_stage1_pairwise_feature_dict(
                    candidate_features=candidate_features,
                    anchor_features=anchor_features,
                    candidate_name=candidate_name,
                    bank_names=bank_names,
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _build_stage1_pairwise_permissioned_training_frame(
    *,
    bank_df: pd.DataFrame,
    config: GateLearningConfig,
    variant: str = "full",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    bank_names = gate_bank_candidate_names(prototype_candidates=config.bank_prototype_candidates)
    budget_variant = variant if variant in {"fixed_budget", "no_anchor_adaptation"} else "full"
    use_regret_aux = variant != "no_regret_aux"

    for dataset_id, group in bank_df.groupby("dataset_id", sort=True):
        records = group.to_dict(orient="records")
        anchor_row = next(record for record in records if str(record["candidate_name"]) == "stage1_blended")
        anchor_features = {
            key: float(anchor_row[key])
            for key in _bank_feature_columns(prototype_candidates=config.bank_prototype_candidates)
        }
        dataset_stats = {key: float(anchor_row[f"stat_{key}"]) for key in DEFAULT_GATE_FEATURE_KEYS}

        row_buffer: list[dict[str, float | int | str]] = []
        candidate_summaries: list[dict[str, float]] = []

        for record in records:
            candidate_name = str(record["candidate_name"])
            candidate_features = {
                key: float(record[key])
                for key in _bank_feature_columns(prototype_candidates=config.bank_prototype_candidates)
            }
            raw_utility_gain = float(
                np.clip(
                    max(0.0, float(record["route_reward"]) - float(anchor_row["route_reward"])),
                    0.0,
                    1.0,
                )
            )
            fallback_regret_raw = float(
                np.clip(
                    max(0.0, float(record["route_adjusted_reward"]) - float(anchor_row["route_adjusted_reward"])),
                    0.0,
                    1.0,
                )
            )
            failure_risk = 0.0 if candidate_name == "stage1_blended" else _stage1_relative_failure_risk(
                candidate_row=record,
                anchor_row=anchor_row,
            )
            route_margin_raw = float(record["route_adjusted_reward"]) - float(anchor_row["route_adjusted_reward"])
            route_consistency = float(np.clip(record.get("route_view_consistency", 1.0), 0.0, 1.0))
            calibrated = _build_pairwise_calibration_targets(
                raw_utility_gain=raw_utility_gain,
                failure_risk=failure_risk,
                fallback_regret_raw=fallback_regret_raw,
                route_margin_raw=route_margin_raw,
                route_consistency=route_consistency,
                config=config,
            )
            if not use_regret_aux:
                calibrated["pairwise_fallback_regret_calibrated"] = 0.0
                calibrated["pairwise_regret_supervision_weight"] = config.pairwise_min_regret_weight
                calibrated["pairwise_final_route_score"] = float(
                    calibrated["pairwise_safe_utility_gain"]
                    - config.pairwise_risk_weight * failure_risk
                    + config.pairwise_margin_weight * calibrated["pairwise_route_margin_calibrated"]
                )

            route_value_raw, route_value_target = _pairperm_route_value_target(
                safe_utility_gain=calibrated["pairwise_safe_utility_gain"],
                fallback_regret_calibrated=calibrated["pairwise_fallback_regret_calibrated"],
                failure_risk=failure_risk,
                route_margin_calibrated=calibrated["pairwise_route_margin_calibrated"],
                config=config,
                use_regret_aux=use_regret_aux,
            )
            escape_support = _pairperm_candidate_escape_support(
                safe_utility_gain=calibrated["pairwise_safe_utility_gain"],
                fallback_regret_calibrated=calibrated["pairwise_fallback_regret_calibrated"],
                failure_risk=failure_risk,
                route_consistency=route_consistency,
                route_margin_raw=route_margin_raw,
                safe_positive_mask=calibrated["pairwise_safe_positive_mask"],
                config=config,
            )

            sample_weight = float(
                np.clip(
                    0.12
                    + abs(route_value_raw)
                    + 0.45 * escape_support
                    + 0.35 * calibrated["pairwise_route_constraint_score"]
                    + 0.20 * calibrated["pairwise_regret_supervision_weight"],
                    0.10,
                    3.0,
                )
            )
            row: dict[str, float | int | str] = {
                "dataset_id": int(dataset_id),
                "scenario": str(record["scenario"]),
                "seed": int(record["seed"]),
                "candidate_name": candidate_name,
                "anchor_name": "stage1_blended",
                "pairwise_utility_gain_target": calibrated["pairwise_safe_utility_gain"],
                "pairwise_failure_risk_target": failure_risk,
                "pairwise_fallback_regret_target": calibrated["pairwise_fallback_regret_calibrated"],
                "pairwise_route_margin_raw_target": route_margin_raw,
                "pairwise_route_margin_target": calibrated["pairwise_route_margin_calibrated"],
                "pairwise_route_value_raw": route_value_raw,
                "pairwise_route_value_target": route_value_target,
                "pairwise_sample_weight": sample_weight,
                "pairwise_stage1_route_adjusted": float(anchor_row["route_adjusted_reward"]),
                "pairwise_candidate_route_adjusted": float(record["route_adjusted_reward"]),
                "pairwise_route_consistency": route_consistency,
                "pairwise_escape_support": escape_support,
            }
            row.update(calibrated)
            row.update(
                _build_stage1_pairwise_feature_dict(
                    candidate_features=candidate_features,
                    anchor_features=anchor_features,
                    candidate_name=candidate_name,
                    bank_names=bank_names,
                )
            )
            row_buffer.append(row)
            candidate_summaries.append(
                {
                    "candidate_is_stage1": 1.0 if candidate_name == "stage1_blended" else 0.0,
                    "safe_utility_gain": calibrated["pairwise_safe_utility_gain"],
                    "fallback_regret_calibrated": calibrated["pairwise_fallback_regret_calibrated"],
                    "failure_risk": failure_risk,
                    "route_consistency": route_consistency,
                    "route_margin_raw": route_margin_raw,
                    "safe_positive_mask": calibrated["pairwise_safe_positive_mask"],
                    "escape_support": escape_support,
                    "route_value_target": route_value_target,
                }
            )

        anchor_context = _pairperm_anchor_context(
            candidate_summaries=candidate_summaries,
            anchor_features=anchor_features,
        )
        adaptive_risk_budget, adaptive_consistency_budget = _pairperm_budget_targets(
            anchor_context=anchor_context,
            dataset_stats=dataset_stats,
            config=config,
            variant=budget_variant,
        )

        for row in row_buffer:
            candidate_name = str(row["candidate_name"])
            budget_stats = _pairperm_budget_score(
                failure_risk=float(row["pairwise_failure_risk_target"]),
                route_consistency=float(row["pairwise_route_consistency"]),
                adaptive_risk_budget=adaptive_risk_budget,
                adaptive_consistency_budget=adaptive_consistency_budget,
                config=config,
            )
            route_permission_target = float(
                np.clip(
                    0.40 * float(row["pairwise_escape_support"])
                    + 0.25 * anchor_context["anchor_escape_pressure"]
                    + 0.15 * anchor_context["anchor_routability"]
                    + 0.10 * float(row["pairwise_safe_positive_mask"])
                    + 0.10 * _sigmoid_np(3.5 * float(row["pairwise_route_value_target"])),
                    0.0,
                    1.0,
                )
            )
            route_permission_calibrated = float(
                np.clip(
                    route_permission_target * (0.35 + 0.65 * budget_stats["permission_budget_score"]),
                    0.0,
                    1.0,
                )
            )
            route_permission_prob = route_permission_target
            route_permission_logit = float(np.log(np.clip(route_permission_prob, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - route_permission_prob, 1e-6, 1.0)))
            final_permission_pass = float(
                candidate_name != "stage1_blended"
                and route_permission_calibrated >= config.pairperm_permission_threshold
            )
            if route_permission_prob <= 1e-6:
                route_permission_logit = -12.0
            elif route_permission_prob >= 1.0 - 1e-6:
                route_permission_logit = 12.0
            row["pairwise_route_permission_target"] = route_permission_target
            row["pairwise_route_permission_logit"] = route_permission_logit
            row["pairwise_route_permission_prob"] = route_permission_prob
            row["pairwise_route_permission_calibrated"] = route_permission_calibrated
            row["pairwise_route_permission_margin"] = budget_stats["permission_margin"]
            row["pairwise_route_permission_confidence"] = budget_stats["permission_confidence"]
            row["pairwise_anchor_escape_pressure"] = anchor_context["anchor_escape_pressure"]
            row["pairwise_anchor_routability"] = anchor_context["anchor_routability"]
            row["pairwise_anchor_fragility"] = anchor_context["anchor_fragility"]
            row["pairwise_adaptive_risk_budget"] = adaptive_risk_budget
            row["pairwise_adaptive_consistency_budget"] = adaptive_consistency_budget
            row["pairwise_permission_budget_score"] = budget_stats["permission_budget_score"]
            row["pairwise_permission_block_reason"] = _pairperm_block_reason(
                candidate_name=candidate_name,
                safe_positive_mask=float(row["pairwise_safe_positive_mask"]),
                failure_risk=float(row["pairwise_failure_risk_target"]),
                route_consistency=float(row["pairwise_route_consistency"]),
                adaptive_risk_budget=adaptive_risk_budget,
                adaptive_consistency_budget=adaptive_consistency_budget,
                anchor_escape_pressure=anchor_context["anchor_escape_pressure"],
                permission_calibrated=route_permission_calibrated,
                config=config,
            )
            row["pairwise_final_route_permission_pass"] = final_permission_pass
            row["pairwise_route_win_target"] = final_permission_pass
            row["pairwise_routed_away_from_stage1"] = final_permission_pass
            row["pairwise_anchor_selection_outcome"] = 1.0 - final_permission_pass
            row["pairwise_final_route_score"] = float(row["pairwise_route_value_raw"])
            row["pairwise_value_among_allowed_candidates"] = (
                float(row["pairwise_route_value_target"]) if final_permission_pass > 0.5 else 0.0
            )
            row["pairwise_route_selected_under_permission"] = 0.0

        allowed_rows = [
            row
            for row in row_buffer
            if str(row["candidate_name"]) != "stage1_blended" and float(row["pairwise_final_route_permission_pass"]) > 0.5
        ]
        if allowed_rows:
            selected_row = max(allowed_rows, key=lambda current: float(current["pairwise_route_value_target"]))
            selected_row["pairwise_route_selected_under_permission"] = 1.0

        rows.extend(row_buffer)

    return pd.DataFrame(rows)


def _build_stage1_pairwise_escapecert_training_frame(
    *,
    bank_df: pd.DataFrame,
    config: GateLearningConfig,
    variant: str = "full",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    bank_names = gate_bank_candidate_names(prototype_candidates=config.bank_prototype_candidates)
    flags = _escapecert_variant_flags(variant)

    for dataset_id, group in bank_df.groupby("dataset_id", sort=True):
        records = group.to_dict(orient="records")
        anchor_row = next(record for record in records if str(record["candidate_name"]) == "stage1_blended")
        anchor_features = {
            key: float(anchor_row[key])
            for key in _bank_feature_columns(prototype_candidates=config.bank_prototype_candidates)
        }
        dataset_stats = {key: float(anchor_row[f"stat_{key}"]) for key in DEFAULT_GATE_FEATURE_KEYS}

        row_buffer: list[dict[str, float | int | str]] = []
        candidate_summaries: list[dict[str, float | str]] = []

        for record in records:
            candidate_name = str(record["candidate_name"])
            candidate_features = {
                key: float(record[key])
                for key in _bank_feature_columns(prototype_candidates=config.bank_prototype_candidates)
            }
            raw_utility_gain = float(
                np.clip(
                    max(0.0, float(record["route_reward"]) - float(anchor_row["route_reward"])),
                    0.0,
                    1.0,
                )
            )
            fallback_regret_raw = float(
                np.clip(
                    max(0.0, float(record["route_adjusted_reward"]) - float(anchor_row["route_adjusted_reward"])),
                    0.0,
                    1.0,
                )
            )
            failure_risk = 0.0 if candidate_name == "stage1_blended" else _stage1_relative_failure_risk(
                candidate_row=record,
                anchor_row=anchor_row,
            )
            route_margin_raw = float(record["route_adjusted_reward"]) - float(anchor_row["route_adjusted_reward"])
            route_consistency = float(np.clip(record.get("route_view_consistency", 1.0), 0.0, 1.0))
            calibrated = _build_pairwise_calibration_targets(
                raw_utility_gain=raw_utility_gain,
                failure_risk=failure_risk,
                fallback_regret_raw=fallback_regret_raw,
                route_margin_raw=route_margin_raw,
                route_consistency=route_consistency,
                config=config,
            )
            if not flags["use_regret_aux"]:
                calibrated["pairwise_fallback_regret_calibrated"] = 0.0
                calibrated["pairwise_regret_supervision_weight"] = config.pairwise_min_regret_weight
                calibrated["pairwise_final_route_score"] = float(
                    calibrated["pairwise_safe_utility_gain"]
                    - config.pairwise_risk_weight * failure_risk
                    + config.pairwise_margin_weight * calibrated["pairwise_route_margin_calibrated"]
                )

            route_value_raw, route_value_target = _pairperm_route_value_target(
                safe_utility_gain=calibrated["pairwise_safe_utility_gain"],
                fallback_regret_calibrated=calibrated["pairwise_fallback_regret_calibrated"],
                failure_risk=failure_risk,
                route_margin_calibrated=calibrated["pairwise_route_margin_calibrated"],
                config=config,
                use_regret_aux=flags["use_regret_aux"],
            )
            escape_support = _pairperm_candidate_escape_support(
                safe_utility_gain=calibrated["pairwise_safe_utility_gain"],
                fallback_regret_calibrated=calibrated["pairwise_fallback_regret_calibrated"],
                failure_risk=failure_risk,
                route_consistency=route_consistency,
                route_margin_raw=route_margin_raw,
                safe_positive_mask=calibrated["pairwise_safe_positive_mask"],
                config=config,
            )
            admissibility_oracle = 0.0 if candidate_name == "stage1_blended" else _escapecert_candidate_membership_oracle(
                safe_utility_gain=calibrated["pairwise_safe_utility_gain"],
                fallback_regret_calibrated=calibrated["pairwise_fallback_regret_calibrated"],
                failure_risk=failure_risk,
                route_consistency=route_consistency,
                route_value_target=route_value_target,
                route_constraint_score=calibrated["pairwise_route_constraint_score"],
                escape_support=escape_support,
                config=config,
                use_regret_aux=flags["use_regret_aux"],
            )
            sample_weight = float(
                np.clip(
                    0.18
                    + abs(route_value_raw)
                    + 0.42 * admissibility_oracle
                    + 0.28 * escape_support
                    + 0.22 * calibrated["pairwise_route_constraint_score"],
                    0.10,
                    3.0,
                )
            )
            row: dict[str, float | int | str] = {
                "dataset_id": int(dataset_id),
                "scenario": str(record["scenario"]),
                "seed": int(record["seed"]),
                "candidate_name": candidate_name,
                "anchor_name": "stage1_blended",
                "pairwise_utility_gain_target": calibrated["pairwise_safe_utility_gain"],
                "pairwise_failure_risk_target": failure_risk,
                "pairwise_fallback_regret_target": calibrated["pairwise_fallback_regret_calibrated"],
                "pairwise_route_margin_raw_target": route_margin_raw,
                "pairwise_route_margin_target": calibrated["pairwise_route_margin_calibrated"],
                "pairwise_route_value_raw": route_value_raw,
                "pairwise_route_value_target": route_value_target,
                "pairwise_sample_weight": sample_weight,
                "pairwise_stage1_route_adjusted": float(anchor_row["route_adjusted_reward"]),
                "pairwise_candidate_route_adjusted": float(record["route_adjusted_reward"]),
                "pairwise_route_consistency": route_consistency,
                "pairwise_escape_support": escape_support,
                "admissibility_oracle": admissibility_oracle,
            }
            row.update(calibrated)
            row.update(
                _build_stage1_pairwise_feature_dict(
                    candidate_features=candidate_features,
                    anchor_features=anchor_features,
                    candidate_name=candidate_name,
                    bank_names=bank_names,
                )
            )
            row_buffer.append(row)
            candidate_summaries.append(
                {
                    "candidate_name": candidate_name,
                    "candidate_is_stage1": 1.0 if candidate_name == "stage1_blended" else 0.0,
                    "safe_utility_gain": calibrated["pairwise_safe_utility_gain"],
                    "fallback_regret_calibrated": calibrated["pairwise_fallback_regret_calibrated"],
                    "failure_risk": failure_risk,
                    "route_consistency": route_consistency,
                    "route_margin_raw": route_margin_raw,
                    "escape_support": escape_support,
                    "route_value_target": route_value_target,
                    "admissibility_oracle": admissibility_oracle,
                    "legacy_permission_target": calibrated["pairwise_route_permission_target"],
                    "route_constraint_score": calibrated["pairwise_route_constraint_score"],
                }
            )

        anchor_targets = _escapecert_anchor_targets(
            candidate_summaries=candidate_summaries,
            anchor_features=anchor_features,
            config=config,
            use_set_supervision=flags["use_set_supervision"],
        )
        budget_anchor_context = {
            "anchor_strength": float(anchor_targets["anchor_strength"]),
            "anchor_escape_pressure": float(anchor_targets["anchor_escape_target"]),
            "anchor_fragility": float(anchor_targets["anchor_escape_uncertainty_target"]),
            "anchor_routability": float(anchor_targets["anchor_escape_topm_mass"]),
        }
        adaptive_risk_budget, adaptive_consistency_budget = _pairperm_budget_targets(
            anchor_context=budget_anchor_context,
            dataset_stats=dataset_stats,
            config=config,
            variant="fixed_budget" if flags["fixed_budget"] else "full",
        )

        allowed_threshold = max(config.escapecert_admissibility_threshold, 0.40)
        anchor_threshold = config.pairperm_permission_threshold
        top_candidate_names = set(anchor_targets["top_candidate_names"])  # type: ignore[arg-type]

        for row in row_buffer:
            candidate_name = str(row["candidate_name"])
            base_admissibility = float(row["admissibility_oracle"])
            top_candidate_bonus = 1.0 if candidate_name in top_candidate_names else 0.0
            legacy_permission = float(row["pairwise_route_permission_target"])
            candidate_admissibility_target = 0.0
            if candidate_name != "stage1_blended":
                if flags["use_set_supervision"]:
                    candidate_admissibility_target = float(
                        np.clip(
                            0.68 * base_admissibility
                            + 0.20 * float(anchor_targets["anchor_escape_target"])
                            + 0.12 * top_candidate_bonus,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    candidate_admissibility_target = float(
                        np.clip(
                            0.60 * legacy_permission + 0.40 * base_admissibility,
                            0.0,
                            1.0,
                        )
                    )

            budget_stats = _pairperm_budget_score(
                failure_risk=float(row["pairwise_failure_risk_target"]),
                route_consistency=float(row["pairwise_route_consistency"]),
                adaptive_risk_budget=adaptive_risk_budget,
                adaptive_consistency_budget=adaptive_consistency_budget,
                config=config,
            )
            candidate_allowed_target = float(
                np.clip(
                    candidate_admissibility_target
                    * (0.25 + 0.75 * float(anchor_targets["anchor_escape_target"])),
                    0.0,
                    1.0,
                )
            )
            route_permission_calibrated = float(
                np.clip(
                    candidate_allowed_target * (0.35 + 0.65 * budget_stats["permission_budget_score"]),
                    0.0,
                    1.0,
                )
            )
            final_permission_pass = float(
                candidate_name != "stage1_blended"
                and float(anchor_targets["anchor_escape_target"]) >= anchor_threshold
                and candidate_allowed_target >= allowed_threshold
            )
            if candidate_name == "stage1_blended":
                block_reason = "anchor_stage1"
            elif float(anchor_targets["anchor_escape_target"]) < anchor_threshold:
                block_reason = "anchor_escape_low"
            elif budget_stats["permission_budget_score"] < 0.25:
                block_reason = "escape_budget_low"
            elif candidate_allowed_target < allowed_threshold:
                block_reason = "not_in_admissible_set"
            else:
                block_reason = "allowed"

            route_permission_prob = candidate_allowed_target
            route_permission_logit = float(
                np.log(np.clip(route_permission_prob, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - route_permission_prob, 1e-6, 1.0))
            )
            if route_permission_prob <= 1e-6:
                route_permission_logit = -12.0
            elif route_permission_prob >= 1.0 - 1e-6:
                route_permission_logit = 12.0

            row["anchor_escape_target"] = float(anchor_targets["anchor_escape_target"])
            row["anchor_escape_soft_oracle_gain"] = float(anchor_targets["anchor_escape_soft_oracle_gain"])
            row["anchor_escape_safe_candidate_count"] = float(anchor_targets["anchor_escape_safe_candidate_count"])
            row["anchor_escape_topm_mass"] = float(anchor_targets["anchor_escape_topm_mass"])
            row["anchor_escape_counterfactual_margin"] = float(anchor_targets["anchor_escape_counterfactual_margin"])
            row["anchor_escape_uncertainty_target"] = float(anchor_targets["anchor_escape_uncertainty_target"])
            row["allowed_set_target_mass"] = float(anchor_targets["allowed_set_target_mass"])
            row["permission_set_consistency_target"] = float(anchor_targets["permission_set_consistency_target"])
            row["counterfactual_escape_support"] = float(anchor_targets["counterfactual_escape_support"])
            row["candidate_admissibility_target"] = candidate_admissibility_target
            row["candidate_allowed_target"] = candidate_allowed_target
            row["pairwise_route_permission_target"] = route_permission_prob
            row["pairwise_route_permission_logit"] = route_permission_logit
            row["pairwise_route_permission_prob"] = route_permission_prob
            row["pairwise_route_permission_calibrated"] = route_permission_calibrated
            row["pairwise_route_permission_margin"] = budget_stats["permission_margin"]
            row["pairwise_route_permission_confidence"] = budget_stats["permission_confidence"]
            row["pairwise_anchor_escape_pressure"] = float(anchor_targets["anchor_escape_target"])
            row["pairwise_anchor_routability"] = float(anchor_targets["anchor_escape_topm_mass"])
            row["pairwise_anchor_fragility"] = float(anchor_targets["anchor_escape_uncertainty_target"])
            row["pairwise_adaptive_risk_budget"] = adaptive_risk_budget
            row["pairwise_adaptive_consistency_budget"] = adaptive_consistency_budget
            row["pairwise_permission_budget_score"] = budget_stats["permission_budget_score"]
            row["pairwise_permission_block_reason"] = block_reason
            row["pairwise_final_route_permission_pass"] = final_permission_pass
            row["pairwise_route_win_target"] = final_permission_pass
            row["pairwise_routed_away_from_stage1"] = final_permission_pass
            row["pairwise_anchor_selection_outcome"] = 1.0 - final_permission_pass
            row["pairwise_final_route_score"] = float(row["pairwise_route_value_raw"])
            row["pairwise_value_among_allowed_candidates"] = (
                float(row["pairwise_route_value_target"]) if final_permission_pass > 0.5 else 0.0
            )
            row["pairwise_route_selected_under_permission"] = 0.0

        allowed_rows = [
            row
            for row in row_buffer
            if str(row["candidate_name"]) != "stage1_blended" and float(row["pairwise_final_route_permission_pass"]) > 0.5
        ]
        if allowed_rows:
            selected_row = max(allowed_rows, key=lambda current: float(current["pairwise_route_value_target"]))
            selected_row["pairwise_route_selected_under_permission"] = 1.0

        rows.extend(row_buffer)

    return pd.DataFrame(rows)


def _build_stage1_pairwise_frontier_training_frame(
    *,
    bank_df: pd.DataFrame,
    config: GateLearningConfig,
    variant: str = "full",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    bank_names = gate_bank_candidate_names(prototype_candidates=config.bank_prototype_candidates)
    flags = _frontier_variant_flags(variant)

    for dataset_id, group in bank_df.groupby("dataset_id", sort=True):
        records = group.to_dict(orient="records")
        anchor_row = next(record for record in records if str(record["candidate_name"]) == "stage1_blended")
        anchor_features = {
            key: float(anchor_row[key])
            for key in _bank_feature_columns(prototype_candidates=config.bank_prototype_candidates)
        }
        dataset_stats = {key: float(anchor_row[f"stat_{key}"]) for key in DEFAULT_GATE_FEATURE_KEYS}

        row_buffer: list[dict[str, float | int | str]] = []
        candidate_summaries: list[dict[str, float | str]] = []

        for record in records:
            candidate_name = str(record["candidate_name"])
            candidate_features = {
                key: float(record[key])
                for key in _bank_feature_columns(prototype_candidates=config.bank_prototype_candidates)
            }
            raw_utility_gain = float(
                np.clip(
                    max(0.0, float(record["route_reward"]) - float(anchor_row["route_reward"])),
                    0.0,
                    1.0,
                )
            )
            fallback_regret_raw = float(
                np.clip(
                    max(0.0, float(record["route_adjusted_reward"]) - float(anchor_row["route_adjusted_reward"])),
                    0.0,
                    1.0,
                )
            )
            failure_risk = 0.0 if candidate_name == "stage1_blended" else _stage1_relative_failure_risk(
                candidate_row=record,
                anchor_row=anchor_row,
            )
            route_margin_raw = float(record["route_adjusted_reward"]) - float(anchor_row["route_adjusted_reward"])
            route_consistency = float(np.clip(record.get("route_view_consistency", 1.0), 0.0, 1.0))
            calibrated = _build_pairwise_calibration_targets(
                raw_utility_gain=raw_utility_gain,
                failure_risk=failure_risk,
                fallback_regret_raw=fallback_regret_raw,
                route_margin_raw=route_margin_raw,
                route_consistency=route_consistency,
                config=config,
            )
            if not flags["use_regret_aux"]:
                calibrated["pairwise_fallback_regret_calibrated"] = 0.0
                calibrated["pairwise_regret_supervision_weight"] = config.pairwise_min_regret_weight
                calibrated["pairwise_final_route_score"] = float(
                    calibrated["pairwise_safe_utility_gain"]
                    - config.pairwise_risk_weight * failure_risk
                    + config.pairwise_margin_weight * calibrated["pairwise_route_margin_calibrated"]
                )

            route_value_raw, route_value_target = _pairperm_route_value_target(
                safe_utility_gain=calibrated["pairwise_safe_utility_gain"],
                fallback_regret_calibrated=calibrated["pairwise_fallback_regret_calibrated"],
                failure_risk=failure_risk,
                route_margin_calibrated=calibrated["pairwise_route_margin_calibrated"],
                config=config,
                use_regret_aux=flags["use_regret_aux"],
            )
            escape_support = _pairperm_candidate_escape_support(
                safe_utility_gain=calibrated["pairwise_safe_utility_gain"],
                fallback_regret_calibrated=calibrated["pairwise_fallback_regret_calibrated"],
                failure_risk=failure_risk,
                route_consistency=route_consistency,
                route_margin_raw=route_margin_raw,
                safe_positive_mask=calibrated["pairwise_safe_positive_mask"],
                config=config,
            )
            teacher_oracle = 0.0
            provisional_frontier_reserve = 0.0
            if candidate_name != "stage1_blended":
                teacher_oracle = _frontier_teacher_membership_oracle(
                    safe_utility_gain=calibrated["pairwise_safe_utility_gain"],
                    fallback_regret_calibrated=calibrated["pairwise_fallback_regret_calibrated"],
                    failure_risk=failure_risk,
                    route_consistency=route_consistency,
                    route_value_target=route_value_target,
                    route_constraint_score=calibrated["pairwise_route_constraint_score"],
                    escape_support=escape_support,
                    config=config,
                    use_regret_aux=flags["use_regret_aux"],
                )
                base_risk_reserve = _sigmoid_np(
                    (config.pairwise_safe_risk_budget - failure_risk) / max(config.pairperm_risk_budget_temperature, 1e-6)
                )
                base_consistency_reserve = _sigmoid_np(
                    (route_consistency - config.pairwise_consistency_threshold)
                    / max(config.pairperm_consistency_budget_temperature, 1e-6)
                )
                provisional_frontier_reserve = float(
                    np.clip(0.56 * base_risk_reserve + 0.44 * base_consistency_reserve, 0.0, 1.0)
                )

            sample_weight = float(
                np.clip(
                    0.22
                    + abs(route_value_raw)
                    + 0.34 * teacher_oracle
                    + 0.24 * provisional_frontier_reserve
                    + 0.20 * escape_support
                    + 0.16 * calibrated["pairwise_route_constraint_score"],
                    0.10,
                    3.2,
                )
            )
            row: dict[str, float | int | str] = {
                "dataset_id": int(dataset_id),
                "scenario": str(record["scenario"]),
                "seed": int(record["seed"]),
                "candidate_name": candidate_name,
                "anchor_name": "stage1_blended",
                "pairwise_utility_gain_target": calibrated["pairwise_safe_utility_gain"],
                "pairwise_failure_risk_target": failure_risk,
                "pairwise_fallback_regret_target": calibrated["pairwise_fallback_regret_calibrated"],
                "pairwise_route_margin_raw_target": route_margin_raw,
                "pairwise_route_margin_target": calibrated["pairwise_route_margin_calibrated"],
                "pairwise_route_value_raw": route_value_raw,
                "pairwise_route_value_target": route_value_target,
                "pairwise_sample_weight": sample_weight,
                "pairwise_stage1_route_adjusted": float(anchor_row["route_adjusted_reward"]),
                "pairwise_candidate_route_adjusted": float(record["route_adjusted_reward"]),
                "pairwise_route_consistency": route_consistency,
                "pairwise_escape_support": escape_support,
                "teacher_oracle": teacher_oracle,
                "frontier_risk_reserve_oracle": provisional_frontier_reserve,
            }
            row.update(calibrated)
            row.update(
                _build_stage1_pairwise_feature_dict(
                    candidate_features=candidate_features,
                    anchor_features=anchor_features,
                    candidate_name=candidate_name,
                    bank_names=bank_names,
                )
            )
            row_buffer.append(row)
            candidate_summaries.append(
                {
                    "candidate_name": candidate_name,
                    "candidate_is_stage1": 1.0 if candidate_name == "stage1_blended" else 0.0,
                    "teacher_oracle": teacher_oracle,
                    "frontier_risk_reserve": provisional_frontier_reserve,
                    "safe_utility_gain": calibrated["pairwise_safe_utility_gain"],
                    "fallback_regret_calibrated": calibrated["pairwise_fallback_regret_calibrated"],
                    "failure_risk": failure_risk,
                    "route_consistency": route_consistency,
                    "escape_support": escape_support,
                    "route_value_target": route_value_target,
                }
            )

        anchor_targets = _frontier_anchor_targets(
            candidate_summaries=candidate_summaries,
            anchor_features=anchor_features,
            config=config,
        )
        budget_anchor_context = {
            "anchor_strength": float(anchor_targets["anchor_strength"]),
            "anchor_escape_pressure": float(anchor_targets["anchor_escape_target"]),
            "anchor_fragility": float(anchor_targets["anchor_escape_uncertainty_target"]),
            "anchor_routability": float(anchor_targets["teacher_topm_mass"]),
        }
        adaptive_risk_budget, adaptive_consistency_budget = _pairperm_budget_targets(
            anchor_context=budget_anchor_context,
            dataset_stats=dataset_stats,
            config=config,
            variant="fixed_budget" if flags["fixed_frontier"] else "full",
        )

        anchor_threshold = config.pairperm_permission_threshold
        frontier_threshold = max(config.frontier_accept_threshold, 0.40)
        top_candidate_names = set(anchor_targets["top_candidate_names"])  # type: ignore[arg-type]

        for row in row_buffer:
            candidate_name = str(row["candidate_name"])
            teacher_oracle = float(row["teacher_oracle"])
            top_candidate_bonus = 1.0 if candidate_name in top_candidate_names else 0.0
            teacher_set_target = 0.0
            frontier_target = 0.0
            frontier_risk_reserve = 0.0
            frontier_false_release_risk = 0.0
            frontier_missed_escape_risk = 0.0
            frontier_margin = 0.0
            frontier_uncertainty_target = 1.0
            if candidate_name != "stage1_blended":
                risk_reserve = _sigmoid_np(
                    (adaptive_risk_budget - float(row["pairwise_failure_risk_target"]))
                    / max(config.pairperm_risk_budget_temperature, 1e-6)
                )
                consistency_reserve = _sigmoid_np(
                    (float(row["pairwise_route_consistency"]) - adaptive_consistency_budget)
                    / max(config.pairperm_consistency_budget_temperature, 1e-6)
                )
                frontier_risk_reserve = float(np.clip(0.56 * risk_reserve + 0.44 * consistency_reserve, 0.0, 1.0))
                teacher_set_target = float(
                    np.clip(
                        0.68 * teacher_oracle
                        + 0.18 * top_candidate_bonus
                        + 0.14 * float(anchor_targets["anchor_escape_target"]),
                        0.0,
                        1.0,
                    )
                )
                frontier_margin = float(
                    1.05 * (teacher_set_target - 0.50)
                    + 0.62 * (frontier_risk_reserve - 0.50)
                    + 0.22 * (float(row["pairwise_route_consistency"]) - 0.50)
                    + 0.18 * float(row["pairwise_route_value_target"])
                    + 0.10 * float(row["pairwise_fallback_regret_target"])
                    - 0.18 * float(row["pairwise_failure_risk_target"])
                )
                frontier_target = float(
                    np.clip(
                        _sigmoid_np(frontier_margin / max(config.frontier_margin_temperature, 1e-6))
                        + 0.06 * top_candidate_bonus,
                        0.0,
                        1.0,
                    )
                )
                frontier_false_release_risk = float(
                    np.clip(
                        (1.0 - frontier_risk_reserve)
                        * (
                            0.60 * float(row["pairwise_failure_risk_target"])
                            + 0.40 * (1.0 - float(row["pairwise_route_consistency"]))
                        ),
                        0.0,
                        1.0,
                    )
                )
                frontier_missed_escape_risk = float(
                    np.clip(
                        teacher_set_target
                        * (0.55 + 0.45 * max(float(row["pairwise_route_value_target"]), 0.0))
                        * (1.0 - frontier_target),
                        0.0,
                        1.0,
                    )
                )
                frontier_uncertainty_target = float(np.clip(1.0 - 2.0 * abs(frontier_target - 0.5), 0.0, 1.0))

            route_permission_prob = frontier_target
            route_permission_calibrated = float(
                np.clip(
                    0.74 * frontier_target + 0.16 * frontier_risk_reserve + 0.10 * teacher_set_target,
                    0.0,
                    1.0,
                )
            )
            route_permission_logit = float(
                np.log(
                    np.clip(route_permission_prob, 1e-6, 1.0 - 1e-6)
                    / np.clip(1.0 - route_permission_prob, 1e-6, 1.0)
                )
            )
            if route_permission_prob <= 1e-6:
                route_permission_logit = -12.0
            elif route_permission_prob >= 1.0 - 1e-6:
                route_permission_logit = 12.0

            final_permission_pass = float(
                candidate_name != "stage1_blended"
                and float(anchor_targets["anchor_escape_target"]) >= anchor_threshold
                and frontier_target >= frontier_threshold
            )
            block_reason = _frontier_block_reason(
                candidate_name=candidate_name,
                anchor_escape_calibrated=float(anchor_targets["anchor_escape_target"]),
                teacher_set_prob=teacher_set_target,
                frontier_accept_prob=frontier_target,
                frontier_risk_reserve=frontier_risk_reserve,
                anchor_threshold=anchor_threshold,
                frontier_threshold=frontier_threshold,
            )

            row["anchor_escape_target"] = float(anchor_targets["anchor_escape_target"])
            row["anchor_escape_soft_oracle_gain"] = float(anchor_targets["anchor_escape_soft_oracle_gain"])
            row["anchor_escape_safe_candidate_count"] = float(anchor_targets["anchor_escape_safe_candidate_count"])
            row["anchor_escape_topm_mass"] = float(anchor_targets["anchor_escape_topm_mass"])
            row["anchor_escape_counterfactual_margin"] = float(anchor_targets["anchor_escape_counterfactual_margin"])
            row["anchor_escape_uncertainty_target"] = float(anchor_targets["anchor_escape_uncertainty_target"])
            row["allowed_set_target_mass"] = float(anchor_targets["frontier_target_mass"])
            row["permission_set_consistency_target"] = float(anchor_targets["frontier_consistency_target"])
            row["counterfactual_escape_support"] = float(anchor_targets["counterfactual_escape_support"])
            row["candidate_admissibility_target"] = teacher_set_target
            row["candidate_allowed_target"] = frontier_target
            row["teacher_set_score"] = teacher_oracle
            row["teacher_set_prob"] = teacher_oracle
            row["teacher_set_target"] = teacher_set_target
            row["teacher_set_mass"] = float(anchor_targets["teacher_set_mass"])
            row["teacher_topm_mass"] = float(anchor_targets["teacher_topm_mass"])
            row["frontier_target"] = frontier_target
            row["frontier_margin"] = frontier_margin
            row["frontier_accept_prob"] = frontier_target
            row["frontier_uncertainty_target"] = frontier_uncertainty_target
            row["frontier_risk_reserve"] = frontier_risk_reserve
            row["frontier_coverage_score"] = float(anchor_targets["frontier_coverage_target"])
            row["frontier_teacher_agreement"] = float(np.clip(1.0 - abs(frontier_target - teacher_set_target), 0.0, 1.0))
            row["frontier_false_release_risk"] = frontier_false_release_risk
            row["frontier_missed_escape_risk"] = frontier_missed_escape_risk
            row["frontier_reject_reason"] = block_reason
            row["pairwise_route_permission_target"] = route_permission_prob
            row["pairwise_route_permission_logit"] = route_permission_logit
            row["pairwise_route_permission_prob"] = route_permission_prob
            row["pairwise_route_permission_calibrated"] = route_permission_calibrated
            row["pairwise_route_permission_margin"] = frontier_target - frontier_threshold
            row["pairwise_route_permission_confidence"] = float(np.clip(abs(2.0 * frontier_target - 1.0), 0.0, 1.0))
            row["pairwise_anchor_escape_pressure"] = float(anchor_targets["anchor_escape_target"])
            row["pairwise_anchor_routability"] = float(anchor_targets["teacher_topm_mass"])
            row["pairwise_anchor_fragility"] = float(anchor_targets["anchor_escape_uncertainty_target"])
            row["pairwise_adaptive_risk_budget"] = adaptive_risk_budget
            row["pairwise_adaptive_consistency_budget"] = adaptive_consistency_budget
            row["pairwise_permission_budget_score"] = frontier_risk_reserve
            row["pairwise_permission_block_reason"] = block_reason
            row["pairwise_final_route_permission_pass"] = final_permission_pass
            row["pairwise_route_win_target"] = final_permission_pass
            row["pairwise_routed_away_from_stage1"] = final_permission_pass
            row["pairwise_anchor_selection_outcome"] = 1.0 - final_permission_pass
            row["pairwise_final_route_score"] = float(row["pairwise_route_value_raw"])
            row["pairwise_value_among_allowed_candidates"] = (
                float(row["pairwise_route_value_target"]) if final_permission_pass > 0.5 else 0.0
            )
            row["pairwise_route_selected_under_permission"] = 0.0

        allowed_rows = [
            row
            for row in row_buffer
            if str(row["candidate_name"]) != "stage1_blended" and float(row["pairwise_final_route_permission_pass"]) > 0.5
        ]
        if allowed_rows:
            selected_row = max(allowed_rows, key=lambda current: float(current["pairwise_route_value_target"]))
            selected_row["pairwise_route_selected_under_permission"] = 1.0

        rows.extend(row_buffer)

    return pd.DataFrame(rows)


def collect_gate_training_data(
    *,
    config: GateLearningConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[dict[str, float | int | str]] = []
    target_rows: list[dict[str, float | int | str]] = []

    for scenario_idx, scenario in enumerate(config.train_scenarios):
        for seed_idx, seed in enumerate(config.train_seeds):
            n_cells = config.cell_options[(seed_idx + scenario_idx) % len(config.cell_options)]
            n_genes = config.gene_options[(2 * seed_idx + scenario_idx) % len(config.gene_options)]
            data = generate_synthetic_scrna(
                scenario=scenario,
                n_cells=n_cells,
                n_genes=n_genes,
                random_state=config.random_state + 37 * scenario_idx + seed,
            )

            selector = RefineMoEHVGSelector(
                top_k=min(config.top_k, data.counts.shape[1]),
                refine_epochs=6,
                random_state=config.random_state + seed,
                mode="full",
            )
            x, base_features, dataset_stats, expert_scores = selector.prepare_context(data.counts, data.batches)
            routed_gate = selector._dataset_gate(dataset_stats)
            heuristic_gate = _blend_heuristic_gate(routed_gate)
            candidate_names, candidate_gates = build_gate_candidates(
                heuristic_gate=heuristic_gate,
                n_random=config.candidate_random_gates,
                rng=np.random.default_rng(config.random_state + seed + 101 * scenario_idx),
            )

            candidate_records: list[dict[str, object]] = []
            top_k = min(config.top_k, data.counts.shape[1])
            for candidate_name, gate in zip(candidate_names, candidate_gates, strict=False):
                scores = selector.score_with_context(
                    x=x,
                    base_features=base_features,
                    dataset_stats=dataset_stats,
                    expert_scores=expert_scores,
                    gate=gate,
                    apply_refine=config.train_with_refine,
                )
                selected = np.argsort(scores)[-top_k:]
                metrics = _fast_weak_metrics(
                    counts=data.counts,
                    selected_genes=selected,
                    labels=data.cell_types,
                    batches=data.batches,
                    random_state=config.random_state + seed,
                )
                reward = downstream_reward(metrics)
                candidate_records.append(
                    {
                        "candidate_name": candidate_name,
                        "gate": gate,
                        "metrics": metrics,
                        "reward": reward,
                    }
                )

            baseline_record, ari_floor, nmi_floor, structure_floor = _baseline_metric_floors(
                candidate_records,
                floor_ratio=config.floor_ratio,
                ari_margin=config.floor_margin_ari,
                nmi_margin=config.floor_margin_nmi,
                structure_margin=config.floor_margin_structure,
            )
            baseline_metrics = baseline_record["metrics"]  # type: ignore[assignment]
            baseline_gate = _normalize_gate_np(np.asarray(baseline_record["gate"], dtype=np.float64))  # type: ignore[index]
            baseline_reward = float(baseline_record["reward"])
            baseline_adjusted_reward = baseline_reward
            batch_tradeoff_pressure = _batch_tradeoff_pressure(dataset_stats)
            adjusted_rewards: list[float] = []
            ari_gaps: list[float] = []
            nmi_gaps: list[float] = []
            structure_gaps: list[float] = []
            safe_mask: list[bool] = []
            tradeoff_penalties: list[float] = []
            for record in candidate_records:
                metrics = record["metrics"]  # type: ignore[assignment]
                reward = float(record["reward"])
                ari_gap = max(0.0, ari_floor - float(metrics["ari"]))
                nmi_gap = max(0.0, nmi_floor - float(metrics["nmi"]))
                structure_gap = max(0.0, structure_floor - structure_preservation_score(metrics))
                mixing_gain = max(0.0, float(metrics.get("batch_mixing", 0.0)) - float(baseline_metrics.get("batch_mixing", 0.0)))
                tradeoff_penalty = batch_tradeoff_pressure * mixing_gain * (
                    1.25 * ari_gap + 1.05 * nmi_gap + 0.85 * structure_gap
                )
                adjusted_reward = (
                    reward
                    - config.floor_weight * (1.60 * ari_gap + 1.25 * nmi_gap + 0.95 * structure_gap)
                    - config.tradeoff_penalty_weight * tradeoff_penalty
                )
                record["adjusted_reward"] = adjusted_reward
                record["structure_score"] = structure_preservation_score(metrics)
                adjusted_rewards.append(adjusted_reward)
                ari_gaps.append(ari_gap)
                nmi_gaps.append(nmi_gap)
                structure_gaps.append(structure_gap)
                safe_mask.append(ari_gap <= 1e-8 and nmi_gap <= 1e-8 and structure_gap <= 1e-8)
                tradeoff_penalties.append(tradeoff_penalty)

                row: dict[str, float | int | str] = {
                    "scenario": scenario,
                    "seed": seed,
                    "n_cells": int(n_cells),
                    "n_genes": int(n_genes),
                    "candidate_name": str(record["candidate_name"]),
                    "reward": reward,
                    "adjusted_reward": adjusted_reward,
                    "structure_score": float(record["structure_score"]),
                    "ari_floor": ari_floor,
                    "nmi_floor": nmi_floor,
                    "structure_floor": structure_floor,
                    "ari_gap": ari_gap,
                    "nmi_gap": nmi_gap,
                    "structure_gap": structure_gap,
                    "tradeoff_penalty": tradeoff_penalty,
                    "baseline_reward": baseline_reward,
                    "baseline_batch_mixing": float(baseline_metrics.get("batch_mixing", 0.0)),
                    "is_safe_candidate": int(safe_mask[-1]),
                }
                gate = np.asarray(record["gate"], dtype=np.float64)
                row.update({f"gate_{expert}": float(weight) for expert, weight in zip(EXPERT_NAMES, gate, strict=False)})
                row.update({f"stat_{key}": float(dataset_stats[key]) for key in DEFAULT_GATE_FEATURE_KEYS})
                row.update({metric_name: float(metric_value) for metric_name, metric_value in record["metrics"].items()})  # type: ignore[arg-type]
                candidate_rows.append(row)

            adjusted_rewards_np = np.asarray(adjusted_rewards, dtype=np.float64)
            safe_mask_np = np.asarray(safe_mask, dtype=bool)
            safe_indices = np.flatnonzero(safe_mask_np)
            if safe_indices.size == 0:
                baseline_idx = next(
                    idx
                    for idx, record in enumerate(candidate_records)
                    if str(record["candidate_name"]) == str(baseline_record["candidate_name"])
                )
                safe_indices = np.asarray([baseline_idx], dtype=np.int64)

            safe_reward_probs = _softmax_np(adjusted_rewards_np[safe_indices] / max(config.reward_temperature, 1e-6))
            safe_target_gate = (safe_reward_probs[:, None] * candidate_gates[safe_indices]).sum(axis=0)
            safe_target_gate = _normalize_gate_np(safe_target_gate)

            best_safe_idx = int(safe_indices[np.argmax(adjusted_rewards_np[safe_indices])])
            improvement = max(0.0, adjusted_rewards_np[best_safe_idx] - baseline_adjusted_reward)
            improvement_scale = improvement / (improvement + config.baseline_blend_scale)
            target_gate = _normalize_gate_np(
                (1.0 - improvement_scale) * baseline_gate + improvement_scale * safe_target_gate
            )

            violating_indices = np.flatnonzero(
                (np.asarray(ari_gaps, dtype=np.float64) > 1e-8)
                | (np.asarray(nmi_gaps, dtype=np.float64) > 1e-8)
                | (np.asarray(structure_gaps, dtype=np.float64) > 1e-8)
            )
            negative_pool = violating_indices if violating_indices.size else np.arange(len(candidate_records))
            worst_idx = int(negative_pool[np.argmin(adjusted_rewards_np[negative_pool])])
            preferred_gate = _normalize_gate_np(candidate_gates[best_safe_idx])
            dispreferred_gate = _normalize_gate_np(candidate_gates[worst_idx])
            pair_margin = float(np.clip(config.preference_margin + 0.45 * improvement, config.preference_margin, 0.18))
            ranking_scale = float(0.25 + improvement_scale)
            baseline_keep = float(1.0 - improvement_scale)
            sample_weight = float(
                max(np.max(adjusted_rewards_np) - np.mean(adjusted_rewards_np), 0.05)
                * (1.0 + 1.25 * float(dataset_stats.get("batch_strength", 0.0)) + 0.35 * float(dataset_stats.get("trajectory_strength", 0.0)))
            )

            target_row: dict[str, float | int | str] = {
                "scenario": scenario,
                "seed": seed,
                "n_cells": int(n_cells),
                "n_genes": int(n_genes),
                "best_reward": float(np.max(adjusted_rewards_np)),
                "mean_reward": float(np.mean(adjusted_rewards_np)),
                "reward_margin": float(np.max(adjusted_rewards_np) - np.mean(adjusted_rewards_np)),
                "ari_floor": ari_floor,
                "nmi_floor": nmi_floor,
                "structure_floor": structure_floor,
                "baseline_reward": baseline_reward,
                "best_safe_reward": float(adjusted_rewards_np[best_safe_idx]),
                "baseline_keep": baseline_keep,
                "ranking_scale": ranking_scale,
                "pair_margin": pair_margin,
                "sample_weight": sample_weight,
            }
            target_row.update({f"stat_{key}": float(dataset_stats[key]) for key in DEFAULT_GATE_FEATURE_KEYS})
            target_row.update({f"target_{expert}": float(weight) for expert, weight in zip(EXPERT_NAMES, target_gate, strict=False)})
            target_row.update({f"pref_{expert}": float(weight) for expert, weight in zip(EXPERT_NAMES, preferred_gate, strict=False)})
            target_row.update({f"anti_{expert}": float(weight) for expert, weight in zip(EXPERT_NAMES, dispreferred_gate, strict=False)})
            target_row.update({f"heuristic_{expert}": float(weight) for expert, weight in zip(EXPERT_NAMES, heuristic_gate, strict=False)})
            target_row.update({f"baseline_{expert}": float(weight) for expert, weight in zip(EXPERT_NAMES, baseline_gate, strict=False)})
            target_row["feature_vector"] = json.dumps(dataset_stats_to_vector(dataset_stats).tolist())
            target_rows.append(target_row)

    return pd.DataFrame(candidate_rows), pd.DataFrame(target_rows)


def train_gate_model(
    *,
    output_dir: str,
    config: GateLearningConfig,
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    collect_start = time.perf_counter()
    candidate_df, target_df = collect_gate_training_data(config=config)
    collect_elapsed = time.perf_counter() - collect_start

    candidate_csv = output_path / "gate_candidate_rewards.csv"
    target_csv = output_path / "gate_training_targets.csv"
    candidate_df.to_csv(candidate_csv, index=False)
    target_df.to_csv(target_csv, index=False)

    feature_matrix = np.stack(target_df["feature_vector"].map(json.loads).to_list()).astype(np.float32)
    target_matrix = target_df[[f"target_{expert}" for expert in EXPERT_NAMES]].to_numpy(dtype=np.float32)
    pref_matrix = target_df[[f"pref_{expert}" for expert in EXPERT_NAMES]].to_numpy(dtype=np.float32)
    anti_matrix = target_df[[f"anti_{expert}" for expert in EXPERT_NAMES]].to_numpy(dtype=np.float32)
    heuristic_matrix = target_df[[f"heuristic_{expert}" for expert in EXPERT_NAMES]].to_numpy(dtype=np.float32)
    baseline_matrix = target_df[[f"baseline_{expert}" for expert in EXPERT_NAMES]].to_numpy(dtype=np.float32)
    pair_margin_vec = target_df["pair_margin"].to_numpy(dtype=np.float32)
    ranking_scale_vec = target_df["ranking_scale"].to_numpy(dtype=np.float32)
    baseline_keep_vec = target_df["baseline_keep"].to_numpy(dtype=np.float32)
    sample_weights = np.maximum(target_df["sample_weight"].to_numpy(dtype=np.float32), 0.05)

    rng = np.random.default_rng(config.random_state)
    indices = rng.permutation(len(target_df))
    val_size = 0
    if len(indices) >= 4:
        val_size = max(1, int(round(len(indices) * config.val_fraction)))
        val_size = min(val_size, len(indices) - 1)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        train_idx = indices
        val_idx = np.asarray([], dtype=np.int64)

    feature_mean = feature_matrix[train_idx].mean(axis=0)
    feature_std = feature_matrix[train_idx].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    x_train = ((feature_matrix[train_idx] - feature_mean) / feature_std).astype(np.float32)
    y_train = target_matrix[train_idx].astype(np.float32)
    pref_train = pref_matrix[train_idx].astype(np.float32)
    anti_train = anti_matrix[train_idx].astype(np.float32)
    heuristic_train = heuristic_matrix[train_idx].astype(np.float32)
    baseline_train = baseline_matrix[train_idx].astype(np.float32)
    pair_margin_train = pair_margin_vec[train_idx].astype(np.float32)
    ranking_scale_train = ranking_scale_vec[train_idx].astype(np.float32)
    baseline_keep_train = baseline_keep_vec[train_idx].astype(np.float32)
    w_train = sample_weights[train_idx].astype(np.float32)

    x_val = (
        ((feature_matrix[val_idx] - feature_mean) / feature_std).astype(np.float32)
        if len(val_idx)
        else np.zeros((0, feature_matrix.shape[1]), dtype=np.float32)
    )
    y_val = target_matrix[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0, target_matrix.shape[1]), dtype=np.float32)
    pref_val = pref_matrix[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0, pref_matrix.shape[1]), dtype=np.float32)
    anti_val = anti_matrix[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0, anti_matrix.shape[1]), dtype=np.float32)
    heuristic_val = heuristic_matrix[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0, heuristic_matrix.shape[1]), dtype=np.float32)
    baseline_val = baseline_matrix[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0, baseline_matrix.shape[1]), dtype=np.float32)
    pair_margin_val = pair_margin_vec[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    ranking_scale_val = ranking_scale_vec[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    baseline_keep_val = baseline_keep_vec[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    w_val = sample_weights[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)

    device = choose_torch_device()
    base_model = LearnableDatasetGate(
        input_dim=feature_matrix.shape[1],
        output_dim=len(EXPERT_NAMES),
        gate_type=config.gate_type,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )
    if torch.cuda.is_available():
        base_model = base_model.to(device)
        model: nn.Module = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model
    else:
        base_model = base_model.to(device)
        model = base_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(pref_train),
        torch.from_numpy(anti_train),
        torch.from_numpy(heuristic_train),
        torch.from_numpy(baseline_train),
        torch.from_numpy(pair_margin_train),
        torch.from_numpy(ranking_scale_train),
        torch.from_numpy(baseline_keep_train),
        torch.from_numpy(w_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False,
    )

    best_state = copy.deepcopy(_unwrap_model(model).state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0
    history_rows: list[dict[str, float | int]] = []
    train_start = time.perf_counter()

    for epoch in range(config.train_epochs):
        model.train()
        running_loss = 0.0
        running_count = 0

        for (
            batch_x,
            batch_y,
            batch_pref,
            batch_anti,
            batch_heuristic,
            batch_baseline,
            batch_pair_margin,
            batch_ranking_scale,
            batch_baseline_keep,
            batch_w,
        ) in train_loader:
            batch_x = batch_x.to(device, non_blocking=torch.cuda.is_available())
            batch_y = batch_y.to(device, non_blocking=torch.cuda.is_available())
            batch_pref = batch_pref.to(device, non_blocking=torch.cuda.is_available())
            batch_anti = batch_anti.to(device, non_blocking=torch.cuda.is_available())
            batch_heuristic = batch_heuristic.to(device, non_blocking=torch.cuda.is_available())
            batch_baseline = batch_baseline.to(device, non_blocking=torch.cuda.is_available())
            batch_pair_margin = batch_pair_margin.to(device, non_blocking=torch.cuda.is_available())
            batch_ranking_scale = batch_ranking_scale.to(device, non_blocking=torch.cuda.is_available())
            batch_baseline_keep = batch_baseline_keep.to(device, non_blocking=torch.cuda.is_available())
            batch_w = batch_w.to(device, non_blocking=torch.cuda.is_available())

            delta_logits = model(batch_x)
            prob = _compose_gate_from_delta_torch(
                delta_logits=delta_logits,
                heuristic_gate=batch_heuristic,
                residual_scale=config.residual_scale,
            )
            loss = _gate_training_objective(
                prob=prob,
                target=batch_y,
                preferred=batch_pref,
                anti=batch_anti,
                heuristic=batch_heuristic,
                baseline=batch_baseline,
                weights=batch_w,
                preference_weight=config.preference_weight,
                anchor_weight=config.anchor_weight,
                baseline_ce_weight=config.baseline_ce_weight,
                pair_margin=batch_pair_margin,
                ranking_scale=batch_ranking_scale,
                baseline_keep=batch_baseline_keep,
                entropy_bonus=config.entropy_bonus,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu()) * len(batch_x)
            running_count += len(batch_x)

        train_loss = running_loss / max(running_count, 1)
        val_loss = _evaluate_gate_loss(
            model=model,
            x_val=x_val,
            y_val=y_val,
            pref_val=pref_val,
            anti_val=anti_val,
            heuristic_val=heuristic_val,
            baseline_val=baseline_val,
            pair_margin_val=pair_margin_val,
            ranking_scale_val=ranking_scale_val,
            baseline_keep_val=baseline_keep_val,
            w_val=w_val,
            device=device,
            config=config,
        )
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(_unwrap_model(model).state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= config.patience:
            break

    train_elapsed = time.perf_counter() - train_start
    _unwrap_model(model).load_state_dict(best_state)
    model.eval()

    normalized_features = ((feature_matrix - feature_mean) / feature_std).astype(np.float32)
    prototype_gate_matrix = _normalize_gate_matrix_np(0.65 * pref_matrix + 0.35 * target_matrix).astype(np.float32)
    checkpoint_path = output_path / "learnable_gate.pt"
    bundle = {
        "model_state": best_state,
        "feature_keys": list(DEFAULT_GATE_FEATURE_KEYS),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "expert_names": list(EXPERT_NAMES),
        "gate_type": config.gate_type,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "config": asdict(config),
        "train_examples": int(len(train_idx)),
        "val_examples": int(len(val_idx)),
        "collect_time_sec": collect_elapsed,
        "train_time_sec": train_elapsed,
        "best_val_loss": float(best_val_loss if np.isfinite(best_val_loss) else history_rows[-1]["train_loss"]),
        "device": str(device),
        "prototype_features": normalized_features,
        "prototype_gates": prototype_gate_matrix,
    }
    torch.save(bundle, checkpoint_path)
    _RUNTIME_CACHE.clear()

    bank_training = _train_gate_bank_model(
        output_dir=output_path,
        checkpoint_path=str(checkpoint_path),
        config=config,
    )
    curr_training = _train_counterfactual_tri_factor_model(
        output_dir=output_path,
        config=config,
    )
    pairwise_training = _train_stage1_pairwise_router(
        output_dir=output_path,
        config=config,
    )
    pairwise_cal_variants = {
        variant: _train_stage1_pairwise_router_calibrated(
            output_dir=output_path,
            config=config,
            variant=variant,
        )
        for variant in (
            "full",
            "no_regret_calibration",
            "no_consistency_calibration",
            "no_route_constraint",
            "utility_only",
        )
    }
    pairwise_cal_training = pairwise_cal_variants["full"]
    pairperm_variants = {
        variant: _train_stage1_pairwise_router_permissioned(
            output_dir=output_path,
            config=config,
            variant=variant,
        )
        for variant in (
            "full",
            "no_permission_head",
            "fixed_budget",
            "no_anchor_adaptation",
            "no_permission_value_decoupling",
            "no_regret_aux",
            "permission_only",
            "refine_on",
        )
    }
    pairperm_training = pairperm_variants["full"]
    escapecert_variants = {
        variant: _train_stage1_pairwise_router_escapecert(
            output_dir=output_path,
            config=config,
            variant=variant,
        )
        for variant in (
            "full",
            "no_anchor_escape_head",
            "no_set_supervision",
            "no_candidate_admissibility",
            "fixed_budget",
            "no_escape_uncertainty",
            "no_regret_aux",
            "anchor_escape_only",
            "admissibility_only",
        )
    }
    escapecert_training = escapecert_variants["full"]
    frontier_variants = {
        variant: _train_stage1_pairwise_router_frontier(
            output_dir=output_path,
            config=config,
            variant=variant,
        )
        for variant in (
            "full",
            "no_teacher_distill",
            "no_frontier_head",
            "no_frontier_uncertainty",
            "fixed_frontier",
            "no_anchor_escape_head",
            "no_regret_aux",
            "teacher_only",
            "no_teacher_frontier_only",
        )
    }
    frontier_training = frontier_variants["full"]
    bundle.update(
        {
            "bank_model_state": bank_training["bank_model_state"],
            "bank_feature_keys": bank_training["bank_feature_keys"],
            "bank_feature_mean": bank_training["bank_feature_mean"],
            "bank_feature_std": bank_training["bank_feature_std"],
            "bank_hidden_dim": bank_training["bank_hidden_dim"],
            "bank_dropout": bank_training["bank_dropout"],
            "bank_candidate_names": bank_training["bank_candidate_names"],
            "bank_best_val_loss": bank_training["bank_best_val_loss"],
            "curr_model_state": curr_training["curr_model_state"],
            "curr_feature_keys": curr_training["curr_feature_keys"],
            "curr_feature_mean": curr_training["curr_feature_mean"],
            "curr_feature_std": curr_training["curr_feature_std"],
            "curr_hidden_dim": curr_training["curr_hidden_dim"],
            "curr_dropout": curr_training["curr_dropout"],
            "curr_best_val_loss": curr_training["curr_best_val_loss"],
            "pairwise_model_state": pairwise_training["pairwise_model_state"],
            "pairwise_feature_keys": pairwise_training["pairwise_feature_keys"],
            "pairwise_feature_mean": pairwise_training["pairwise_feature_mean"],
            "pairwise_feature_std": pairwise_training["pairwise_feature_std"],
            "pairwise_hidden_dim": pairwise_training["pairwise_hidden_dim"],
            "pairwise_dropout": pairwise_training["pairwise_dropout"],
            "pairwise_best_val_loss": pairwise_training["pairwise_best_val_loss"],
            "pairwise_calibration_models": {
                variant: {
                    "model_state": result["pairwise_cal_model_state"],
                    "feature_keys": result["pairwise_cal_feature_keys"],
                    "feature_mean": result["pairwise_cal_feature_mean"],
                    "feature_std": result["pairwise_cal_feature_std"],
                    "hidden_dim": result["pairwise_cal_hidden_dim"],
                    "dropout": result["pairwise_cal_dropout"],
                    "best_val_loss": result["pairwise_cal_best_val_loss"],
                    "history_csv": result["pairwise_cal_history_csv"],
                    "candidate_csv": result["pairwise_cal_candidate_csv"],
                    "train_examples": result["pairwise_cal_train_examples"],
                    "val_examples": result["pairwise_cal_val_examples"],
                    "train_time_sec": result["pairwise_cal_train_time_sec"],
                }
                for variant, result in pairwise_cal_variants.items()
            },
            "pairwise_permission_models": {
                variant: {
                    "model_state": result["pairperm_model_state"],
                    "feature_keys": result["pairperm_feature_keys"],
                    "feature_mean": result["pairperm_feature_mean"],
                    "feature_std": result["pairperm_feature_std"],
                    "hidden_dim": result["pairperm_hidden_dim"],
                    "dropout": result["pairperm_dropout"],
                    "best_val_loss": result["pairperm_best_val_loss"],
                    "history_csv": result["pairperm_history_csv"],
                    "candidate_csv": result["pairperm_candidate_csv"],
                    "train_examples": result["pairperm_train_examples"],
                    "val_examples": result["pairperm_val_examples"],
                    "train_time_sec": result["pairperm_train_time_sec"],
                }
                for variant, result in pairperm_variants.items()
            },
            "pairwise_escapecert_models": {
                variant: {
                    "model_state": result["escapecert_model_state"],
                    "feature_keys": result["escapecert_feature_keys"],
                    "feature_mean": result["escapecert_feature_mean"],
                    "feature_std": result["escapecert_feature_std"],
                    "hidden_dim": result["escapecert_hidden_dim"],
                    "dropout": result["escapecert_dropout"],
                    "best_val_loss": result["escapecert_best_val_loss"],
                    "history_csv": result["escapecert_history_csv"],
                    "candidate_csv": result["escapecert_candidate_csv"],
                    "train_examples": result["escapecert_train_examples"],
                    "val_examples": result["escapecert_val_examples"],
                    "train_anchors": result["escapecert_train_anchors"],
                    "val_anchors": result["escapecert_val_anchors"],
                    "train_time_sec": result["escapecert_train_time_sec"],
                }
                for variant, result in escapecert_variants.items()
            },
            "pairwise_frontier_models": {
                variant: {
                    "model_state": result["frontier_model_state"],
                    "feature_keys": result["frontier_feature_keys"],
                    "feature_mean": result["frontier_feature_mean"],
                    "feature_std": result["frontier_feature_std"],
                    "hidden_dim": result["frontier_hidden_dim"],
                    "dropout": result["frontier_dropout"],
                    "best_val_loss": result["frontier_best_val_loss"],
                    "history_csv": result["frontier_history_csv"],
                    "candidate_csv": result["frontier_candidate_csv"],
                    "train_examples": result["frontier_train_examples"],
                    "val_examples": result["frontier_val_examples"],
                    "train_anchors": result["frontier_train_anchors"],
                    "val_anchors": result["frontier_val_anchors"],
                    "train_time_sec": result["frontier_train_time_sec"],
                }
                for variant, result in frontier_variants.items()
            },
        }
    )
    torch.save(bundle, checkpoint_path)
    _RUNTIME_CACHE.clear()

    history_df = pd.DataFrame(history_rows)
    history_csv = output_path / "gate_training_history.csv"
    history_df.to_csv(history_csv, index=False)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "candidate_csv": str(candidate_csv),
        "target_csv": str(target_csv),
        "history_csv": str(history_csv),
        "collect_time_sec": collect_elapsed,
        "train_time_sec": train_elapsed,
        "train_examples": int(len(train_idx)),
        "val_examples": int(len(val_idx)),
        "best_val_loss": float(best_val_loss if np.isfinite(best_val_loss) else history_rows[-1]["train_loss"]),
        "bank_candidate_csv": bank_training["bank_candidate_csv"],
        "bank_history_csv": bank_training["bank_history_csv"],
        "bank_best_val_loss": bank_training["bank_best_val_loss"],
        "bank_train_examples": bank_training["bank_train_examples"],
        "bank_val_examples": bank_training["bank_val_examples"],
        "bank_train_time_sec": bank_training["bank_train_time_sec"],
        "curr_history_csv": curr_training["curr_history_csv"],
        "curr_best_val_loss": curr_training["curr_best_val_loss"],
        "curr_train_examples": curr_training["curr_train_examples"],
        "curr_val_examples": curr_training["curr_val_examples"],
        "curr_train_time_sec": curr_training["curr_train_time_sec"],
        "pairwise_candidate_csv": pairwise_training["pairwise_candidate_csv"],
        "pairwise_history_csv": pairwise_training["pairwise_history_csv"],
        "pairwise_best_val_loss": pairwise_training["pairwise_best_val_loss"],
        "pairwise_train_examples": pairwise_training["pairwise_train_examples"],
        "pairwise_val_examples": pairwise_training["pairwise_val_examples"],
        "pairwise_train_time_sec": pairwise_training["pairwise_train_time_sec"],
        "pairwise_cal_candidate_csv": pairwise_cal_training["pairwise_cal_candidate_csv"],
        "pairwise_cal_history_csv": pairwise_cal_training["pairwise_cal_history_csv"],
        "pairwise_cal_best_val_loss": pairwise_cal_training["pairwise_cal_best_val_loss"],
        "pairwise_cal_train_examples": pairwise_cal_training["pairwise_cal_train_examples"],
        "pairwise_cal_val_examples": pairwise_cal_training["pairwise_cal_val_examples"],
        "pairwise_cal_train_time_sec": pairwise_cal_training["pairwise_cal_train_time_sec"],
        "pairwise_cal_variant_history": {
            variant: {
                "history_csv": result["pairwise_cal_history_csv"],
                "candidate_csv": result["pairwise_cal_candidate_csv"],
                "best_val_loss": result["pairwise_cal_best_val_loss"],
            }
            for variant, result in pairwise_cal_variants.items()
        },
        "pairperm_candidate_csv": pairperm_training["pairperm_candidate_csv"],
        "pairperm_history_csv": pairperm_training["pairperm_history_csv"],
        "pairperm_best_val_loss": pairperm_training["pairperm_best_val_loss"],
        "pairperm_train_examples": pairperm_training["pairperm_train_examples"],
        "pairperm_val_examples": pairperm_training["pairperm_val_examples"],
        "pairperm_train_time_sec": pairperm_training["pairperm_train_time_sec"],
        "pairperm_variant_history": {
            variant: {
                "history_csv": result["pairperm_history_csv"],
                "candidate_csv": result["pairperm_candidate_csv"],
                "best_val_loss": result["pairperm_best_val_loss"],
            }
            for variant, result in pairperm_variants.items()
        },
        "escapecert_candidate_csv": escapecert_training["escapecert_candidate_csv"],
        "escapecert_history_csv": escapecert_training["escapecert_history_csv"],
        "escapecert_best_val_loss": escapecert_training["escapecert_best_val_loss"],
        "escapecert_train_examples": escapecert_training["escapecert_train_examples"],
        "escapecert_val_examples": escapecert_training["escapecert_val_examples"],
        "escapecert_train_anchors": escapecert_training["escapecert_train_anchors"],
        "escapecert_val_anchors": escapecert_training["escapecert_val_anchors"],
        "escapecert_train_time_sec": escapecert_training["escapecert_train_time_sec"],
        "escapecert_variant_history": {
            variant: {
                "history_csv": result["escapecert_history_csv"],
                "candidate_csv": result["escapecert_candidate_csv"],
                "best_val_loss": result["escapecert_best_val_loss"],
                "train_anchors": result["escapecert_train_anchors"],
                "val_anchors": result["escapecert_val_anchors"],
            }
            for variant, result in escapecert_variants.items()
        },
        "frontier_candidate_csv": frontier_training["frontier_candidate_csv"],
        "frontier_history_csv": frontier_training["frontier_history_csv"],
        "frontier_best_val_loss": frontier_training["frontier_best_val_loss"],
        "frontier_train_examples": frontier_training["frontier_train_examples"],
        "frontier_val_examples": frontier_training["frontier_val_examples"],
        "frontier_train_anchors": frontier_training["frontier_train_anchors"],
        "frontier_val_anchors": frontier_training["frontier_val_anchors"],
        "frontier_train_time_sec": frontier_training["frontier_train_time_sec"],
        "frontier_variant_history": {
            variant: {
                "history_csv": result["frontier_history_csv"],
                "candidate_csv": result["frontier_candidate_csv"],
                "best_val_loss": result["frontier_best_val_loss"],
                "train_anchors": result["frontier_train_anchors"],
                "val_anchors": result["frontier_val_anchors"],
            }
            for variant, result in frontier_variants.items()
        },
        "device": str(device),
        "gate_variant": "stage1_anchored_pairwise_permissioned_escapecert_frontier_with_teacher_guided_risk_controlled_admissible_frontier_learning",
    }
    summary_path = output_path / "gate_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def collect_gate_bank_training_data(
    *,
    config: GateLearningConfig,
    checkpoint_path: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    bank_names = gate_bank_candidate_names(prototype_candidates=config.bank_prototype_candidates)

    dataset_id = 0
    for scenario_idx, scenario in enumerate(config.train_scenarios):
        for seed_idx, seed in enumerate(config.train_seeds):
            n_cells = config.cell_options[(seed_idx + scenario_idx) % len(config.cell_options)]
            n_genes = config.gene_options[(2 * seed_idx + scenario_idx) % len(config.gene_options)]
            data = generate_synthetic_scrna(
                scenario=scenario,
                n_cells=n_cells,
                n_genes=n_genes,
                random_state=config.random_state + 37 * scenario_idx + seed,
            )

            selector = RefineMoEHVGSelector(
                top_k=min(config.top_k, data.counts.shape[1]),
                refine_epochs=6,
                random_state=config.random_state + seed,
                mode="full",
            )
            x, base_features, dataset_stats, expert_scores = selector.prepare_context(data.counts, data.batches)
            routed_gate = selector._dataset_gate(dataset_stats)
            heuristic_gate = _blend_heuristic_gate(routed_gate)
            point_components = _point_gate_components_from_runtime(
                dataset_stats=dataset_stats,
                checkpoint_path=checkpoint_path,
                heuristic_gate=heuristic_gate,
                device=choose_torch_device(),
            )
            candidate_names, candidate_gates, prototype_distances = _build_inference_gate_bank(
                dataset_stats=dataset_stats,
                heuristic_gate=heuristic_gate,
                residual_gate=np.asarray(point_components["residual_gate"], dtype=np.float64),
                stage1_gate=np.asarray(point_components["stage1_blended"], dtype=np.float64),
                prototype_gate=np.asarray(point_components["prototype_gate"], dtype=np.float64),
                prototype_neighbor_gates=np.asarray(point_components["prototype_neighbor_gates"], dtype=np.float64),
                prototype_neighbor_distances=np.asarray(point_components["prototype_neighbor_distances"], dtype=np.float64),
                runtime_config=point_components["runtime"]["config"],  # type: ignore[index]
            )
            proxy_context = _prepare_bank_proxy_context(
                x=x,
                batches=data.batches,
                random_state=config.random_state + seed,
            )

            candidate_records: list[dict[str, object]] = []
            top_k = min(config.top_k, data.counts.shape[1])
            for candidate_name, gate, prototype_distance in zip(
                candidate_names,
                candidate_gates,
                prototype_distances,
                strict=False,
            ):
                no_refine_scores = selector.score_with_context(
                    x=x,
                    base_features=base_features,
                    dataset_stats=dataset_stats,
                    expert_scores=expert_scores,
                    gate=gate,
                    apply_refine=False,
                )
                selected_no_refine = np.argsort(no_refine_scores)[-top_k:]
                boundary_size = min(len(no_refine_scores), max(top_k, int(1.5 * top_k)))
                boundary_idx = np.argsort(no_refine_scores)[-boundary_size:]
                route_metrics = _fast_weak_metrics(
                    counts=data.counts,
                    selected_genes=selected_no_refine,
                    labels=data.cell_types,
                    batches=data.batches,
                    random_state=config.random_state + seed,
                )
                route_reward = downstream_reward(route_metrics)
                feature_dict = _gate_bank_candidate_features(
                    candidate_name=str(candidate_name),
                    gate=gate,
                    score_vector=no_refine_scores,
                    selected_idx=selected_no_refine,
                    boundary_idx=boundary_idx,
                    dataset_stats=dataset_stats,
                    base_features=base_features,
                    x=x,
                    proxy_context=proxy_context,
                    heuristic_gate=heuristic_gate,
                    stage1_gate=np.asarray(point_components["stage1_blended"], dtype=np.float64),
                    prototype_distance=float(prototype_distance),
                    bank_names=bank_names,
                    random_state=config.random_state + seed,
                )
                refine_scores = selector.score_with_context(
                    x=x,
                    base_features=base_features,
                    dataset_stats=dataset_stats,
                    expert_scores=expert_scores,
                    gate=gate,
                    apply_refine=True,
                )
                selected = np.argsort(refine_scores)[-top_k:]
                final_metrics = _fast_weak_metrics(
                    counts=data.counts,
                    selected_genes=selected,
                    labels=data.cell_types,
                    batches=data.batches,
                    random_state=config.random_state + seed,
                )
                final_reward = downstream_reward(final_metrics)
                candidate_records.append(
                    {
                        "candidate_name": str(candidate_name),
                        "gate": np.asarray(gate, dtype=np.float64),
                        "route_metrics": route_metrics,
                        "route_reward": route_reward,
                        "metrics": final_metrics,
                        "reward": final_reward,
                        "features": feature_dict,
                    }
                )

            baseline_record, ari_floor, nmi_floor, structure_floor = _baseline_metric_floors(
                candidate_records,
                floor_ratio=config.floor_ratio,
                ari_margin=config.floor_margin_ari,
                nmi_margin=config.floor_margin_nmi,
                structure_margin=config.floor_margin_structure,
                baseline_names={"heuristic", "stage1_blended", "prototype_mean"},
            )
            baseline_metrics = baseline_record["metrics"]  # type: ignore[assignment]
            route_baseline_record, route_ari_floor, route_nmi_floor, route_structure_floor = _baseline_metric_floors(
                [
                    {
                        "candidate_name": str(record["candidate_name"]),
                        "gate": record["gate"],
                        "metrics": record["route_metrics"],
                        "reward": record["route_reward"],
                    }
                    for record in candidate_records
                ],
                floor_ratio=config.floor_ratio,
                ari_margin=config.floor_margin_ari,
                nmi_margin=config.floor_margin_nmi,
                structure_margin=config.floor_margin_structure,
                baseline_names={"heuristic", "stage1_blended", "prototype_mean"},
            )
            route_baseline_metrics = route_baseline_record["metrics"]  # type: ignore[assignment]
            batch_tradeoff_pressure = _batch_tradeoff_pressure(dataset_stats)
            adjusted_rewards: list[float] = []
            route_adjusted_rewards: list[float] = []
            safe_mask: list[bool] = []

            for record in candidate_records:
                final_metrics = record["metrics"]  # type: ignore[assignment]
                final_reward = float(record["reward"])
                adjusted_reward, _, _, _, _ = _adjust_reward_from_metrics(
                    metrics=final_metrics,
                    reward=final_reward,
                    ari_floor=ari_floor,
                    nmi_floor=nmi_floor,
                    structure_floor=structure_floor,
                    baseline_metrics=baseline_metrics,
                    batch_tradeoff_pressure=batch_tradeoff_pressure,
                    floor_weight=config.floor_weight,
                    tradeoff_penalty_weight=config.tradeoff_penalty_weight,
                )
                adjusted_rewards.append(adjusted_reward)
                route_metrics = record["route_metrics"]  # type: ignore[assignment]
                route_reward = float(record["route_reward"])
                route_adjusted_reward, _, _, _, _ = _adjust_reward_from_metrics(
                    metrics=route_metrics,
                    reward=route_reward,
                    ari_floor=route_ari_floor,
                    nmi_floor=route_nmi_floor,
                    structure_floor=route_structure_floor,
                    baseline_metrics=route_baseline_metrics,
                    batch_tradeoff_pressure=batch_tradeoff_pressure,
                    floor_weight=config.floor_weight,
                    tradeoff_penalty_weight=config.tradeoff_penalty_weight,
                )
                route_adjusted_rewards.append(route_adjusted_reward)
                final_ari_gap = max(0.0, ari_floor - float(final_metrics["ari"]))
                final_nmi_gap = max(0.0, nmi_floor - float(final_metrics["nmi"]))
                final_structure_gap = max(0.0, structure_floor - structure_preservation_score(final_metrics))
                safe_mask.append(final_ari_gap <= 1e-8 and final_nmi_gap <= 1e-8 and final_structure_gap <= 1e-8)

            for record, adjusted_reward, route_adjusted_reward in zip(
                candidate_records,
                adjusted_rewards,
                route_adjusted_rewards,
                strict=False,
            ):
                record["adjusted_reward"] = adjusted_reward
                record["route_adjusted_reward"] = route_adjusted_reward

            stage1_record = next(
                record for record in candidate_records if str(record["candidate_name"]) == "stage1_blended"
            )
            stage1_final_adjusted = float(stage1_record["adjusted_reward"])
            stage1_route_adjusted = float(stage1_record["route_adjusted_reward"])
            route_view_stats = _candidate_route_view_statistics(
                counts=data.counts,
                batches=data.batches,
                labels=data.cell_types,
                selector=selector,
                candidate_names=np.asarray(candidate_names),
                candidate_gates=np.asarray(candidate_gates, dtype=np.float64),
                config=config,
                random_state=config.random_state + seed + 17 * scenario_idx,
            )

            for candidate_order, record in enumerate(candidate_records):
                metrics = record["metrics"]  # type: ignore[assignment]
                route_metrics = record["route_metrics"]  # type: ignore[assignment]
                reward = float(record["reward"])
                route_reward = float(record["route_reward"])
                adjusted_reward = float(record["adjusted_reward"])
                route_adjusted_reward = float(record["route_adjusted_reward"])
                ari_gap = max(0.0, ari_floor - float(metrics["ari"]))
                nmi_gap = max(0.0, nmi_floor - float(metrics["nmi"]))
                structure_gap = max(0.0, structure_floor - structure_preservation_score(metrics))
                mixing_gain = max(0.0, float(metrics.get("batch_mixing", 0.0)) - float(baseline_metrics.get("batch_mixing", 0.0)))
                tradeoff_penalty = batch_tradeoff_pressure * mixing_gain * (
                    1.25 * ari_gap + 1.05 * nmi_gap + 0.85 * structure_gap
                )
                route_utility_target = float(np.clip(route_reward, 0.0, 1.0))
                route_risk_target = float(np.clip(stage1_route_adjusted - route_adjusted_reward, 0.0, 1.0))
                route_regret_target = float(np.clip(route_adjusted_reward - stage1_route_adjusted, 0.0, 1.0))
                refine_target = float(np.clip(adjusted_reward - route_adjusted_reward, -1.0, 1.0))
                is_safe = bool(safe_mask[candidate_order])

                row: dict[str, float | int | str] = {
                    "dataset_id": dataset_id,
                    "scenario": scenario,
                    "seed": seed,
                    "n_cells": int(n_cells),
                    "n_genes": int(n_genes),
                    "candidate_order": candidate_order,
                    "candidate_name": str(record["candidate_name"]),
                    "reward": reward,
                    "adjusted_reward": adjusted_reward,
                    "is_safe_candidate": int(is_safe),
                    "route_reward": route_reward,
                    "route_adjusted_reward": route_adjusted_reward,
                    "route_ari": float(route_metrics["ari"]),
                    "route_nmi": float(route_metrics["nmi"]),
                    "route_batch_mixing": float(route_metrics["batch_mixing"]),
                    "route_neighbor_preservation": float(route_metrics["neighbor_preservation"]),
                    "route_cluster_silhouette": float(route_metrics["cluster_silhouette"]),
                    "route_label_silhouette": float(route_metrics["label_silhouette"]),
                    "route_structure_score": structure_preservation_score(route_metrics),
                    "route_utility_target": route_utility_target,
                    "risk_target": route_risk_target,
                    "regret_target": route_regret_target,
                    "refine_target": refine_target,
                    "stage1_route_adjusted": stage1_route_adjusted,
                    "stage1_final_adjusted": stage1_final_adjusted,
                    "route_margin_vs_stage1": route_adjusted_reward - stage1_route_adjusted,
                    "final_margin_vs_stage1": adjusted_reward - stage1_final_adjusted,
                    "ari": float(metrics["ari"]),
                    "nmi": float(metrics["nmi"]),
                    "batch_mixing": float(metrics["batch_mixing"]),
                    "neighbor_preservation": float(metrics["neighbor_preservation"]),
                    "cluster_silhouette": float(metrics["cluster_silhouette"]),
                    "label_silhouette": float(metrics["label_silhouette"]),
                    "structure_score": structure_preservation_score(metrics),
                }
                row.update(route_view_stats.get(str(record["candidate_name"]), {}))
                row.update(record["features"])  # type: ignore[arg-type]
                rows.append(row)
            dataset_id += 1

    return pd.DataFrame(rows)


def _bank_feature_columns(*, prototype_candidates: int) -> list[str]:
    return (
        [f"stat_{key}" for key in DEFAULT_GATE_FEATURE_KEYS]
        + [f"gate_{expert}" for expert in EXPERT_NAMES]
        + list(BANK_PROXY_FEATURE_KEYS)
        + [f"bank_id_{name}" for name in gate_bank_candidate_names(prototype_candidates=prototype_candidates)]
    )


def _listwise_soft_ce(
    *,
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_probs * log_probs).sum(dim=1)
    return (loss * weights).mean()


def _evaluate_bank_loss(
    *,
    model: nn.Module,
    x_val: np.ndarray,
    target_val: np.ndarray,
    weight_val: np.ndarray,
    device: torch.device,
) -> float:
    if len(x_val) == 0:
        return 0.0

    model.eval()
    with torch.no_grad():
        batch_x = torch.from_numpy(x_val).to(device, non_blocking=torch.cuda.is_available())
        batch_target = torch.from_numpy(target_val).to(device, non_blocking=torch.cuda.is_available())
        batch_weight = torch.from_numpy(weight_val).to(device, non_blocking=torch.cuda.is_available())
        logits = model(batch_x)
        loss = _listwise_soft_ce(logits=logits, target_probs=batch_target, weights=batch_weight)
    return float(loss.detach().cpu())


def _train_gate_bank_model(
    *,
    output_dir: Path,
    checkpoint_path: str,
    config: GateLearningConfig,
) -> dict[str, object]:
    train_start = time.perf_counter()
    bank_df = collect_gate_bank_training_data(config=config, checkpoint_path=checkpoint_path)
    bank_csv = output_dir / "gate_bank_candidates.csv"
    bank_df.to_csv(bank_csv, index=False)

    feature_columns = _bank_feature_columns(prototype_candidates=config.bank_prototype_candidates)
    bank_size = len(gate_bank_candidate_names(prototype_candidates=config.bank_prototype_candidates))
    grouped_count = bank_df.groupby("dataset_id").size().to_numpy()
    if not np.all(grouped_count == bank_size):
        raise ValueError("Gate bank candidate count is not consistent across datasets.")

    feature_matrix = bank_df[feature_columns].to_numpy(dtype=np.float32).reshape(-1, bank_size, len(feature_columns))
    reward_matrix = bank_df["adjusted_reward"].to_numpy(dtype=np.float32).reshape(-1, bank_size)
    safe_matrix = bank_df["is_safe_candidate"].to_numpy(dtype=np.float32).reshape(-1, bank_size)
    batch_strength = bank_df["stat_batch_strength"].to_numpy(dtype=np.float32).reshape(-1, bank_size)[:, 0]
    reward_margin = reward_matrix.max(axis=1) - reward_matrix.mean(axis=1)
    dataset_weights = np.maximum(0.15 + reward_margin * (1.0 + batch_strength), 0.10).astype(np.float32)

    target_probs = np.zeros_like(reward_matrix, dtype=np.float32)
    for idx in range(len(target_probs)):
        safe_idx = np.flatnonzero(safe_matrix[idx] > 0.5)
        candidate_idx = safe_idx if len(safe_idx) else np.arange(bank_size)
        candidate_probs = _softmax_np(reward_matrix[idx, candidate_idx] / max(config.bank_temperature, 1e-6))
        target_probs[idx, candidate_idx] = candidate_probs.astype(np.float32)

    rng = np.random.default_rng(config.random_state + 911)
    indices = rng.permutation(len(feature_matrix))
    val_size = 0
    if len(indices) >= 4:
        val_size = max(1, int(round(len(indices) * config.val_fraction)))
        val_size = min(val_size, len(indices) - 1)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        train_idx = indices
        val_idx = np.asarray([], dtype=np.int64)

    feature_mean = feature_matrix[train_idx].reshape(-1, feature_matrix.shape[-1]).mean(axis=0)
    feature_std = feature_matrix[train_idx].reshape(-1, feature_matrix.shape[-1]).std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    x_train = ((feature_matrix[train_idx] - feature_mean[None, None, :]) / feature_std[None, None, :]).astype(np.float32)
    y_train = target_probs[train_idx].astype(np.float32)
    w_train = dataset_weights[train_idx].astype(np.float32)
    x_val = (
        ((feature_matrix[val_idx] - feature_mean[None, None, :]) / feature_std[None, None, :]).astype(np.float32)
        if len(val_idx)
        else np.zeros((0, bank_size, feature_matrix.shape[-1]), dtype=np.float32)
    )
    y_val = target_probs[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0, bank_size), dtype=np.float32)
    w_val = dataset_weights[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)

    device = choose_torch_device()
    base_model = GateBankScorer(
        input_dim=feature_matrix.shape[-1],
        hidden_dim=config.bank_hidden_dim,
        dropout=config.bank_dropout,
    )
    if torch.cuda.is_available():
        base_model = base_model.to(device)
        model: nn.Module = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model
    else:
        base_model = base_model.to(device)
        model = base_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.bank_learning_rate, weight_decay=config.bank_weight_decay)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(w_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False,
    )

    best_state = copy.deepcopy(_unwrap_model(model).state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(config.bank_train_epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device, non_blocking=torch.cuda.is_available())
            batch_y = batch_y.to(device, non_blocking=torch.cuda.is_available())
            batch_w = batch_w.to(device, non_blocking=torch.cuda.is_available())

            logits = model(batch_x)
            loss = _listwise_soft_ce(logits=logits, target_probs=batch_y, weights=batch_w)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu()) * len(batch_x)
            running_count += len(batch_x)

        train_loss = running_loss / max(running_count, 1)
        val_loss = _evaluate_bank_loss(
            model=model,
            x_val=x_val,
            target_val=y_val,
            weight_val=w_val,
            device=device,
        )
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(_unwrap_model(model).state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= config.bank_patience:
            break

    _unwrap_model(model).load_state_dict(best_state)
    history_csv = output_dir / "gate_bank_training_history.csv"
    pd.DataFrame(history_rows).to_csv(history_csv, index=False)
    train_elapsed = time.perf_counter() - train_start
    return {
        "bank_candidate_csv": str(bank_csv),
        "bank_history_csv": str(history_csv),
        "bank_model_state": best_state,
        "bank_feature_keys": feature_columns,
        "bank_feature_mean": feature_mean.astype(np.float32),
        "bank_feature_std": feature_std.astype(np.float32),
        "bank_hidden_dim": config.bank_hidden_dim,
        "bank_dropout": config.bank_dropout,
        "bank_best_val_loss": float(best_val_loss if np.isfinite(best_val_loss) else history_rows[-1]["train_loss"]),
        "bank_train_examples": int(len(train_idx)),
        "bank_val_examples": int(len(val_idx)),
        "bank_candidate_names": list(gate_bank_candidate_names(prototype_candidates=config.bank_prototype_candidates)),
        "bank_train_time_sec": train_elapsed,
    }


def _curr_policy_score_np(
    *,
    utility: np.ndarray,
    risk: np.ndarray,
    regret: np.ndarray,
    config: GateLearningConfig,
) -> np.ndarray:
    return utility - config.curr_risk_weight * risk + config.curr_regret_weight * regret


def _curr_policy_score_torch(
    *,
    utility: torch.Tensor,
    risk: torch.Tensor,
    regret: torch.Tensor,
    config: GateLearningConfig,
) -> torch.Tensor:
    return utility - config.curr_risk_weight * risk + config.curr_regret_weight * regret


def _pairwise_decision_score_np(
    *,
    utility: np.ndarray,
    risk: np.ndarray,
    regret: np.ndarray,
    route_margin: np.ndarray,
    config: GateLearningConfig,
    use_risk: bool = True,
    use_regret: bool = True,
    use_margin_term: bool = True,
) -> np.ndarray:
    score = np.asarray(utility, dtype=np.float64).copy()
    if use_regret:
        score += config.pairwise_regret_weight * np.asarray(regret, dtype=np.float64)
    if use_risk:
        score -= config.pairwise_risk_weight * np.asarray(risk, dtype=np.float64)
    if use_margin_term:
        score += config.pairwise_margin_weight * np.asarray(route_margin, dtype=np.float64)
    return score


def _pairwise_decision_score_torch(
    *,
    utility: torch.Tensor,
    risk: torch.Tensor,
    regret: torch.Tensor,
    route_margin: torch.Tensor,
    config: GateLearningConfig,
) -> torch.Tensor:
    return (
        utility
        + config.pairwise_regret_weight * regret
        - config.pairwise_risk_weight * risk
        + config.pairwise_margin_weight * route_margin
    )


def _counterfactual_policy_objective(
    *,
    utility_pred: torch.Tensor,
    risk_pred: torch.Tensor,
    regret_pred: torch.Tensor,
    refine_pred: torch.Tensor,
    utility_target: torch.Tensor,
    risk_target: torch.Tensor,
    regret_target: torch.Tensor,
    refine_target: torch.Tensor,
    policy_target: torch.Tensor,
    weights: torch.Tensor,
    config: GateLearningConfig,
) -> torch.Tensor:
    regression = (
        F.smooth_l1_loss(utility_pred, utility_target, reduction="none")
        + F.smooth_l1_loss(risk_pred, risk_target, reduction="none")
        + F.smooth_l1_loss(regret_pred, regret_target, reduction="none")
        + config.curr_refine_weight * F.smooth_l1_loss(refine_pred, refine_target, reduction="none")
    ).mean(dim=1)
    policy_logits = _curr_policy_score_torch(
        utility=utility_pred,
        risk=risk_pred,
        regret=regret_pred,
        config=config,
    )
    ranking = -(policy_target * F.log_softmax(policy_logits, dim=1)).sum(dim=1)
    total = config.curr_regression_weight * regression + config.curr_ranking_weight * ranking
    return (total * weights).mean()


def _evaluate_curr_loss(
    *,
    model: nn.Module,
    x_val: np.ndarray,
    utility_val: np.ndarray,
    risk_val: np.ndarray,
    regret_val: np.ndarray,
    refine_val: np.ndarray,
    policy_val: np.ndarray,
    weight_val: np.ndarray,
    device: torch.device,
    config: GateLearningConfig,
) -> float:
    if len(x_val) == 0:
        return 0.0

    model.eval()
    with torch.no_grad():
        batch_x = torch.from_numpy(x_val).to(device, non_blocking=torch.cuda.is_available())
        batch_utility = torch.from_numpy(utility_val).to(device, non_blocking=torch.cuda.is_available())
        batch_risk = torch.from_numpy(risk_val).to(device, non_blocking=torch.cuda.is_available())
        batch_regret = torch.from_numpy(regret_val).to(device, non_blocking=torch.cuda.is_available())
        batch_refine = torch.from_numpy(refine_val).to(device, non_blocking=torch.cuda.is_available())
        batch_policy = torch.from_numpy(policy_val).to(device, non_blocking=torch.cuda.is_available())
        batch_weight = torch.from_numpy(weight_val).to(device, non_blocking=torch.cuda.is_available())
        utility_pred, risk_pred, regret_pred, refine_pred = model(batch_x)
        loss = _counterfactual_policy_objective(
            utility_pred=utility_pred,
            risk_pred=risk_pred,
            regret_pred=regret_pred,
            refine_pred=refine_pred,
            utility_target=batch_utility,
            risk_target=batch_risk,
            regret_target=batch_regret,
            refine_target=batch_refine,
            policy_target=batch_policy,
            weights=batch_weight,
            config=config,
        )
    return float(loss.detach().cpu())


def _train_counterfactual_tri_factor_model(
    *,
    output_dir: Path,
    config: GateLearningConfig,
) -> dict[str, object]:
    bank_csv = output_dir / "gate_bank_candidates.csv"
    bank_df = pd.read_csv(bank_csv)
    feature_columns = _bank_feature_columns(prototype_candidates=config.bank_prototype_candidates)
    bank_size = len(gate_bank_candidate_names(prototype_candidates=config.bank_prototype_candidates))

    grouped_count = bank_df.groupby("dataset_id").size().to_numpy()
    if not np.all(grouped_count == bank_size):
        raise ValueError("Counterfactual tri-factor training requires fixed gate bank candidate counts.")

    feature_matrix = bank_df[feature_columns].to_numpy(dtype=np.float32).reshape(-1, bank_size, len(feature_columns))
    utility_target = bank_df["route_utility_target"].to_numpy(dtype=np.float32).reshape(-1, bank_size)
    risk_target = bank_df["risk_target"].to_numpy(dtype=np.float32).reshape(-1, bank_size)
    regret_target = bank_df["regret_target"].to_numpy(dtype=np.float32).reshape(-1, bank_size)
    refine_target = bank_df["refine_target"].to_numpy(dtype=np.float32).reshape(-1, bank_size)
    batch_strength = bank_df["stat_batch_strength"].to_numpy(dtype=np.float32).reshape(-1, bank_size)[:, 0]
    target_scores = _curr_policy_score_np(
        utility=utility_target,
        risk=risk_target,
        regret=regret_target,
        config=config,
    )
    score_margin = target_scores.max(axis=1) - target_scores.mean(axis=1)
    dataset_weights = np.maximum(0.15 + score_margin * (1.0 + batch_strength), 0.10).astype(np.float32)

    policy_target = np.zeros_like(target_scores, dtype=np.float32)
    for idx in range(len(policy_target)):
        policy_target[idx] = _softmax_np(target_scores[idx] / max(config.curr_policy_temperature, 1e-6)).astype(np.float32)

    rng = np.random.default_rng(config.random_state + 1917)
    indices = rng.permutation(len(feature_matrix))
    val_size = 0
    if len(indices) >= 4:
        val_size = max(1, int(round(len(indices) * config.val_fraction)))
        val_size = min(val_size, len(indices) - 1)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        train_idx = indices
        val_idx = np.asarray([], dtype=np.int64)

    feature_mean = feature_matrix[train_idx].reshape(-1, feature_matrix.shape[-1]).mean(axis=0)
    feature_std = feature_matrix[train_idx].reshape(-1, feature_matrix.shape[-1]).std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    x_train = ((feature_matrix[train_idx] - feature_mean[None, None, :]) / feature_std[None, None, :]).astype(np.float32)
    utility_train = utility_target[train_idx].astype(np.float32)
    risk_train = risk_target[train_idx].astype(np.float32)
    regret_train = regret_target[train_idx].astype(np.float32)
    refine_train = refine_target[train_idx].astype(np.float32)
    policy_train = policy_target[train_idx].astype(np.float32)
    weight_train = dataset_weights[train_idx].astype(np.float32)

    x_val = (
        ((feature_matrix[val_idx] - feature_mean[None, None, :]) / feature_std[None, None, :]).astype(np.float32)
        if len(val_idx)
        else np.zeros((0, bank_size, feature_matrix.shape[-1]), dtype=np.float32)
    )
    utility_val = utility_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0, bank_size), dtype=np.float32)
    risk_val = risk_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0, bank_size), dtype=np.float32)
    regret_val = regret_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0, bank_size), dtype=np.float32)
    refine_val = refine_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0, bank_size), dtype=np.float32)
    policy_val = policy_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0, bank_size), dtype=np.float32)
    weight_val = dataset_weights[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)

    device = choose_torch_device()
    base_model = CounterfactualTriFactorPolicy(
        input_dim=feature_matrix.shape[-1],
        hidden_dim=config.curr_hidden_dim,
        dropout=config.curr_dropout,
    )
    if torch.cuda.is_available():
        base_model = base_model.to(device)
        model: nn.Module = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model
    else:
        base_model = base_model.to(device)
        model = base_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.curr_learning_rate, weight_decay=config.curr_weight_decay)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(utility_train),
        torch.from_numpy(risk_train),
        torch.from_numpy(regret_train),
        torch.from_numpy(refine_train),
        torch.from_numpy(policy_train),
        torch.from_numpy(weight_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False,
    )

    best_state = copy.deepcopy(_unwrap_model(model).state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0
    history_rows: list[dict[str, float | int]] = []
    train_start = time.perf_counter()

    for epoch in range(config.curr_train_epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        for batch_x, batch_utility, batch_risk, batch_regret, batch_refine, batch_policy, batch_weight in train_loader:
            batch_x = batch_x.to(device, non_blocking=torch.cuda.is_available())
            batch_utility = batch_utility.to(device, non_blocking=torch.cuda.is_available())
            batch_risk = batch_risk.to(device, non_blocking=torch.cuda.is_available())
            batch_regret = batch_regret.to(device, non_blocking=torch.cuda.is_available())
            batch_refine = batch_refine.to(device, non_blocking=torch.cuda.is_available())
            batch_policy = batch_policy.to(device, non_blocking=torch.cuda.is_available())
            batch_weight = batch_weight.to(device, non_blocking=torch.cuda.is_available())

            utility_pred, risk_pred, regret_pred, refine_pred = model(batch_x)
            loss = _counterfactual_policy_objective(
                utility_pred=utility_pred,
                risk_pred=risk_pred,
                regret_pred=regret_pred,
                refine_pred=refine_pred,
                utility_target=batch_utility,
                risk_target=batch_risk,
                regret_target=batch_regret,
                refine_target=batch_refine,
                policy_target=batch_policy,
                weights=batch_weight,
                config=config,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu()) * len(batch_x)
            running_count += len(batch_x)

        train_loss = running_loss / max(running_count, 1)
        val_loss = _evaluate_curr_loss(
            model=model,
            x_val=x_val,
            utility_val=utility_val,
            risk_val=risk_val,
            regret_val=regret_val,
            refine_val=refine_val,
            policy_val=policy_val,
            weight_val=weight_val,
            device=device,
            config=config,
        )
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(_unwrap_model(model).state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= config.curr_patience:
            break

    train_elapsed = time.perf_counter() - train_start
    history_csv = output_dir / "gate_curr_training_history.csv"
    pd.DataFrame(history_rows).to_csv(history_csv, index=False)
    return {
        "curr_model_state": best_state,
        "curr_feature_keys": feature_columns,
        "curr_feature_mean": feature_mean.astype(np.float32),
        "curr_feature_std": feature_std.astype(np.float32),
        "curr_hidden_dim": config.curr_hidden_dim,
        "curr_dropout": config.curr_dropout,
        "curr_best_val_loss": float(best_val_loss if np.isfinite(best_val_loss) else history_rows[-1]["train_loss"]),
        "curr_train_examples": int(len(train_idx)),
        "curr_val_examples": int(len(val_idx)),
        "curr_train_time_sec": train_elapsed,
        "curr_history_csv": str(history_csv),
    }


def _pairwise_router_objective(
    *,
    utility_pred: torch.Tensor,
    risk_pred: torch.Tensor,
    regret_pred: torch.Tensor,
    margin_pred: torch.Tensor,
    utility_target: torch.Tensor,
    risk_target: torch.Tensor,
    regret_target: torch.Tensor,
    margin_target: torch.Tensor,
    route_win_target: torch.Tensor,
    weights: torch.Tensor,
    config: GateLearningConfig,
) -> torch.Tensor:
    regression = (
        F.smooth_l1_loss(utility_pred, utility_target, reduction="none")
        + F.smooth_l1_loss(risk_pred, risk_target, reduction="none")
        + F.smooth_l1_loss(regret_pred, regret_target, reduction="none")
        + F.smooth_l1_loss(margin_pred, margin_target, reduction="none")
    )
    decision_score = _pairwise_decision_score_torch(
        utility=utility_pred,
        risk=risk_pred,
        regret=regret_pred,
        route_margin=margin_pred,
        config=config,
    )
    counterfactual_logits = 4.0 * (decision_score - config.pairwise_route_threshold)
    counterfactual = F.binary_cross_entropy_with_logits(
        counterfactual_logits,
        route_win_target,
        reduction="none",
    )
    total = config.pairwise_regression_weight * regression + config.pairwise_counterfactual_weight * counterfactual
    return (total * weights).mean()


def _evaluate_pairwise_router_loss(
    *,
    model: nn.Module,
    x_val: np.ndarray,
    utility_val: np.ndarray,
    risk_val: np.ndarray,
    regret_val: np.ndarray,
    margin_val: np.ndarray,
    route_win_val: np.ndarray,
    weight_val: np.ndarray,
    device: torch.device,
    config: GateLearningConfig,
) -> float:
    if len(x_val) == 0:
        return 0.0

    model.eval()
    with torch.no_grad():
        batch_x = torch.from_numpy(x_val).to(device, non_blocking=torch.cuda.is_available())
        batch_utility = torch.from_numpy(utility_val).to(device, non_blocking=torch.cuda.is_available())
        batch_risk = torch.from_numpy(risk_val).to(device, non_blocking=torch.cuda.is_available())
        batch_regret = torch.from_numpy(regret_val).to(device, non_blocking=torch.cuda.is_available())
        batch_margin = torch.from_numpy(margin_val).to(device, non_blocking=torch.cuda.is_available())
        batch_route_win = torch.from_numpy(route_win_val).to(device, non_blocking=torch.cuda.is_available())
        batch_weight = torch.from_numpy(weight_val).to(device, non_blocking=torch.cuda.is_available())
        utility_pred, risk_pred, regret_pred, margin_pred = model(batch_x)
        loss = _pairwise_router_objective(
            utility_pred=utility_pred,
            risk_pred=risk_pred,
            regret_pred=regret_pred,
            margin_pred=margin_pred,
            utility_target=batch_utility,
            risk_target=batch_risk,
            regret_target=batch_regret,
            margin_target=batch_margin,
            route_win_target=batch_route_win,
            weights=batch_weight,
            config=config,
        )
    return float(loss.detach().cpu())


def _train_stage1_pairwise_router(
    *,
    output_dir: Path,
    config: GateLearningConfig,
) -> dict[str, object]:
    bank_csv = output_dir / "gate_bank_candidates.csv"
    bank_df = pd.read_csv(bank_csv)
    pairwise_df = _build_stage1_pairwise_training_frame(bank_df=bank_df, config=config)
    pairwise_csv = output_dir / "gate_pairwise_candidates.csv"
    pairwise_df.to_csv(pairwise_csv, index=False)

    feature_columns = _pairwise_feature_columns(prototype_candidates=config.bank_prototype_candidates)
    feature_matrix = pairwise_df[feature_columns].to_numpy(dtype=np.float32)
    utility_target = pairwise_df["pairwise_utility_gain_target"].to_numpy(dtype=np.float32)
    risk_target = pairwise_df["pairwise_failure_risk_target"].to_numpy(dtype=np.float32)
    regret_target = pairwise_df["pairwise_fallback_regret_target"].to_numpy(dtype=np.float32)
    margin_target = pairwise_df["pairwise_route_margin_target"].to_numpy(dtype=np.float32)
    route_win_target = pairwise_df["pairwise_route_win_target"].to_numpy(dtype=np.float32)
    sample_weight = pairwise_df["pairwise_sample_weight"].to_numpy(dtype=np.float32)

    rng = np.random.default_rng(config.random_state + 2917)
    indices = rng.permutation(len(feature_matrix))
    val_size = 0
    if len(indices) >= 8:
        val_size = max(1, int(round(len(indices) * config.val_fraction)))
        val_size = min(val_size, len(indices) - 1)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        train_idx = indices
        val_idx = np.asarray([], dtype=np.int64)

    feature_mean = feature_matrix[train_idx].mean(axis=0)
    feature_std = feature_matrix[train_idx].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    x_train = ((feature_matrix[train_idx] - feature_mean[None, :]) / feature_std[None, :]).astype(np.float32)
    utility_train = utility_target[train_idx].astype(np.float32)
    risk_train = risk_target[train_idx].astype(np.float32)
    regret_train = regret_target[train_idx].astype(np.float32)
    margin_train = margin_target[train_idx].astype(np.float32)
    route_win_train = route_win_target[train_idx].astype(np.float32)
    weight_train = sample_weight[train_idx].astype(np.float32)

    x_val = (
        ((feature_matrix[val_idx] - feature_mean[None, :]) / feature_std[None, :]).astype(np.float32)
        if len(val_idx)
        else np.zeros((0, feature_matrix.shape[-1]), dtype=np.float32)
    )
    utility_val = utility_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    risk_val = risk_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    regret_val = regret_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    margin_val = margin_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    route_win_val = route_win_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    weight_val = sample_weight[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)

    device = choose_torch_device()
    base_model = Stage1AnchoredPairwiseRouter(
        input_dim=feature_matrix.shape[-1],
        hidden_dim=config.pairwise_hidden_dim,
        dropout=config.pairwise_dropout,
    )
    if torch.cuda.is_available():
        base_model = base_model.to(device)
        model: nn.Module = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model
    else:
        base_model = base_model.to(device)
        model = base_model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pairwise_learning_rate,
        weight_decay=config.pairwise_weight_decay,
    )
    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(utility_train),
        torch.from_numpy(risk_train),
        torch.from_numpy(regret_train),
        torch.from_numpy(margin_train),
        torch.from_numpy(route_win_train),
        torch.from_numpy(weight_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size * 2, len(train_dataset)),
        shuffle=True,
        drop_last=False,
    )

    best_state = copy.deepcopy(_unwrap_model(model).state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0
    history_rows: list[dict[str, float | int]] = []
    train_start = time.perf_counter()

    for epoch in range(config.pairwise_train_epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        for batch_x, batch_utility, batch_risk, batch_regret, batch_margin, batch_route_win, batch_weight in train_loader:
            batch_x = batch_x.to(device, non_blocking=torch.cuda.is_available())
            batch_utility = batch_utility.to(device, non_blocking=torch.cuda.is_available())
            batch_risk = batch_risk.to(device, non_blocking=torch.cuda.is_available())
            batch_regret = batch_regret.to(device, non_blocking=torch.cuda.is_available())
            batch_margin = batch_margin.to(device, non_blocking=torch.cuda.is_available())
            batch_route_win = batch_route_win.to(device, non_blocking=torch.cuda.is_available())
            batch_weight = batch_weight.to(device, non_blocking=torch.cuda.is_available())

            utility_pred, risk_pred, regret_pred, margin_pred = model(batch_x)
            loss = _pairwise_router_objective(
                utility_pred=utility_pred,
                risk_pred=risk_pred,
                regret_pred=regret_pred,
                margin_pred=margin_pred,
                utility_target=batch_utility,
                risk_target=batch_risk,
                regret_target=batch_regret,
                margin_target=batch_margin,
                route_win_target=batch_route_win,
                weights=batch_weight,
                config=config,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu()) * len(batch_x)
            running_count += len(batch_x)

        train_loss = running_loss / max(running_count, 1)
        val_loss = _evaluate_pairwise_router_loss(
            model=model,
            x_val=x_val,
            utility_val=utility_val,
            risk_val=risk_val,
            regret_val=regret_val,
            margin_val=margin_val,
            route_win_val=route_win_val,
            weight_val=weight_val,
            device=device,
            config=config,
        )
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(_unwrap_model(model).state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= config.pairwise_patience:
            break

    train_elapsed = time.perf_counter() - train_start
    history_csv = output_dir / "gate_pairwise_training_history.csv"
    pd.DataFrame(history_rows).to_csv(history_csv, index=False)
    return {
        "pairwise_candidate_csv": str(pairwise_csv),
        "pairwise_history_csv": str(history_csv),
        "pairwise_model_state": best_state,
        "pairwise_feature_keys": feature_columns,
        "pairwise_feature_mean": feature_mean.astype(np.float32),
        "pairwise_feature_std": feature_std.astype(np.float32),
        "pairwise_hidden_dim": config.pairwise_hidden_dim,
        "pairwise_dropout": config.pairwise_dropout,
        "pairwise_best_val_loss": float(best_val_loss if np.isfinite(best_val_loss) else history_rows[-1]["train_loss"]),
        "pairwise_train_examples": int(len(train_idx)),
        "pairwise_val_examples": int(len(val_idx)),
        "pairwise_train_time_sec": train_elapsed,
    }


def _pairwise_calibrated_router_objective(
    *,
    utility_pred: torch.Tensor,
    risk_pred: torch.Tensor,
    regret_pred: torch.Tensor,
    margin_pred: torch.Tensor,
    utility_target: torch.Tensor,
    risk_target: torch.Tensor,
    regret_target: torch.Tensor,
    margin_target: torch.Tensor,
    route_win_target: torch.Tensor,
    regret_weight: torch.Tensor,
    constraint_score: torch.Tensor,
    weights: torch.Tensor,
    config: GateLearningConfig,
) -> torch.Tensor:
    utility_loss = F.smooth_l1_loss(utility_pred, utility_target, reduction="none")
    risk_loss = F.smooth_l1_loss(risk_pred, risk_target, reduction="none")
    regret_loss = regret_weight * F.smooth_l1_loss(regret_pred, regret_target, reduction="none")
    margin_loss = (0.35 + 0.65 * constraint_score) * F.smooth_l1_loss(margin_pred, margin_target, reduction="none")
    regression = utility_loss + risk_loss + regret_loss + margin_loss

    decision_score = _pairwise_decision_score_torch(
        utility=utility_pred,
        risk=risk_pred,
        regret=regret_pred,
        route_margin=margin_pred,
        config=config,
    )
    counterfactual_logits = 4.0 * (decision_score - config.pairwise_route_threshold)
    counterfactual = F.binary_cross_entropy_with_logits(
        counterfactual_logits,
        route_win_target,
        reduction="none",
    )
    counterfactual = (0.25 + 0.75 * constraint_score) * counterfactual
    total = config.pairwise_regression_weight * regression + config.pairwise_counterfactual_weight * counterfactual
    return (total * weights).mean()


def _evaluate_pairwise_calibrated_router_loss(
    *,
    model: nn.Module,
    x_val: np.ndarray,
    utility_val: np.ndarray,
    risk_val: np.ndarray,
    regret_val: np.ndarray,
    margin_val: np.ndarray,
    route_win_val: np.ndarray,
    regret_weight_val: np.ndarray,
    constraint_score_val: np.ndarray,
    weight_val: np.ndarray,
    device: torch.device,
    config: GateLearningConfig,
) -> float:
    if len(x_val) == 0:
        return 0.0

    model.eval()
    with torch.no_grad():
        batch_x = torch.from_numpy(x_val).to(device, non_blocking=torch.cuda.is_available())
        batch_utility = torch.from_numpy(utility_val).to(device, non_blocking=torch.cuda.is_available())
        batch_risk = torch.from_numpy(risk_val).to(device, non_blocking=torch.cuda.is_available())
        batch_regret = torch.from_numpy(regret_val).to(device, non_blocking=torch.cuda.is_available())
        batch_margin = torch.from_numpy(margin_val).to(device, non_blocking=torch.cuda.is_available())
        batch_route_win = torch.from_numpy(route_win_val).to(device, non_blocking=torch.cuda.is_available())
        batch_regret_weight = torch.from_numpy(regret_weight_val).to(device, non_blocking=torch.cuda.is_available())
        batch_constraint_score = torch.from_numpy(constraint_score_val).to(
            device,
            non_blocking=torch.cuda.is_available(),
        )
        batch_weight = torch.from_numpy(weight_val).to(device, non_blocking=torch.cuda.is_available())
        utility_pred, risk_pred, regret_pred, margin_pred = model(batch_x)
        loss = _pairwise_calibrated_router_objective(
            utility_pred=utility_pred,
            risk_pred=risk_pred,
            regret_pred=regret_pred,
            margin_pred=margin_pred,
            utility_target=batch_utility,
            risk_target=batch_risk,
            regret_target=batch_regret,
            margin_target=batch_margin,
            route_win_target=batch_route_win,
            regret_weight=batch_regret_weight,
            constraint_score=batch_constraint_score,
            weights=batch_weight,
            config=config,
        )
    return float(loss.detach().cpu())


def _train_stage1_pairwise_router_calibrated(
    *,
    output_dir: Path,
    config: GateLearningConfig,
    variant: str = "full",
) -> dict[str, object]:
    bank_csv = output_dir / "gate_bank_candidates.csv"
    bank_df = pd.read_csv(bank_csv)
    pairwise_df = _build_stage1_pairwise_calibrated_training_frame(bank_df=bank_df, config=config, variant=variant)
    variant_suffix = variant.replace("no_", "no-").replace("_", "-")
    pairwise_csv = output_dir / f"gate_pairwise_cal_{variant_suffix}_candidates.csv"
    pairwise_df.to_csv(pairwise_csv, index=False)

    feature_columns = _pairwise_feature_columns(prototype_candidates=config.bank_prototype_candidates)
    feature_matrix = pairwise_df[feature_columns].to_numpy(dtype=np.float32)
    utility_target = pairwise_df["pairwise_utility_gain_target"].to_numpy(dtype=np.float32)
    risk_target = pairwise_df["pairwise_failure_risk_target"].to_numpy(dtype=np.float32)
    regret_target = pairwise_df["pairwise_fallback_regret_target"].to_numpy(dtype=np.float32)
    margin_target = pairwise_df["pairwise_route_margin_target"].to_numpy(dtype=np.float32)
    route_win_target = pairwise_df["pairwise_route_permission_target"].to_numpy(dtype=np.float32)
    regret_weight = pairwise_df["pairwise_regret_supervision_weight"].to_numpy(dtype=np.float32)
    constraint_score = pairwise_df["pairwise_route_constraint_score"].to_numpy(dtype=np.float32)
    sample_weight = pairwise_df["pairwise_sample_weight"].to_numpy(dtype=np.float32)

    rng = np.random.default_rng(config.random_state + 3917)
    indices = rng.permutation(len(feature_matrix))
    val_size = 0
    if len(indices) >= 8:
        val_size = max(1, int(round(len(indices) * config.val_fraction)))
        val_size = min(val_size, len(indices) - 1)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        train_idx = indices
        val_idx = np.asarray([], dtype=np.int64)

    feature_mean = feature_matrix[train_idx].mean(axis=0)
    feature_std = feature_matrix[train_idx].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    x_train = ((feature_matrix[train_idx] - feature_mean[None, :]) / feature_std[None, :]).astype(np.float32)
    utility_train = utility_target[train_idx].astype(np.float32)
    risk_train = risk_target[train_idx].astype(np.float32)
    regret_train = regret_target[train_idx].astype(np.float32)
    margin_train = margin_target[train_idx].astype(np.float32)
    route_win_train = route_win_target[train_idx].astype(np.float32)
    regret_weight_train = regret_weight[train_idx].astype(np.float32)
    constraint_score_train = constraint_score[train_idx].astype(np.float32)
    weight_train = sample_weight[train_idx].astype(np.float32)

    x_val = (
        ((feature_matrix[val_idx] - feature_mean[None, :]) / feature_std[None, :]).astype(np.float32)
        if len(val_idx)
        else np.zeros((0, feature_matrix.shape[-1]), dtype=np.float32)
    )
    utility_val = utility_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    risk_val = risk_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    regret_val = regret_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    margin_val = margin_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    route_win_val = route_win_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    regret_weight_val = regret_weight[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    constraint_score_val = (
        constraint_score[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    )
    weight_val = sample_weight[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)

    device = choose_torch_device()
    base_model = Stage1AnchoredPairwiseRouter(
        input_dim=feature_matrix.shape[-1],
        hidden_dim=config.pairwise_hidden_dim,
        dropout=config.pairwise_dropout,
    )
    if torch.cuda.is_available():
        base_model = base_model.to(device)
        model: nn.Module = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model
    else:
        base_model = base_model.to(device)
        model = base_model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pairwise_learning_rate,
        weight_decay=config.pairwise_weight_decay,
    )
    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(utility_train),
        torch.from_numpy(risk_train),
        torch.from_numpy(regret_train),
        torch.from_numpy(margin_train),
        torch.from_numpy(route_win_train),
        torch.from_numpy(regret_weight_train),
        torch.from_numpy(constraint_score_train),
        torch.from_numpy(weight_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size * 2, len(train_dataset)),
        shuffle=True,
        drop_last=False,
    )

    best_state = copy.deepcopy(_unwrap_model(model).state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0
    history_rows: list[dict[str, float | int]] = []
    train_start = time.perf_counter()

    for epoch in range(config.pairwise_train_epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        for (
            batch_x,
            batch_utility,
            batch_risk,
            batch_regret,
            batch_margin,
            batch_route_win,
            batch_regret_weight,
            batch_constraint_score,
            batch_weight,
        ) in train_loader:
            batch_x = batch_x.to(device, non_blocking=torch.cuda.is_available())
            batch_utility = batch_utility.to(device, non_blocking=torch.cuda.is_available())
            batch_risk = batch_risk.to(device, non_blocking=torch.cuda.is_available())
            batch_regret = batch_regret.to(device, non_blocking=torch.cuda.is_available())
            batch_margin = batch_margin.to(device, non_blocking=torch.cuda.is_available())
            batch_route_win = batch_route_win.to(device, non_blocking=torch.cuda.is_available())
            batch_regret_weight = batch_regret_weight.to(device, non_blocking=torch.cuda.is_available())
            batch_constraint_score = batch_constraint_score.to(device, non_blocking=torch.cuda.is_available())
            batch_weight = batch_weight.to(device, non_blocking=torch.cuda.is_available())

            utility_pred, risk_pred, regret_pred, margin_pred = model(batch_x)
            loss = _pairwise_calibrated_router_objective(
                utility_pred=utility_pred,
                risk_pred=risk_pred,
                regret_pred=regret_pred,
                margin_pred=margin_pred,
                utility_target=batch_utility,
                risk_target=batch_risk,
                regret_target=batch_regret,
                margin_target=batch_margin,
                route_win_target=batch_route_win,
                regret_weight=batch_regret_weight,
                constraint_score=batch_constraint_score,
                weights=batch_weight,
                config=config,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu()) * len(batch_x)
            running_count += len(batch_x)

        train_loss = running_loss / max(running_count, 1)
        val_loss = _evaluate_pairwise_calibrated_router_loss(
            model=model,
            x_val=x_val,
            utility_val=utility_val,
            risk_val=risk_val,
            regret_val=regret_val,
            margin_val=margin_val,
            route_win_val=route_win_val,
            regret_weight_val=regret_weight_val,
            constraint_score_val=constraint_score_val,
            weight_val=weight_val,
            device=device,
            config=config,
        )
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(_unwrap_model(model).state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= config.pairwise_patience:
            break

    train_elapsed = time.perf_counter() - train_start
    history_csv = output_dir / f"gate_pairwise_cal_{variant_suffix}_training_history.csv"
    pd.DataFrame(history_rows).to_csv(history_csv, index=False)
    return {
        "variant": variant,
        "pairwise_cal_candidate_csv": str(pairwise_csv),
        "pairwise_cal_history_csv": str(history_csv),
        "pairwise_cal_model_state": best_state,
        "pairwise_cal_feature_keys": feature_columns,
        "pairwise_cal_feature_mean": feature_mean.astype(np.float32),
        "pairwise_cal_feature_std": feature_std.astype(np.float32),
        "pairwise_cal_hidden_dim": config.pairwise_hidden_dim,
        "pairwise_cal_dropout": config.pairwise_dropout,
        "pairwise_cal_best_val_loss": float(
            best_val_loss if np.isfinite(best_val_loss) else history_rows[-1]["train_loss"]
        ),
        "pairwise_cal_train_examples": int(len(train_idx)),
        "pairwise_cal_val_examples": int(len(val_idx)),
        "pairwise_cal_train_time_sec": train_elapsed,
    }


def _pairperm_permission_calibrated_torch(
    *,
    permission_logit: torch.Tensor,
    value_pred: torch.Tensor,
    risk_pred: torch.Tensor,
    route_consistency: torch.Tensor,
    risk_budget_pred: torch.Tensor,
    consistency_budget_pred: torch.Tensor,
    config: GateLearningConfig,
    use_permission_head: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    permission_prob = (
        torch.sigmoid(permission_logit)
        if use_permission_head
        else torch.sigmoid(4.0 * value_pred)
    )
    risk_budget_score = torch.sigmoid(
        (risk_budget_pred - risk_pred) / max(config.pairperm_risk_budget_temperature, 1e-6)
    )
    consistency_budget_score = torch.sigmoid(
        (route_consistency - consistency_budget_pred) / max(config.pairperm_consistency_budget_temperature, 1e-6)
    )
    permission_budget_score = torch.clamp(0.54 * risk_budget_score + 0.46 * consistency_budget_score, 0.0, 1.0)
    permission_calibrated = torch.clamp(
        permission_prob * (0.35 + 0.65 * permission_budget_score),
        1e-6,
        1.0 - 1e-6,
    )
    return permission_prob, permission_budget_score, permission_calibrated, consistency_budget_score


def _pairperm_router_objective(
    *,
    utility_pred: torch.Tensor,
    risk_pred: torch.Tensor,
    regret_pred: torch.Tensor,
    value_pred: torch.Tensor,
    permission_logit: torch.Tensor,
    risk_budget_pred: torch.Tensor,
    consistency_budget_pred: torch.Tensor,
    utility_target: torch.Tensor,
    risk_target: torch.Tensor,
    regret_target: torch.Tensor,
    value_target: torch.Tensor,
    permission_target: torch.Tensor,
    permission_calibrated_target: torch.Tensor,
    risk_budget_target: torch.Tensor,
    consistency_budget_target: torch.Tensor,
    route_consistency: torch.Tensor,
    final_permission_pass: torch.Tensor,
    selection_target: torch.Tensor,
    regret_weight: torch.Tensor,
    weights: torch.Tensor,
    config: GateLearningConfig,
    variant: str,
) -> torch.Tensor:
    use_permission_head = variant != "no_permission_head"
    utility_loss = F.smooth_l1_loss(utility_pred, utility_target, reduction="none")
    risk_loss = F.smooth_l1_loss(risk_pred, risk_target, reduction="none")
    regret_loss = regret_weight * F.smooth_l1_loss(regret_pred, regret_target, reduction="none")
    value_loss = F.smooth_l1_loss(value_pred, value_target, reduction="none")
    risk_budget_loss = F.smooth_l1_loss(risk_budget_pred, risk_budget_target, reduction="none")
    consistency_budget_loss = F.smooth_l1_loss(
        consistency_budget_pred,
        consistency_budget_target,
        reduction="none",
    )

    permission_prob, permission_budget_score, permission_calibrated, _ = _pairperm_permission_calibrated_torch(
        permission_logit=permission_logit,
        value_pred=value_pred,
        risk_pred=risk_pred,
        route_consistency=route_consistency,
        risk_budget_pred=risk_budget_pred,
        consistency_budget_pred=consistency_budget_pred,
        config=config,
        use_permission_head=use_permission_head,
    )
    if use_permission_head:
        permission_loss = F.binary_cross_entropy_with_logits(
            permission_logit,
            permission_target,
            reduction="none",
        )
    else:
        permission_loss = torch.zeros_like(permission_target)
    calibrated_loss = F.binary_cross_entropy(
        permission_calibrated,
        permission_calibrated_target,
        reduction="none",
    )
    final_pass_loss = F.binary_cross_entropy(
        permission_calibrated,
        final_permission_pass,
        reduction="none",
    )
    selection_loss = F.binary_cross_entropy_with_logits(
        4.0 * value_pred,
        selection_target,
        reduction="none",
    )

    value_weight = config.pairperm_value_weight if variant != "permission_only" else 0.25
    selection_weight = 0.25 if variant != "permission_only" else 0.08
    if variant == "no_permission_value_decoupling":
        coupled_score = torch.clamp(permission_calibrated * torch.sigmoid(4.0 * value_pred), 1e-6, 1.0 - 1e-6)
        calibrated_loss = F.binary_cross_entropy(
            coupled_score,
            permission_calibrated_target,
            reduction="none",
        )
        selection_loss = F.binary_cross_entropy(
            coupled_score,
            torch.clamp(selection_target + 0.50 * final_permission_pass, 0.0, 1.0),
            reduction="none",
        )
        selection_weight = 0.35

    regression = utility_loss + risk_loss + value_weight * value_loss
    if variant != "no_regret_aux":
        regression = regression + config.pairperm_regret_weight * regret_loss
    budget_loss = risk_budget_loss + consistency_budget_loss
    total = (
        regression
        + config.pairperm_permission_weight * (permission_loss + calibrated_loss + 0.45 * final_pass_loss)
        + config.pairperm_budget_weight * budget_loss
        + selection_weight * selection_loss
    )
    return (total * weights).mean()


def _evaluate_pairperm_router_loss(
    *,
    model: nn.Module,
    x_val: np.ndarray,
    utility_val: np.ndarray,
    risk_val: np.ndarray,
    regret_val: np.ndarray,
    value_val: np.ndarray,
    permission_val: np.ndarray,
    permission_cal_val: np.ndarray,
    risk_budget_val: np.ndarray,
    consistency_budget_val: np.ndarray,
    route_consistency_val: np.ndarray,
    final_permission_pass_val: np.ndarray,
    selection_val: np.ndarray,
    regret_weight_val: np.ndarray,
    weight_val: np.ndarray,
    device: torch.device,
    config: GateLearningConfig,
    variant: str,
) -> float:
    if len(x_val) == 0:
        return 0.0

    model.eval()
    with torch.no_grad():
        batch_x = torch.from_numpy(x_val).to(device, non_blocking=torch.cuda.is_available())
        batch_utility = torch.from_numpy(utility_val).to(device, non_blocking=torch.cuda.is_available())
        batch_risk = torch.from_numpy(risk_val).to(device, non_blocking=torch.cuda.is_available())
        batch_regret = torch.from_numpy(regret_val).to(device, non_blocking=torch.cuda.is_available())
        batch_value = torch.from_numpy(value_val).to(device, non_blocking=torch.cuda.is_available())
        batch_permission = torch.from_numpy(permission_val).to(device, non_blocking=torch.cuda.is_available())
        batch_permission_cal = torch.from_numpy(permission_cal_val).to(
            device,
            non_blocking=torch.cuda.is_available(),
        )
        batch_risk_budget = torch.from_numpy(risk_budget_val).to(device, non_blocking=torch.cuda.is_available())
        batch_consistency_budget = torch.from_numpy(consistency_budget_val).to(
            device,
            non_blocking=torch.cuda.is_available(),
        )
        batch_route_consistency = torch.from_numpy(route_consistency_val).to(
            device,
            non_blocking=torch.cuda.is_available(),
        )
        batch_final_permission_pass = torch.from_numpy(final_permission_pass_val).to(
            device,
            non_blocking=torch.cuda.is_available(),
        )
        batch_selection = torch.from_numpy(selection_val).to(device, non_blocking=torch.cuda.is_available())
        batch_regret_weight = torch.from_numpy(regret_weight_val).to(device, non_blocking=torch.cuda.is_available())
        batch_weight = torch.from_numpy(weight_val).to(device, non_blocking=torch.cuda.is_available())

        (
            utility_pred,
            risk_pred,
            regret_pred,
            value_pred,
            permission_logit,
            risk_budget_pred,
            consistency_budget_pred,
        ) = model(batch_x)
        loss = _pairperm_router_objective(
            utility_pred=utility_pred,
            risk_pred=risk_pred,
            regret_pred=regret_pred,
            value_pred=value_pred,
            permission_logit=permission_logit,
            risk_budget_pred=risk_budget_pred,
            consistency_budget_pred=consistency_budget_pred,
            utility_target=batch_utility,
            risk_target=batch_risk,
            regret_target=batch_regret,
            value_target=batch_value,
            permission_target=batch_permission,
            permission_calibrated_target=batch_permission_cal,
            risk_budget_target=batch_risk_budget,
            consistency_budget_target=batch_consistency_budget,
            route_consistency=batch_route_consistency,
            final_permission_pass=batch_final_permission_pass,
            selection_target=batch_selection,
            regret_weight=batch_regret_weight,
            weights=batch_weight,
            config=config,
            variant=variant,
        )
    return float(loss.detach().cpu())


def _train_stage1_pairwise_router_permissioned(
    *,
    output_dir: Path,
    config: GateLearningConfig,
    variant: str = "full",
) -> dict[str, object]:
    bank_csv = output_dir / "gate_bank_candidates.csv"
    bank_df = pd.read_csv(bank_csv)
    pairperm_df = _build_stage1_pairwise_permissioned_training_frame(
        bank_df=bank_df,
        config=config,
        variant=variant,
    )
    variant_suffix = variant.replace("no_", "no-").replace("_", "-")
    pairperm_csv = output_dir / f"gate_pairperm_{variant_suffix}_candidates.csv"
    pairperm_df.to_csv(pairperm_csv, index=False)

    feature_columns = _pairwise_feature_columns(prototype_candidates=config.bank_prototype_candidates)
    feature_matrix = pairperm_df[feature_columns].to_numpy(dtype=np.float32)
    utility_target = pairperm_df["pairwise_utility_gain_target"].to_numpy(dtype=np.float32)
    risk_target = pairperm_df["pairwise_failure_risk_target"].to_numpy(dtype=np.float32)
    regret_target = pairperm_df["pairwise_fallback_regret_target"].to_numpy(dtype=np.float32)
    value_target = pairperm_df["pairwise_route_value_target"].to_numpy(dtype=np.float32)
    permission_target = pairperm_df["pairwise_route_permission_target"].to_numpy(dtype=np.float32)
    permission_cal_target = pairperm_df["pairwise_route_permission_calibrated"].to_numpy(dtype=np.float32)
    risk_budget_target = pairperm_df["pairwise_adaptive_risk_budget"].to_numpy(dtype=np.float32)
    consistency_budget_target = pairperm_df["pairwise_adaptive_consistency_budget"].to_numpy(dtype=np.float32)
    route_consistency = pairperm_df["pairwise_route_consistency"].to_numpy(dtype=np.float32)
    final_permission_pass = pairperm_df["pairwise_final_route_permission_pass"].to_numpy(dtype=np.float32)
    selection_target = pairperm_df["pairwise_route_selected_under_permission"].to_numpy(dtype=np.float32)
    regret_weight = pairperm_df["pairwise_regret_supervision_weight"].to_numpy(dtype=np.float32)
    sample_weight = pairperm_df["pairwise_sample_weight"].to_numpy(dtype=np.float32)

    variant_seed = sum(ord(ch) for ch in variant)
    rng = np.random.default_rng(config.random_state + 4917 + variant_seed)
    indices = rng.permutation(len(feature_matrix))
    val_size = 0
    if len(indices) >= 8:
        val_size = max(1, int(round(len(indices) * config.val_fraction)))
        val_size = min(val_size, len(indices) - 1)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        train_idx = indices
        val_idx = np.asarray([], dtype=np.int64)

    feature_mean = feature_matrix[train_idx].mean(axis=0)
    feature_std = feature_matrix[train_idx].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    x_train = ((feature_matrix[train_idx] - feature_mean[None, :]) / feature_std[None, :]).astype(np.float32)
    utility_train = utility_target[train_idx].astype(np.float32)
    risk_train = risk_target[train_idx].astype(np.float32)
    regret_train = regret_target[train_idx].astype(np.float32)
    value_train = value_target[train_idx].astype(np.float32)
    permission_train = permission_target[train_idx].astype(np.float32)
    permission_cal_train = permission_cal_target[train_idx].astype(np.float32)
    risk_budget_train = risk_budget_target[train_idx].astype(np.float32)
    consistency_budget_train = consistency_budget_target[train_idx].astype(np.float32)
    route_consistency_train = route_consistency[train_idx].astype(np.float32)
    final_permission_pass_train = final_permission_pass[train_idx].astype(np.float32)
    selection_train = selection_target[train_idx].astype(np.float32)
    regret_weight_train = regret_weight[train_idx].astype(np.float32)
    weight_train = sample_weight[train_idx].astype(np.float32)

    x_val = (
        ((feature_matrix[val_idx] - feature_mean[None, :]) / feature_std[None, :]).astype(np.float32)
        if len(val_idx)
        else np.zeros((0, feature_matrix.shape[-1]), dtype=np.float32)
    )
    utility_val = utility_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    risk_val = risk_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    regret_val = regret_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    value_val = value_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    permission_val = permission_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    permission_cal_val = (
        permission_cal_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    )
    risk_budget_val = risk_budget_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    consistency_budget_val = (
        consistency_budget_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    )
    route_consistency_val = route_consistency[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    final_permission_pass_val = (
        final_permission_pass[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    )
    selection_val = selection_target[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    regret_weight_val = regret_weight[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)
    weight_val = sample_weight[val_idx].astype(np.float32) if len(val_idx) else np.zeros((0,), dtype=np.float32)

    device = choose_torch_device()
    base_model = PermissionedStage1AnchoredPairwiseRouter(
        input_dim=feature_matrix.shape[-1],
        hidden_dim=config.pairperm_hidden_dim,
        dropout=config.pairperm_dropout,
        risk_budget_floor=config.pairperm_risk_budget_floor,
        risk_budget_ceiling=config.pairperm_risk_budget_ceiling,
        consistency_budget_floor=config.pairperm_consistency_budget_floor,
        consistency_budget_ceiling=config.pairperm_consistency_budget_ceiling,
    )
    if torch.cuda.is_available():
        base_model = base_model.to(device)
        model: nn.Module = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model
    else:
        base_model = base_model.to(device)
        model = base_model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pairperm_learning_rate,
        weight_decay=config.pairperm_weight_decay,
    )
    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(utility_train),
        torch.from_numpy(risk_train),
        torch.from_numpy(regret_train),
        torch.from_numpy(value_train),
        torch.from_numpy(permission_train),
        torch.from_numpy(permission_cal_train),
        torch.from_numpy(risk_budget_train),
        torch.from_numpy(consistency_budget_train),
        torch.from_numpy(route_consistency_train),
        torch.from_numpy(final_permission_pass_train),
        torch.from_numpy(selection_train),
        torch.from_numpy(regret_weight_train),
        torch.from_numpy(weight_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size * 2, len(train_dataset)),
        shuffle=True,
        drop_last=False,
    )

    best_state = copy.deepcopy(_unwrap_model(model).state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0
    history_rows: list[dict[str, float | int]] = []
    train_start = time.perf_counter()

    for epoch in range(config.pairperm_train_epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        for (
            batch_x,
            batch_utility,
            batch_risk,
            batch_regret,
            batch_value,
            batch_permission,
            batch_permission_cal,
            batch_risk_budget,
            batch_consistency_budget,
            batch_route_consistency,
            batch_final_permission_pass,
            batch_selection,
            batch_regret_weight,
            batch_weight,
        ) in train_loader:
            batch_x = batch_x.to(device, non_blocking=torch.cuda.is_available())
            batch_utility = batch_utility.to(device, non_blocking=torch.cuda.is_available())
            batch_risk = batch_risk.to(device, non_blocking=torch.cuda.is_available())
            batch_regret = batch_regret.to(device, non_blocking=torch.cuda.is_available())
            batch_value = batch_value.to(device, non_blocking=torch.cuda.is_available())
            batch_permission = batch_permission.to(device, non_blocking=torch.cuda.is_available())
            batch_permission_cal = batch_permission_cal.to(device, non_blocking=torch.cuda.is_available())
            batch_risk_budget = batch_risk_budget.to(device, non_blocking=torch.cuda.is_available())
            batch_consistency_budget = batch_consistency_budget.to(device, non_blocking=torch.cuda.is_available())
            batch_route_consistency = batch_route_consistency.to(device, non_blocking=torch.cuda.is_available())
            batch_final_permission_pass = batch_final_permission_pass.to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
            batch_selection = batch_selection.to(device, non_blocking=torch.cuda.is_available())
            batch_regret_weight = batch_regret_weight.to(device, non_blocking=torch.cuda.is_available())
            batch_weight = batch_weight.to(device, non_blocking=torch.cuda.is_available())

            (
                utility_pred,
                risk_pred,
                regret_pred,
                value_pred,
                permission_logit,
                risk_budget_pred,
                consistency_budget_pred,
            ) = model(batch_x)
            loss = _pairperm_router_objective(
                utility_pred=utility_pred,
                risk_pred=risk_pred,
                regret_pred=regret_pred,
                value_pred=value_pred,
                permission_logit=permission_logit,
                risk_budget_pred=risk_budget_pred,
                consistency_budget_pred=consistency_budget_pred,
                utility_target=batch_utility,
                risk_target=batch_risk,
                regret_target=batch_regret,
                value_target=batch_value,
                permission_target=batch_permission,
                permission_calibrated_target=batch_permission_cal,
                risk_budget_target=batch_risk_budget,
                consistency_budget_target=batch_consistency_budget,
                route_consistency=batch_route_consistency,
                final_permission_pass=batch_final_permission_pass,
                selection_target=batch_selection,
                regret_weight=batch_regret_weight,
                weights=batch_weight,
                config=config,
                variant=variant,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu()) * len(batch_x)
            running_count += len(batch_x)

        train_loss = running_loss / max(running_count, 1)
        val_loss = _evaluate_pairperm_router_loss(
            model=model,
            x_val=x_val,
            utility_val=utility_val,
            risk_val=risk_val,
            regret_val=regret_val,
            value_val=value_val,
            permission_val=permission_val,
            permission_cal_val=permission_cal_val,
            risk_budget_val=risk_budget_val,
            consistency_budget_val=consistency_budget_val,
            route_consistency_val=route_consistency_val,
            final_permission_pass_val=final_permission_pass_val,
            selection_val=selection_val,
            regret_weight_val=regret_weight_val,
            weight_val=weight_val,
            device=device,
            config=config,
            variant=variant,
        )
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(_unwrap_model(model).state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= config.pairperm_patience:
            break

    train_elapsed = time.perf_counter() - train_start
    history_csv = output_dir / f"gate_pairperm_{variant_suffix}_training_history.csv"
    pd.DataFrame(history_rows).to_csv(history_csv, index=False)
    return {
        "variant": variant,
        "pairperm_candidate_csv": str(pairperm_csv),
        "pairperm_history_csv": str(history_csv),
        "pairperm_model_state": best_state,
        "pairperm_feature_keys": feature_columns,
        "pairperm_feature_mean": feature_mean.astype(np.float32),
        "pairperm_feature_std": feature_std.astype(np.float32),
        "pairperm_hidden_dim": config.pairperm_hidden_dim,
        "pairperm_dropout": config.pairperm_dropout,
        "pairperm_best_val_loss": float(
            best_val_loss if np.isfinite(best_val_loss) else history_rows[-1]["train_loss"]
        ),
        "pairperm_train_examples": int(len(train_idx)),
        "pairperm_val_examples": int(len(val_idx)),
        "pairperm_train_time_sec": train_elapsed,
    }


def _escapecert_candidate_safe_score_torch(
    *,
    utility_pred: torch.Tensor,
    risk_pred: torch.Tensor,
    regret_pred: torch.Tensor,
    value_pred: torch.Tensor,
    route_consistency: torch.Tensor,
    use_regret_aux: bool,
) -> torch.Tensor:
    regret_component = torch.sigmoid(4.0 * regret_pred) if use_regret_aux else torch.sigmoid(3.0 * value_pred - 0.2)
    return torch.clamp(
        0.24 * torch.sigmoid(6.0 * utility_pred - 1.0)
        + 0.24 * torch.sigmoid(3.5 * value_pred)
        + 0.18 * regret_component
        + 0.18 * (1.0 - risk_pred)
        + 0.16 * route_consistency,
        0.0,
        1.0,
    )


def _escapecert_group_predictions_torch(
    *,
    utility_pred: torch.Tensor,
    risk_pred: torch.Tensor,
    regret_pred: torch.Tensor,
    value_pred: torch.Tensor,
    anchor_escape_logit: torch.Tensor,
    admissibility_logit: torch.Tensor,
    risk_budget_pred: torch.Tensor,
    consistency_budget_pred: torch.Tensor,
    uncertainty_pred: torch.Tensor,
    route_consistency: torch.Tensor,
    candidate_is_stage1: torch.Tensor,
    config: GateLearningConfig,
    variant: str,
) -> dict[str, torch.Tensor]:
    flags = _escapecert_variant_flags(variant)
    non_stage_mask = (candidate_is_stage1 < 0.5).float()
    non_stage_idx = torch.nonzero(non_stage_mask > 0.5, as_tuple=False).squeeze(-1)
    zero = utility_pred.new_tensor(0.0)
    if non_stage_idx.numel() == 0:
        zero_vec = torch.zeros_like(utility_pred)
        return {
            "base_admissibility_prob": zero_vec,
            "candidate_admissibility_calibrated": zero_vec,
            "candidate_budget_score": zero_vec,
            "anchor_escape_logit": zero,
            "anchor_escape_prob": zero,
            "anchor_escape_calibrated": zero,
            "anchor_escape_uncertainty": zero,
            "counterfactual_escape_budget_score": zero,
            "counterfactual_escape_support": zero,
            "anchor_escape_counterfactual_margin": zero,
            "agg_risk_budget": utility_pred.new_tensor(config.pairwise_safe_risk_budget),
            "agg_consistency_budget": utility_pred.new_tensor(config.pairwise_consistency_threshold),
            "allowed_set_total_mass": zero,
            "allowed_set_entropy": zero,
            "allowed_set_topm_mass": zero,
            "allowed_set_best_value": zero,
            "permission_set_consistency": zero,
            "anchor_escape_safe_candidate_count": zero,
        }

    safe_score = _escapecert_candidate_safe_score_torch(
        utility_pred=utility_pred,
        risk_pred=risk_pred,
        regret_pred=regret_pred,
        value_pred=value_pred,
        route_consistency=route_consistency,
        use_regret_aux=flags["use_regret_aux"],
    )
    evidence_prob = torch.sigmoid(anchor_escape_logit) if flags["use_anchor_head"] else safe_score
    support = torch.clamp(
        evidence_prob * (0.35 + 0.65 * safe_score) * non_stage_mask,
        0.0,
        1.0,
    )
    non_stage_support = support[non_stage_idx]
    topm = max(1, min(int(config.escapecert_topm), int(non_stage_idx.numel())))
    top_values, top_local = torch.topk(non_stage_support, k=topm)
    top_idx = non_stage_idx[top_local]
    top_weights = torch.softmax(top_values / max(config.escapecert_anchor_temperature, 1e-6), dim=0)

    if flags["fixed_budget"]:
        agg_risk_budget = utility_pred.new_tensor(config.pairwise_safe_risk_budget)
        agg_consistency_budget = utility_pred.new_tensor(config.pairwise_consistency_threshold)
    else:
        agg_risk_budget = torch.sum(top_weights * risk_budget_pred[top_idx])
        agg_consistency_budget = torch.sum(top_weights * consistency_budget_pred[top_idx])

    risk_pass = torch.sigmoid(
        (agg_risk_budget - risk_pred) / max(config.pairperm_risk_budget_temperature, 1e-6)
    )
    consistency_pass = torch.sigmoid(
        (route_consistency - agg_consistency_budget) / max(config.pairperm_consistency_budget_temperature, 1e-6)
    )
    candidate_budget_score = torch.clamp(0.54 * risk_pass + 0.46 * consistency_pass, 0.0, 1.0) * non_stage_mask
    counterfactual_escape_budget_score = torch.sum(top_weights * candidate_budget_score[top_idx])

    base_admissibility_prob = (
        torch.sigmoid(admissibility_logit)
        if flags["use_admissibility_head"]
        else torch.clamp(0.45 * support + 0.55 * safe_score, 0.0, 1.0)
    ) * non_stage_mask

    anchor_escape_uncertainty = (
        torch.sum(top_weights * uncertainty_pred[top_idx]) if flags["use_uncertainty"] else zero
    )
    if flags["use_anchor_head"]:
        top_escape_mass = torch.clamp(support[top_idx], 1e-6, 1.0 - 1e-6)
        anchor_escape_prob = torch.clamp(1.0 - torch.prod(1.0 - top_escape_mass), 1e-6, 1.0 - 1e-6)
        anchor_escape_logit_agg = torch.logit(anchor_escape_prob)
    else:
        derived_mass = torch.clamp(
            base_admissibility_prob[non_stage_idx]
            * candidate_budget_score[non_stage_idx]
            * (0.35 + 0.65 * safe_score[non_stage_idx]),
            1e-6,
            1.0 - 1e-6,
        )
        anchor_escape_prob = torch.clamp(1.0 - torch.prod(1.0 - derived_mass), 1e-6, 1.0 - 1e-6)
        anchor_escape_logit_agg = torch.logit(anchor_escape_prob)

    anchor_escape_calibrated = torch.clamp(
        0.70 * anchor_escape_prob
        + 0.20 * counterfactual_escape_budget_score
        + 0.10 * (1.0 - anchor_escape_uncertainty),
        1e-6,
        1.0 - 1e-6,
    )
    candidate_admissibility_calibrated = torch.clamp(
        base_admissibility_prob
        * (0.20 + 0.45 * anchor_escape_prob + 0.35 * candidate_budget_score)
        * (0.45 + 0.55 * safe_score),
        0.0,
        1.0,
    ) * non_stage_mask

    allowed_sum = torch.sum(candidate_admissibility_calibrated[non_stage_idx])
    normalized_mass = candidate_admissibility_calibrated[non_stage_idx] / torch.clamp(allowed_sum, min=1e-8)
    entropy_denom = max(float(np.log(max(int(non_stage_idx.numel()), 2))), 1e-6)
    raw_entropy = -torch.sum(normalized_mass * torch.log(torch.clamp(normalized_mass, min=1e-8))) / entropy_denom
    allowed_set_entropy = torch.where(allowed_sum <= 1e-8, zero, raw_entropy)

    permission_set_consistency = torch.clamp(
        1.0 - torch.prod(1.0 - torch.clamp(candidate_admissibility_calibrated[non_stage_idx], 1e-6, 1.0 - 1e-6)),
        1e-6,
        1.0 - 1e-6,
    )
    counterfactual_escape_support = torch.sum(top_weights * support[top_idx])
    anchor_escape_counterfactual_margin = torch.sum(
        top_weights * (value_pred[top_idx] + 0.35 * regret_pred[top_idx] - 0.25 * risk_pred[top_idx])
    )
    allowed_set_total_mass = torch.sum(candidate_admissibility_calibrated[non_stage_idx]) / max(int(non_stage_idx.numel()), 1)
    allowed_set_topm_mass = torch.sum(top_weights * candidate_admissibility_calibrated[top_idx])
    allowed_set_best_value = torch.max(value_pred[non_stage_idx])
    anchor_escape_safe_candidate_count = torch.sum((candidate_admissibility_calibrated[non_stage_idx] >= 0.5).float())
    return {
        "base_admissibility_prob": torch.clamp(base_admissibility_prob, 1e-6, 1.0 - 1e-6),
        "candidate_admissibility_calibrated": torch.clamp(candidate_admissibility_calibrated, 1e-6, 1.0 - 1e-6),
        "candidate_budget_score": candidate_budget_score,
        "anchor_escape_logit": anchor_escape_logit_agg,
        "anchor_escape_prob": torch.clamp(anchor_escape_prob, 1e-6, 1.0 - 1e-6),
        "anchor_escape_calibrated": anchor_escape_calibrated,
        "anchor_escape_uncertainty": torch.clamp(anchor_escape_uncertainty, 0.0, 1.0),
        "counterfactual_escape_budget_score": torch.clamp(counterfactual_escape_budget_score, 0.0, 1.0),
        "counterfactual_escape_support": torch.clamp(counterfactual_escape_support, 0.0, 1.0),
        "anchor_escape_counterfactual_margin": anchor_escape_counterfactual_margin,
        "agg_risk_budget": agg_risk_budget,
        "agg_consistency_budget": agg_consistency_budget,
        "allowed_set_total_mass": torch.clamp(allowed_set_total_mass, 0.0, 1.0),
        "allowed_set_entropy": torch.clamp(allowed_set_entropy, 0.0, 1.0),
        "allowed_set_topm_mass": torch.clamp(allowed_set_topm_mass, 0.0, 1.0),
        "allowed_set_best_value": allowed_set_best_value,
        "permission_set_consistency": permission_set_consistency,
        "anchor_escape_safe_candidate_count": anchor_escape_safe_candidate_count,
    }


def _escapecert_group_loss(
    *,
    utility_pred: torch.Tensor,
    risk_pred: torch.Tensor,
    regret_pred: torch.Tensor,
    value_pred: torch.Tensor,
    anchor_escape_logit: torch.Tensor,
    admissibility_logit: torch.Tensor,
    risk_budget_pred: torch.Tensor,
    consistency_budget_pred: torch.Tensor,
    uncertainty_pred: torch.Tensor,
    utility_target: torch.Tensor,
    risk_target: torch.Tensor,
    regret_target: torch.Tensor,
    value_target: torch.Tensor,
    candidate_admissibility_target: torch.Tensor,
    candidate_allowed_target: torch.Tensor,
    anchor_escape_target: torch.Tensor,
    allowed_set_target_mass: torch.Tensor,
    permission_set_consistency_target: torch.Tensor,
    anchor_escape_topm_mass_target: torch.Tensor,
    anchor_escape_soft_oracle_gain_target: torch.Tensor,
    anchor_escape_uncertainty_target: torch.Tensor,
    risk_budget_target: torch.Tensor,
    consistency_budget_target: torch.Tensor,
    route_consistency: torch.Tensor,
    candidate_is_stage1: torch.Tensor,
    selection_index: int,
    weights: torch.Tensor,
    config: GateLearningConfig,
    variant: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    flags = _escapecert_variant_flags(variant)
    outputs = _escapecert_group_predictions_torch(
        utility_pred=utility_pred,
        risk_pred=risk_pred,
        regret_pred=regret_pred,
        value_pred=value_pred,
        anchor_escape_logit=anchor_escape_logit,
        admissibility_logit=admissibility_logit,
        risk_budget_pred=risk_budget_pred,
        consistency_budget_pred=consistency_budget_pred,
        uncertainty_pred=uncertainty_pred,
        route_consistency=route_consistency,
        candidate_is_stage1=candidate_is_stage1,
        config=config,
        variant=variant,
    )
    non_stage_mask = (candidate_is_stage1 < 0.5).float()
    weight_denom = torch.clamp(torch.sum(weights), min=1e-6)
    non_stage_weight_denom = torch.clamp(torch.sum(weights * non_stage_mask), min=1e-6)

    utility_loss = torch.sum(F.smooth_l1_loss(utility_pred, utility_target, reduction="none") * weights) / weight_denom
    risk_loss = torch.sum(F.smooth_l1_loss(risk_pred, risk_target, reduction="none") * weights) / weight_denom
    value_loss = torch.sum(F.smooth_l1_loss(value_pred, value_target, reduction="none") * weights) / weight_denom
    regret_loss = zero = utility_loss.new_tensor(0.0)
    if flags["use_regret_aux"]:
        regret_loss = torch.sum(
            F.smooth_l1_loss(regret_pred, regret_target, reduction="none") * weights
        ) / weight_denom

    anchor_loss = utility_loss.new_tensor(0.0)
    if flags["use_anchor_head"]:
        anchor_loss = F.binary_cross_entropy(outputs["anchor_escape_calibrated"], anchor_escape_target)
        anchor_loss = anchor_loss + 0.35 * F.binary_cross_entropy(outputs["anchor_escape_prob"], anchor_escape_target)

    admissibility_loss = utility_loss.new_tensor(0.0)
    admissibility_cal_loss = utility_loss.new_tensor(0.0)
    if flags["use_admissibility_head"]:
        admissibility_loss = torch.sum(
            F.binary_cross_entropy(
                outputs["base_admissibility_prob"],
                torch.clamp(candidate_admissibility_target, 1e-6, 1.0 - 1e-6),
                reduction="none",
            )
            * weights
            * non_stage_mask
        ) / non_stage_weight_denom
        admissibility_cal_loss = torch.sum(
            F.binary_cross_entropy(
                outputs["candidate_admissibility_calibrated"],
                torch.clamp(candidate_allowed_target, 1e-6, 1.0 - 1e-6),
                reduction="none",
            )
            * weights
            * non_stage_mask
        ) / non_stage_weight_denom

    budget_loss = F.smooth_l1_loss(outputs["agg_risk_budget"], risk_budget_target)
    budget_loss = budget_loss + F.smooth_l1_loss(outputs["agg_consistency_budget"], consistency_budget_target)

    uncertainty_loss = utility_loss.new_tensor(0.0)
    if flags["use_uncertainty"]:
        uncertainty_loss = F.smooth_l1_loss(outputs["anchor_escape_uncertainty"], anchor_escape_uncertainty_target)

    set_consistency_loss = utility_loss.new_tensor(0.0)
    cardinality_loss = utility_loss.new_tensor(0.0)
    if flags["use_set_supervision"]:
        set_consistency_loss = F.binary_cross_entropy(
            outputs["permission_set_consistency"],
            permission_set_consistency_target,
        )
        set_consistency_loss = set_consistency_loss + 0.35 * F.binary_cross_entropy(
            outputs["anchor_escape_calibrated"],
            permission_set_consistency_target,
        )
        cardinality_loss = F.smooth_l1_loss(outputs["allowed_set_total_mass"], allowed_set_target_mass)
        cardinality_loss = cardinality_loss + 0.55 * F.smooth_l1_loss(
            outputs["allowed_set_topm_mass"],
            anchor_escape_topm_mass_target,
        )
        cardinality_loss = cardinality_loss + 0.35 * F.smooth_l1_loss(
            outputs["allowed_set_best_value"],
            anchor_escape_soft_oracle_gain_target,
        )

    ranking_loss = utility_loss.new_tensor(0.0)
    if selection_index >= 0:
        non_stage_idx = torch.nonzero(non_stage_mask > 0.5, as_tuple=False).squeeze(-1)
        if non_stage_idx.numel() > 0:
            selected_tensor = utility_pred.new_tensor(float(selection_index), dtype=torch.long)
            selected_local = torch.nonzero(non_stage_idx == selected_tensor, as_tuple=False)
            if selected_local.numel() > 0:
                ranking_logits = value_pred[non_stage_idx] + torch.log(
                    torch.clamp(outputs["candidate_admissibility_calibrated"][non_stage_idx], min=1e-6)
                )
                ranking_loss = F.cross_entropy(
                    ranking_logits.unsqueeze(0),
                    selected_local[0:1, 0],
                )

    total = utility_loss + risk_loss + config.escapecert_value_weight * (value_loss + 0.35 * ranking_loss)
    if flags["use_regret_aux"]:
        total = total + config.escapecert_regret_weight * regret_loss
    total = total + config.escapecert_anchor_weight * anchor_loss
    total = total + config.escapecert_admissibility_weight * (admissibility_loss + 0.45 * admissibility_cal_loss)
    total = total + config.escapecert_budget_weight * budget_loss
    total = total + config.escapecert_set_consistency_weight * set_consistency_loss
    total = total + config.escapecert_cardinality_weight * cardinality_loss
    total = total + config.escapecert_uncertainty_weight * uncertainty_loss
    return total, outputs


def _evaluate_escapecert_router_loss(
    *,
    model: nn.Module,
    groups: list[dict[str, object]],
    device: torch.device,
    config: GateLearningConfig,
    variant: str,
) -> float:
    if not groups:
        return 0.0

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for group in groups:
            x_group = torch.from_numpy(group["x"]).to(device, non_blocking=torch.cuda.is_available())
            utility_target = torch.from_numpy(group["utility_target"]).to(device, non_blocking=torch.cuda.is_available())
            risk_target = torch.from_numpy(group["risk_target"]).to(device, non_blocking=torch.cuda.is_available())
            regret_target = torch.from_numpy(group["regret_target"]).to(device, non_blocking=torch.cuda.is_available())
            value_target = torch.from_numpy(group["value_target"]).to(device, non_blocking=torch.cuda.is_available())
            candidate_admissibility_target = torch.from_numpy(group["candidate_admissibility_target"]).to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
            candidate_allowed_target = torch.from_numpy(group["candidate_allowed_target"]).to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
            route_consistency = torch.from_numpy(group["route_consistency"]).to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
            candidate_is_stage1 = torch.from_numpy(group["candidate_is_stage1"]).to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
            weights = torch.from_numpy(group["weights"]).to(device, non_blocking=torch.cuda.is_available())
            (
                utility_pred,
                risk_pred,
                regret_pred,
                value_pred,
                anchor_escape_logit,
                admissibility_logit,
                risk_budget_pred,
                consistency_budget_pred,
                uncertainty_pred,
            ) = model(x_group)
            loss, _ = _escapecert_group_loss(
                utility_pred=utility_pred,
                risk_pred=risk_pred,
                regret_pred=regret_pred,
                value_pred=value_pred,
                anchor_escape_logit=anchor_escape_logit,
                admissibility_logit=admissibility_logit,
                risk_budget_pred=risk_budget_pred,
                consistency_budget_pred=consistency_budget_pred,
                uncertainty_pred=uncertainty_pred,
                utility_target=utility_target,
                risk_target=risk_target,
                regret_target=regret_target,
                value_target=value_target,
                candidate_admissibility_target=candidate_admissibility_target,
                candidate_allowed_target=candidate_allowed_target,
                anchor_escape_target=torch.tensor(
                    float(group["anchor_escape_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                allowed_set_target_mass=torch.tensor(
                    float(group["allowed_set_target_mass"]),
                    device=device,
                    dtype=torch.float32,
                ),
                permission_set_consistency_target=torch.tensor(
                    float(group["permission_set_consistency_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                anchor_escape_topm_mass_target=torch.tensor(
                    float(group["anchor_escape_topm_mass_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                anchor_escape_soft_oracle_gain_target=torch.tensor(
                    float(group["anchor_escape_soft_oracle_gain_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                anchor_escape_uncertainty_target=torch.tensor(
                    float(group["anchor_escape_uncertainty_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                risk_budget_target=torch.tensor(
                    float(group["risk_budget_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                consistency_budget_target=torch.tensor(
                    float(group["consistency_budget_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                route_consistency=route_consistency,
                candidate_is_stage1=candidate_is_stage1,
                selection_index=int(group["selection_index"]),
                weights=weights,
                config=config,
                variant=variant,
            )
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def _train_stage1_pairwise_router_escapecert(
    *,
    output_dir: Path,
    config: GateLearningConfig,
    variant: str = "full",
) -> dict[str, object]:
    bank_csv = output_dir / "gate_bank_candidates.csv"
    bank_df = pd.read_csv(bank_csv)
    escapecert_df = _build_stage1_pairwise_escapecert_training_frame(
        bank_df=bank_df,
        config=config,
        variant=variant,
    )
    variant_suffix = variant.replace("no_", "no-").replace("_", "-")
    candidate_csv = output_dir / f"gate_escapecert_{variant_suffix}_candidates.csv"
    escapecert_df.to_csv(candidate_csv, index=False)

    feature_columns = _pairwise_feature_columns(prototype_candidates=config.bank_prototype_candidates)
    feature_matrix = escapecert_df[feature_columns].to_numpy(dtype=np.float32)
    group_ids = escapecert_df["dataset_id"].to_numpy(dtype=np.int64)

    unique_groups = np.unique(group_ids)
    variant_seed = sum(ord(ch) for ch in variant)
    rng = np.random.default_rng(config.random_state + 7949 + variant_seed)
    shuffled_groups = rng.permutation(unique_groups)
    val_size = 0
    if len(shuffled_groups) >= 5:
        val_size = max(1, int(round(len(shuffled_groups) * config.val_fraction)))
        val_size = min(val_size, len(shuffled_groups) - 1)
    val_groups = set(int(group_id) for group_id in shuffled_groups[:val_size])
    train_groups = set(int(group_id) for group_id in shuffled_groups[val_size:])
    if not train_groups:
        train_groups = set(int(group_id) for group_id in shuffled_groups)
        val_groups = set()

    train_mask = np.asarray([int(group_id) in train_groups for group_id in group_ids], dtype=bool)
    feature_mean = feature_matrix[train_mask].mean(axis=0)
    feature_std = feature_matrix[train_mask].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    normalized = ((feature_matrix - feature_mean[None, :]) / feature_std[None, :]).astype(np.float32)

    grouped_data: list[dict[str, object]] = []
    for dataset_id, group in escapecert_df.reset_index(drop=True).groupby("dataset_id", sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        selection_rows = np.where(group["pairwise_route_selected_under_permission"].to_numpy(dtype=np.float32) > 0.5)[0]
        grouped_data.append(
            {
                "dataset_id": int(dataset_id),
                "x": normalized[idx],
                "utility_target": group["pairwise_utility_gain_target"].to_numpy(dtype=np.float32),
                "risk_target": group["pairwise_failure_risk_target"].to_numpy(dtype=np.float32),
                "regret_target": group["pairwise_fallback_regret_target"].to_numpy(dtype=np.float32),
                "value_target": group["pairwise_route_value_target"].to_numpy(dtype=np.float32),
                "candidate_admissibility_target": group["candidate_admissibility_target"].to_numpy(dtype=np.float32),
                "candidate_allowed_target": group["candidate_allowed_target"].to_numpy(dtype=np.float32),
                "route_consistency": group["pairwise_route_consistency"].to_numpy(dtype=np.float32),
                "candidate_is_stage1": (group["candidate_name"] == "stage1_blended").to_numpy(dtype=np.float32),
                "weights": group["pairwise_sample_weight"].to_numpy(dtype=np.float32),
                "anchor_escape_target": float(group["anchor_escape_target"].iloc[0]),
                "allowed_set_target_mass": float(group["allowed_set_target_mass"].iloc[0]),
                "permission_set_consistency_target": float(group["permission_set_consistency_target"].iloc[0]),
                "anchor_escape_topm_mass_target": float(group["anchor_escape_topm_mass"].iloc[0]),
                "anchor_escape_soft_oracle_gain_target": float(group["anchor_escape_soft_oracle_gain"].iloc[0]),
                "anchor_escape_uncertainty_target": float(group["anchor_escape_uncertainty_target"].iloc[0]),
                "risk_budget_target": float(group["pairwise_adaptive_risk_budget"].iloc[0]),
                "consistency_budget_target": float(group["pairwise_adaptive_consistency_budget"].iloc[0]),
                "selection_index": int(selection_rows[0]) if selection_rows.size else -1,
            }
        )

    train_group_data = [group for group in grouped_data if int(group["dataset_id"]) in train_groups]
    val_group_data = [group for group in grouped_data if int(group["dataset_id"]) in val_groups]

    device = choose_torch_device()
    base_model = EscapeCertStage1AnchoredPairwiseRouter(
        input_dim=feature_matrix.shape[-1],
        hidden_dim=config.escapecert_hidden_dim,
        dropout=config.escapecert_dropout,
        risk_budget_floor=config.pairperm_risk_budget_floor,
        risk_budget_ceiling=config.pairperm_risk_budget_ceiling,
        consistency_budget_floor=config.pairperm_consistency_budget_floor,
        consistency_budget_ceiling=config.pairperm_consistency_budget_ceiling,
    )
    if torch.cuda.is_available():
        base_model = base_model.to(device)
        model: nn.Module = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model
    else:
        base_model = base_model.to(device)
        model = base_model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.escapecert_learning_rate,
        weight_decay=config.escapecert_weight_decay,
    )

    best_state = copy.deepcopy(_unwrap_model(model).state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0
    history_rows: list[dict[str, float | int]] = []
    train_start = time.perf_counter()

    for epoch in range(config.escapecert_train_epochs):
        model.train()
        epoch_order = rng.permutation(len(train_group_data)) if train_group_data else np.asarray([], dtype=np.int64)
        running_loss = 0.0
        running_count = 0

        for group_idx in epoch_order:
            group = train_group_data[int(group_idx)]
            x_group = torch.from_numpy(group["x"]).to(device, non_blocking=torch.cuda.is_available())
            utility_target = torch.from_numpy(group["utility_target"]).to(device, non_blocking=torch.cuda.is_available())
            risk_target = torch.from_numpy(group["risk_target"]).to(device, non_blocking=torch.cuda.is_available())
            regret_target = torch.from_numpy(group["regret_target"]).to(device, non_blocking=torch.cuda.is_available())
            value_target = torch.from_numpy(group["value_target"]).to(device, non_blocking=torch.cuda.is_available())
            candidate_admissibility_target = torch.from_numpy(group["candidate_admissibility_target"]).to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
            candidate_allowed_target = torch.from_numpy(group["candidate_allowed_target"]).to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
            route_consistency = torch.from_numpy(group["route_consistency"]).to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
            candidate_is_stage1 = torch.from_numpy(group["candidate_is_stage1"]).to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
            weights = torch.from_numpy(group["weights"]).to(device, non_blocking=torch.cuda.is_available())

            (
                utility_pred,
                risk_pred,
                regret_pred,
                value_pred,
                anchor_escape_logit,
                admissibility_logit,
                risk_budget_pred,
                consistency_budget_pred,
                uncertainty_pred,
            ) = model(x_group)
            loss, _ = _escapecert_group_loss(
                utility_pred=utility_pred,
                risk_pred=risk_pred,
                regret_pred=regret_pred,
                value_pred=value_pred,
                anchor_escape_logit=anchor_escape_logit,
                admissibility_logit=admissibility_logit,
                risk_budget_pred=risk_budget_pred,
                consistency_budget_pred=consistency_budget_pred,
                uncertainty_pred=uncertainty_pred,
                utility_target=utility_target,
                risk_target=risk_target,
                regret_target=regret_target,
                value_target=value_target,
                candidate_admissibility_target=candidate_admissibility_target,
                candidate_allowed_target=candidate_allowed_target,
                anchor_escape_target=torch.tensor(
                    float(group["anchor_escape_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                allowed_set_target_mass=torch.tensor(
                    float(group["allowed_set_target_mass"]),
                    device=device,
                    dtype=torch.float32,
                ),
                permission_set_consistency_target=torch.tensor(
                    float(group["permission_set_consistency_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                anchor_escape_topm_mass_target=torch.tensor(
                    float(group["anchor_escape_topm_mass_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                anchor_escape_soft_oracle_gain_target=torch.tensor(
                    float(group["anchor_escape_soft_oracle_gain_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                anchor_escape_uncertainty_target=torch.tensor(
                    float(group["anchor_escape_uncertainty_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                risk_budget_target=torch.tensor(
                    float(group["risk_budget_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                consistency_budget_target=torch.tensor(
                    float(group["consistency_budget_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                route_consistency=route_consistency,
                candidate_is_stage1=candidate_is_stage1,
                selection_index=int(group["selection_index"]),
                weights=weights,
                config=config,
                variant=variant,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            running_count += 1

        train_loss = running_loss / max(running_count, 1)
        val_loss = _evaluate_escapecert_router_loss(
            model=model,
            groups=val_group_data,
            device=device,
            config=config,
            variant=variant,
        )
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(_unwrap_model(model).state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= config.escapecert_patience:
            break

    train_elapsed = time.perf_counter() - train_start
    history_csv = output_dir / f"gate_escapecert_{variant_suffix}_training_history.csv"
    pd.DataFrame(history_rows).to_csv(history_csv, index=False)
    return {
        "variant": variant,
        "escapecert_candidate_csv": str(candidate_csv),
        "escapecert_history_csv": str(history_csv),
        "escapecert_model_state": best_state,
        "escapecert_feature_keys": feature_columns,
        "escapecert_feature_mean": feature_mean.astype(np.float32),
        "escapecert_feature_std": feature_std.astype(np.float32),
        "escapecert_hidden_dim": config.escapecert_hidden_dim,
        "escapecert_dropout": config.escapecert_dropout,
        "escapecert_best_val_loss": float(
            best_val_loss if np.isfinite(best_val_loss) else history_rows[-1]["train_loss"]
        ),
        "escapecert_train_examples": int(sum(len(group["x"]) for group in train_group_data)),
        "escapecert_val_examples": int(sum(len(group["x"]) for group in val_group_data)),
        "escapecert_train_anchors": int(len(train_group_data)),
        "escapecert_val_anchors": int(len(val_group_data)),
        "escapecert_train_time_sec": train_elapsed,
    }


def _frontier_group_predictions_torch(
    *,
    utility_pred: torch.Tensor,
    risk_pred: torch.Tensor,
    regret_pred: torch.Tensor,
    value_pred: torch.Tensor,
    anchor_escape_logit: torch.Tensor,
    teacher_support_logit: torch.Tensor,
    frontier_logit: torch.Tensor,
    risk_budget_pred: torch.Tensor,
    consistency_budget_pred: torch.Tensor,
    frontier_uncertainty_pred: torch.Tensor,
    route_consistency: torch.Tensor,
    candidate_is_stage1: torch.Tensor,
    config: GateLearningConfig,
    variant: str,
) -> dict[str, torch.Tensor]:
    flags = _frontier_variant_flags(variant)
    non_stage_mask = (candidate_is_stage1 < 0.5).float()
    non_stage_idx = torch.nonzero(non_stage_mask > 0.5, as_tuple=False).squeeze(-1)
    zero = utility_pred.new_tensor(0.0)
    zero_vec = torch.zeros_like(utility_pred)
    if non_stage_idx.numel() == 0:
        return {
            "teacher_set_prob": zero_vec,
            "frontier_prob": zero_vec,
            "frontier_margin_vec": zero_vec,
            "frontier_accept_prob": zero_vec,
            "frontier_risk_reserve": zero_vec,
            "frontier_uncertainty_vec": zero_vec,
            "base_admissibility_prob": zero_vec,
            "candidate_admissibility_calibrated": zero_vec,
            "candidate_budget_score": zero_vec,
            "anchor_escape_logit": zero,
            "anchor_escape_prob": zero,
            "anchor_escape_calibrated": zero,
            "anchor_escape_uncertainty": zero,
            "counterfactual_escape_budget_score": zero,
            "counterfactual_escape_support": zero,
            "anchor_escape_counterfactual_margin": zero,
            "agg_risk_budget": utility_pred.new_tensor(config.pairwise_safe_risk_budget),
            "agg_consistency_budget": utility_pred.new_tensor(config.pairwise_consistency_threshold),
            "teacher_set_mass": zero,
            "teacher_topm_mass": zero,
            "allowed_set_total_mass": zero,
            "allowed_set_entropy": zero,
            "allowed_set_topm_mass": zero,
            "allowed_set_best_value": zero,
            "permission_set_consistency": zero,
            "frontier_coverage_score": zero,
            "frontier_teacher_agreement": zero,
            "frontier_false_release_risk": zero,
            "frontier_missed_escape_risk": zero,
            "anchor_escape_safe_candidate_count": zero,
        }

    safe_score = _escapecert_candidate_safe_score_torch(
        utility_pred=utility_pred,
        risk_pred=risk_pred,
        regret_pred=regret_pred,
        value_pred=value_pred,
        route_consistency=route_consistency,
        use_regret_aux=flags["use_regret_aux"],
    )
    value_support = torch.sigmoid(3.2 * value_pred + 0.10)
    teacher_set_prob = (
        torch.sigmoid(teacher_support_logit)
        if flags["use_teacher_head"]
        else torch.clamp(0.52 * safe_score + 0.28 * value_support + 0.20 * route_consistency, 0.0, 1.0)
    ) * non_stage_mask
    teacher_support = torch.clamp(
        teacher_set_prob * (0.40 + 0.60 * safe_score) * non_stage_mask,
        0.0,
        1.0,
    )
    topm = max(1, min(int(config.frontier_teacher_topm), int(non_stage_idx.numel())))
    top_values, top_local = torch.topk(teacher_support[non_stage_idx], k=topm)
    top_idx = non_stage_idx[top_local]
    top_weights = torch.softmax(top_values / max(config.frontier_teacher_temperature, 1e-6), dim=0)

    if flags["fixed_frontier"]:
        agg_risk_budget = utility_pred.new_tensor(config.pairwise_safe_risk_budget)
        agg_consistency_budget = utility_pred.new_tensor(config.pairwise_consistency_threshold)
    else:
        agg_risk_budget = torch.sum(top_weights * risk_budget_pred[top_idx])
        agg_consistency_budget = torch.sum(top_weights * consistency_budget_pred[top_idx])

    risk_reserve = torch.sigmoid(
        (agg_risk_budget - risk_pred) / max(config.pairperm_risk_budget_temperature, 1e-6)
    ) * non_stage_mask
    consistency_reserve = torch.sigmoid(
        (route_consistency - agg_consistency_budget) / max(config.pairperm_consistency_budget_temperature, 1e-6)
    ) * non_stage_mask
    frontier_risk_reserve = torch.clamp(0.56 * risk_reserve + 0.44 * consistency_reserve, 0.0, 1.0) * non_stage_mask

    frontier_prob = (
        torch.sigmoid(frontier_logit)
        if flags["use_frontier_head"]
        else teacher_set_prob
    ) * non_stage_mask
    frontier_uncertainty_vec = (
        torch.clamp(frontier_uncertainty_pred, 0.0, 1.0) * non_stage_mask
        if flags["use_frontier_uncertainty"]
        else zero_vec
    )

    anchor_evidence = (
        torch.sigmoid(anchor_escape_logit)
        if flags["use_anchor_head"]
        else torch.clamp(0.55 * teacher_set_prob + 0.45 * safe_score, 0.0, 1.0)
    ) * non_stage_mask
    top_escape_mass = torch.clamp(
        0.65 * anchor_evidence[top_idx] + 0.35 * teacher_support[top_idx],
        1e-6,
        1.0 - 1e-6,
    )
    anchor_escape_prob = torch.clamp(1.0 - torch.prod(1.0 - top_escape_mass), 1e-6, 1.0 - 1e-6)
    anchor_escape_logit_agg = torch.logit(anchor_escape_prob)
    anchor_escape_uncertainty = (
        torch.sum(top_weights * frontier_uncertainty_vec[top_idx]) if flags["use_frontier_uncertainty"] else zero
    )
    counterfactual_escape_budget_score = torch.sum(top_weights * frontier_risk_reserve[top_idx])
    counterfactual_escape_support = torch.sum(top_weights * teacher_support[top_idx])
    anchor_escape_counterfactual_margin = torch.sum(
        top_weights * (value_pred[top_idx] + 0.30 * regret_pred[top_idx] - 0.22 * risk_pred[top_idx])
    )
    anchor_escape_calibrated = torch.clamp(
        0.64 * anchor_escape_prob
        + 0.22 * torch.sum(top_weights * teacher_set_prob[top_idx])
        + 0.14 * counterfactual_escape_budget_score,
        1e-6,
        1.0 - 1e-6,
    )

    if flags["teacher_only"]:
        frontier_margin_vec = teacher_set_prob - 0.5
        frontier_accept_prob = teacher_set_prob
    else:
        frontier_margin_vec = (
            0.85 * (frontier_prob - 0.5)
            + 0.60 * (teacher_set_prob - 0.5)
            + 0.42 * (frontier_risk_reserve - 0.5)
            + 0.18 * value_pred
            + 0.10 * regret_pred
            - 0.20 * frontier_uncertainty_vec
        )
        if not flags["use_teacher_head"]:
            frontier_margin_vec = frontier_margin_vec + 0.12 * (safe_score - 0.5)
        frontier_accept_prob = torch.sigmoid(
            frontier_margin_vec / max(config.frontier_margin_temperature, 1e-6)
        ) * non_stage_mask

    allowed_sum = torch.sum(frontier_accept_prob[non_stage_idx])
    normalized_mass = frontier_accept_prob[non_stage_idx] / torch.clamp(allowed_sum, min=1e-8)
    entropy_denom = max(float(np.log(max(int(non_stage_idx.numel()), 2))), 1e-6)
    raw_entropy = -torch.sum(normalized_mass * torch.log(torch.clamp(normalized_mass, min=1e-8))) / entropy_denom
    allowed_set_entropy = torch.where(allowed_sum <= 1e-8, zero, raw_entropy)
    teacher_set_mass = torch.mean(teacher_set_prob[non_stage_idx])
    teacher_topm_mass = torch.sum(top_weights * teacher_set_prob[top_idx])
    allowed_set_total_mass = torch.mean(frontier_accept_prob[non_stage_idx])
    allowed_set_topm_mass = torch.sum(top_weights * frontier_accept_prob[top_idx])
    allowed_set_best_value = torch.max(value_pred[non_stage_idx])
    frontier_coverage_score = torch.clamp(
        1.0 - torch.prod(1.0 - torch.clamp(frontier_accept_prob[non_stage_idx], 1e-6, 1.0 - 1e-6)),
        1e-6,
        1.0 - 1e-6,
    )
    frontier_teacher_agreement = torch.clamp(
        1.0
        - torch.mean(torch.abs(frontier_accept_prob[non_stage_idx] - teacher_set_prob[non_stage_idx])),
        0.0,
        1.0,
    )
    permission_set_consistency = torch.clamp(
        0.55 * frontier_coverage_score + 0.45 * frontier_teacher_agreement,
        1e-6,
        1.0 - 1e-6,
    )
    frontier_false_release_risk = torch.mean(
        frontier_accept_prob[non_stage_idx]
        * (1.0 - frontier_risk_reserve[non_stage_idx])
        * (0.65 * risk_pred[non_stage_idx] + 0.35 * (1.0 - route_consistency[non_stage_idx]))
    )
    frontier_missed_escape_risk = torch.mean(
        (1.0 - frontier_accept_prob[non_stage_idx])
        * teacher_set_prob[non_stage_idx]
        * torch.clamp(value_pred[non_stage_idx] + 0.30 * regret_pred[non_stage_idx], 0.0, 1.0)
    )
    anchor_escape_safe_candidate_count = torch.sum((frontier_accept_prob[non_stage_idx] >= 0.5).float())
    return {
        "teacher_set_prob": torch.clamp(teacher_set_prob, 1e-6, 1.0 - 1e-6),
        "frontier_prob": torch.clamp(frontier_prob, 1e-6, 1.0 - 1e-6),
        "frontier_margin_vec": frontier_margin_vec,
        "frontier_accept_prob": torch.clamp(frontier_accept_prob, 1e-6, 1.0 - 1e-6),
        "frontier_risk_reserve": frontier_risk_reserve,
        "frontier_uncertainty_vec": frontier_uncertainty_vec,
        "base_admissibility_prob": torch.clamp(teacher_set_prob, 1e-6, 1.0 - 1e-6),
        "candidate_admissibility_calibrated": torch.clamp(frontier_accept_prob, 1e-6, 1.0 - 1e-6),
        "candidate_budget_score": frontier_risk_reserve,
        "anchor_escape_logit": anchor_escape_logit_agg,
        "anchor_escape_prob": torch.clamp(anchor_escape_prob, 1e-6, 1.0 - 1e-6),
        "anchor_escape_calibrated": anchor_escape_calibrated,
        "anchor_escape_uncertainty": torch.clamp(anchor_escape_uncertainty, 0.0, 1.0),
        "counterfactual_escape_budget_score": torch.clamp(counterfactual_escape_budget_score, 0.0, 1.0),
        "counterfactual_escape_support": torch.clamp(counterfactual_escape_support, 0.0, 1.0),
        "anchor_escape_counterfactual_margin": anchor_escape_counterfactual_margin,
        "agg_risk_budget": agg_risk_budget,
        "agg_consistency_budget": agg_consistency_budget,
        "teacher_set_mass": torch.clamp(teacher_set_mass, 0.0, 1.0),
        "teacher_topm_mass": torch.clamp(teacher_topm_mass, 0.0, 1.0),
        "allowed_set_total_mass": torch.clamp(allowed_set_total_mass, 0.0, 1.0),
        "allowed_set_entropy": torch.clamp(allowed_set_entropy, 0.0, 1.0),
        "allowed_set_topm_mass": torch.clamp(allowed_set_topm_mass, 0.0, 1.0),
        "allowed_set_best_value": allowed_set_best_value,
        "permission_set_consistency": permission_set_consistency,
        "frontier_coverage_score": frontier_coverage_score,
        "frontier_teacher_agreement": frontier_teacher_agreement,
        "frontier_false_release_risk": torch.clamp(frontier_false_release_risk, 0.0, 1.0),
        "frontier_missed_escape_risk": torch.clamp(frontier_missed_escape_risk, 0.0, 1.0),
        "anchor_escape_safe_candidate_count": anchor_escape_safe_candidate_count,
    }


def _frontier_group_loss(
    *,
    utility_pred: torch.Tensor,
    risk_pred: torch.Tensor,
    regret_pred: torch.Tensor,
    value_pred: torch.Tensor,
    anchor_escape_logit: torch.Tensor,
    teacher_support_logit: torch.Tensor,
    frontier_logit: torch.Tensor,
    risk_budget_pred: torch.Tensor,
    consistency_budget_pred: torch.Tensor,
    frontier_uncertainty_pred: torch.Tensor,
    utility_target: torch.Tensor,
    risk_target: torch.Tensor,
    regret_target: torch.Tensor,
    value_target: torch.Tensor,
    teacher_set_target: torch.Tensor,
    frontier_target: torch.Tensor,
    frontier_uncertainty_target: torch.Tensor,
    anchor_escape_target: torch.Tensor,
    teacher_set_mass_target: torch.Tensor,
    teacher_topm_mass_target: torch.Tensor,
    allowed_set_target_mass: torch.Tensor,
    frontier_coverage_target: torch.Tensor,
    permission_set_consistency_target: torch.Tensor,
    anchor_escape_soft_oracle_gain_target: torch.Tensor,
    anchor_escape_uncertainty_target: torch.Tensor,
    frontier_false_release_target: torch.Tensor,
    frontier_missed_escape_target: torch.Tensor,
    risk_budget_target: torch.Tensor,
    consistency_budget_target: torch.Tensor,
    route_consistency: torch.Tensor,
    candidate_is_stage1: torch.Tensor,
    selection_index: int,
    weights: torch.Tensor,
    config: GateLearningConfig,
    variant: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    flags = _frontier_variant_flags(variant)
    outputs = _frontier_group_predictions_torch(
        utility_pred=utility_pred,
        risk_pred=risk_pred,
        regret_pred=regret_pred,
        value_pred=value_pred,
        anchor_escape_logit=anchor_escape_logit,
        teacher_support_logit=teacher_support_logit,
        frontier_logit=frontier_logit,
        risk_budget_pred=risk_budget_pred,
        consistency_budget_pred=consistency_budget_pred,
        frontier_uncertainty_pred=frontier_uncertainty_pred,
        route_consistency=route_consistency,
        candidate_is_stage1=candidate_is_stage1,
        config=config,
        variant=variant,
    )
    non_stage_mask = (candidate_is_stage1 < 0.5).float()
    weight_denom = torch.clamp(torch.sum(weights), min=1e-6)
    non_stage_weight_denom = torch.clamp(torch.sum(weights * non_stage_mask), min=1e-6)

    utility_loss = torch.sum(F.smooth_l1_loss(utility_pred, utility_target, reduction="none") * weights) / weight_denom
    risk_loss = torch.sum(F.smooth_l1_loss(risk_pred, risk_target, reduction="none") * weights) / weight_denom
    value_loss = torch.sum(F.smooth_l1_loss(value_pred, value_target, reduction="none") * weights) / weight_denom
    regret_loss = utility_loss.new_tensor(0.0)
    if flags["use_regret_aux"]:
        regret_loss = torch.sum(
            F.smooth_l1_loss(regret_pred, regret_target, reduction="none") * weights
        ) / weight_denom

    anchor_loss = utility_loss.new_tensor(0.0)
    if flags["use_anchor_head"]:
        anchor_loss = F.binary_cross_entropy(outputs["anchor_escape_calibrated"], anchor_escape_target)
        anchor_loss = anchor_loss + 0.35 * F.binary_cross_entropy(outputs["anchor_escape_prob"], anchor_escape_target)

    teacher_loss = utility_loss.new_tensor(0.0)
    if flags["use_teacher_distill"]:
        teacher_loss = torch.sum(
            F.binary_cross_entropy(
                outputs["teacher_set_prob"],
                torch.clamp(teacher_set_target, 1e-6, 1.0 - 1e-6),
                reduction="none",
            )
            * weights
            * non_stage_mask
        ) / non_stage_weight_denom

    frontier_loss = utility_loss.new_tensor(0.0)
    if flags["use_frontier_head"] or flags["teacher_only"] or not flags["use_teacher_head"]:
        base_term = F.binary_cross_entropy(
            outputs["frontier_prob"],
            torch.clamp(frontier_target, 1e-6, 1.0 - 1e-6),
            reduction="none",
        )
        accept_term = F.binary_cross_entropy(
            outputs["frontier_accept_prob"],
            torch.clamp(frontier_target, 1e-6, 1.0 - 1e-6),
            reduction="none",
        )
        asym_weight = (
            1.0
            + config.frontier_missed_escape_weight * teacher_set_target * torch.clamp(value_target + 0.25, 0.0, 1.5)
            + config.frontier_false_release_weight * frontier_false_release_target * (1.0 - frontier_target)
        )
        frontier_loss = torch.sum((base_term + 0.65 * accept_term) * asym_weight * weights * non_stage_mask) / torch.clamp(
            torch.sum(asym_weight * weights * non_stage_mask),
            min=1e-6,
        )

    budget_loss = F.smooth_l1_loss(outputs["agg_risk_budget"], risk_budget_target)
    budget_loss = budget_loss + F.smooth_l1_loss(outputs["agg_consistency_budget"], consistency_budget_target)

    uncertainty_loss = utility_loss.new_tensor(0.0)
    if flags["use_frontier_uncertainty"]:
        uncertainty_loss = torch.sum(
            F.smooth_l1_loss(
                outputs["frontier_uncertainty_vec"],
                frontier_uncertainty_target,
                reduction="none",
            )
            * weights
            * non_stage_mask
        ) / non_stage_weight_denom
        uncertainty_loss = uncertainty_loss + 0.35 * F.smooth_l1_loss(
            outputs["anchor_escape_uncertainty"],
            anchor_escape_uncertainty_target,
        )

    coverage_loss = F.smooth_l1_loss(outputs["teacher_set_mass"], teacher_set_mass_target)
    coverage_loss = coverage_loss + 0.65 * F.smooth_l1_loss(outputs["teacher_topm_mass"], teacher_topm_mass_target)
    coverage_loss = coverage_loss + F.smooth_l1_loss(outputs["allowed_set_total_mass"], allowed_set_target_mass)
    coverage_loss = coverage_loss + 0.55 * F.binary_cross_entropy(
        outputs["frontier_coverage_score"],
        frontier_coverage_target,
    )
    coverage_loss = coverage_loss + 0.55 * F.binary_cross_entropy(
        outputs["permission_set_consistency"],
        permission_set_consistency_target,
    )
    coverage_loss = coverage_loss + 0.35 * F.smooth_l1_loss(
        outputs["allowed_set_best_value"],
        anchor_escape_soft_oracle_gain_target,
    )

    tradeoff_loss = F.smooth_l1_loss(outputs["frontier_false_release_risk"], frontier_false_release_target)
    tradeoff_loss = tradeoff_loss + F.smooth_l1_loss(
        outputs["frontier_missed_escape_risk"],
        frontier_missed_escape_target,
    )

    ranking_loss = utility_loss.new_tensor(0.0)
    if selection_index >= 0:
        non_stage_idx = torch.nonzero(non_stage_mask > 0.5, as_tuple=False).squeeze(-1)
        if non_stage_idx.numel() > 0:
            selected_tensor = utility_pred.new_tensor(float(selection_index), dtype=torch.long)
            selected_local = torch.nonzero(non_stage_idx == selected_tensor, as_tuple=False)
            if selected_local.numel() > 0:
                ranking_logits = value_pred[non_stage_idx] + torch.log(
                    torch.clamp(outputs["frontier_accept_prob"][non_stage_idx], min=1e-6)
                )
                ranking_loss = F.cross_entropy(
                    ranking_logits.unsqueeze(0),
                    selected_local[0:1, 0],
                )

    total = utility_loss + risk_loss + config.frontier_value_weight * (value_loss + 0.40 * ranking_loss)
    if flags["use_regret_aux"]:
        total = total + config.frontier_regret_weight * regret_loss
    total = total + config.frontier_anchor_weight * anchor_loss
    total = total + config.frontier_teacher_weight * teacher_loss
    total = total + config.frontier_accept_weight * frontier_loss
    total = total + config.frontier_budget_weight * budget_loss
    total = total + config.frontier_coverage_weight * coverage_loss
    total = total + config.frontier_uncertainty_weight * uncertainty_loss
    total = total + 0.25 * tradeoff_loss
    return total, outputs


def _evaluate_frontier_router_loss(
    *,
    model: nn.Module,
    groups: list[dict[str, object]],
    device: torch.device,
    config: GateLearningConfig,
    variant: str,
) -> float:
    if not groups:
        return 0.0

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for group in groups:
            x_group = torch.from_numpy(group["x"]).to(device, non_blocking=torch.cuda.is_available())
            utility_target = torch.from_numpy(group["utility_target"]).to(device, non_blocking=torch.cuda.is_available())
            risk_target = torch.from_numpy(group["risk_target"]).to(device, non_blocking=torch.cuda.is_available())
            regret_target = torch.from_numpy(group["regret_target"]).to(device, non_blocking=torch.cuda.is_available())
            value_target = torch.from_numpy(group["value_target"]).to(device, non_blocking=torch.cuda.is_available())
            teacher_set_target = torch.from_numpy(group["teacher_set_target"]).to(device, non_blocking=torch.cuda.is_available())
            frontier_target = torch.from_numpy(group["frontier_target"]).to(device, non_blocking=torch.cuda.is_available())
            frontier_uncertainty_target = torch.from_numpy(group["frontier_uncertainty_target"]).to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
            route_consistency = torch.from_numpy(group["route_consistency"]).to(device, non_blocking=torch.cuda.is_available())
            candidate_is_stage1 = torch.from_numpy(group["candidate_is_stage1"]).to(device, non_blocking=torch.cuda.is_available())
            weights = torch.from_numpy(group["weights"]).to(device, non_blocking=torch.cuda.is_available())
            (
                utility_pred,
                risk_pred,
                regret_pred,
                value_pred,
                anchor_escape_logit,
                teacher_support_logit,
                frontier_logit,
                risk_budget_pred,
                consistency_budget_pred,
                frontier_uncertainty_pred,
            ) = model(x_group)
            loss, _ = _frontier_group_loss(
                utility_pred=utility_pred,
                risk_pred=risk_pred,
                regret_pred=regret_pred,
                value_pred=value_pred,
                anchor_escape_logit=anchor_escape_logit,
                teacher_support_logit=teacher_support_logit,
                frontier_logit=frontier_logit,
                risk_budget_pred=risk_budget_pred,
                consistency_budget_pred=consistency_budget_pred,
                frontier_uncertainty_pred=frontier_uncertainty_pred,
                utility_target=utility_target,
                risk_target=risk_target,
                regret_target=regret_target,
                value_target=value_target,
                teacher_set_target=teacher_set_target,
                frontier_target=frontier_target,
                frontier_uncertainty_target=frontier_uncertainty_target,
                anchor_escape_target=torch.tensor(float(group["anchor_escape_target"]), device=device, dtype=torch.float32),
                teacher_set_mass_target=torch.tensor(float(group["teacher_set_mass_target"]), device=device, dtype=torch.float32),
                teacher_topm_mass_target=torch.tensor(float(group["teacher_topm_mass_target"]), device=device, dtype=torch.float32),
                allowed_set_target_mass=torch.tensor(float(group["allowed_set_target_mass"]), device=device, dtype=torch.float32),
                frontier_coverage_target=torch.tensor(float(group["frontier_coverage_target"]), device=device, dtype=torch.float32),
                permission_set_consistency_target=torch.tensor(
                    float(group["permission_set_consistency_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                anchor_escape_soft_oracle_gain_target=torch.tensor(
                    float(group["anchor_escape_soft_oracle_gain_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                anchor_escape_uncertainty_target=torch.tensor(
                    float(group["anchor_escape_uncertainty_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                frontier_false_release_target=torch.tensor(
                    float(group["frontier_false_release_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                frontier_missed_escape_target=torch.tensor(
                    float(group["frontier_missed_escape_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                risk_budget_target=torch.tensor(float(group["risk_budget_target"]), device=device, dtype=torch.float32),
                consistency_budget_target=torch.tensor(
                    float(group["consistency_budget_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                route_consistency=route_consistency,
                candidate_is_stage1=candidate_is_stage1,
                selection_index=int(group["selection_index"]),
                weights=weights,
                config=config,
                variant=variant,
            )
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def _train_stage1_pairwise_router_frontier(
    *,
    output_dir: Path,
    config: GateLearningConfig,
    variant: str = "full",
) -> dict[str, object]:
    bank_csv = output_dir / "gate_bank_candidates.csv"
    bank_df = pd.read_csv(bank_csv)
    frontier_df = _build_stage1_pairwise_frontier_training_frame(
        bank_df=bank_df,
        config=config,
        variant=variant,
    )
    variant_suffix = variant.replace("no_", "no-").replace("_", "-")
    candidate_csv = output_dir / f"gate_frontier_{variant_suffix}_candidates.csv"
    frontier_df.to_csv(candidate_csv, index=False)

    feature_columns = _pairwise_feature_columns(prototype_candidates=config.bank_prototype_candidates)
    feature_matrix = frontier_df[feature_columns].to_numpy(dtype=np.float32)
    group_ids = frontier_df["dataset_id"].to_numpy(dtype=np.int64)

    unique_groups = np.unique(group_ids)
    variant_seed = sum(ord(ch) for ch in variant)
    rng = np.random.default_rng(config.random_state + 10991 + variant_seed)
    shuffled_groups = rng.permutation(unique_groups)
    val_size = 0
    if len(shuffled_groups) >= 5:
        val_size = max(1, int(round(len(shuffled_groups) * config.val_fraction)))
        val_size = min(val_size, len(shuffled_groups) - 1)
    val_groups = set(int(group_id) for group_id in shuffled_groups[:val_size])
    train_groups = set(int(group_id) for group_id in shuffled_groups[val_size:])
    if not train_groups:
        train_groups = set(int(group_id) for group_id in shuffled_groups)
        val_groups = set()

    train_mask = np.asarray([int(group_id) in train_groups for group_id in group_ids], dtype=bool)
    feature_mean = feature_matrix[train_mask].mean(axis=0)
    feature_std = feature_matrix[train_mask].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    normalized = ((feature_matrix - feature_mean[None, :]) / feature_std[None, :]).astype(np.float32)

    grouped_data: list[dict[str, object]] = []
    for dataset_id, group in frontier_df.reset_index(drop=True).groupby("dataset_id", sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        selection_rows = np.where(group["pairwise_route_selected_under_permission"].to_numpy(dtype=np.float32) > 0.5)[0]
        grouped_data.append(
            {
                "dataset_id": int(dataset_id),
                "x": normalized[idx],
                "utility_target": group["pairwise_utility_gain_target"].to_numpy(dtype=np.float32),
                "risk_target": group["pairwise_failure_risk_target"].to_numpy(dtype=np.float32),
                "regret_target": group["pairwise_fallback_regret_target"].to_numpy(dtype=np.float32),
                "value_target": group["pairwise_route_value_target"].to_numpy(dtype=np.float32),
                "teacher_set_target": group["teacher_set_target"].to_numpy(dtype=np.float32),
                "frontier_target": group["frontier_target"].to_numpy(dtype=np.float32),
                "frontier_uncertainty_target": group["frontier_uncertainty_target"].to_numpy(dtype=np.float32),
                "route_consistency": group["pairwise_route_consistency"].to_numpy(dtype=np.float32),
                "candidate_is_stage1": (group["candidate_name"] == "stage1_blended").to_numpy(dtype=np.float32),
                "weights": group["pairwise_sample_weight"].to_numpy(dtype=np.float32),
                "anchor_escape_target": float(group["anchor_escape_target"].iloc[0]),
                "teacher_set_mass_target": float(group["teacher_set_mass"].iloc[0]),
                "teacher_topm_mass_target": float(group["teacher_topm_mass"].iloc[0]),
                "allowed_set_target_mass": float(group["allowed_set_target_mass"].iloc[0]),
                "frontier_coverage_target": float(group["frontier_coverage_score"].iloc[0]),
                "permission_set_consistency_target": float(group["permission_set_consistency_target"].iloc[0]),
                "anchor_escape_soft_oracle_gain_target": float(group["anchor_escape_soft_oracle_gain"].iloc[0]),
                "anchor_escape_uncertainty_target": float(group["anchor_escape_uncertainty_target"].iloc[0]),
                "frontier_false_release_target": float(group["frontier_false_release_risk"].mean()),
                "frontier_missed_escape_target": float(group["frontier_missed_escape_risk"].mean()),
                "risk_budget_target": float(group["pairwise_adaptive_risk_budget"].iloc[0]),
                "consistency_budget_target": float(group["pairwise_adaptive_consistency_budget"].iloc[0]),
                "selection_index": int(selection_rows[0]) if selection_rows.size else -1,
            }
        )

    train_group_data = [group for group in grouped_data if int(group["dataset_id"]) in train_groups]
    val_group_data = [group for group in grouped_data if int(group["dataset_id"]) in val_groups]

    device = choose_torch_device()
    base_model = FrontierStage1AnchoredPairwiseRouter(
        input_dim=feature_matrix.shape[-1],
        hidden_dim=config.frontier_hidden_dim,
        dropout=config.frontier_dropout,
        risk_budget_floor=config.pairperm_risk_budget_floor,
        risk_budget_ceiling=config.pairperm_risk_budget_ceiling,
        consistency_budget_floor=config.pairperm_consistency_budget_floor,
        consistency_budget_ceiling=config.pairperm_consistency_budget_ceiling,
    )
    if torch.cuda.is_available():
        base_model = base_model.to(device)
        model: nn.Module = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model
    else:
        base_model = base_model.to(device)
        model = base_model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.frontier_learning_rate,
        weight_decay=config.frontier_weight_decay,
    )

    best_state = copy.deepcopy(_unwrap_model(model).state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0
    history_rows: list[dict[str, float | int]] = []
    train_start = time.perf_counter()

    for epoch in range(config.frontier_train_epochs):
        model.train()
        epoch_order = rng.permutation(len(train_group_data)) if train_group_data else np.asarray([], dtype=np.int64)
        running_loss = 0.0
        running_count = 0

        for group_idx in epoch_order:
            group = train_group_data[int(group_idx)]
            x_group = torch.from_numpy(group["x"]).to(device, non_blocking=torch.cuda.is_available())
            utility_target = torch.from_numpy(group["utility_target"]).to(device, non_blocking=torch.cuda.is_available())
            risk_target = torch.from_numpy(group["risk_target"]).to(device, non_blocking=torch.cuda.is_available())
            regret_target = torch.from_numpy(group["regret_target"]).to(device, non_blocking=torch.cuda.is_available())
            value_target = torch.from_numpy(group["value_target"]).to(device, non_blocking=torch.cuda.is_available())
            teacher_set_target = torch.from_numpy(group["teacher_set_target"]).to(device, non_blocking=torch.cuda.is_available())
            frontier_target = torch.from_numpy(group["frontier_target"]).to(device, non_blocking=torch.cuda.is_available())
            frontier_uncertainty_target = torch.from_numpy(group["frontier_uncertainty_target"]).to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
            route_consistency = torch.from_numpy(group["route_consistency"]).to(device, non_blocking=torch.cuda.is_available())
            candidate_is_stage1 = torch.from_numpy(group["candidate_is_stage1"]).to(device, non_blocking=torch.cuda.is_available())
            weights = torch.from_numpy(group["weights"]).to(device, non_blocking=torch.cuda.is_available())
            (
                utility_pred,
                risk_pred,
                regret_pred,
                value_pred,
                anchor_escape_logit,
                teacher_support_logit,
                frontier_logit,
                risk_budget_pred,
                consistency_budget_pred,
                frontier_uncertainty_pred,
            ) = model(x_group)
            loss, _ = _frontier_group_loss(
                utility_pred=utility_pred,
                risk_pred=risk_pred,
                regret_pred=regret_pred,
                value_pred=value_pred,
                anchor_escape_logit=anchor_escape_logit,
                teacher_support_logit=teacher_support_logit,
                frontier_logit=frontier_logit,
                risk_budget_pred=risk_budget_pred,
                consistency_budget_pred=consistency_budget_pred,
                frontier_uncertainty_pred=frontier_uncertainty_pred,
                utility_target=utility_target,
                risk_target=risk_target,
                regret_target=regret_target,
                value_target=value_target,
                teacher_set_target=teacher_set_target,
                frontier_target=frontier_target,
                frontier_uncertainty_target=frontier_uncertainty_target,
                anchor_escape_target=torch.tensor(float(group["anchor_escape_target"]), device=device, dtype=torch.float32),
                teacher_set_mass_target=torch.tensor(float(group["teacher_set_mass_target"]), device=device, dtype=torch.float32),
                teacher_topm_mass_target=torch.tensor(float(group["teacher_topm_mass_target"]), device=device, dtype=torch.float32),
                allowed_set_target_mass=torch.tensor(float(group["allowed_set_target_mass"]), device=device, dtype=torch.float32),
                frontier_coverage_target=torch.tensor(float(group["frontier_coverage_target"]), device=device, dtype=torch.float32),
                permission_set_consistency_target=torch.tensor(
                    float(group["permission_set_consistency_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                anchor_escape_soft_oracle_gain_target=torch.tensor(
                    float(group["anchor_escape_soft_oracle_gain_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                anchor_escape_uncertainty_target=torch.tensor(
                    float(group["anchor_escape_uncertainty_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                frontier_false_release_target=torch.tensor(
                    float(group["frontier_false_release_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                frontier_missed_escape_target=torch.tensor(
                    float(group["frontier_missed_escape_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                risk_budget_target=torch.tensor(float(group["risk_budget_target"]), device=device, dtype=torch.float32),
                consistency_budget_target=torch.tensor(
                    float(group["consistency_budget_target"]),
                    device=device,
                    dtype=torch.float32,
                ),
                route_consistency=route_consistency,
                candidate_is_stage1=candidate_is_stage1,
                selection_index=int(group["selection_index"]),
                weights=weights,
                config=config,
                variant=variant,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            running_count += 1

        train_loss = running_loss / max(running_count, 1)
        val_loss = _evaluate_frontier_router_loss(
            model=model,
            groups=val_group_data,
            device=device,
            config=config,
            variant=variant,
        )
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(_unwrap_model(model).state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= config.frontier_patience:
            break

    train_elapsed = time.perf_counter() - train_start
    history_csv = output_dir / f"gate_frontier_{variant_suffix}_training_history.csv"
    pd.DataFrame(history_rows).to_csv(history_csv, index=False)
    return {
        "variant": variant,
        "frontier_candidate_csv": str(candidate_csv),
        "frontier_history_csv": str(history_csv),
        "frontier_model_state": best_state,
        "frontier_feature_keys": feature_columns,
        "frontier_feature_mean": feature_mean.astype(np.float32),
        "frontier_feature_std": feature_std.astype(np.float32),
        "frontier_hidden_dim": config.frontier_hidden_dim,
        "frontier_dropout": config.frontier_dropout,
        "frontier_best_val_loss": float(
            best_val_loss if np.isfinite(best_val_loss) else history_rows[-1]["train_loss"]
        ),
        "frontier_train_examples": int(sum(len(group["x"]) for group in train_group_data)),
        "frontier_val_examples": int(sum(len(group["x"]) for group in val_group_data)),
        "frontier_train_anchors": int(len(train_group_data)),
        "frontier_val_anchors": int(len(val_group_data)),
        "frontier_train_time_sec": train_elapsed,
    }


def _point_gate_components_from_runtime(
    *,
    dataset_stats: dict[str, float],
    checkpoint_path: str,
    heuristic_gate: np.ndarray,
    device: torch.device | None = None,
) -> dict[str, object]:
    if device is None:
        device = choose_torch_device()

    runtime = _load_runtime_checkpoint(checkpoint_path=checkpoint_path, device=device)
    feature_keys = tuple(runtime["feature_keys"])
    features = dataset_stats_to_vector(dataset_stats, feature_keys=feature_keys).astype(np.float32)
    feature_mean = runtime["feature_mean"]
    feature_std = runtime["feature_std"]
    normalized = (features - feature_mean) / feature_std

    x_tensor = torch.from_numpy(normalized[None, :]).to(device, non_blocking=torch.cuda.is_available())
    heuristic_gate = _normalize_gate_np(heuristic_gate)

    model: nn.Module = runtime["model"]  # type: ignore[assignment]
    model.eval()
    with torch.no_grad():
        delta_logits = model(x_tensor)
        residual_gate = _compose_gate_from_delta_torch(
            delta_logits=delta_logits,
            heuristic_gate=torch.from_numpy(heuristic_gate[None, :].astype(np.float32)).to(
                device,
                non_blocking=torch.cuda.is_available(),
            ),
            residual_scale=float(runtime["config"]["residual_scale"]),
        )[0].detach().cpu().numpy()

    prototype_gate, nearest_distance, prototype_neighbor_gates, prototype_neighbor_distances = _prototype_memory_bank(
        normalized_feature=normalized,
        prototype_features=np.asarray(runtime["prototype_features"], dtype=np.float32),
        prototype_gates=np.asarray(runtime["prototype_gates"], dtype=np.float32),
        k=int(runtime["config"]["prototype_k"]),
        candidate_count=int(runtime["config"].get("bank_prototype_candidates", 3)),
    )
    prototype_gate = _normalize_gate_np(prototype_gate)

    memory_weight = float(runtime["config"]["prototype_blend"]) * np.exp(-0.35 * nearest_distance)
    fallback_weight = float(runtime["config"]["heuristic_fallback"]) * (1.0 - np.exp(-0.35 * nearest_distance))
    model_weight = max(1e-6, 1.0 - memory_weight - fallback_weight)

    blended = model_weight * residual_gate + memory_weight * prototype_gate + fallback_weight * heuristic_gate
    blended = _apply_gate_safety_projection(
        gate=blended,
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        safety_floor_strength=float(runtime["config"]["safety_floor_strength"]),
    )
    return {
        "runtime": runtime,
        "normalized_feature": normalized.astype(np.float32),
        "residual_gate": _normalize_gate_np(residual_gate),
        "prototype_gate": prototype_gate,
        "stage1_blended": _normalize_gate_np(blended),
        "nearest_distance": float(nearest_distance),
        "prototype_neighbor_gates": _normalize_gate_matrix_np(prototype_neighbor_gates),
        "prototype_neighbor_distances": np.asarray(prototype_neighbor_distances, dtype=np.float32),
    }


def predict_gate_weights(
    *,
    dataset_stats: dict[str, float],
    checkpoint_path: str,
    heuristic_gate: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    components = _point_gate_components_from_runtime(
        dataset_stats=dataset_stats,
        checkpoint_path=checkpoint_path,
        heuristic_gate=heuristic_gate,
        device=device,
    )
    return np.asarray(components["stage1_blended"], dtype=np.float64)


def _subsample_bank_view_indices(
    *,
    n_cells: int,
    batches: np.ndarray | None,
    ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_cells <= 4:
        return np.arange(n_cells, dtype=np.int64)

    target_size = int(round(n_cells * ratio))
    target_size = min(n_cells, max(target_size, min(n_cells, 48)))
    if target_size >= n_cells:
        return np.arange(n_cells, dtype=np.int64)

    if batches is None or len(np.unique(batches)) <= 1:
        return np.sort(rng.choice(n_cells, size=target_size, replace=False).astype(np.int64))

    selected_parts: list[np.ndarray] = []
    remaining_budget = target_size
    batches = np.asarray(batches)
    unique_batches = np.unique(batches)
    for batch_idx, batch_value in enumerate(unique_batches):
        cell_idx = np.flatnonzero(batches == batch_value)
        if cell_idx.size == 0:
            continue
        batch_target = int(round(cell_idx.size * ratio))
        batch_target = min(cell_idx.size, max(batch_target, 1))
        if batch_idx == len(unique_batches) - 1:
            batch_target = min(cell_idx.size, max(remaining_budget, 1))
        selected = rng.choice(cell_idx, size=batch_target, replace=False).astype(np.int64)
        selected_parts.append(selected)
        remaining_budget -= batch_target

    chosen = np.concatenate(selected_parts) if selected_parts else np.arange(n_cells, dtype=np.int64)
    if chosen.size > target_size:
        chosen = rng.choice(chosen, size=target_size, replace=False).astype(np.int64)
    elif chosen.size < target_size:
        missing = np.setdiff1d(np.arange(n_cells, dtype=np.int64), chosen, assume_unique=False)
        if missing.size:
            extra = rng.choice(missing, size=min(target_size - chosen.size, missing.size), replace=False).astype(np.int64)
            chosen = np.concatenate([chosen, extra])
    return np.sort(chosen.astype(np.int64))


def _candidate_route_view_statistics(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None,
    labels: np.ndarray,
    selector: RefineMoEHVGSelector,
    candidate_names: np.ndarray,
    candidate_gates: np.ndarray,
    config: GateLearningConfig,
    random_state: int,
) -> dict[str, dict[str, float]]:
    margins_by_name: dict[str, list[float]] = {str(name): [] for name in candidate_names}
    view_rng = np.random.default_rng(random_state + 4049)
    n_views = max(1, int(config.bank_uncertainty_views))
    use_subsampled_views = counts.shape[0] > 64 and config.bank_subsample_ratio < 0.999

    for _ in range(n_views):
        if use_subsampled_views:
            view_idx = _subsample_bank_view_indices(
                n_cells=counts.shape[0],
                batches=batches,
                ratio=float(config.bank_subsample_ratio),
                rng=view_rng,
            )
        else:
            view_idx = np.arange(counts.shape[0], dtype=np.int64)

        view_counts = np.asarray(counts[view_idx], dtype=np.float64)
        view_batches = None if batches is None else np.asarray(batches)[view_idx]
        view_labels = np.asarray(labels)[view_idx]
        current_x, current_base_features, current_dataset_stats, current_expert_scores = selector.prepare_context(
            view_counts,
            view_batches,
        )

        view_records: list[dict[str, object]] = []
        current_top_k = min(selector.top_k, view_counts.shape[1])
        for candidate_name, gate in zip(candidate_names, candidate_gates, strict=False):
            no_refine_scores = selector.score_with_context(
                x=current_x,
                base_features=current_base_features,
                dataset_stats=current_dataset_stats,
                expert_scores=current_expert_scores,
                gate=np.asarray(gate, dtype=np.float64),
                apply_refine=False,
            )
            selected_idx = np.argsort(no_refine_scores)[-current_top_k:]
            route_metrics = _fast_weak_metrics(
                counts=view_counts,
                selected_genes=selected_idx,
                labels=view_labels,
                batches=view_batches,
                random_state=random_state,
            )
            view_records.append(
                {
                    "candidate_name": str(candidate_name),
                    "gate": np.asarray(gate, dtype=np.float64),
                    "metrics": route_metrics,
                    "reward": downstream_reward(route_metrics),
                }
            )

        route_baseline_record, route_ari_floor, route_nmi_floor, route_structure_floor = _baseline_metric_floors(
            view_records,
            floor_ratio=config.floor_ratio,
            ari_margin=config.floor_margin_ari,
            nmi_margin=config.floor_margin_nmi,
            structure_margin=config.floor_margin_structure,
            baseline_names={"heuristic", "stage1_blended", "prototype_mean"},
        )
        route_baseline_metrics = route_baseline_record["metrics"]  # type: ignore[assignment]
        batch_tradeoff_pressure = _batch_tradeoff_pressure(current_dataset_stats)

        route_adjusted: dict[str, float] = {}
        for record in view_records:
            metrics = record["metrics"]  # type: ignore[assignment]
            reward = float(record["reward"])
            adjusted_reward, _, _, _, _ = _adjust_reward_from_metrics(
                metrics=metrics,
                reward=reward,
                ari_floor=route_ari_floor,
                nmi_floor=route_nmi_floor,
                structure_floor=route_structure_floor,
                baseline_metrics=route_baseline_metrics,
                batch_tradeoff_pressure=batch_tradeoff_pressure,
                floor_weight=config.floor_weight,
                tradeoff_penalty_weight=config.tradeoff_penalty_weight,
            )
            route_adjusted[str(record["candidate_name"])] = adjusted_reward

        stage1_adjusted = float(route_adjusted.get("stage1_blended", 0.0))
        for candidate_name in margins_by_name:
            margins_by_name[candidate_name].append(float(route_adjusted.get(candidate_name, stage1_adjusted) - stage1_adjusted))

    statistics: dict[str, dict[str, float]] = {}
    for candidate_name, margins in margins_by_name.items():
        margin_array = np.asarray(margins if margins else [0.0], dtype=np.float64)
        margin_mean = float(np.mean(margin_array))
        margin_std = float(np.std(margin_array))
        if candidate_name == "stage1_blended":
            win_rate = 0.0
            consistency = 1.0
        else:
            win_rate = float(np.mean(margin_array > 0.0))
            consistency = _pairwise_consistency_score(
                win_rate=win_rate,
                margin_mean=margin_mean,
                margin_std=margin_std,
                config=config,
            )
        statistics[candidate_name] = {
            "route_view_win_rate": win_rate,
            "route_view_margin_mean": margin_mean,
            "route_view_margin_std": margin_std,
            "route_view_consistency": consistency,
        }
    return statistics


def select_gate_bank_weights(
    *,
    x: np.ndarray,
    counts: np.ndarray,
    batches: np.ndarray | None,
    dataset_stats: dict[str, float],
    base_features: dict[str, np.ndarray],
    expert_scores: dict[str, np.ndarray],
    checkpoint_path: str,
    heuristic_gate: np.ndarray,
    selector: RefineMoEHVGSelector,
    device: torch.device | None = None,
    reliability_aware: bool = False,
) -> tuple[np.ndarray, str, dict[str, float]]:
    components = _point_gate_components_from_runtime(
        dataset_stats=dataset_stats,
        checkpoint_path=checkpoint_path,
        heuristic_gate=heuristic_gate,
        device=device,
    )
    runtime = components["runtime"]
    bank_model: nn.Module | None = runtime.get("bank_model")  # type: ignore[assignment]
    if bank_model is None or len(runtime.get("bank_feature_keys", [])) == 0:
        return (
            np.asarray(components["stage1_blended"], dtype=np.float64),
            "bank_fallback_stage1",
            {
                "bank_reliability": 0.0,
                "bank_consensus": 0.0,
                "bank_uncertainty": 1.0,
                "bank_transfer_risk": 1.0,
                "bank_blend_weight": 0.0,
                "bank_logit_margin": 0.0,
                "bank_view_top1_share": 0.0,
                "bank_view_rank_std": 0.0,
                "bank_view_margin_mean": 0.0,
                "bank_view_margin_std": 0.0,
                "bank_structure_support": 0.0,
                "bank_refine_support": 0.0,
                "bank_boundary_batch_risk": 1.0,
                "bank_selected_is_stage1": 1.0,
            },
        )

    candidate_names, candidate_gates, prototype_distances = _build_inference_gate_bank(
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        residual_gate=np.asarray(components["residual_gate"], dtype=np.float64),
        stage1_gate=np.asarray(components["stage1_blended"], dtype=np.float64),
        prototype_gate=np.asarray(components["prototype_gate"], dtype=np.float64),
        prototype_neighbor_gates=np.asarray(components["prototype_neighbor_gates"], dtype=np.float64),
        prototype_neighbor_distances=np.asarray(components["prototype_neighbor_distances"], dtype=np.float64),
        runtime_config=runtime["config"],
    )
    bank_names = tuple(runtime["bank_candidate_names"])
    if device is None:
        device = choose_torch_device()

    stage1_gate = np.asarray(components["stage1_blended"], dtype=np.float64)
    stage1_idx = int(np.where(candidate_names == "stage1_blended")[0][0])
    prototype_idx = int(np.where(candidate_names == "prototype_mean")[0][0])

    def build_feature_matrix_for_candidates(
        *,
        view_counts: np.ndarray,
        view_batches: np.ndarray | None,
        candidate_idx: np.ndarray,
        prepared_context: tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        if prepared_context is None:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = selector.prepare_context(
                counts=view_counts,
                batches=view_batches,
            )
        else:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = prepared_context
        proxy_context = _prepare_bank_proxy_context(
            x=current_x,
            batches=view_batches,
            random_state=selector.random_state,
        )
        feature_dicts: list[dict[str, float]] = []
        feature_rows: list[list[float]] = []

        for idx in candidate_idx:
            gate = candidate_gates[int(idx)]
            no_refine_scores = selector.score_with_context(
                x=current_x,
                base_features=current_base_features,
                dataset_stats=current_dataset_stats,
                expert_scores=current_expert_scores,
                gate=gate,
                apply_refine=False,
            )
            current_top_k = min(selector.top_k, no_refine_scores.size)
            selected_idx = np.argsort(no_refine_scores)[-current_top_k:]
            boundary_size = min(len(no_refine_scores), max(current_top_k, int(1.5 * current_top_k)))
            boundary_idx = np.argsort(no_refine_scores)[-boundary_size:]
            feature_dict = _gate_bank_candidate_features(
                candidate_name=str(candidate_names[int(idx)]),
                gate=gate,
                score_vector=no_refine_scores,
                selected_idx=selected_idx,
                boundary_idx=boundary_idx,
                dataset_stats=current_dataset_stats,
                base_features=current_base_features,
                x=current_x,
                proxy_context=proxy_context,
                heuristic_gate=heuristic_gate,
                stage1_gate=stage1_gate,
                prototype_distance=float(prototype_distances[int(idx)]),
                bank_names=bank_names,
                random_state=selector.random_state,
            )
            feature_dicts.append(feature_dict)
            feature_rows.append([float(feature_dict[key]) for key in runtime["bank_feature_keys"]])
        return np.asarray(feature_rows, dtype=np.float32), feature_dicts

    full_candidate_idx = np.arange(len(candidate_names), dtype=np.int64)
    feature_matrix, feature_dicts = build_feature_matrix_for_candidates(
        view_counts=np.asarray(counts, dtype=np.float64),
        view_batches=None if batches is None else np.asarray(batches),
        candidate_idx=full_candidate_idx,
        prepared_context=(x, base_features, dataset_stats, expert_scores),
    )
    feature_mean = np.asarray(runtime["bank_feature_mean"], dtype=np.float32)
    feature_std = np.asarray(runtime["bank_feature_std"], dtype=np.float32)
    normalized = (feature_matrix - feature_mean[None, :]) / feature_std[None, :]

    bank_model.eval()
    with torch.no_grad():
        logits = bank_model(
            torch.from_numpy(normalized.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
        ).detach().cpu().numpy().reshape(-1)

    ranked_idx = np.argsort(logits)[::-1]
    best_idx = int(ranked_idx[0])
    second_idx = int(ranked_idx[1]) if len(ranked_idx) > 1 else best_idx
    full_margin = float(logits[best_idx] - logits[second_idx]) if len(ranked_idx) > 1 else 0.0
    best_feature = feature_dicts[best_idx]

    if not reliability_aware:
        final_gate = candidate_gates[best_idx].copy()
        source = f"bank_select:{candidate_names[best_idx]}"
        if len(ranked_idx) > 1:
            logit_gap = float(logits[best_idx] - logits[second_idx])
            if logit_gap < 0.15:
                top_idx = np.asarray([best_idx, second_idx], dtype=np.int64)
                mix_weights = _softmax_np(logits[top_idx] / max(float(runtime["config"].get("bank_mix_temperature", 0.12)), 1e-6))
                final_gate = (mix_weights[:, None] * candidate_gates[top_idx]).sum(axis=0)
                source = "bank_mix:" + "+".join(str(candidate_names[idx]) for idx in top_idx)
        final_gate = _apply_gate_safety_projection(
            gate=final_gate,
            dataset_stats=dataset_stats,
            heuristic_gate=heuristic_gate,
            safety_floor_strength=float(runtime["config"]["safety_floor_strength"]),
        )
        return (
            _normalize_gate_np(final_gate),
            source,
            {
                "bank_reliability": 1.0,
                "bank_consensus": 1.0,
                "bank_uncertainty": 0.0,
                "bank_transfer_risk": 0.0,
                "bank_blend_weight": 1.0,
                "bank_logit_margin": full_margin,
                "bank_view_top1_share": 1.0,
                "bank_view_rank_std": 0.0,
                "bank_view_margin_mean": full_margin,
                "bank_view_margin_std": 0.0,
                "bank_structure_support": _sigmoid_np(float(best_feature["proxy_structure_support"])),
                "bank_refine_support": _sigmoid_np(
                    0.75 * float(best_feature["proxy_safe_refine_support"])
                    + 0.25 * float(best_feature["proxy_structure_support"])
                    - 0.20 * float(best_feature["proxy_transfer_risk"])
                ),
                "bank_boundary_batch_risk": _sigmoid_np(float(best_feature["boundary_batch_mi_mean"])),
                "bank_selected_is_stage1": 1.0 if best_idx == stage1_idx else 0.0,
            },
        )

    audit_indices = np.unique(np.asarray([best_idx, second_idx, stage1_idx, prototype_idx], dtype=np.int64))
    audit_rng = np.random.default_rng(selector.random_state + 97 * x.shape[0] + 13 * x.shape[1])
    view_top_matches: list[float] = []
    view_best_ranks: list[float] = []
    view_best_margins: list[float] = []

    n_views = int(runtime["config"].get("bank_uncertainty_views", 3))
    view_ratio = float(runtime["config"].get("bank_subsample_ratio", 0.78))
    if n_views > 0 and x.shape[0] > 64:
        for _ in range(n_views):
            view_idx = _subsample_bank_view_indices(
                n_cells=x.shape[0],
                batches=None if batches is None else np.asarray(batches),
                ratio=view_ratio,
                rng=audit_rng,
            )
            view_counts = np.asarray(counts[view_idx], dtype=np.float64)
            view_batches = None if batches is None else np.asarray(batches)[view_idx]
            view_feature_matrix, _ = build_feature_matrix_for_candidates(
                view_counts=view_counts,
                view_batches=view_batches,
                candidate_idx=audit_indices,
            )
            view_normalized = (view_feature_matrix - feature_mean[None, :]) / feature_std[None, :]
            with torch.no_grad():
                view_logits = bank_model(
                    torch.from_numpy(view_normalized.astype(np.float32)).to(
                        device,
                        non_blocking=torch.cuda.is_available(),
                    ),
                ).detach().cpu().numpy().reshape(-1)
            view_rank_order = audit_indices[np.argsort(view_logits)[::-1]]
            view_top_matches.append(1.0 if int(view_rank_order[0]) == best_idx else 0.0)
            view_best_ranks.append(float(np.where(view_rank_order == best_idx)[0][0]))
            view_best_logit = float(view_logits[np.where(audit_indices == best_idx)[0][0]])
            other_logits = [float(logit) for idx, logit in zip(audit_indices, view_logits, strict=False) if int(idx) != best_idx]
            competitor_logit = max(other_logits) if other_logits else view_best_logit
            view_best_margins.append(float(view_best_logit - competitor_logit))

    top1_share = float(np.mean(view_top_matches)) if view_top_matches else 1.0
    rank_mean = float(np.mean(view_best_ranks)) if view_best_ranks else 0.0
    rank_std = float(np.std(view_best_ranks)) if view_best_ranks else 0.0
    view_margin_mean = float(np.mean(view_best_margins)) if view_best_margins else full_margin
    view_margin_std = float(np.std(view_best_margins)) if view_best_margins else 0.0

    transfer_risk = _sigmoid_np(
        0.95 * float(best_feature["proxy_transfer_risk"])
        - 0.55 * float(best_feature["proxy_structure_support"])
        - 0.35 * float(best_feature["proxy_safe_refine_support"])
        + 0.25 * float(dataset_stats.get("batch_strength", 0.0))
        + 0.12 * float(best_feature["prototype_distance"])
    )
    consensus = float(
        np.clip(
            0.48 * top1_share
            + 0.18 * (1.0 / (1.0 + rank_mean))
            + 0.10 * (1.0 / (1.0 + rank_std))
            + 0.14 * _sigmoid_np(7.0 * (full_margin - 0.08))
            + 0.10 * _sigmoid_np(7.0 * (view_margin_mean - 0.03)),
            0.0,
            1.0,
        )
    )
    uncertainty = float(
        np.clip(
            0.45 * (1.0 - top1_share)
            + 0.20 * min(rank_std / 2.0, 1.0)
            + 0.15 * _sigmoid_np(6.0 * (0.08 - full_margin))
            + 0.20 * _sigmoid_np(6.0 * (view_margin_std - 0.04)),
            0.0,
            1.0,
        )
    )
    reliability = float(
        np.clip(
            consensus * (1.0 - float(runtime["config"].get("bank_transfer_penalty", 0.45)) * transfer_risk)
            + 0.18
            - 0.35 * uncertainty,
            float(runtime["config"].get("bank_min_reliability", 0.12)),
            0.98,
        )
    )
    refine_support = float(
        np.clip(
            _sigmoid_np(
                0.75 * float(best_feature["proxy_safe_refine_support"])
                + 0.25 * float(best_feature["proxy_structure_support"])
                - 0.20 * float(best_feature["proxy_transfer_risk"])
            ),
            0.0,
            1.0,
        )
    )
    structure_support = float(np.clip(_sigmoid_np(float(best_feature["proxy_structure_support"])), 0.0, 1.0))
    boundary_batch_risk = float(np.clip(_sigmoid_np(float(best_feature["boundary_batch_mi_mean"])), 0.0, 1.0))

    candidate_gate = candidate_gates[best_idx].copy()
    source = f"reliable_bank:{candidate_names[best_idx]}"
    if len(ranked_idx) > 1:
        logit_gap = float(logits[best_idx] - logits[second_idx])
        if logit_gap < 0.15:
            top_idx = np.asarray([best_idx, second_idx], dtype=np.int64)
            mix_weights = _softmax_np(logits[top_idx] / max(float(runtime["config"].get("bank_mix_temperature", 0.12)), 1e-6))
            candidate_gate = (mix_weights[:, None] * candidate_gates[top_idx]).sum(axis=0)
            source = "reliable_bank_mix:" + "+".join(str(candidate_names[idx]) for idx in top_idx)

    if best_idx == stage1_idx:
        blend_weight = 1.0
        final_gate = candidate_gate
    else:
        blend_weight = float(
            np.clip(
                0.18 + float(runtime["config"].get("bank_reliability_blend", 0.72)) * reliability,
                float(runtime["config"].get("bank_min_reliability", 0.12)),
                0.96,
            )
        )
        final_gate = blend_weight * candidate_gate + (1.0 - blend_weight) * stage1_gate
        source = f"{source}->stage1({blend_weight:.2f})"
    final_gate = _apply_gate_safety_projection(
        gate=final_gate,
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        safety_floor_strength=float(runtime["config"]["safety_floor_strength"]),
    )
    return (
        _normalize_gate_np(final_gate),
        source,
        {
            "bank_reliability": reliability,
            "bank_consensus": consensus,
            "bank_uncertainty": uncertainty,
            "bank_transfer_risk": transfer_risk,
            "bank_blend_weight": blend_weight,
            "bank_logit_margin": full_margin,
            "bank_view_top1_share": top1_share,
            "bank_view_rank_std": rank_std,
            "bank_view_margin_mean": view_margin_mean,
            "bank_view_margin_std": view_margin_std,
            "bank_structure_support": structure_support,
            "bank_refine_support": refine_support,
            "bank_boundary_batch_risk": boundary_batch_risk,
            "bank_selected_is_stage1": 1.0 if best_idx == stage1_idx else 0.0,
        },
    )


def select_counterfactual_gate_bank_weights(
    *,
    x: np.ndarray,
    counts: np.ndarray,
    batches: np.ndarray | None,
    dataset_stats: dict[str, float],
    base_features: dict[str, np.ndarray],
    expert_scores: dict[str, np.ndarray],
    checkpoint_path: str,
    heuristic_gate: np.ndarray,
    selector: RefineMoEHVGSelector,
    device: torch.device | None = None,
    variant: str = "full",
) -> tuple[np.ndarray, str, dict[str, float]]:
    components = _point_gate_components_from_runtime(
        dataset_stats=dataset_stats,
        checkpoint_path=checkpoint_path,
        heuristic_gate=heuristic_gate,
        device=device,
    )
    runtime = components["runtime"]
    curr_model: nn.Module | None = runtime.get("curr_model")  # type: ignore[assignment]
    if curr_model is None or len(runtime.get("curr_feature_keys", [])) == 0:
        return select_gate_bank_weights(
            x=x,
            counts=counts,
            batches=batches,
            dataset_stats=dataset_stats,
            base_features=base_features,
            expert_scores=expert_scores,
            checkpoint_path=checkpoint_path,
            heuristic_gate=heuristic_gate,
            selector=selector,
            device=device,
            reliability_aware=True,
        )

    candidate_names, candidate_gates, prototype_distances = _build_inference_gate_bank(
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        residual_gate=np.asarray(components["residual_gate"], dtype=np.float64),
        stage1_gate=np.asarray(components["stage1_blended"], dtype=np.float64),
        prototype_gate=np.asarray(components["prototype_gate"], dtype=np.float64),
        prototype_neighbor_gates=np.asarray(components["prototype_neighbor_gates"], dtype=np.float64),
        prototype_neighbor_distances=np.asarray(components["prototype_neighbor_distances"], dtype=np.float64),
        runtime_config=runtime["config"],
    )
    bank_names = tuple(runtime["bank_candidate_names"])
    if device is None:
        device = choose_torch_device()

    runtime_config = GateLearningConfig(**runtime["config"])
    stage1_gate = np.asarray(components["stage1_blended"], dtype=np.float64)
    stage1_idx = int(np.where(candidate_names == "stage1_blended")[0][0])
    prototype_idx = int(np.where(candidate_names == "prototype_mean")[0][0])
    use_risk = variant not in {"no_risk", "utility_only"}
    use_regret = variant not in {"no_regret", "utility_only"}
    use_refine_policy = variant != "no_refine_policy"

    def build_feature_matrix_for_candidates(
        *,
        view_counts: np.ndarray,
        view_batches: np.ndarray | None,
        candidate_idx: np.ndarray,
        prepared_context: tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        if prepared_context is None:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = selector.prepare_context(
                counts=view_counts,
                batches=view_batches,
            )
        else:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = prepared_context
        proxy_context = _prepare_bank_proxy_context(
            x=current_x,
            batches=view_batches,
            random_state=selector.random_state,
        )
        feature_dicts: list[dict[str, float]] = []
        for idx in candidate_idx:
            gate = candidate_gates[int(idx)]
            no_refine_scores = selector.score_with_context(
                x=current_x,
                base_features=current_base_features,
                dataset_stats=current_dataset_stats,
                expert_scores=current_expert_scores,
                gate=gate,
                apply_refine=False,
            )
            current_top_k = min(selector.top_k, no_refine_scores.size)
            selected_idx = np.argsort(no_refine_scores)[-current_top_k:]
            boundary_size = min(len(no_refine_scores), max(current_top_k, int(1.5 * current_top_k)))
            boundary_idx = np.argsort(no_refine_scores)[-boundary_size:]
            feature_dicts.append(
                _gate_bank_candidate_features(
                    candidate_name=str(candidate_names[int(idx)]),
                    gate=gate,
                    score_vector=no_refine_scores,
                    selected_idx=selected_idx,
                    boundary_idx=boundary_idx,
                    dataset_stats=current_dataset_stats,
                    base_features=current_base_features,
                    x=current_x,
                    proxy_context=proxy_context,
                    heuristic_gate=heuristic_gate,
                    stage1_gate=stage1_gate,
                    prototype_distance=float(prototype_distances[int(idx)]),
                    bank_names=bank_names,
                    random_state=selector.random_state,
                )
            )
        feature_rows = [
            [float(feature_dict[key]) for key in runtime["curr_feature_keys"]]
            for feature_dict in feature_dicts
        ]
        return np.asarray(feature_rows, dtype=np.float32), feature_dicts

    full_candidate_idx = np.arange(len(candidate_names), dtype=np.int64)
    feature_matrix, feature_dicts = build_feature_matrix_for_candidates(
        view_counts=np.asarray(counts, dtype=np.float64),
        view_batches=None if batches is None else np.asarray(batches),
        candidate_idx=full_candidate_idx,
        prepared_context=(x, base_features, dataset_stats, expert_scores),
    )
    feature_mean = np.asarray(runtime["curr_feature_mean"], dtype=np.float32)
    feature_std = np.asarray(runtime["curr_feature_std"], dtype=np.float32)
    normalized = (feature_matrix - feature_mean[None, :]) / feature_std[None, :]

    curr_model.eval()
    with torch.no_grad():
        utility, risk, regret, refine_delta = curr_model(
            torch.from_numpy(normalized.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
        )
    utility_np = utility.detach().cpu().numpy().reshape(-1)
    risk_np = risk.detach().cpu().numpy().reshape(-1)
    regret_np = regret.detach().cpu().numpy().reshape(-1)
    refine_delta_np = refine_delta.detach().cpu().numpy().reshape(-1)

    decision_scores = _curr_policy_score_np(
        utility=utility_np,
        risk=risk_np if use_risk else np.zeros_like(risk_np),
        regret=regret_np if use_regret else np.zeros_like(regret_np),
        config=runtime_config,
    )
    ranked_idx = np.argsort(decision_scores)[::-1]
    best_idx = int(ranked_idx[0])
    second_idx = int(ranked_idx[1]) if len(ranked_idx) > 1 else best_idx
    full_margin = float(decision_scores[best_idx] - decision_scores[second_idx]) if len(ranked_idx) > 1 else 0.0
    route_margin = float(decision_scores[best_idx] - decision_scores[stage1_idx])
    best_feature = feature_dicts[best_idx]

    audit_indices = np.unique(np.asarray([best_idx, second_idx, stage1_idx, prototype_idx], dtype=np.int64))
    audit_rng = np.random.default_rng(selector.random_state + 131 * x.shape[0] + 17 * x.shape[1])
    view_top_matches: list[float] = []
    view_best_ranks: list[float] = []
    view_best_margins: list[float] = []

    n_views = int(runtime_config.bank_uncertainty_views)
    if n_views > 0 and x.shape[0] > 64:
        for _ in range(n_views):
            view_idx = _subsample_bank_view_indices(
                n_cells=x.shape[0],
                batches=None if batches is None else np.asarray(batches),
                ratio=float(runtime_config.bank_subsample_ratio),
                rng=audit_rng,
            )
            view_counts = np.asarray(counts[view_idx], dtype=np.float64)
            view_batches = None if batches is None else np.asarray(batches)[view_idx]
            view_feature_matrix, _ = build_feature_matrix_for_candidates(
                view_counts=view_counts,
                view_batches=view_batches,
                candidate_idx=audit_indices,
            )
            view_normalized = (view_feature_matrix - feature_mean[None, :]) / feature_std[None, :]
            with torch.no_grad():
                view_utility, view_risk, view_regret, _ = curr_model(
                    torch.from_numpy(view_normalized.astype(np.float32)).to(
                        device,
                        non_blocking=torch.cuda.is_available(),
                    ),
                )
            view_scores = _curr_policy_score_np(
                utility=view_utility.detach().cpu().numpy().reshape(-1),
                risk=view_risk.detach().cpu().numpy().reshape(-1) if use_risk else np.zeros(len(audit_indices), dtype=np.float64),
                regret=view_regret.detach().cpu().numpy().reshape(-1) if use_regret else np.zeros(len(audit_indices), dtype=np.float64),
                config=runtime_config,
            )
            view_rank_order = audit_indices[np.argsort(view_scores)[::-1]]
            view_top_matches.append(1.0 if int(view_rank_order[0]) == best_idx else 0.0)
            view_best_ranks.append(float(np.where(view_rank_order == best_idx)[0][0]))
            view_best_score = float(view_scores[np.where(audit_indices == best_idx)[0][0]])
            other_scores = [
                float(score)
                for idx, score in zip(audit_indices, view_scores, strict=False)
                if int(idx) != best_idx
            ]
            competitor_score = max(other_scores) if other_scores else view_best_score
            view_best_margins.append(float(view_best_score - competitor_score))

    top1_share = float(np.mean(view_top_matches)) if view_top_matches else 1.0
    rank_mean = float(np.mean(view_best_ranks)) if view_best_ranks else 0.0
    rank_std = float(np.std(view_best_ranks)) if view_best_ranks else 0.0
    view_margin_mean = float(np.mean(view_best_margins)) if view_best_margins else full_margin
    view_margin_std = float(np.std(view_best_margins)) if view_best_margins else 0.0
    consensus = float(
        np.clip(
            0.50 * top1_share
            + 0.18 * (1.0 / (1.0 + rank_mean))
            + 0.12 * (1.0 / (1.0 + rank_std))
            + 0.12 * _sigmoid_np(6.0 * (full_margin - 0.05))
            + 0.08 * _sigmoid_np(6.0 * (view_margin_mean - 0.02)),
            0.0,
            1.0,
        )
    )
    uncertainty = float(
        np.clip(
            0.45 * (1.0 - top1_share)
            + 0.20 * min(rank_std / 2.0, 1.0)
            + 0.15 * _sigmoid_np(6.0 * (0.06 - full_margin))
            + 0.20 * _sigmoid_np(6.0 * (view_margin_std - 0.04)),
            0.0,
            1.0,
        )
    )

    utility_score = float(utility_np[best_idx])
    risk_score = float(risk_np[best_idx] if use_risk else 0.0)
    regret_score = float(regret_np[best_idx] if use_regret else 0.0)
    policy_score = float(decision_scores[best_idx])

    if best_idx == stage1_idx:
        conservative_coeff = 1.0
        aggressive_weight = 0.0
        final_gate = stage1_gate.copy()
        source = "curr_select:stage1"
    else:
        conservative_logit = (
            runtime_config.curr_risk_weight * risk_score
            - runtime_config.curr_regret_weight * regret_score
            - route_margin
            + runtime_config.curr_uncertainty_weight * uncertainty
        )
        conservative_coeff = float(
            np.clip(
                _sigmoid_np(conservative_logit / max(runtime_config.curr_conservative_temperature, 1e-6)),
                0.05,
                0.95,
            )
        )
        aggressive_weight = 1.0 - conservative_coeff
        final_gate = aggressive_weight * candidate_gates[best_idx] + conservative_coeff * stage1_gate
        source = f"curr:{candidate_names[best_idx]}->stage1({conservative_coeff:.2f})"

    final_gate = _apply_gate_safety_projection(
        gate=final_gate,
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        safety_floor_strength=float(runtime["config"]["safety_floor_strength"]),
    )

    transfer_risk = float(
        np.clip(
            0.70 * risk_score + 0.30 * _sigmoid_np(float(best_feature["proxy_transfer_risk"])),
            0.0,
            1.0,
        )
    )
    structure_support = float(np.clip(_sigmoid_np(float(best_feature["proxy_structure_support"])), 0.0, 1.0))
    boundary_batch_risk = float(np.clip(_sigmoid_np(float(best_feature["boundary_batch_mi_mean"])), 0.0, 1.0))
    reliability = float(
        np.clip(
            0.55 * (1.0 - transfer_risk) + 0.25 * consensus + 0.20 * (1.0 - uncertainty),
            runtime_config.bank_min_reliability,
            0.99,
        )
    )
    proxy_refine_support = float(
        np.clip(
            _sigmoid_np(
                0.75 * float(best_feature["proxy_safe_refine_support"])
                + 0.25 * float(best_feature["proxy_structure_support"])
                - 0.20 * float(best_feature["proxy_transfer_risk"])
            ),
            0.0,
            1.0,
        )
    )

    metadata: dict[str, float] = {
        "bank_reliability": reliability,
        "bank_consensus": consensus,
        "bank_uncertainty": uncertainty,
        "bank_transfer_risk": transfer_risk,
        "bank_blend_weight": aggressive_weight,
        "bank_logit_margin": full_margin,
        "bank_view_top1_share": top1_share,
        "bank_view_rank_std": rank_std,
        "bank_view_margin_mean": view_margin_mean,
        "bank_view_margin_std": view_margin_std,
        "bank_structure_support": structure_support,
        "bank_refine_support": proxy_refine_support,
        "bank_boundary_batch_risk": boundary_batch_risk,
        "bank_selected_is_stage1": 1.0 if best_idx == stage1_idx else 0.0,
        "curr_utility_score": utility_score,
        "curr_risk_score": risk_score,
        "curr_regret_score": regret_score,
        "curr_policy_score": policy_score,
        "curr_route_margin": route_margin,
        "curr_conservative_coeff": conservative_coeff,
        "curr_aggressive_weight": aggressive_weight,
        "curr_refine_delta": float(refine_delta_np[best_idx]),
    }

    if use_refine_policy:
        refine_delta_score = float(refine_delta_np[best_idx])
        refine_gain = max(refine_delta_score, 0.0)
        refine_risk = max(-refine_delta_score, 0.0)
        route_freedom = 0.45 + 0.55 * aggressive_weight
        refine_logit = (
            refine_delta_score
            + 0.30 * utility_score
            + 0.45 * regret_score
            - 0.80 * risk_score
            - runtime_config.curr_refine_conservative_weight * conservative_coeff
            - 0.30 * uncertainty
        )
        refine_probability = float(
            np.clip(
                _sigmoid_np(refine_logit / max(runtime_config.curr_refine_temperature, 1e-6)),
                0.0,
                1.0,
            )
        )
        refine_intensity = float(
            np.clip(
                route_freedom * refine_probability * (0.25 + 0.75 * refine_gain),
                0.0,
                1.0,
            )
        )
        metadata.update(
            {
                "policy_refine_probability": refine_probability,
                "policy_refine_intensity": refine_intensity,
                "policy_refine_decision": 1.0 if refine_probability >= 0.5 else 0.0,
                "policy_refine_gain": refine_gain,
                "policy_refine_risk": refine_risk,
                "bank_refine_support": float(np.clip(0.5 * proxy_refine_support + 0.5 * refine_probability, 0.0, 1.0)),
            }
        )

    return _normalize_gate_np(final_gate), source, metadata


def select_stage1_anchored_pairwise_gate_bank_weights(
    *,
    x: np.ndarray,
    counts: np.ndarray,
    batches: np.ndarray | None,
    dataset_stats: dict[str, float],
    base_features: dict[str, np.ndarray],
    expert_scores: dict[str, np.ndarray],
    checkpoint_path: str,
    heuristic_gate: np.ndarray,
    selector: RefineMoEHVGSelector,
    device: torch.device | None = None,
    variant: str = "full",
) -> tuple[np.ndarray, str, dict[str, float]]:
    components = _point_gate_components_from_runtime(
        dataset_stats=dataset_stats,
        checkpoint_path=checkpoint_path,
        heuristic_gate=heuristic_gate,
        device=device,
    )
    runtime = components["runtime"]
    pairwise_model: nn.Module | None = runtime.get("pairwise_model")  # type: ignore[assignment]
    if pairwise_model is None or len(runtime.get("pairwise_feature_keys", [])) == 0:
        return select_counterfactual_gate_bank_weights(
            x=x,
            counts=counts,
            batches=batches,
            dataset_stats=dataset_stats,
            base_features=base_features,
            expert_scores=expert_scores,
            checkpoint_path=checkpoint_path,
            heuristic_gate=heuristic_gate,
            selector=selector,
            device=device,
            variant="no_refine_policy",
        )

    candidate_names, candidate_gates, prototype_distances = _build_inference_gate_bank(
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        residual_gate=np.asarray(components["residual_gate"], dtype=np.float64),
        stage1_gate=np.asarray(components["stage1_blended"], dtype=np.float64),
        prototype_gate=np.asarray(components["prototype_gate"], dtype=np.float64),
        prototype_neighbor_gates=np.asarray(components["prototype_neighbor_gates"], dtype=np.float64),
        prototype_neighbor_distances=np.asarray(components["prototype_neighbor_distances"], dtype=np.float64),
        runtime_config=runtime["config"],
    )
    bank_names = tuple(runtime["bank_candidate_names"])
    if device is None:
        device = choose_torch_device()

    runtime_config = GateLearningConfig(**runtime["config"])
    stage1_gate = np.asarray(components["stage1_blended"], dtype=np.float64)
    stage1_idx = int(np.where(candidate_names == "stage1_blended")[0][0])
    use_risk = variant not in {"no_risk", "utility_only"}
    use_regret = variant not in {"no_regret", "utility_only"}
    use_margin_term = variant not in {"no_pairwise_term", "utility_only"}
    use_explicit_conservative = variant != "no_conservative_routing"
    refine_enabled = variant == "refine_on"

    def build_candidate_features(
        *,
        view_counts: np.ndarray,
        view_batches: np.ndarray | None,
        prepared_context: tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]] | None = None,
    ) -> list[dict[str, float]]:
        if prepared_context is None:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = selector.prepare_context(
                counts=view_counts,
                batches=view_batches,
            )
        else:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = prepared_context
        proxy_context = _prepare_bank_proxy_context(
            x=current_x,
            batches=view_batches,
            random_state=selector.random_state,
        )
        feature_dicts: list[dict[str, float]] = []
        for idx, gate in enumerate(candidate_gates):
            no_refine_scores = selector.score_with_context(
                x=current_x,
                base_features=current_base_features,
                dataset_stats=current_dataset_stats,
                expert_scores=current_expert_scores,
                gate=gate,
                apply_refine=False,
            )
            current_top_k = min(selector.top_k, no_refine_scores.size)
            selected_idx = np.argsort(no_refine_scores)[-current_top_k:]
            boundary_size = min(len(no_refine_scores), max(current_top_k, int(1.5 * current_top_k)))
            boundary_idx = np.argsort(no_refine_scores)[-boundary_size:]
            feature_dicts.append(
                _gate_bank_candidate_features(
                    candidate_name=str(candidate_names[idx]),
                    gate=gate,
                    score_vector=no_refine_scores,
                    selected_idx=selected_idx,
                    boundary_idx=boundary_idx,
                    dataset_stats=current_dataset_stats,
                    base_features=current_base_features,
                    x=current_x,
                    proxy_context=proxy_context,
                    heuristic_gate=heuristic_gate,
                    stage1_gate=stage1_gate,
                    prototype_distance=float(prototype_distances[idx]),
                    bank_names=bank_names,
                    random_state=selector.random_state,
                )
            )
        return feature_dicts

    candidate_feature_dicts = build_candidate_features(
        view_counts=np.asarray(counts, dtype=np.float64),
        view_batches=None if batches is None else np.asarray(batches),
        prepared_context=(x, base_features, dataset_stats, expert_scores),
    )
    anchor_features = candidate_feature_dicts[stage1_idx]
    pairwise_feature_dicts = [
        _build_stage1_pairwise_feature_dict(
            candidate_features=feature_dict,
            anchor_features=anchor_features,
            candidate_name=str(candidate_names[idx]),
            bank_names=bank_names,
        )
        for idx, feature_dict in enumerate(candidate_feature_dicts)
    ]
    feature_rows = [
        [float(feature_dict[key]) for key in runtime["pairwise_feature_keys"]]
        for feature_dict in pairwise_feature_dicts
    ]
    feature_matrix = np.asarray(feature_rows, dtype=np.float32)
    feature_mean = np.asarray(runtime["pairwise_feature_mean"], dtype=np.float32)
    feature_std = np.asarray(runtime["pairwise_feature_std"], dtype=np.float32)
    normalized = (feature_matrix - feature_mean[None, :]) / feature_std[None, :]

    pairwise_model.eval()
    with torch.no_grad():
        utility, risk, regret, route_margin = pairwise_model(
            torch.from_numpy(normalized.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
        )
    utility_np = utility.detach().cpu().numpy().reshape(-1)
    risk_np = risk.detach().cpu().numpy().reshape(-1)
    regret_np = regret.detach().cpu().numpy().reshape(-1)
    route_margin_np = route_margin.detach().cpu().numpy().reshape(-1)

    utility_np[stage1_idx] = 0.0
    risk_np[stage1_idx] = 0.0
    regret_np[stage1_idx] = 0.0
    route_margin_np[stage1_idx] = 0.0

    decision_scores = _pairwise_decision_score_np(
        utility=utility_np,
        risk=risk_np,
        regret=regret_np,
        route_margin=route_margin_np,
        config=runtime_config,
        use_risk=use_risk,
        use_regret=use_regret,
        use_margin_term=use_margin_term,
    )
    decision_scores[stage1_idx] = 0.0

    non_stage1_idx = np.asarray([idx for idx in range(len(candidate_names)) if idx != stage1_idx], dtype=np.int64)
    if non_stage1_idx.size == 0:
        best_idx = stage1_idx
    else:
        best_idx = int(non_stage1_idx[np.argmax(decision_scores[non_stage1_idx])])

    best_candidate_name = str(candidate_names[best_idx])
    best_feature = candidate_feature_dicts[best_idx]
    utility_score = float(utility_np[best_idx]) if best_idx != stage1_idx else 0.0
    risk_score = float(risk_np[best_idx]) if best_idx != stage1_idx and use_risk else 0.0
    regret_score = float(regret_np[best_idx]) if best_idx != stage1_idx and use_regret else 0.0
    route_margin_score = float(route_margin_np[best_idx]) if best_idx != stage1_idx else 0.0
    decision_score = float(decision_scores[best_idx]) if best_idx != stage1_idx else 0.0
    threshold_pass = float(decision_score > runtime_config.pairwise_route_threshold)
    margin_pass = float(route_margin_score > 0.0) if use_margin_term else 1.0

    if best_idx == stage1_idx:
        routed_away = False
    elif use_explicit_conservative:
        routed_away = bool(threshold_pass > 0.5 and margin_pass > 0.5)
    else:
        routed_away = True

    if routed_away:
        final_gate = candidate_gates[best_idx].copy()
        source = f"pairregret_route:{best_candidate_name}"
    else:
        final_gate = stage1_gate.copy()
        source = "pairregret_stay:stage1"

    final_gate = _apply_gate_safety_projection(
        gate=final_gate,
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        safety_floor_strength=float(runtime["config"]["safety_floor_strength"]),
    )

    transfer_risk = float(
        np.clip(
            0.75 * risk_score + 0.25 * _sigmoid_np(float(best_feature["proxy_transfer_risk"])),
            0.0,
            1.0,
        )
    )
    structure_support = float(np.clip(_sigmoid_np(float(best_feature["proxy_structure_support"])), 0.0, 1.0))
    refine_support = float(np.clip(_sigmoid_np(float(best_feature["proxy_safe_refine_support"])), 0.0, 1.0))
    boundary_batch_risk = float(np.clip(_sigmoid_np(float(best_feature["boundary_batch_mi_mean"])), 0.0, 1.0))
    reliability = float(
        np.clip(
            0.60 * (1.0 - transfer_risk)
            + 0.20 * _sigmoid_np(4.0 * decision_score)
            + 0.20 * _sigmoid_np(4.0 * route_margin_score),
            runtime_config.bank_min_reliability,
            0.99,
        )
    )
    refine_intensity = 0.0
    if refine_enabled and routed_away:
        refine_intensity = float(
            np.clip(
                _sigmoid_np(4.0 * decision_score) * (1.0 - 0.55 * risk_score) * (0.35 + 0.65 * regret_score),
                0.0,
                1.0,
            )
        )

    metadata: dict[str, float] = {
        "bank_reliability": reliability,
        "bank_consensus": float(np.clip(_sigmoid_np(4.0 * decision_score), 0.0, 1.0)),
        "bank_uncertainty": float(np.clip(1.0 - _sigmoid_np(4.0 * decision_score), 0.0, 1.0)),
        "bank_transfer_risk": transfer_risk,
        "bank_blend_weight": 1.0 if routed_away else 0.0,
        "bank_logit_margin": decision_score,
        "bank_view_top1_share": 1.0 if routed_away else 0.0,
        "bank_view_rank_std": 0.0,
        "bank_view_margin_mean": route_margin_score,
        "bank_view_margin_std": 0.0,
        "bank_structure_support": structure_support,
        "bank_refine_support": refine_support,
        "bank_boundary_batch_risk": boundary_batch_risk,
        "bank_selected_is_stage1": 0.0 if routed_away else 1.0,
        "pairwise_utility_gain": utility_score,
        "pairwise_failure_risk": risk_score,
        "pairwise_fallback_regret": regret_score,
        "pairwise_route_margin": route_margin_score,
        "pairwise_decision_score": decision_score,
        "pairwise_anchor_selection_outcome": 1.0 if not routed_away else 0.0,
        "pairwise_routed_away_from_stage1": 1.0 if routed_away else 0.0,
        "pairwise_route_threshold_pass": threshold_pass,
        "pairwise_margin_pass": margin_pass,
        "pairwise_refine_enabled": 1.0 if refine_enabled else 0.0,
        "pairwise_refine_intensity": refine_intensity,
        "pairwise_conservative_routing_enabled": 1.0 if use_explicit_conservative else 0.0,
    }
    return _normalize_gate_np(final_gate), source, metadata


def select_stage1_anchored_pairwise_gate_bank_weights_calibrated(
    *,
    x: np.ndarray,
    counts: np.ndarray,
    batches: np.ndarray | None,
    dataset_stats: dict[str, float],
    base_features: dict[str, np.ndarray],
    expert_scores: dict[str, np.ndarray],
    checkpoint_path: str,
    heuristic_gate: np.ndarray,
    selector: RefineMoEHVGSelector,
    device: torch.device | None = None,
    variant: str = "full",
) -> tuple[np.ndarray, str, dict[str, float]]:
    components = _point_gate_components_from_runtime(
        dataset_stats=dataset_stats,
        checkpoint_path=checkpoint_path,
        heuristic_gate=heuristic_gate,
        device=device,
    )
    runtime = components["runtime"]
    pairwise_packages: dict[str, dict[str, object]] = runtime.get("pairwise_cal_models", {})  # type: ignore[assignment]
    model_variant = variant if variant in pairwise_packages else "full"
    pairwise_package = pairwise_packages.get(model_variant) or pairwise_packages.get("full")
    if not pairwise_package:
        return select_stage1_anchored_pairwise_gate_bank_weights(
            x=x,
            counts=counts,
            batches=batches,
            dataset_stats=dataset_stats,
            base_features=base_features,
            expert_scores=expert_scores,
            checkpoint_path=checkpoint_path,
            heuristic_gate=heuristic_gate,
            selector=selector,
            device=device,
            variant="full",
        )

    pairwise_model: nn.Module = pairwise_package["model"]  # type: ignore[assignment]
    pairwise_feature_keys = list(pairwise_package["feature_keys"])
    pairwise_feature_mean = np.asarray(pairwise_package["feature_mean"], dtype=np.float32)
    pairwise_feature_std = np.asarray(pairwise_package["feature_std"], dtype=np.float32)

    candidate_names, candidate_gates, prototype_distances = _build_inference_gate_bank(
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        residual_gate=np.asarray(components["residual_gate"], dtype=np.float64),
        stage1_gate=np.asarray(components["stage1_blended"], dtype=np.float64),
        prototype_gate=np.asarray(components["prototype_gate"], dtype=np.float64),
        prototype_neighbor_gates=np.asarray(components["prototype_neighbor_gates"], dtype=np.float64),
        prototype_neighbor_distances=np.asarray(components["prototype_neighbor_distances"], dtype=np.float64),
        runtime_config=runtime["config"],
    )
    bank_names = tuple(runtime["bank_candidate_names"])
    if device is None:
        device = choose_torch_device()

    runtime_config = GateLearningConfig(**runtime["config"])
    stage1_gate = np.asarray(components["stage1_blended"], dtype=np.float64)
    stage1_idx = int(np.where(candidate_names == "stage1_blended")[0][0])
    prototype_idx = int(np.where(candidate_names == "prototype_mean")[0][0])
    use_regret = variant not in {"no_regret_calibration", "utility_only"}
    use_risk = variant != "utility_only"
    use_route_constraint = variant != "no_route_constraint"
    use_consistency = variant != "no_consistency_calibration"
    use_explicit_conservative = variant != "no_conservative_routing"
    refine_enabled = variant == "refine_on"

    def build_candidate_features(
        *,
        view_counts: np.ndarray,
        view_batches: np.ndarray | None,
        candidate_idx: np.ndarray | None = None,
        prepared_context: tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        if prepared_context is None:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = selector.prepare_context(
                counts=view_counts,
                batches=view_batches,
            )
        else:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = prepared_context
        proxy_context = _prepare_bank_proxy_context(
            x=current_x,
            batches=view_batches,
            random_state=selector.random_state,
        )
        feature_dicts: list[dict[str, float]] = []
        indices = np.arange(len(candidate_names), dtype=np.int64) if candidate_idx is None else np.asarray(candidate_idx, dtype=np.int64)
        for idx in indices:
            gate = candidate_gates[int(idx)]
            no_refine_scores = selector.score_with_context(
                x=current_x,
                base_features=current_base_features,
                dataset_stats=current_dataset_stats,
                expert_scores=current_expert_scores,
                gate=gate,
                apply_refine=False,
            )
            current_top_k = min(selector.top_k, no_refine_scores.size)
            selected_idx = np.argsort(no_refine_scores)[-current_top_k:]
            boundary_size = min(len(no_refine_scores), max(current_top_k, int(1.5 * current_top_k)))
            boundary_idx = np.argsort(no_refine_scores)[-boundary_size:]
            feature_dicts.append(
                _gate_bank_candidate_features(
                    candidate_name=str(candidate_names[int(idx)]),
                    gate=gate,
                    score_vector=no_refine_scores,
                    selected_idx=selected_idx,
                    boundary_idx=boundary_idx,
                    dataset_stats=current_dataset_stats,
                    base_features=current_base_features,
                    x=current_x,
                    proxy_context=proxy_context,
                    heuristic_gate=heuristic_gate,
                    stage1_gate=stage1_gate,
                    prototype_distance=float(prototype_distances[int(idx)]),
                    bank_names=bank_names,
                    random_state=selector.random_state,
                )
            )
        return indices, feature_dicts

    def pairwise_outputs_for_indices(
        *,
        candidate_idx: np.ndarray,
        view_counts: np.ndarray,
        view_batches: np.ndarray | None,
        prepared_context: tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices, feature_dicts = build_candidate_features(
            view_counts=view_counts,
            view_batches=view_batches,
            candidate_idx=candidate_idx,
            prepared_context=prepared_context,
        )
        anchor_pos = int(np.where(indices == stage1_idx)[0][0])
        anchor_features = feature_dicts[anchor_pos]
        feature_rows = [
            [
                float(
                    _build_stage1_pairwise_feature_dict(
                        candidate_features=feature_dict,
                        anchor_features=anchor_features,
                        candidate_name=str(candidate_names[int(idx)]),
                        bank_names=bank_names,
                    )[key]
                )
                for key in pairwise_feature_keys
            ]
            for idx, feature_dict in zip(indices, feature_dicts, strict=False)
        ]
        normalized = (np.asarray(feature_rows, dtype=np.float32) - pairwise_feature_mean[None, :]) / pairwise_feature_std[None, :]
        with torch.no_grad():
            utility, risk, regret, route_margin = pairwise_model(
                torch.from_numpy(normalized.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
            )
        utility_np = utility.detach().cpu().numpy().reshape(-1)
        risk_np = risk.detach().cpu().numpy().reshape(-1)
        regret_np = regret.detach().cpu().numpy().reshape(-1)
        route_margin_np = route_margin.detach().cpu().numpy().reshape(-1)
        stage1_local_pos = int(np.where(indices == stage1_idx)[0][0])
        utility_np[stage1_local_pos] = 0.0
        risk_np[stage1_local_pos] = 0.0
        regret_np[stage1_local_pos] = 0.0
        route_margin_np[stage1_local_pos] = 0.0
        decision_np = _pairwise_decision_score_np(
            utility=utility_np,
            risk=risk_np,
            regret=regret_np,
            route_margin=route_margin_np,
            config=runtime_config,
            use_risk=use_risk,
            use_regret=use_regret,
            use_margin_term=True,
        )
        decision_np[stage1_local_pos] = 0.0
        return indices, utility_np, risk_np, regret_np, route_margin_np, decision_np

    full_indices, utility_np, risk_np, regret_np, route_margin_np, decision_scores = pairwise_outputs_for_indices(
        candidate_idx=np.arange(len(candidate_names), dtype=np.int64),
        view_counts=np.asarray(counts, dtype=np.float64),
        view_batches=None if batches is None else np.asarray(batches),
        prepared_context=(x, base_features, dataset_stats, expert_scores),
    )

    non_stage1_idx = np.asarray([idx for idx in range(len(candidate_names)) if idx != stage1_idx], dtype=np.int64)
    if non_stage1_idx.size == 0:
        best_idx = stage1_idx
    else:
        best_idx = int(non_stage1_idx[np.argmax(decision_scores[non_stage1_idx])])
    ranked_idx = non_stage1_idx[np.argsort(decision_scores[non_stage1_idx])[::-1]] if non_stage1_idx.size else np.asarray([], dtype=np.int64)
    second_idx = int(ranked_idx[1]) if ranked_idx.size > 1 else (int(ranked_idx[0]) if ranked_idx.size == 1 else stage1_idx)

    best_candidate_name = str(candidate_names[best_idx])
    utility_score = float(utility_np[best_idx]) if best_idx != stage1_idx else 0.0
    risk_score = float(risk_np[best_idx]) if best_idx != stage1_idx and use_risk else 0.0
    regret_score = float(regret_np[best_idx]) if best_idx != stage1_idx and use_regret else 0.0
    route_margin_score = float(route_margin_np[best_idx]) if best_idx != stage1_idx else 0.0
    base_decision_score = float(decision_scores[best_idx]) if best_idx != stage1_idx else 0.0

    consistency_stats = {
        "top1_share": 1.0,
        "rank_mean": 0.0,
        "rank_std": 0.0,
        "route_allow_share": 1.0 if route_margin_score > 0.0 else 0.0,
        "margin_mean": route_margin_score,
        "margin_std": 0.0,
        "consistency": _pairwise_consistency_score(
            win_rate=1.0 if route_margin_score > 0.0 else 0.0,
            margin_mean=route_margin_score,
            margin_std=0.0,
            config=runtime_config,
        ),
    }
    if best_idx != stage1_idx and use_consistency and counts.shape[0] > 48:
        audit_indices = np.unique(np.asarray([best_idx, second_idx, stage1_idx, prototype_idx], dtype=np.int64))
        view_rng = np.random.default_rng(selector.random_state + 701 * counts.shape[0] + 29 * counts.shape[1])
        top1_matches: list[float] = []
        ranks: list[float] = []
        route_allow: list[float] = []
        margins: list[float] = []
        for _ in range(max(1, int(runtime_config.bank_uncertainty_views))):
            view_idx = _subsample_bank_view_indices(
                n_cells=counts.shape[0],
                batches=None if batches is None else np.asarray(batches),
                ratio=float(runtime_config.bank_subsample_ratio),
                rng=view_rng,
            )
            view_counts = np.asarray(counts[view_idx], dtype=np.float64)
            view_batches = None if batches is None else np.asarray(batches)[view_idx]
            view_indices, view_utility, view_risk, view_regret, view_margin, view_decision = pairwise_outputs_for_indices(
                candidate_idx=audit_indices,
                view_counts=view_counts,
                view_batches=view_batches,
            )
            local_best_pos = int(np.argmax(view_decision))
            local_best_idx = int(view_indices[local_best_pos])
            candidate_pos = int(np.where(view_indices == best_idx)[0][0])
            candidate_rank = int(np.where(view_indices[np.argsort(view_decision)[::-1]] == best_idx)[0][0])
            local_consistency = _pairwise_consistency_score(
                win_rate=1.0 if view_margin[candidate_pos] > 0.0 else 0.0,
                margin_mean=float(view_margin[candidate_pos]),
                margin_std=0.0,
                config=runtime_config,
            )
            local_constraint = _pairwise_route_constraint_score(
                raw_utility_gain=float(view_utility[candidate_pos]),
                failure_risk=float(view_risk[candidate_pos]) if use_risk else 0.0,
                route_consistency=local_consistency,
                config=runtime_config,
            )
            allowed = float(
                view_decision[candidate_pos] > runtime_config.pairwise_route_threshold
                and view_margin[candidate_pos] > 0.0
                and local_constraint >= runtime_config.pairwise_constraint_floor
            )
            top1_matches.append(1.0 if local_best_idx == best_idx else 0.0)
            ranks.append(float(candidate_rank))
            route_allow.append(allowed)
            margins.append(float(view_margin[candidate_pos]))

        route_allow_share = float(np.mean(route_allow)) if route_allow else 0.0
        top1_share = float(np.mean(top1_matches)) if top1_matches else 0.0
        rank_mean = float(np.mean(ranks)) if ranks else 0.0
        rank_std = float(np.std(ranks)) if ranks else 0.0
        margin_mean = float(np.mean(margins)) if margins else route_margin_score
        margin_std = float(np.std(margins)) if margins else 0.0
        base_consistency = _pairwise_consistency_score(
            win_rate=route_allow_share,
            margin_mean=margin_mean,
            margin_std=margin_std,
            config=runtime_config,
        )
        consistency_stats = {
            "top1_share": top1_share,
            "rank_mean": rank_mean,
            "rank_std": rank_std,
            "route_allow_share": route_allow_share,
            "margin_mean": margin_mean,
            "margin_std": margin_std,
            "consistency": float(
                np.clip(
                    0.45 * route_allow_share
                    + 0.25 * top1_share
                    + 0.15 * (1.0 / (1.0 + rank_std))
                    + 0.15 * base_consistency,
                    0.0,
                    1.0,
                )
            ),
        }

    route_consistency = float(consistency_stats["consistency"]) if use_consistency else 1.0
    route_constraint_score = (
        _pairwise_route_constraint_score(
            raw_utility_gain=utility_score,
            failure_risk=risk_score,
            route_consistency=route_consistency,
            config=runtime_config,
        )
        if use_route_constraint
        else 1.0
    )
    safe_mask_regret_input = max(route_margin_score, 0.0) if variant == "utility_only" else regret_score
    safe_positive_mask = (
        _pairwise_safe_positive_mask(
            raw_utility_gain=utility_score,
            failure_risk=risk_score,
            fallback_regret_raw=safe_mask_regret_input,
            route_consistency=route_consistency,
            route_margin_raw=route_margin_score,
            config=runtime_config,
        )
        if use_route_constraint
        else float(route_margin_score > 0.0)
    )
    route_win_raw = float(route_margin_score > 0.0)
    route_permission_target = float(route_win_raw * route_constraint_score) if use_route_constraint else route_win_raw
    regret_supervision_weight = float(
        np.clip(
            runtime_config.pairwise_min_regret_weight
            + (1.0 - runtime_config.pairwise_min_regret_weight) * np.power(route_constraint_score, 1.5),
            0.0,
            1.0,
        )
    )
    final_route_score = base_decision_score * route_constraint_score if use_route_constraint else base_decision_score
    threshold_pass = float(final_route_score > runtime_config.pairwise_route_threshold)
    margin_pass = float(route_margin_score > 0.0)
    route_win_safe = float(threshold_pass > 0.5 and margin_pass > 0.5 and safe_positive_mask > 0.5)

    if best_idx == stage1_idx:
        routed_away = False
    elif use_explicit_conservative:
        routed_away = bool(route_win_safe > 0.5)
    else:
        routed_away = True

    if routed_away:
        final_gate = candidate_gates[best_idx].copy()
        source = f"pairregret_cal_route:{best_candidate_name}"
    else:
        final_gate = stage1_gate.copy()
        source = "pairregret_cal_stay:stage1"

    final_gate = _apply_gate_safety_projection(
        gate=final_gate,
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        safety_floor_strength=float(runtime["config"]["safety_floor_strength"]),
    )

    transfer_risk = float(
        np.clip(
            0.70 * risk_score + 0.30 * _sigmoid_np(float(final_route_score - route_margin_score)),
            0.0,
            1.0,
        )
    )
    reliability = float(
        np.clip(
            0.30 * (1.0 - transfer_risk)
            + 0.30 * route_constraint_score
            + 0.25 * route_consistency
            + 0.15 * _sigmoid_np(4.0 * final_route_score),
            runtime_config.bank_min_reliability,
            0.99,
        )
    )
    refine_intensity = 0.0
    if refine_enabled and routed_away:
        refine_intensity = float(
            np.clip(
                _sigmoid_np(4.0 * final_route_score) * (1.0 - 0.55 * risk_score) * (0.35 + 0.65 * regret_score),
                0.0,
                1.0,
            )
        )

    metadata: dict[str, float] = {
        "bank_reliability": reliability,
        "bank_consensus": float(consistency_stats["top1_share"]),
        "bank_uncertainty": float(np.clip(1.0 - route_consistency, 0.0, 1.0)),
        "bank_transfer_risk": transfer_risk,
        "bank_blend_weight": 1.0 if routed_away else 0.0,
        "bank_logit_margin": final_route_score,
        "bank_view_top1_share": float(consistency_stats["top1_share"]),
        "bank_view_rank_std": float(consistency_stats["rank_std"]),
        "bank_view_margin_mean": float(consistency_stats["margin_mean"]),
        "bank_view_margin_std": float(consistency_stats["margin_std"]),
        "bank_structure_support": float(route_constraint_score),
        "bank_refine_support": float(route_consistency),
        "bank_boundary_batch_risk": float(np.clip(transfer_risk, 0.0, 1.0)),
        "bank_selected_is_stage1": 0.0 if routed_away else 1.0,
        "pairwise_utility_gain": utility_score,
        "pairwise_safe_utility_gain": utility_score,
        "pairwise_failure_risk": risk_score,
        "pairwise_fallback_regret": regret_score,
        "pairwise_fallback_regret_calibrated": regret_score,
        "pairwise_route_margin": route_margin_score,
        "pairwise_route_margin_calibrated": route_margin_score,
        "pairwise_route_win_raw": route_win_raw,
        "pairwise_route_permission_target": route_permission_target,
        "pairwise_decision_score": base_decision_score,
        "pairwise_route_consistency": route_consistency,
        "pairwise_safe_positive_mask": safe_positive_mask,
        "pairwise_regret_supervision_weight": regret_supervision_weight,
        "pairwise_route_constraint_score": route_constraint_score,
        "pairwise_final_route_score": final_route_score,
        "pairwise_route_win_safe": route_win_safe,
        "pairwise_anchor_selection_outcome": 1.0 if not routed_away else 0.0,
        "pairwise_routed_away_from_stage1": 1.0 if routed_away else 0.0,
        "pairwise_route_threshold_pass": threshold_pass,
        "pairwise_margin_pass": margin_pass,
        "pairwise_refine_enabled": 1.0 if refine_enabled else 0.0,
        "pairwise_refine_intensity": refine_intensity,
        "pairwise_conservative_routing_enabled": 1.0 if use_explicit_conservative else 0.0,
    }
    return _normalize_gate_np(final_gate), source, metadata


def select_stage1_anchored_pairwise_gate_bank_weights_permissioned(
    *,
    x: np.ndarray,
    counts: np.ndarray,
    batches: np.ndarray | None,
    dataset_stats: dict[str, float],
    base_features: dict[str, np.ndarray],
    expert_scores: dict[str, np.ndarray],
    checkpoint_path: str,
    heuristic_gate: np.ndarray,
    selector: RefineMoEHVGSelector,
    device: torch.device | None = None,
    variant: str = "full",
) -> tuple[np.ndarray, str, dict[str, float | str]]:
    components = _point_gate_components_from_runtime(
        dataset_stats=dataset_stats,
        checkpoint_path=checkpoint_path,
        heuristic_gate=heuristic_gate,
        device=device,
    )
    runtime = components["runtime"]
    pairperm_packages: dict[str, dict[str, object]] = runtime.get("pairwise_permission_models", {})  # type: ignore[assignment]
    model_variant = variant if variant in pairperm_packages else "full"
    pairperm_package = pairperm_packages.get(model_variant) or pairperm_packages.get("full")
    if not pairperm_package:
        return select_stage1_anchored_pairwise_gate_bank_weights_calibrated(
            x=x,
            counts=counts,
            batches=batches,
            dataset_stats=dataset_stats,
            base_features=base_features,
            expert_scores=expert_scores,
            checkpoint_path=checkpoint_path,
            heuristic_gate=heuristic_gate,
            selector=selector,
            device=device,
            variant="full",
        )

    pairperm_model: nn.Module = pairperm_package["model"]  # type: ignore[assignment]
    pairperm_feature_keys = list(pairperm_package["feature_keys"])
    pairperm_feature_mean = np.asarray(pairperm_package["feature_mean"], dtype=np.float32)
    pairperm_feature_std = np.asarray(pairperm_package["feature_std"], dtype=np.float32)

    candidate_names, candidate_gates, prototype_distances = _build_inference_gate_bank(
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        residual_gate=np.asarray(components["residual_gate"], dtype=np.float64),
        stage1_gate=np.asarray(components["stage1_blended"], dtype=np.float64),
        prototype_gate=np.asarray(components["prototype_gate"], dtype=np.float64),
        prototype_neighbor_gates=np.asarray(components["prototype_neighbor_gates"], dtype=np.float64),
        prototype_neighbor_distances=np.asarray(components["prototype_neighbor_distances"], dtype=np.float64),
        runtime_config=runtime["config"],
    )
    bank_names = tuple(runtime["bank_candidate_names"])
    if device is None:
        device = choose_torch_device()

    runtime_config = GateLearningConfig(**runtime["config"])
    stage1_gate = np.asarray(components["stage1_blended"], dtype=np.float64)
    stage1_idx = int(np.where(candidate_names == "stage1_blended")[0][0])
    non_stage1_idx = np.asarray([idx for idx in range(len(candidate_names)) if idx != stage1_idx], dtype=np.int64)
    use_permission_head = variant != "no_permission_head"
    use_regret_aux = variant != "no_regret_aux"
    use_value_decoupling = variant not in {"no_permission_value_decoupling", "permission_only"}
    permission_only = variant == "permission_only"
    refine_enabled = variant == "refine_on"

    def build_candidate_features(
        *,
        view_counts: np.ndarray,
        view_batches: np.ndarray | None,
        candidate_idx: np.ndarray | None = None,
        prepared_context: tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        if prepared_context is None:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = selector.prepare_context(
                counts=view_counts,
                batches=view_batches,
            )
        else:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = prepared_context
        proxy_context = _prepare_bank_proxy_context(
            x=current_x,
            batches=view_batches,
            random_state=selector.random_state,
        )
        feature_dicts: list[dict[str, float]] = []
        indices = np.arange(len(candidate_names), dtype=np.int64) if candidate_idx is None else np.asarray(candidate_idx, dtype=np.int64)
        for idx in indices:
            gate = candidate_gates[int(idx)]
            no_refine_scores = selector.score_with_context(
                x=current_x,
                base_features=current_base_features,
                dataset_stats=current_dataset_stats,
                expert_scores=current_expert_scores,
                gate=gate,
                apply_refine=False,
            )
            current_top_k = min(selector.top_k, no_refine_scores.size)
            selected_idx = np.argsort(no_refine_scores)[-current_top_k:]
            boundary_size = min(len(no_refine_scores), max(current_top_k, int(1.5 * current_top_k)))
            boundary_idx = np.argsort(no_refine_scores)[-boundary_size:]
            feature_dicts.append(
                _gate_bank_candidate_features(
                    candidate_name=str(candidate_names[int(idx)]),
                    gate=gate,
                    score_vector=no_refine_scores,
                    selected_idx=selected_idx,
                    boundary_idx=boundary_idx,
                    dataset_stats=current_dataset_stats,
                    base_features=current_base_features,
                    x=current_x,
                    proxy_context=proxy_context,
                    heuristic_gate=heuristic_gate,
                    stage1_gate=stage1_gate,
                    prototype_distance=float(prototype_distances[int(idx)]),
                    bank_names=bank_names,
                    random_state=selector.random_state,
                )
            )
        return indices, feature_dicts

    def permission_outputs_for_indices(
        *,
        candidate_idx: np.ndarray,
        view_counts: np.ndarray,
        view_batches: np.ndarray | None,
        prepared_context: tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]] | None = None,
    ) -> tuple[
        np.ndarray,
        list[dict[str, float]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        indices, feature_dicts = build_candidate_features(
            view_counts=view_counts,
            view_batches=view_batches,
            candidate_idx=candidate_idx,
            prepared_context=prepared_context,
        )
        anchor_pos = int(np.where(indices == stage1_idx)[0][0])
        anchor_features = feature_dicts[anchor_pos]
        feature_rows = [
            [
                float(
                    _build_stage1_pairwise_feature_dict(
                        candidate_features=feature_dict,
                        anchor_features=anchor_features,
                        candidate_name=str(candidate_names[int(idx)]),
                        bank_names=bank_names,
                    )[key]
                )
                for key in pairperm_feature_keys
            ]
            for idx, feature_dict in zip(indices, feature_dicts, strict=False)
        ]
        normalized = (np.asarray(feature_rows, dtype=np.float32) - pairperm_feature_mean[None, :]) / pairperm_feature_std[None, :]
        with torch.no_grad():
            (
                utility,
                risk,
                regret,
                value,
                permission_logit,
                risk_budget,
                consistency_budget,
            ) = pairperm_model(
                torch.from_numpy(normalized.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
            )
        utility_np = utility.detach().cpu().numpy().reshape(-1)
        risk_np = risk.detach().cpu().numpy().reshape(-1)
        regret_np = regret.detach().cpu().numpy().reshape(-1)
        value_np = value.detach().cpu().numpy().reshape(-1)
        permission_logit_np = permission_logit.detach().cpu().numpy().reshape(-1)
        risk_budget_np = risk_budget.detach().cpu().numpy().reshape(-1)
        consistency_budget_np = consistency_budget.detach().cpu().numpy().reshape(-1)

        stage1_local_pos = int(np.where(indices == stage1_idx)[0][0])
        utility_np[stage1_local_pos] = 0.0
        risk_np[stage1_local_pos] = 0.0
        regret_np[stage1_local_pos] = 0.0
        value_np[stage1_local_pos] = 0.0
        permission_logit_np[stage1_local_pos] = 0.0
        risk_budget_np[stage1_local_pos] = runtime_config.pairwise_safe_risk_budget
        consistency_budget_np[stage1_local_pos] = runtime_config.pairwise_consistency_threshold
        if not use_regret_aux:
            regret_np[:] = 0.0
        return (
            indices,
            feature_dicts,
            utility_np,
            risk_np,
            regret_np,
            value_np,
            permission_logit_np,
            risk_budget_np,
            consistency_budget_np,
        )

    (
        full_indices,
        full_feature_dicts,
        utility_np,
        risk_np,
        regret_np,
        value_np,
        permission_logit_np,
        risk_budget_np,
        consistency_budget_np,
    ) = permission_outputs_for_indices(
        candidate_idx=np.arange(len(candidate_names), dtype=np.int64),
        view_counts=np.asarray(counts, dtype=np.float64),
        view_batches=None if batches is None else np.asarray(batches),
        prepared_context=(x, base_features, dataset_stats, expert_scores),
    )

    candidate_metrics: dict[int, dict[str, float]] = {
        int(idx): {
            "top1_share": 1.0 if idx != stage1_idx else 0.0,
            "rank_std": 0.0,
            "margin_mean": float(value_np[int(np.where(full_indices == idx)[0][0])]) if idx != stage1_idx else 0.0,
            "margin_std": 0.0,
            "route_consistency": 1.0 if idx == stage1_idx else 0.0,
        }
        for idx in full_indices
    }
    if counts.shape[0] > 48 and non_stage1_idx.size:
        view_rng = np.random.default_rng(selector.random_state + 881 * counts.shape[0] + 37 * counts.shape[1])
        per_candidate_top1: dict[int, list[float]] = {int(idx): [] for idx in non_stage1_idx}
        per_candidate_rank: dict[int, list[float]] = {int(idx): [] for idx in non_stage1_idx}
        per_candidate_margin: dict[int, list[float]] = {int(idx): [] for idx in non_stage1_idx}

        for _ in range(max(1, int(runtime_config.bank_uncertainty_views))):
            view_idx = _subsample_bank_view_indices(
                n_cells=counts.shape[0],
                batches=None if batches is None else np.asarray(batches),
                ratio=float(runtime_config.bank_subsample_ratio),
                rng=view_rng,
            )
            view_counts = np.asarray(counts[view_idx], dtype=np.float64)
            view_batches = None if batches is None else np.asarray(batches)[view_idx]
            (
                view_indices,
                _,
                _view_utility,
                _view_risk,
                _view_regret,
                view_value,
                view_permission_logit,
                _view_risk_budget,
                _view_consistency_budget,
            ) = permission_outputs_for_indices(
                candidate_idx=np.arange(len(candidate_names), dtype=np.int64),
                view_counts=view_counts,
                view_batches=view_batches,
            )
            if permission_only:
                view_signal = torch.sigmoid(torch.from_numpy(view_permission_logit)).numpy() - 0.5
            elif variant == "no_permission_value_decoupling":
                coupled = torch.sigmoid(torch.from_numpy(view_permission_logit)).numpy() * (
                    torch.sigmoid(4.0 * torch.from_numpy(view_value)).numpy()
                )
                view_signal = coupled - 0.5
            else:
                view_signal = view_value
            ranked_view = view_indices[np.argsort(view_signal)[::-1]]
            best_view_idx = int(ranked_view[0])
            for idx in non_stage1_idx:
                local_pos = int(np.where(view_indices == idx)[0][0])
                rank = int(np.where(ranked_view == idx)[0][0])
                per_candidate_top1[int(idx)].append(1.0 if best_view_idx == int(idx) else 0.0)
                per_candidate_rank[int(idx)].append(float(rank))
                per_candidate_margin[int(idx)].append(float(view_signal[local_pos]))

        for idx in non_stage1_idx:
            margin_array = np.asarray(per_candidate_margin[int(idx)], dtype=np.float64)
            top1_share = float(np.mean(per_candidate_top1[int(idx)])) if per_candidate_top1[int(idx)] else 0.0
            rank_std = float(np.std(per_candidate_rank[int(idx)])) if per_candidate_rank[int(idx)] else 0.0
            margin_mean = float(np.mean(margin_array)) if margin_array.size else 0.0
            margin_std = float(np.std(margin_array)) if margin_array.size else 0.0
            base_consistency = _pairwise_consistency_score(
                win_rate=float(np.mean(margin_array > 0.0)) if margin_array.size else 0.0,
                margin_mean=margin_mean,
                margin_std=margin_std,
                config=runtime_config,
            )
            route_consistency = float(
                np.clip(
                    0.45 * float(np.mean(margin_array > 0.0) if margin_array.size else 0.0)
                    + 0.25 * top1_share
                    + 0.15 * (1.0 / (1.0 + rank_std))
                    + 0.15 * base_consistency,
                    0.0,
                    1.0,
                )
            )
            candidate_metrics[int(idx)] = {
                "top1_share": top1_share,
                "rank_std": rank_std,
                "margin_mean": margin_mean,
                "margin_std": margin_std,
                "route_consistency": route_consistency,
            }

    anchor_pos = int(np.where(full_indices == stage1_idx)[0][0])
    anchor_features = full_feature_dicts[anchor_pos]
    candidate_infos: list[dict[str, float | str | int]] = []
    candidate_summaries: list[dict[str, float]] = []
    for pos, idx in enumerate(full_indices):
        idx_int = int(idx)
        candidate_name = str(candidate_names[idx_int])
        route_consistency = 1.0 if idx_int == stage1_idx else float(candidate_metrics[idx_int]["route_consistency"])
        route_signal = (
            float(value_np[pos])
            if use_value_decoupling
            else float(torch.sigmoid(torch.tensor(permission_logit_np[pos])).item() * torch.sigmoid(4.0 * torch.tensor(value_np[pos])).item() - 0.5)
        )
        if permission_only:
            route_signal = float(torch.sigmoid(torch.tensor(permission_logit_np[pos])).item() - 0.5)
        fallback_regret_input = regret_np[pos] if use_regret_aux else max(route_signal, 0.0)
        safe_positive_mask = (
            0.0
            if idx_int == stage1_idx
            else _pairwise_safe_positive_mask(
                raw_utility_gain=float(utility_np[pos]),
                failure_risk=float(risk_np[pos]),
                fallback_regret_raw=float(fallback_regret_input),
                route_consistency=route_consistency,
                route_margin_raw=route_signal,
                config=runtime_config,
            )
        )
        escape_support = _pairperm_candidate_escape_support(
            safe_utility_gain=float(utility_np[pos]),
            fallback_regret_calibrated=float(regret_np[pos]) if use_regret_aux else 0.0,
            failure_risk=float(risk_np[pos]),
            route_consistency=route_consistency,
            route_margin_raw=route_signal,
            safe_positive_mask=safe_positive_mask,
            config=runtime_config,
        )
        candidate_summaries.append(
            {
                "candidate_is_stage1": 1.0 if idx_int == stage1_idx else 0.0,
                "safe_utility_gain": float(utility_np[pos]),
                "fallback_regret_calibrated": float(regret_np[pos]) if use_regret_aux else 0.0,
                "failure_risk": float(risk_np[pos]),
                "route_consistency": route_consistency,
                "route_margin_raw": route_signal,
                "safe_positive_mask": safe_positive_mask,
                "escape_support": escape_support,
                "route_value_target": float(value_np[pos]),
            }
        )
        candidate_infos.append(
            {
                "idx": idx_int,
                "candidate_name": candidate_name,
                "utility": float(utility_np[pos]),
                "risk": float(risk_np[pos]),
                "regret": float(regret_np[pos]) if use_regret_aux else 0.0,
                "value": float(value_np[pos]),
                "permission_logit": float(permission_logit_np[pos]),
                "risk_budget": float(risk_budget_np[pos]),
                "consistency_budget": float(consistency_budget_np[pos]),
                "route_signal": route_signal,
                "route_consistency": route_consistency,
                "top1_share": float(candidate_metrics[idx_int]["top1_share"]),
                "rank_std": float(candidate_metrics[idx_int]["rank_std"]),
                "margin_mean": float(candidate_metrics[idx_int]["margin_mean"]),
                "margin_std": float(candidate_metrics[idx_int]["margin_std"]),
                "safe_positive_mask": safe_positive_mask,
                "escape_support": escape_support,
            }
        )

    anchor_context = _pairperm_anchor_context(
        candidate_summaries=candidate_summaries,
        anchor_features=anchor_features,
    )

    for info in candidate_infos:
        idx_int = int(info["idx"])
        permission_prob = float(_sigmoid_np(float(info["permission_logit"])) if use_permission_head else _sigmoid_np(4.0 * float(info["value"])))
        if variant == "fixed_budget":
            risk_budget_eval = float(runtime_config.pairwise_safe_risk_budget)
            consistency_budget_eval = float(runtime_config.pairwise_consistency_threshold)
        else:
            risk_budget_eval = float(info["risk_budget"])
            consistency_budget_eval = float(info["consistency_budget"])
        budget_stats = _pairperm_budget_score(
            failure_risk=float(info["risk"]),
            route_consistency=float(info["route_consistency"]),
            adaptive_risk_budget=risk_budget_eval,
            adaptive_consistency_budget=consistency_budget_eval,
            config=runtime_config,
        )
        permission_calibrated = float(
            np.clip(
                permission_prob * (0.35 + 0.65 * budget_stats["permission_budget_score"]),
                0.0,
                1.0,
            )
        )
        if permission_only:
            final_permission_pass = float(
                idx_int != stage1_idx and permission_calibrated >= runtime_config.pairperm_permission_threshold
            )
            ranking_score = permission_calibrated
        elif variant == "no_permission_value_decoupling":
            ranking_score = float(permission_calibrated * _sigmoid_np(4.0 * float(info["value"])))
            final_permission_pass = float(
                idx_int != stage1_idx and ranking_score >= runtime_config.pairperm_permission_threshold
            )
        else:
            final_permission_pass = float(
                idx_int != stage1_idx and permission_calibrated >= runtime_config.pairperm_permission_threshold
            )
            ranking_score = float(info["value"])

        block_reason = _pairperm_block_reason(
            candidate_name=str(info["candidate_name"]),
            safe_positive_mask=float(info["safe_positive_mask"]),
            failure_risk=float(info["risk"]),
            route_consistency=float(info["route_consistency"]),
            adaptive_risk_budget=risk_budget_eval,
            adaptive_consistency_budget=consistency_budget_eval,
            anchor_escape_pressure=anchor_context["anchor_escape_pressure"],
            permission_calibrated=permission_calibrated,
            config=runtime_config,
        )
        info.update(
            {
                "permission_prob": permission_prob,
                "permission_calibrated": permission_calibrated,
                "permission_budget_score": budget_stats["permission_budget_score"],
                "permission_margin": budget_stats["permission_margin"],
                "permission_confidence": budget_stats["permission_confidence"],
                "adaptive_risk_budget": risk_budget_eval,
                "adaptive_consistency_budget": consistency_budget_eval,
                "ranking_score": ranking_score,
                "final_permission_pass": final_permission_pass,
                "block_reason": block_reason,
                "route_constraint_score": _pairwise_route_constraint_score(
                    raw_utility_gain=float(info["utility"]),
                    failure_risk=float(info["risk"]),
                    route_consistency=float(info["route_consistency"]),
                    config=runtime_config,
                ),
            }
        )

    non_stage1_infos = [info for info in candidate_infos if int(info["idx"]) != stage1_idx]
    if not non_stage1_infos:
        final_gate = stage1_gate.copy()
        source = "pairperm_stay:stage1"
        metadata = {
            "pairwise_route_permission_block_reason": "no_candidate",
            "pairwise_anchor_selection_outcome": 1.0,
            "pairwise_routed_away_from_stage1": 0.0,
            "bank_selected_is_stage1": 1.0,
        }
        return _normalize_gate_np(final_gate), source, metadata

    if variant == "no_permission_value_decoupling":
        best_info = max(non_stage1_infos, key=lambda current: float(current["ranking_score"]))
        routed_away = bool(float(best_info["final_permission_pass"]) > 0.5)
    else:
        allowed_infos = [info for info in non_stage1_infos if float(info["final_permission_pass"]) > 0.5]
        if allowed_infos:
            best_info = max(allowed_infos, key=lambda current: float(current["ranking_score"]))
            routed_away = True
        else:
            best_info = max(non_stage1_infos, key=lambda current: float(current["permission_calibrated"]))
            routed_away = False

    best_idx = int(best_info["idx"])
    if routed_away:
        final_gate = candidate_gates[best_idx].copy()
        source = f"pairperm_route:{best_info['candidate_name']}"
    else:
        final_gate = stage1_gate.copy()
        source = "pairperm_stay:stage1"

    final_gate = _apply_gate_safety_projection(
        gate=final_gate,
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        safety_floor_strength=float(runtime["config"]["safety_floor_strength"]),
    )

    transfer_risk = float(
        np.clip(
            0.60 * float(best_info["risk"])
            + 0.25 * (1.0 - float(best_info["permission_budget_score"]))
            + 0.15 * (1.0 - float(best_info["permission_prob"])),
            0.0,
            1.0,
        )
    )
    reliability = float(
        np.clip(
            0.28 * (1.0 - transfer_risk)
            + 0.24 * float(best_info["permission_calibrated"])
            + 0.20 * float(best_info["route_consistency"])
            + 0.16 * float(best_info["permission_confidence"])
            + 0.12 * max(float(best_info["value"]), 0.0),
            runtime_config.bank_min_reliability,
            0.99,
        )
    )
    refine_intensity = 0.0
    if refine_enabled and routed_away:
        regret_factor = float(best_info["regret"]) if use_regret_aux else float(best_info["permission_prob"])
        refine_intensity = float(
            np.clip(
                float(best_info["permission_calibrated"])
                * max(float(best_info["value"]), 0.0)
                * (0.35 + 0.65 * (1.0 - float(best_info["risk"])))
                * (0.45 + 0.55 * regret_factor),
                0.0,
                1.0,
            )
        )

    metadata: dict[str, float | str] = {
        "bank_reliability": reliability,
        "bank_consensus": float(best_info["top1_share"]),
        "bank_uncertainty": float(np.clip(1.0 - float(best_info["route_consistency"]), 0.0, 1.0)),
        "bank_transfer_risk": transfer_risk,
        "bank_blend_weight": 1.0 if routed_away else 0.0,
        "bank_logit_margin": float(best_info["ranking_score"]),
        "bank_view_top1_share": float(best_info["top1_share"]),
        "bank_view_rank_std": float(best_info["rank_std"]),
        "bank_view_margin_mean": float(best_info["margin_mean"]),
        "bank_view_margin_std": float(best_info["margin_std"]),
        "bank_structure_support": float(best_info["route_constraint_score"]),
        "bank_refine_support": float(best_info["route_consistency"]),
        "bank_boundary_batch_risk": float(np.clip(transfer_risk, 0.0, 1.0)),
        "bank_selected_is_stage1": 0.0 if routed_away else 1.0,
        "pairwise_utility_gain": float(best_info["utility"]),
        "pairwise_safe_utility_gain": float(best_info["utility"]),
        "pairwise_failure_risk": float(best_info["risk"]),
        "pairwise_fallback_regret": float(best_info["regret"]),
        "pairwise_fallback_regret_calibrated": float(best_info["regret"]),
        "pairwise_route_margin": float(best_info["route_signal"]),
        "pairwise_route_margin_calibrated": float(best_info["route_signal"]),
        "pairwise_route_consistency": float(best_info["route_consistency"]),
        "pairwise_safe_positive_mask": float(best_info["safe_positive_mask"]),
        "pairwise_regret_supervision_weight": 1.0 if use_regret_aux else float(runtime_config.pairwise_min_regret_weight),
        "pairwise_route_constraint_score": float(best_info["route_constraint_score"]),
        "pairwise_final_route_score": float(best_info["value"]),
        "pairwise_route_permission_logit": float(best_info["permission_logit"]),
        "pairwise_route_permission_prob": float(best_info["permission_prob"]),
        "pairwise_route_permission_target": float(best_info["permission_prob"]),
        "pairwise_route_permission_calibrated": float(best_info["permission_calibrated"]),
        "pairwise_route_permission_margin": float(best_info["permission_margin"]),
        "pairwise_route_permission_confidence": float(best_info["permission_confidence"]),
        "pairwise_anchor_escape_pressure": anchor_context["anchor_escape_pressure"],
        "pairwise_anchor_routability": anchor_context["anchor_routability"],
        "pairwise_anchor_fragility": anchor_context["anchor_fragility"],
        "pairwise_adaptive_risk_budget": float(best_info["adaptive_risk_budget"]),
        "pairwise_adaptive_consistency_budget": float(best_info["adaptive_consistency_budget"]),
        "pairwise_permission_budget_score": float(best_info["permission_budget_score"]),
        "pairwise_permission_block_reason": str(best_info["block_reason"]),
        "pairwise_route_permission_block_reason": str(best_info["block_reason"]),
        "pairwise_final_route_permission_pass": 1.0 if routed_away else 0.0,
        "pairwise_value_among_allowed_candidates": float(best_info["ranking_score"]) if routed_away else 0.0,
        "pairwise_route_selected_under_permission": 1.0 if routed_away else 0.0,
        "pairwise_anchor_selection_outcome": 1.0 if not routed_away else 0.0,
        "pairwise_routed_away_from_stage1": 1.0 if routed_away else 0.0,
        "pairwise_refine_enabled": 1.0 if refine_enabled else 0.0,
        "pairwise_refine_intensity": refine_intensity,
    }
    return _normalize_gate_np(final_gate), source, metadata


def select_stage1_anchored_pairwise_gate_bank_weights_escapecert(
    *,
    x: np.ndarray,
    counts: np.ndarray,
    batches: np.ndarray | None,
    dataset_stats: dict[str, float],
    base_features: dict[str, np.ndarray],
    expert_scores: dict[str, np.ndarray],
    checkpoint_path: str,
    heuristic_gate: np.ndarray,
    selector: RefineMoEHVGSelector,
    device: torch.device | None = None,
    variant: str = "full",
) -> tuple[np.ndarray, str, dict[str, float | str]]:
    components = _point_gate_components_from_runtime(
        dataset_stats=dataset_stats,
        checkpoint_path=checkpoint_path,
        heuristic_gate=heuristic_gate,
        device=device,
    )
    runtime = components["runtime"]
    escapecert_packages: dict[str, dict[str, object]] = runtime.get("pairwise_escapecert_models", {})  # type: ignore[assignment]
    model_variant = variant if variant in escapecert_packages else "full"
    escapecert_package = escapecert_packages.get(model_variant) or escapecert_packages.get("full")
    if not escapecert_package:
        return select_stage1_anchored_pairwise_gate_bank_weights_permissioned(
            x=x,
            counts=counts,
            batches=batches,
            dataset_stats=dataset_stats,
            base_features=base_features,
            expert_scores=expert_scores,
            checkpoint_path=checkpoint_path,
            heuristic_gate=heuristic_gate,
            selector=selector,
            device=device,
            variant="full",
        )

    escapecert_model: nn.Module = escapecert_package["model"]  # type: ignore[assignment]
    escapecert_feature_keys = list(escapecert_package["feature_keys"])
    escapecert_feature_mean = np.asarray(escapecert_package["feature_mean"], dtype=np.float32)
    escapecert_feature_std = np.asarray(escapecert_package["feature_std"], dtype=np.float32)

    candidate_names, candidate_gates, prototype_distances = _build_inference_gate_bank(
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        residual_gate=np.asarray(components["residual_gate"], dtype=np.float64),
        stage1_gate=np.asarray(components["stage1_blended"], dtype=np.float64),
        prototype_gate=np.asarray(components["prototype_gate"], dtype=np.float64),
        prototype_neighbor_gates=np.asarray(components["prototype_neighbor_gates"], dtype=np.float64),
        prototype_neighbor_distances=np.asarray(components["prototype_neighbor_distances"], dtype=np.float64),
        runtime_config=runtime["config"],
    )
    bank_names = tuple(runtime["bank_candidate_names"])
    if device is None:
        device = choose_torch_device()

    runtime_config = GateLearningConfig(**runtime["config"])
    flags = _escapecert_variant_flags(model_variant)
    stage1_gate = np.asarray(components["stage1_blended"], dtype=np.float64)
    stage1_idx = int(np.where(candidate_names == "stage1_blended")[0][0])
    non_stage1_idx = np.asarray([idx for idx in range(len(candidate_names)) if idx != stage1_idx], dtype=np.int64)
    anchor_threshold = runtime_config.pairperm_permission_threshold
    allowed_threshold = runtime_config.escapecert_admissibility_threshold

    def build_candidate_features(
        *,
        view_counts: np.ndarray,
        view_batches: np.ndarray | None,
        candidate_idx: np.ndarray | None = None,
        prepared_context: tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        if prepared_context is None:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = selector.prepare_context(
                counts=view_counts,
                batches=view_batches,
            )
        else:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = prepared_context
        proxy_context = _prepare_bank_proxy_context(
            x=current_x,
            batches=view_batches,
            random_state=selector.random_state,
        )
        feature_dicts: list[dict[str, float]] = []
        indices = np.arange(len(candidate_names), dtype=np.int64) if candidate_idx is None else np.asarray(candidate_idx, dtype=np.int64)
        for idx in indices:
            gate = candidate_gates[int(idx)]
            no_refine_scores = selector.score_with_context(
                x=current_x,
                base_features=current_base_features,
                dataset_stats=current_dataset_stats,
                expert_scores=current_expert_scores,
                gate=gate,
                apply_refine=False,
            )
            current_top_k = min(selector.top_k, no_refine_scores.size)
            selected_idx = np.argsort(no_refine_scores)[-current_top_k:]
            boundary_size = min(len(no_refine_scores), max(current_top_k, int(1.5 * current_top_k)))
            boundary_idx = np.argsort(no_refine_scores)[-boundary_size:]
            feature_dicts.append(
                _gate_bank_candidate_features(
                    candidate_name=str(candidate_names[int(idx)]),
                    gate=gate,
                    score_vector=no_refine_scores,
                    selected_idx=selected_idx,
                    boundary_idx=boundary_idx,
                    dataset_stats=current_dataset_stats,
                    base_features=current_base_features,
                    x=current_x,
                    proxy_context=proxy_context,
                    heuristic_gate=heuristic_gate,
                    stage1_gate=stage1_gate,
                    prototype_distance=float(prototype_distances[int(idx)]),
                    bank_names=bank_names,
                    random_state=selector.random_state,
                )
            )
        return indices, feature_dicts

    def escapecert_outputs_for_indices(
        *,
        candidate_idx: np.ndarray,
        view_counts: np.ndarray,
        view_batches: np.ndarray | None,
        prepared_context: tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]] | None = None,
    ) -> tuple[
        np.ndarray,
        list[dict[str, float]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        indices, feature_dicts = build_candidate_features(
            view_counts=view_counts,
            view_batches=view_batches,
            candidate_idx=candidate_idx,
            prepared_context=prepared_context,
        )
        anchor_pos = int(np.where(indices == stage1_idx)[0][0])
        anchor_features = feature_dicts[anchor_pos]
        feature_rows = [
            [
                float(
                    _build_stage1_pairwise_feature_dict(
                        candidate_features=feature_dict,
                        anchor_features=anchor_features,
                        candidate_name=str(candidate_names[int(idx)]),
                        bank_names=bank_names,
                    )[key]
                )
                for key in escapecert_feature_keys
            ]
            for idx, feature_dict in zip(indices, feature_dicts, strict=False)
        ]
        normalized = (np.asarray(feature_rows, dtype=np.float32) - escapecert_feature_mean[None, :]) / escapecert_feature_std[None, :]
        with torch.no_grad():
            (
                utility,
                risk,
                regret,
                value,
                anchor_escape_logit,
                admissibility_logit,
                risk_budget,
                consistency_budget,
                uncertainty,
            ) = escapecert_model(
                torch.from_numpy(normalized.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
            )
        utility_np = utility.detach().cpu().numpy().reshape(-1)
        risk_np = risk.detach().cpu().numpy().reshape(-1)
        regret_np = regret.detach().cpu().numpy().reshape(-1)
        value_np = value.detach().cpu().numpy().reshape(-1)
        anchor_escape_logit_np = anchor_escape_logit.detach().cpu().numpy().reshape(-1)
        admissibility_logit_np = admissibility_logit.detach().cpu().numpy().reshape(-1)
        risk_budget_np = risk_budget.detach().cpu().numpy().reshape(-1)
        consistency_budget_np = consistency_budget.detach().cpu().numpy().reshape(-1)
        uncertainty_np = uncertainty.detach().cpu().numpy().reshape(-1)

        stage1_local_pos = int(np.where(indices == stage1_idx)[0][0])
        utility_np[stage1_local_pos] = 0.0
        risk_np[stage1_local_pos] = 0.0
        regret_np[stage1_local_pos] = 0.0
        value_np[stage1_local_pos] = 0.0
        anchor_escape_logit_np[stage1_local_pos] = 0.0
        admissibility_logit_np[stage1_local_pos] = -12.0
        risk_budget_np[stage1_local_pos] = runtime_config.pairwise_safe_risk_budget
        consistency_budget_np[stage1_local_pos] = runtime_config.pairwise_consistency_threshold
        uncertainty_np[stage1_local_pos] = 1.0
        if not flags["use_regret_aux"]:
            regret_np[:] = 0.0
        return (
            indices,
            feature_dicts,
            utility_np,
            risk_np,
            regret_np,
            value_np,
            anchor_escape_logit_np,
            admissibility_logit_np,
            uncertainty_np,
            risk_budget_np,
            consistency_budget_np,
        )

    (
        full_indices,
        full_feature_dicts,
        utility_np,
        risk_np,
        regret_np,
        value_np,
        anchor_escape_logit_np,
        admissibility_logit_np,
        uncertainty_np,
        risk_budget_np,
        consistency_budget_np,
    ) = escapecert_outputs_for_indices(
        candidate_idx=np.arange(len(candidate_names), dtype=np.int64),
        view_counts=np.asarray(counts, dtype=np.float64),
        view_batches=None if batches is None else np.asarray(batches),
        prepared_context=(x, base_features, dataset_stats, expert_scores),
    )

    candidate_metrics: dict[int, dict[str, float]] = {
        int(idx): {
            "top1_share": 1.0 if idx != stage1_idx else 0.0,
            "rank_std": 0.0,
            "margin_mean": 0.0 if idx == stage1_idx else float(value_np[int(np.where(full_indices == idx)[0][0])]),
            "margin_std": 0.0,
            "route_consistency": 1.0 if idx == stage1_idx else 0.0,
        }
        for idx in full_indices
    }
    if counts.shape[0] > 48 and non_stage1_idx.size:
        view_rng = np.random.default_rng(selector.random_state + 997 * counts.shape[0] + 41 * counts.shape[1])
        per_candidate_top1: dict[int, list[float]] = {int(idx): [] for idx in non_stage1_idx}
        per_candidate_rank: dict[int, list[float]] = {int(idx): [] for idx in non_stage1_idx}
        per_candidate_margin: dict[int, list[float]] = {int(idx): [] for idx in non_stage1_idx}

        for _ in range(max(1, int(runtime_config.bank_uncertainty_views))):
            view_idx = _subsample_bank_view_indices(
                n_cells=counts.shape[0],
                batches=None if batches is None else np.asarray(batches),
                ratio=float(runtime_config.bank_subsample_ratio),
                rng=view_rng,
            )
            view_counts = np.asarray(counts[view_idx], dtype=np.float64)
            view_batches = None if batches is None else np.asarray(batches)[view_idx]
            (
                view_indices,
                _,
                view_utility,
                view_risk,
                view_regret,
                view_value,
                view_anchor_logit,
                view_admissibility_logit,
                _view_uncertainty,
                _view_risk_budget,
                _view_consistency_budget,
            ) = escapecert_outputs_for_indices(
                candidate_idx=np.arange(len(candidate_names), dtype=np.int64),
                view_counts=view_counts,
                view_batches=view_batches,
            )
            ranked_scores: list[float] = []
            for idx in view_indices:
                local_pos = int(np.where(view_indices == idx)[0][0])
                if int(idx) == stage1_idx:
                    ranked_scores.append(-1.0)
                    continue
                utility_support = _pairwise_positive_score(
                    raw_utility_gain=float(view_utility[local_pos]),
                    config=runtime_config,
                )
                value_support = _sigmoid_np(3.0 * float(view_value[local_pos]))
                regret_support = (
                    _sigmoid_np((float(view_regret[local_pos]) - 0.015) / 0.05)
                    if flags["use_regret_aux"]
                    else _sigmoid_np(3.0 * float(view_value[local_pos]) - 0.2)
                )
                risk_safety = _sigmoid_np((0.34 - float(view_risk[local_pos])) / 0.08)
                anchor_support = _sigmoid_np(float(view_anchor_logit[local_pos])) if flags["use_anchor_head"] else 1.0
                admissibility_support = (
                    _sigmoid_np(float(view_admissibility_logit[local_pos]))
                    if flags["use_admissibility_head"]
                    else np.clip(
                        0.26 * utility_support
                        + 0.30 * value_support
                        + 0.18 * regret_support
                        + 0.26 * risk_safety,
                        0.0,
                        1.0,
                    )
                )
                view_margin = (
                    anchor_support
                    * admissibility_support
                    * (0.35 + 0.65 * value_support)
                    - allowed_threshold
                )
                ranked_scores.append(float(view_margin))
            ranked_scores_np = np.asarray(ranked_scores, dtype=np.float64)
            ranked_view = view_indices[np.argsort(ranked_scores_np)[::-1]]
            best_view_idx = int(ranked_view[0])
            for idx in non_stage1_idx:
                local_pos = int(np.where(view_indices == idx)[0][0])
                rank = int(np.where(ranked_view == idx)[0][0])
                per_candidate_top1[int(idx)].append(1.0 if best_view_idx == int(idx) else 0.0)
                per_candidate_rank[int(idx)].append(float(rank))
                per_candidate_margin[int(idx)].append(float(ranked_scores_np[local_pos]))

        for idx in non_stage1_idx:
            margin_array = np.asarray(per_candidate_margin[int(idx)], dtype=np.float64)
            top1_share = float(np.mean(per_candidate_top1[int(idx)])) if per_candidate_top1[int(idx)] else 0.0
            rank_std = float(np.std(per_candidate_rank[int(idx)])) if per_candidate_rank[int(idx)] else 0.0
            margin_mean = float(np.mean(margin_array)) if margin_array.size else 0.0
            margin_std = float(np.std(margin_array)) if margin_array.size else 0.0
            base_consistency = _pairwise_consistency_score(
                win_rate=float(np.mean(margin_array > 0.0)) if margin_array.size else 0.0,
                margin_mean=margin_mean,
                margin_std=margin_std,
                config=runtime_config,
            )
            route_consistency = float(
                np.clip(
                    0.45 * float(np.mean(margin_array > 0.0) if margin_array.size else 0.0)
                    + 0.25 * top1_share
                    + 0.15 * (1.0 / (1.0 + rank_std))
                    + 0.15 * base_consistency,
                    0.0,
                    1.0,
                )
            )
            candidate_metrics[int(idx)] = {
                "top1_share": top1_share,
                "rank_std": rank_std,
                "margin_mean": margin_mean,
                "margin_std": margin_std,
                "route_consistency": route_consistency,
            }

    route_consistency_np = np.asarray(
        [
            1.0 if int(idx) == stage1_idx else float(candidate_metrics[int(idx)]["route_consistency"])
            for idx in full_indices
        ],
        dtype=np.float32,
    )
    candidate_is_stage1_np = np.asarray([1.0 if int(idx) == stage1_idx else 0.0 for idx in full_indices], dtype=np.float32)

    with torch.no_grad():
        group_outputs = _escapecert_group_predictions_torch(
            utility_pred=torch.from_numpy(utility_np.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
            risk_pred=torch.from_numpy(risk_np.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
            regret_pred=torch.from_numpy(regret_np.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
            value_pred=torch.from_numpy(value_np.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
            anchor_escape_logit=torch.from_numpy(anchor_escape_logit_np.astype(np.float32)).to(
                device,
                non_blocking=torch.cuda.is_available(),
            ),
            admissibility_logit=torch.from_numpy(admissibility_logit_np.astype(np.float32)).to(
                device,
                non_blocking=torch.cuda.is_available(),
            ),
            risk_budget_pred=torch.from_numpy(risk_budget_np.astype(np.float32)).to(
                device,
                non_blocking=torch.cuda.is_available(),
            ),
            consistency_budget_pred=torch.from_numpy(consistency_budget_np.astype(np.float32)).to(
                device,
                non_blocking=torch.cuda.is_available(),
            ),
            uncertainty_pred=torch.from_numpy(uncertainty_np.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
            route_consistency=torch.from_numpy(route_consistency_np).to(device, non_blocking=torch.cuda.is_available()),
            candidate_is_stage1=torch.from_numpy(candidate_is_stage1_np).to(device, non_blocking=torch.cuda.is_available()),
            config=runtime_config,
            variant=model_variant,
        )

    base_admissibility_prob_np = group_outputs["base_admissibility_prob"].detach().cpu().numpy().reshape(-1)
    candidate_admissibility_calibrated_np = group_outputs["candidate_admissibility_calibrated"].detach().cpu().numpy().reshape(-1)
    candidate_budget_score_np = group_outputs["candidate_budget_score"].detach().cpu().numpy().reshape(-1)
    anchor_escape_logit_agg = float(group_outputs["anchor_escape_logit"].detach().cpu().item())
    anchor_escape_prob = float(group_outputs["anchor_escape_prob"].detach().cpu().item())
    anchor_escape_calibrated = float(group_outputs["anchor_escape_calibrated"].detach().cpu().item())
    anchor_escape_uncertainty = float(group_outputs["anchor_escape_uncertainty"].detach().cpu().item())
    counterfactual_escape_budget_score = float(group_outputs["counterfactual_escape_budget_score"].detach().cpu().item())
    counterfactual_escape_support = float(group_outputs["counterfactual_escape_support"].detach().cpu().item())
    anchor_escape_counterfactual_margin = float(group_outputs["anchor_escape_counterfactual_margin"].detach().cpu().item())
    agg_risk_budget = float(group_outputs["agg_risk_budget"].detach().cpu().item())
    agg_consistency_budget = float(group_outputs["agg_consistency_budget"].detach().cpu().item())
    allowed_set_total_mass = float(group_outputs["allowed_set_total_mass"].detach().cpu().item())
    allowed_set_entropy = float(group_outputs["allowed_set_entropy"].detach().cpu().item())
    allowed_set_topm_mass = float(group_outputs["allowed_set_topm_mass"].detach().cpu().item())
    allowed_set_best_value = float(group_outputs["allowed_set_best_value"].detach().cpu().item())
    permission_set_consistency = float(group_outputs["permission_set_consistency"].detach().cpu().item())
    anchor_escape_safe_candidate_count = float(group_outputs["anchor_escape_safe_candidate_count"].detach().cpu().item())

    candidate_infos: list[dict[str, float | str | int]] = []
    for pos, idx in enumerate(full_indices):
        idx_int = int(idx)
        candidate_name = str(candidate_names[idx_int])
        route_consistency = float(route_consistency_np[pos])
        fallback_regret_input = float(regret_np[pos]) if flags["use_regret_aux"] else max(float(value_np[pos]), 0.0)
        safe_positive_mask = (
            0.0
            if idx_int == stage1_idx
            else _pairwise_safe_positive_mask(
                raw_utility_gain=float(utility_np[pos]),
                failure_risk=float(risk_np[pos]),
                fallback_regret_raw=fallback_regret_input,
                route_consistency=route_consistency,
                route_margin_raw=float(value_np[pos]),
                config=runtime_config,
            )
        )
        escape_support = _pairperm_candidate_escape_support(
            safe_utility_gain=float(utility_np[pos]),
            fallback_regret_calibrated=float(regret_np[pos]) if flags["use_regret_aux"] else 0.0,
            failure_risk=float(risk_np[pos]),
            route_consistency=route_consistency,
            route_margin_raw=float(value_np[pos]),
            safe_positive_mask=safe_positive_mask,
            config=runtime_config,
        )
        block_reason = _escapecert_block_reason(
            candidate_name=candidate_name,
            anchor_escape_calibrated=anchor_escape_calibrated,
            candidate_admissibility_calibrated=float(candidate_admissibility_calibrated_np[pos]),
            candidate_budget_score=float(candidate_budget_score_np[pos]),
            anchor_threshold=anchor_threshold,
            admissibility_threshold=allowed_threshold,
        )
        final_permission_pass = float(
            idx_int != stage1_idx
            and anchor_escape_calibrated >= anchor_threshold
            and float(candidate_admissibility_calibrated_np[pos]) >= allowed_threshold
        )
        candidate_infos.append(
            {
                "idx": idx_int,
                "candidate_name": candidate_name,
                "utility": float(utility_np[pos]),
                "risk": float(risk_np[pos]),
                "regret": float(regret_np[pos]) if flags["use_regret_aux"] else 0.0,
                "value": float(value_np[pos]),
                "anchor_logit": float(anchor_escape_logit_np[pos]),
                "admissibility_logit": float(admissibility_logit_np[pos]),
                "admissibility_prob": float(base_admissibility_prob_np[pos]),
                "admissibility_calibrated": float(candidate_admissibility_calibrated_np[pos]),
                "candidate_budget_score": float(candidate_budget_score_np[pos]),
                "route_consistency": route_consistency,
                "top1_share": float(candidate_metrics[idx_int]["top1_share"]),
                "rank_std": float(candidate_metrics[idx_int]["rank_std"]),
                "margin_mean": float(candidate_metrics[idx_int]["margin_mean"]),
                "margin_std": float(candidate_metrics[idx_int]["margin_std"]),
                "safe_positive_mask": safe_positive_mask,
                "escape_support": escape_support,
                "final_permission_pass": final_permission_pass,
                "block_reason": block_reason,
                "route_constraint_score": _pairwise_route_constraint_score(
                    raw_utility_gain=float(utility_np[pos]),
                    failure_risk=float(risk_np[pos]),
                    route_consistency=route_consistency,
                    config=runtime_config,
                ),
            }
        )

    non_stage1_infos = [info for info in candidate_infos if int(info["idx"]) != stage1_idx]
    if not non_stage1_infos:
        final_gate = stage1_gate.copy()
        source = "escapecert_stay:stage1"
        metadata = {
            "anchor_escape_logit": 0.0,
            "anchor_escape_prob": 0.0,
            "anchor_escape_calibrated": 0.0,
            "anchor_escape_counterfactual_margin": 0.0,
            "anchor_escape_soft_oracle_gain": 0.0,
            "anchor_escape_safe_candidate_count": 0.0,
            "anchor_escape_topm_mass": 0.0,
            "anchor_escape_uncertainty": 1.0,
            "allowed_set_total_mass": 0.0,
            "allowed_set_entropy": 0.0,
            "allowed_set_best_value": 0.0,
            "allowed_set_selected_candidate": "stage1_blended",
            "permission_set_consistency": 0.0,
            "counterfactual_escape_support": 0.0,
            "counterfactual_escape_budget_score": 0.0,
            "pairwise_route_permission_block_reason": "no_candidate",
            "pairwise_anchor_selection_outcome": 1.0,
            "pairwise_routed_away_from_stage1": 0.0,
            "bank_selected_is_stage1": 1.0,
        }
        return _normalize_gate_np(final_gate), source, metadata

    allowed_infos = [info for info in non_stage1_infos if float(info["final_permission_pass"]) > 0.5]
    if allowed_infos:
        best_info = max(allowed_infos, key=lambda current: float(current["value"]))
        routed_away = True
    else:
        best_info = max(
            non_stage1_infos,
            key=lambda current: (
                float(current["admissibility_calibrated"]),
                float(current["value"]),
            ),
        )
        routed_away = False

    best_idx = int(best_info["idx"])
    if routed_away:
        final_gate = candidate_gates[best_idx].copy()
        source = f"escapecert_route:{best_info['candidate_name']}"
    else:
        final_gate = stage1_gate.copy()
        source = "escapecert_stay:stage1"

    final_gate = _apply_gate_safety_projection(
        gate=final_gate,
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        safety_floor_strength=float(runtime["config"]["safety_floor_strength"]),
    )

    transfer_risk = float(
        np.clip(
            0.52 * float(best_info["risk"])
            + 0.23 * (1.0 - float(best_info["candidate_budget_score"]))
            + 0.15 * (1.0 - float(best_info["admissibility_calibrated"]))
            + 0.10 * anchor_escape_uncertainty,
            0.0,
            1.0,
        )
    )
    reliability = float(
        np.clip(
            0.24 * (1.0 - transfer_risk)
            + 0.18 * float(best_info["admissibility_calibrated"])
            + 0.18 * float(best_info["route_consistency"])
            + 0.18 * anchor_escape_calibrated
            + 0.12 * (1.0 - anchor_escape_uncertainty)
            + 0.10 * max(float(best_info["value"]), 0.0),
            runtime_config.bank_min_reliability,
            0.99,
        )
    )

    metadata: dict[str, float | str] = {
        "bank_reliability": reliability,
        "bank_consensus": float(best_info["top1_share"]),
        "bank_uncertainty": float(np.clip(1.0 - float(best_info["route_consistency"]), 0.0, 1.0)),
        "bank_transfer_risk": transfer_risk,
        "bank_blend_weight": 1.0 if routed_away else 0.0,
        "bank_logit_margin": float(best_info["value"]),
        "bank_view_top1_share": float(best_info["top1_share"]),
        "bank_view_rank_std": float(best_info["rank_std"]),
        "bank_view_margin_mean": float(best_info["margin_mean"]),
        "bank_view_margin_std": float(best_info["margin_std"]),
        "bank_structure_support": float(best_info["route_constraint_score"]),
        "bank_refine_support": float(best_info["route_consistency"]),
        "bank_boundary_batch_risk": float(np.clip(transfer_risk, 0.0, 1.0)),
        "bank_selected_is_stage1": 0.0 if routed_away else 1.0,
        "anchor_escape_logit": anchor_escape_logit_agg,
        "anchor_escape_prob": anchor_escape_prob,
        "anchor_escape_calibrated": anchor_escape_calibrated,
        "anchor_escape_counterfactual_margin": anchor_escape_counterfactual_margin,
        "anchor_escape_soft_oracle_gain": allowed_set_best_value,
        "anchor_escape_safe_candidate_count": anchor_escape_safe_candidate_count,
        "anchor_escape_topm_mass": allowed_set_topm_mass,
        "anchor_escape_uncertainty": anchor_escape_uncertainty,
        "candidate_admissibility_logit": float(best_info["admissibility_logit"]),
        "candidate_admissibility_prob": float(best_info["admissibility_prob"]),
        "candidate_admissibility_calibrated": float(best_info["admissibility_calibrated"]),
        "candidate_admissibility_margin": float(best_info["admissibility_calibrated"]) - allowed_threshold,
        "allowed_set_total_mass": allowed_set_total_mass,
        "allowed_set_entropy": allowed_set_entropy,
        "allowed_set_best_value": allowed_set_best_value,
        "allowed_set_selected_candidate": str(best_info["candidate_name"]) if routed_away else "stage1_blended",
        "permission_set_consistency": permission_set_consistency,
        "counterfactual_escape_support": counterfactual_escape_support,
        "counterfactual_escape_budget_score": counterfactual_escape_budget_score,
        "pairwise_utility_gain": float(best_info["utility"]),
        "pairwise_safe_utility_gain": float(best_info["utility"]),
        "pairwise_failure_risk": float(best_info["risk"]),
        "pairwise_fallback_regret": float(best_info["regret"]),
        "pairwise_fallback_regret_calibrated": float(best_info["regret"]),
        "pairwise_route_margin": float(best_info["value"]),
        "pairwise_route_margin_calibrated": float(best_info["value"]),
        "pairwise_route_consistency": float(best_info["route_consistency"]),
        "pairwise_safe_positive_mask": float(best_info["safe_positive_mask"]),
        "pairwise_regret_supervision_weight": 1.0 if flags["use_regret_aux"] else float(runtime_config.pairwise_min_regret_weight),
        "pairwise_route_constraint_score": float(best_info["route_constraint_score"]),
        "pairwise_final_route_score": float(best_info["value"]),
        "pairwise_route_permission_logit": float(best_info["admissibility_logit"]),
        "pairwise_route_permission_prob": float(best_info["admissibility_prob"]),
        "pairwise_route_permission_target": float(best_info["admissibility_prob"]),
        "pairwise_route_permission_calibrated": float(best_info["admissibility_calibrated"]),
        "pairwise_route_permission_margin": float(best_info["admissibility_calibrated"]) - allowed_threshold,
        "pairwise_route_permission_confidence": float(np.clip(abs(2.0 * float(best_info["admissibility_calibrated"]) - 1.0), 0.0, 1.0)),
        "pairwise_anchor_escape_pressure": anchor_escape_calibrated,
        "pairwise_anchor_routability": allowed_set_topm_mass,
        "pairwise_anchor_fragility": anchor_escape_uncertainty,
        "pairwise_adaptive_risk_budget": agg_risk_budget,
        "pairwise_adaptive_consistency_budget": agg_consistency_budget,
        "pairwise_permission_budget_score": float(best_info["candidate_budget_score"]),
        "pairwise_permission_block_reason": str(best_info["block_reason"]),
        "pairwise_route_permission_block_reason": str(best_info["block_reason"]),
        "pairwise_final_route_permission_pass": 1.0 if routed_away else 0.0,
        "pairwise_value_among_allowed_candidates": float(best_info["value"]) if routed_away else 0.0,
        "pairwise_route_selected_under_permission": 1.0 if routed_away else 0.0,
        "pairwise_anchor_selection_outcome": 1.0 if not routed_away else 0.0,
        "pairwise_routed_away_from_stage1": 1.0 if routed_away else 0.0,
        "pairwise_refine_enabled": 0.0,
        "pairwise_refine_intensity": 0.0,
    }
    return _normalize_gate_np(final_gate), source, metadata


def select_stage1_anchored_pairwise_gate_bank_weights_frontier(
    *,
    x: np.ndarray,
    counts: np.ndarray,
    batches: np.ndarray | None,
    dataset_stats: dict[str, float],
    base_features: dict[str, np.ndarray],
    expert_scores: dict[str, np.ndarray],
    checkpoint_path: str,
    heuristic_gate: np.ndarray,
    selector: RefineMoEHVGSelector,
    device: torch.device | None = None,
    variant: str = "full",
) -> tuple[np.ndarray, str, dict[str, float | str]]:
    components = _point_gate_components_from_runtime(
        dataset_stats=dataset_stats,
        checkpoint_path=checkpoint_path,
        heuristic_gate=heuristic_gate,
        device=device,
    )
    runtime = components["runtime"]
    frontier_packages: dict[str, dict[str, object]] = runtime.get("pairwise_frontier_models", {})  # type: ignore[assignment]
    model_variant = variant if variant in frontier_packages else "full"
    frontier_package = frontier_packages.get(model_variant) or frontier_packages.get("full")
    if not frontier_package:
        return select_stage1_anchored_pairwise_gate_bank_weights_escapecert(
            x=x,
            counts=counts,
            batches=batches,
            dataset_stats=dataset_stats,
            base_features=base_features,
            expert_scores=expert_scores,
            checkpoint_path=checkpoint_path,
            heuristic_gate=heuristic_gate,
            selector=selector,
            device=device,
            variant="full",
        )

    frontier_model: nn.Module = frontier_package["model"]  # type: ignore[assignment]
    frontier_feature_keys = list(frontier_package["feature_keys"])
    frontier_feature_mean = np.asarray(frontier_package["feature_mean"], dtype=np.float32)
    frontier_feature_std = np.asarray(frontier_package["feature_std"], dtype=np.float32)

    candidate_names, candidate_gates, prototype_distances = _build_inference_gate_bank(
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        residual_gate=np.asarray(components["residual_gate"], dtype=np.float64),
        stage1_gate=np.asarray(components["stage1_blended"], dtype=np.float64),
        prototype_gate=np.asarray(components["prototype_gate"], dtype=np.float64),
        prototype_neighbor_gates=np.asarray(components["prototype_neighbor_gates"], dtype=np.float64),
        prototype_neighbor_distances=np.asarray(components["prototype_neighbor_distances"], dtype=np.float64),
        runtime_config=runtime["config"],
    )
    bank_names = tuple(runtime["bank_candidate_names"])
    if device is None:
        device = choose_torch_device()

    runtime_config = GateLearningConfig(**runtime["config"])
    flags = _frontier_variant_flags(model_variant)
    stage1_gate = np.asarray(components["stage1_blended"], dtype=np.float64)
    stage1_idx = int(np.where(candidate_names == "stage1_blended")[0][0])
    anchor_threshold = runtime_config.pairperm_permission_threshold
    frontier_threshold = runtime_config.frontier_accept_threshold

    def build_candidate_features(
        *,
        view_counts: np.ndarray,
        view_batches: np.ndarray | None,
        prepared_context: tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]] | None = None,
    ) -> list[dict[str, float]]:
        if prepared_context is None:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = selector.prepare_context(
                counts=view_counts,
                batches=view_batches,
            )
        else:
            current_x, current_base_features, current_dataset_stats, current_expert_scores = prepared_context
        proxy_context = _prepare_bank_proxy_context(
            x=current_x,
            batches=view_batches,
            random_state=selector.random_state,
        )
        feature_dicts: list[dict[str, float]] = []
        for idx in range(len(candidate_names)):
            gate = candidate_gates[int(idx)]
            no_refine_scores = selector.score_with_context(
                x=current_x,
                base_features=current_base_features,
                dataset_stats=current_dataset_stats,
                expert_scores=current_expert_scores,
                gate=gate,
                apply_refine=False,
            )
            current_top_k = min(selector.top_k, no_refine_scores.size)
            selected_idx = np.argsort(no_refine_scores)[-current_top_k:]
            boundary_size = min(len(no_refine_scores), max(current_top_k, int(1.5 * current_top_k)))
            boundary_idx = np.argsort(no_refine_scores)[-boundary_size:]
            feature_dicts.append(
                _gate_bank_candidate_features(
                    candidate_name=str(candidate_names[int(idx)]),
                    gate=gate,
                    score_vector=no_refine_scores,
                    selected_idx=selected_idx,
                    boundary_idx=boundary_idx,
                    dataset_stats=current_dataset_stats,
                    base_features=current_base_features,
                    x=current_x,
                    proxy_context=proxy_context,
                    heuristic_gate=heuristic_gate,
                    stage1_gate=stage1_gate,
                    prototype_distance=float(prototype_distances[int(idx)]),
                    bank_names=bank_names,
                    random_state=selector.random_state,
                )
            )
        return feature_dicts

    feature_dicts = build_candidate_features(
        view_counts=np.asarray(counts, dtype=np.float64),
        view_batches=None if batches is None else np.asarray(batches),
        prepared_context=(x, base_features, dataset_stats, expert_scores),
    )
    anchor_features = feature_dicts[stage1_idx]
    feature_rows = [
        [
            float(
                _build_stage1_pairwise_feature_dict(
                    candidate_features=feature_dict,
                    anchor_features=anchor_features,
                    candidate_name=str(candidate_names[int(idx)]),
                    bank_names=bank_names,
                )[key]
            )
            for key in frontier_feature_keys
        ]
        for idx, feature_dict in enumerate(feature_dicts)
    ]
    normalized = (np.asarray(feature_rows, dtype=np.float32) - frontier_feature_mean[None, :]) / frontier_feature_std[None, :]
    with torch.no_grad():
        (
            utility,
            risk,
            regret,
            value,
            anchor_escape_logit,
            teacher_support_logit,
            frontier_logit,
            risk_budget,
            consistency_budget,
            frontier_uncertainty,
        ) = frontier_model(
            torch.from_numpy(normalized.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
        )

    utility_np = utility.detach().cpu().numpy().reshape(-1)
    risk_np = risk.detach().cpu().numpy().reshape(-1)
    regret_np = regret.detach().cpu().numpy().reshape(-1)
    value_np = value.detach().cpu().numpy().reshape(-1)
    anchor_escape_logit_np = anchor_escape_logit.detach().cpu().numpy().reshape(-1)
    teacher_support_logit_np = teacher_support_logit.detach().cpu().numpy().reshape(-1)
    frontier_logit_np = frontier_logit.detach().cpu().numpy().reshape(-1)
    risk_budget_np = risk_budget.detach().cpu().numpy().reshape(-1)
    consistency_budget_np = consistency_budget.detach().cpu().numpy().reshape(-1)
    frontier_uncertainty_np = frontier_uncertainty.detach().cpu().numpy().reshape(-1)

    utility_np[stage1_idx] = 0.0
    risk_np[stage1_idx] = 0.0
    regret_np[stage1_idx] = 0.0
    value_np[stage1_idx] = 0.0
    anchor_escape_logit_np[stage1_idx] = 0.0
    teacher_support_logit_np[stage1_idx] = -12.0
    frontier_logit_np[stage1_idx] = -12.0
    risk_budget_np[stage1_idx] = runtime_config.pairwise_safe_risk_budget
    consistency_budget_np[stage1_idx] = runtime_config.pairwise_consistency_threshold
    frontier_uncertainty_np[stage1_idx] = 1.0
    if not flags["use_regret_aux"]:
        regret_np[:] = 0.0

    route_consistency_np = np.asarray(
        [
            1.0 if idx == stage1_idx else float(np.clip(feature_dicts[idx].get("route_view_consistency", 1.0), 0.0, 1.0))
            for idx in range(len(candidate_names))
        ],
        dtype=np.float32,
    )
    candidate_is_stage1_np = np.asarray([1.0 if idx == stage1_idx else 0.0 for idx in range(len(candidate_names))], dtype=np.float32)

    with torch.no_grad():
        group_outputs = _frontier_group_predictions_torch(
            utility_pred=torch.from_numpy(utility_np.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
            risk_pred=torch.from_numpy(risk_np.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
            regret_pred=torch.from_numpy(regret_np.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
            value_pred=torch.from_numpy(value_np.astype(np.float32)).to(device, non_blocking=torch.cuda.is_available()),
            anchor_escape_logit=torch.from_numpy(anchor_escape_logit_np.astype(np.float32)).to(
                device,
                non_blocking=torch.cuda.is_available(),
            ),
            teacher_support_logit=torch.from_numpy(teacher_support_logit_np.astype(np.float32)).to(
                device,
                non_blocking=torch.cuda.is_available(),
            ),
            frontier_logit=torch.from_numpy(frontier_logit_np.astype(np.float32)).to(
                device,
                non_blocking=torch.cuda.is_available(),
            ),
            risk_budget_pred=torch.from_numpy(risk_budget_np.astype(np.float32)).to(
                device,
                non_blocking=torch.cuda.is_available(),
            ),
            consistency_budget_pred=torch.from_numpy(consistency_budget_np.astype(np.float32)).to(
                device,
                non_blocking=torch.cuda.is_available(),
            ),
            frontier_uncertainty_pred=torch.from_numpy(frontier_uncertainty_np.astype(np.float32)).to(
                device,
                non_blocking=torch.cuda.is_available(),
            ),
            route_consistency=torch.from_numpy(route_consistency_np).to(device, non_blocking=torch.cuda.is_available()),
            candidate_is_stage1=torch.from_numpy(candidate_is_stage1_np).to(device, non_blocking=torch.cuda.is_available()),
            config=runtime_config,
            variant=model_variant,
        )

    teacher_prob_np = group_outputs["teacher_set_prob"].detach().cpu().numpy().reshape(-1)
    frontier_prob_np = group_outputs["frontier_prob"].detach().cpu().numpy().reshape(-1)
    frontier_margin_np = group_outputs["frontier_margin_vec"].detach().cpu().numpy().reshape(-1)
    frontier_accept_np = group_outputs["frontier_accept_prob"].detach().cpu().numpy().reshape(-1)
    frontier_risk_reserve_np = group_outputs["frontier_risk_reserve"].detach().cpu().numpy().reshape(-1)
    frontier_uncertainty_vec_np = group_outputs["frontier_uncertainty_vec"].detach().cpu().numpy().reshape(-1)
    anchor_escape_logit_agg = float(group_outputs["anchor_escape_logit"].detach().cpu().item())
    anchor_escape_prob = float(group_outputs["anchor_escape_prob"].detach().cpu().item())
    anchor_escape_calibrated = float(group_outputs["anchor_escape_calibrated"].detach().cpu().item())
    anchor_escape_uncertainty = float(group_outputs["anchor_escape_uncertainty"].detach().cpu().item())
    counterfactual_escape_budget_score = float(group_outputs["counterfactual_escape_budget_score"].detach().cpu().item())
    counterfactual_escape_support = float(group_outputs["counterfactual_escape_support"].detach().cpu().item())
    anchor_escape_counterfactual_margin = float(group_outputs["anchor_escape_counterfactual_margin"].detach().cpu().item())
    agg_risk_budget = float(group_outputs["agg_risk_budget"].detach().cpu().item())
    agg_consistency_budget = float(group_outputs["agg_consistency_budget"].detach().cpu().item())
    teacher_set_mass = float(group_outputs["teacher_set_mass"].detach().cpu().item())
    teacher_topm_mass = float(group_outputs["teacher_topm_mass"].detach().cpu().item())
    allowed_set_total_mass = float(group_outputs["allowed_set_total_mass"].detach().cpu().item())
    allowed_set_entropy = float(group_outputs["allowed_set_entropy"].detach().cpu().item())
    allowed_set_topm_mass = float(group_outputs["allowed_set_topm_mass"].detach().cpu().item())
    allowed_set_best_value = float(group_outputs["allowed_set_best_value"].detach().cpu().item())
    permission_set_consistency = float(group_outputs["permission_set_consistency"].detach().cpu().item())
    frontier_coverage_score = float(group_outputs["frontier_coverage_score"].detach().cpu().item())
    frontier_teacher_agreement = float(group_outputs["frontier_teacher_agreement"].detach().cpu().item())
    frontier_false_release_risk = float(group_outputs["frontier_false_release_risk"].detach().cpu().item())
    frontier_missed_escape_risk = float(group_outputs["frontier_missed_escape_risk"].detach().cpu().item())
    anchor_escape_safe_candidate_count = float(group_outputs["anchor_escape_safe_candidate_count"].detach().cpu().item())

    candidate_infos: list[dict[str, float | str | int]] = []
    for idx in range(len(candidate_names)):
        candidate_name = str(candidate_names[idx])
        route_consistency = float(route_consistency_np[idx])
        fallback_regret_input = float(regret_np[idx]) if flags["use_regret_aux"] else max(float(value_np[idx]), 0.0)
        safe_positive_mask = (
            0.0
            if idx == stage1_idx
            else _pairwise_safe_positive_mask(
                raw_utility_gain=float(utility_np[idx]),
                failure_risk=float(risk_np[idx]),
                fallback_regret_raw=fallback_regret_input,
                route_consistency=route_consistency,
                route_margin_raw=float(value_np[idx]),
                config=runtime_config,
            )
        )
        escape_support = _pairperm_candidate_escape_support(
            safe_utility_gain=float(utility_np[idx]),
            fallback_regret_calibrated=float(regret_np[idx]) if flags["use_regret_aux"] else 0.0,
            failure_risk=float(risk_np[idx]),
            route_consistency=route_consistency,
            route_margin_raw=float(value_np[idx]),
            safe_positive_mask=safe_positive_mask,
            config=runtime_config,
        )
        block_reason = _frontier_block_reason(
            candidate_name=candidate_name,
            anchor_escape_calibrated=anchor_escape_calibrated,
            teacher_set_prob=float(teacher_prob_np[idx]),
            frontier_accept_prob=float(frontier_accept_np[idx]),
            frontier_risk_reserve=float(frontier_risk_reserve_np[idx]),
            anchor_threshold=anchor_threshold,
            frontier_threshold=frontier_threshold,
        )
        final_permission_pass = float(
            idx != stage1_idx
            and anchor_escape_calibrated >= anchor_threshold
            and float(frontier_accept_np[idx]) >= frontier_threshold
        )
        candidate_infos.append(
            {
                "idx": idx,
                "candidate_name": candidate_name,
                "utility": float(utility_np[idx]),
                "risk": float(risk_np[idx]),
                "regret": float(regret_np[idx]) if flags["use_regret_aux"] else 0.0,
                "value": float(value_np[idx]),
                "teacher_logit": float(teacher_support_logit_np[idx]),
                "teacher_prob": float(teacher_prob_np[idx]),
                "frontier_logit": float(frontier_logit_np[idx]),
                "frontier_prob": float(frontier_prob_np[idx]),
                "frontier_margin": float(frontier_margin_np[idx]),
                "frontier_accept_prob": float(frontier_accept_np[idx]),
                "frontier_risk_reserve": float(frontier_risk_reserve_np[idx]),
                "frontier_uncertainty": float(frontier_uncertainty_vec_np[idx]),
                "route_consistency": route_consistency,
                "top1_share": float(np.clip(feature_dicts[idx].get("route_view_win_rate", 1.0 if idx != stage1_idx else 0.0), 0.0, 1.0)),
                "rank_std": 0.0,
                "margin_mean": float(feature_dicts[idx].get("route_view_margin_mean", value_np[idx] if idx != stage1_idx else 0.0)),
                "margin_std": float(feature_dicts[idx].get("route_view_margin_std", 0.0)),
                "safe_positive_mask": safe_positive_mask,
                "escape_support": escape_support,
                "final_permission_pass": final_permission_pass,
                "block_reason": block_reason,
                "route_constraint_score": _pairwise_route_constraint_score(
                    raw_utility_gain=float(utility_np[idx]),
                    failure_risk=float(risk_np[idx]),
                    route_consistency=route_consistency,
                    config=runtime_config,
                ),
            }
        )

    non_stage_infos = [info for info in candidate_infos if int(info["idx"]) != stage1_idx]
    if not non_stage_infos:
        return _normalize_gate_np(stage1_gate.copy()), "frontier_stay:stage1", {
            "anchor_escape_logit": 0.0,
            "anchor_escape_prob": 0.0,
            "anchor_escape_calibrated": 0.0,
            "allowed_set_total_mass": 0.0,
            "pairwise_routed_away_from_stage1": 0.0,
            "bank_selected_is_stage1": 1.0,
        }

    allowed_infos = [info for info in non_stage_infos if float(info["final_permission_pass"]) > 0.5]
    if allowed_infos:
        best_info = max(allowed_infos, key=lambda current: float(current["value"]))
        routed_away = True
    else:
        best_info = max(
            non_stage_infos,
            key=lambda current: (
                float(current["frontier_accept_prob"]),
                float(current["value"]),
            ),
        )
        routed_away = False

    best_idx = int(best_info["idx"])
    if routed_away:
        final_gate = candidate_gates[best_idx].copy()
        source = f"frontier_route:{best_info['candidate_name']}"
    else:
        final_gate = stage1_gate.copy()
        source = "frontier_stay:stage1"

    final_gate = _apply_gate_safety_projection(
        gate=final_gate,
        dataset_stats=dataset_stats,
        heuristic_gate=heuristic_gate,
        safety_floor_strength=float(runtime["config"]["safety_floor_strength"]),
    )

    transfer_risk = float(
        np.clip(
            0.48 * float(best_info["risk"])
            + 0.22 * (1.0 - float(best_info["frontier_risk_reserve"]))
            + 0.15 * (1.0 - float(best_info["frontier_accept_prob"]))
            + 0.15 * frontier_false_release_risk,
            0.0,
            1.0,
        )
    )
    reliability = float(
        np.clip(
            0.24 * (1.0 - transfer_risk)
            + 0.16 * float(best_info["frontier_accept_prob"])
            + 0.15 * float(best_info["teacher_prob"])
            + 0.15 * float(best_info["route_consistency"])
            + 0.15 * anchor_escape_calibrated
            + 0.10 * frontier_teacher_agreement
            + 0.05 * max(float(best_info["value"]), 0.0),
            runtime_config.bank_min_reliability,
            0.99,
        )
    )

    metadata: dict[str, float | str] = {
        "bank_reliability": reliability,
        "bank_consensus": float(best_info["top1_share"]),
        "bank_uncertainty": float(np.clip(1.0 - float(best_info["route_consistency"]), 0.0, 1.0)),
        "bank_transfer_risk": transfer_risk,
        "bank_blend_weight": 1.0 if routed_away else 0.0,
        "bank_logit_margin": float(best_info["value"]),
        "bank_view_top1_share": float(best_info["top1_share"]),
        "bank_view_rank_std": float(best_info["rank_std"]),
        "bank_view_margin_mean": float(best_info["margin_mean"]),
        "bank_view_margin_std": float(best_info["margin_std"]),
        "bank_structure_support": float(best_info["route_constraint_score"]),
        "bank_refine_support": float(best_info["route_consistency"]),
        "bank_boundary_batch_risk": float(np.clip(transfer_risk, 0.0, 1.0)),
        "bank_selected_is_stage1": 0.0 if routed_away else 1.0,
        "anchor_escape_logit": anchor_escape_logit_agg,
        "anchor_escape_prob": anchor_escape_prob,
        "anchor_escape_calibrated": anchor_escape_calibrated,
        "anchor_escape_counterfactual_margin": anchor_escape_counterfactual_margin,
        "anchor_escape_soft_oracle_gain": allowed_set_best_value,
        "anchor_escape_safe_candidate_count": anchor_escape_safe_candidate_count,
        "anchor_escape_topm_mass": teacher_topm_mass,
        "anchor_escape_uncertainty": anchor_escape_uncertainty,
        "teacher_set_score": float(best_info["teacher_prob"]),
        "teacher_set_prob": float(best_info["teacher_prob"]),
        "teacher_set_mass": teacher_set_mass,
        "teacher_topm_mass": teacher_topm_mass,
        "frontier_logit": float(best_info["frontier_logit"]),
        "frontier_prob": float(best_info["frontier_prob"]),
        "frontier_margin": float(best_info["frontier_margin"]),
        "frontier_accept_prob": float(best_info["frontier_accept_prob"]),
        "frontier_uncertainty": float(best_info["frontier_uncertainty"]),
        "frontier_risk_reserve": float(best_info["frontier_risk_reserve"]),
        "frontier_coverage_score": frontier_coverage_score,
        "frontier_teacher_agreement": frontier_teacher_agreement,
        "frontier_false_release_risk": frontier_false_release_risk,
        "frontier_missed_escape_risk": frontier_missed_escape_risk,
        "frontier_reject_reason": str(best_info["block_reason"]),
        "candidate_admissibility_logit": float(best_info["frontier_logit"]),
        "candidate_admissibility_prob": float(best_info["frontier_prob"]),
        "candidate_admissibility_calibrated": float(best_info["frontier_accept_prob"]),
        "candidate_admissibility_margin": float(best_info["frontier_accept_prob"]) - frontier_threshold,
        "allowed_set_total_mass": allowed_set_total_mass,
        "allowed_set_entropy": allowed_set_entropy,
        "allowed_set_best_value": allowed_set_best_value,
        "allowed_set_selected_candidate": str(best_info["candidate_name"]) if routed_away else "stage1_blended",
        "allowed_frontier_set_mass": allowed_set_total_mass,
        "allowed_frontier_entropy": allowed_set_entropy,
        "allowed_frontier_best_value": allowed_set_best_value,
        "allowed_frontier_selected_candidate": str(best_info["candidate_name"]) if routed_away else "stage1_blended",
        "permission_set_consistency": permission_set_consistency,
        "counterfactual_escape_support": counterfactual_escape_support,
        "counterfactual_escape_budget_score": counterfactual_escape_budget_score,
        "pairwise_utility_gain": float(best_info["utility"]),
        "pairwise_safe_utility_gain": float(best_info["utility"]),
        "pairwise_failure_risk": float(best_info["risk"]),
        "pairwise_fallback_regret": float(best_info["regret"]),
        "pairwise_fallback_regret_calibrated": float(best_info["regret"]),
        "pairwise_route_margin": float(best_info["value"]),
        "pairwise_route_margin_calibrated": float(best_info["value"]),
        "pairwise_route_consistency": float(best_info["route_consistency"]),
        "pairwise_safe_positive_mask": float(best_info["safe_positive_mask"]),
        "pairwise_regret_supervision_weight": 1.0 if flags["use_regret_aux"] else float(runtime_config.pairwise_min_regret_weight),
        "pairwise_route_constraint_score": float(best_info["route_constraint_score"]),
        "pairwise_final_route_score": float(best_info["value"]),
        "pairwise_route_permission_logit": float(best_info["frontier_logit"]),
        "pairwise_route_permission_prob": float(best_info["frontier_accept_prob"]),
        "pairwise_route_permission_target": float(best_info["frontier_accept_prob"]),
        "pairwise_route_permission_calibrated": float(best_info["frontier_accept_prob"]),
        "pairwise_route_permission_margin": float(best_info["frontier_accept_prob"]) - frontier_threshold,
        "pairwise_route_permission_confidence": float(np.clip(abs(2.0 * float(best_info["frontier_accept_prob"]) - 1.0), 0.0, 1.0)),
        "pairwise_anchor_escape_pressure": anchor_escape_calibrated,
        "pairwise_anchor_routability": teacher_topm_mass,
        "pairwise_anchor_fragility": anchor_escape_uncertainty,
        "pairwise_adaptive_risk_budget": agg_risk_budget,
        "pairwise_adaptive_consistency_budget": agg_consistency_budget,
        "pairwise_permission_budget_score": float(best_info["frontier_risk_reserve"]),
        "pairwise_permission_block_reason": str(best_info["block_reason"]),
        "pairwise_route_permission_block_reason": str(best_info["block_reason"]),
        "pairwise_final_route_permission_pass": 1.0 if routed_away else 0.0,
        "pairwise_value_among_allowed_candidates": float(best_info["value"]) if routed_away else 0.0,
        "pairwise_route_selected_under_permission": 1.0 if routed_away else 0.0,
        "pairwise_anchor_selection_outcome": 1.0 if not routed_away else 0.0,
        "pairwise_routed_away_from_stage1": 1.0 if routed_away else 0.0,
        "pairwise_refine_enabled": 0.0,
        "pairwise_refine_intensity": 0.0,
    }
    return _normalize_gate_np(final_gate), source, metadata


def _load_runtime_checkpoint(
    *,
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, object]:
    cache_key = (str(Path(checkpoint_path).resolve()), str(device))
    if cache_key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[cache_key]

    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_mean = np.asarray(bundle["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(bundle["feature_std"], dtype=np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    bank_feature_mean = np.asarray(bundle.get("bank_feature_mean", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    bank_feature_std = np.asarray(bundle.get("bank_feature_std", np.ones((0,), dtype=np.float32)), dtype=np.float32)
    if bank_feature_std.size:
        bank_feature_std[bank_feature_std < 1e-6] = 1.0
    curr_feature_mean = np.asarray(bundle.get("curr_feature_mean", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    curr_feature_std = np.asarray(bundle.get("curr_feature_std", np.ones((0,), dtype=np.float32)), dtype=np.float32)
    if curr_feature_std.size:
        curr_feature_std[curr_feature_std < 1e-6] = 1.0
    pairwise_feature_mean = np.asarray(
        bundle.get("pairwise_feature_mean", np.zeros((0,), dtype=np.float32)),
        dtype=np.float32,
    )
    pairwise_feature_std = np.asarray(
        bundle.get("pairwise_feature_std", np.ones((0,), dtype=np.float32)),
        dtype=np.float32,
    )
    if pairwise_feature_std.size:
        pairwise_feature_std[pairwise_feature_std < 1e-6] = 1.0
    pairwise_calibration_bundle = dict(bundle.get("pairwise_calibration_models", {}))
    if not pairwise_calibration_bundle and "pairwise_cal_model_state" in bundle:
        pairwise_calibration_bundle = {
            "full": {
                "model_state": bundle["pairwise_cal_model_state"],
                "feature_keys": bundle.get("pairwise_cal_feature_keys", []),
                "feature_mean": bundle.get("pairwise_cal_feature_mean", np.zeros((0,), dtype=np.float32)),
                "feature_std": bundle.get("pairwise_cal_feature_std", np.ones((0,), dtype=np.float32)),
                "hidden_dim": bundle.get("pairwise_cal_hidden_dim", 48),
                "dropout": bundle.get("pairwise_cal_dropout", 0.05),
                "best_val_loss": bundle.get("pairwise_cal_best_val_loss", 0.0),
            }
        }
    pairwise_permission_bundle = dict(bundle.get("pairwise_permission_models", {}))
    pairwise_escapecert_bundle = dict(bundle.get("pairwise_escapecert_models", {}))
    pairwise_frontier_bundle = dict(bundle.get("pairwise_frontier_models", {}))

    base_model = LearnableDatasetGate(
        input_dim=len(bundle["feature_keys"]),
        output_dim=len(bundle["expert_names"]),
        gate_type=bundle.get("gate_type", "mlp"),
        hidden_dim=int(bundle.get("hidden_dim", 32)),
        dropout=float(bundle.get("dropout", 0.10)),
    )
    base_model.load_state_dict(bundle["model_state"])
    base_model = base_model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model: nn.Module = nn.DataParallel(base_model)
    else:
        model = base_model
    model.eval()

    bank_model: nn.Module | None = None
    if "bank_model_state" in bundle and len(bundle.get("bank_feature_keys", [])):
        bank_base_model = GateBankScorer(
            input_dim=len(bundle["bank_feature_keys"]),
            hidden_dim=int(bundle.get("bank_hidden_dim", 32)),
            dropout=float(bundle.get("bank_dropout", 0.05)),
        )
        bank_base_model.load_state_dict(bundle["bank_model_state"])
        bank_base_model = bank_base_model.to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            bank_model = nn.DataParallel(bank_base_model)
        else:
            bank_model = bank_base_model
        bank_model.eval()

    curr_model: nn.Module | None = None
    if "curr_model_state" in bundle and len(bundle.get("curr_feature_keys", [])):
        curr_base_model = CounterfactualTriFactorPolicy(
            input_dim=len(bundle["curr_feature_keys"]),
            hidden_dim=int(bundle.get("curr_hidden_dim", 40)),
            dropout=float(bundle.get("curr_dropout", 0.05)),
        )
        curr_base_model.load_state_dict(bundle["curr_model_state"])
        curr_base_model = curr_base_model.to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            curr_model = nn.DataParallel(curr_base_model)
        else:
            curr_model = curr_base_model
        curr_model.eval()

    pairwise_model: nn.Module | None = None
    if "pairwise_model_state" in bundle and len(bundle.get("pairwise_feature_keys", [])):
        pairwise_base_model = Stage1AnchoredPairwiseRouter(
            input_dim=len(bundle["pairwise_feature_keys"]),
            hidden_dim=int(bundle.get("pairwise_hidden_dim", 48)),
            dropout=float(bundle.get("pairwise_dropout", 0.05)),
        )
        pairwise_base_model.load_state_dict(bundle["pairwise_model_state"])
        pairwise_base_model = pairwise_base_model.to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            pairwise_model = nn.DataParallel(pairwise_base_model)
        else:
            pairwise_model = pairwise_base_model
        pairwise_model.eval()

    pairwise_cal_models: dict[str, dict[str, object]] = {}
    for variant, variant_bundle in pairwise_calibration_bundle.items():
        variant_feature_mean = np.asarray(
            variant_bundle.get("feature_mean", np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        )
        variant_feature_std = np.asarray(
            variant_bundle.get("feature_std", np.ones((0,), dtype=np.float32)),
            dtype=np.float32,
        )
        if variant_feature_std.size:
            variant_feature_std[variant_feature_std < 1e-6] = 1.0
        variant_feature_keys = list(variant_bundle.get("feature_keys", []))
        model_state = variant_bundle.get("model_state")
        if model_state is None or not variant_feature_keys:
            continue
        pairwise_cal_base_model = Stage1AnchoredPairwiseRouter(
            input_dim=len(variant_feature_keys),
            hidden_dim=int(variant_bundle.get("hidden_dim", 48)),
            dropout=float(variant_bundle.get("dropout", 0.05)),
        )
        pairwise_cal_base_model.load_state_dict(model_state)
        pairwise_cal_base_model = pairwise_cal_base_model.to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            pairwise_cal_model: nn.Module = nn.DataParallel(pairwise_cal_base_model)
        else:
            pairwise_cal_model = pairwise_cal_base_model
        pairwise_cal_model.eval()
        pairwise_cal_models[str(variant)] = {
            "model": pairwise_cal_model,
            "feature_keys": variant_feature_keys,
            "feature_mean": variant_feature_mean,
            "feature_std": variant_feature_std,
        }

    pairwise_permission_models: dict[str, dict[str, object]] = {}
    for variant, variant_bundle in pairwise_permission_bundle.items():
        variant_feature_mean = np.asarray(
            variant_bundle.get("feature_mean", np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        )
        variant_feature_std = np.asarray(
            variant_bundle.get("feature_std", np.ones((0,), dtype=np.float32)),
            dtype=np.float32,
        )
        if variant_feature_std.size:
            variant_feature_std[variant_feature_std < 1e-6] = 1.0
        variant_feature_keys = list(variant_bundle.get("feature_keys", []))
        model_state = variant_bundle.get("model_state")
        if model_state is None or not variant_feature_keys:
            continue
        pairperm_base_model = PermissionedStage1AnchoredPairwiseRouter(
            input_dim=len(variant_feature_keys),
            hidden_dim=int(variant_bundle.get("hidden_dim", 56)),
            dropout=float(variant_bundle.get("dropout", 0.05)),
            risk_budget_floor=float(bundle.get("config", {}).get("pairperm_risk_budget_floor", 0.10)),
            risk_budget_ceiling=float(bundle.get("config", {}).get("pairperm_risk_budget_ceiling", 0.52)),
            consistency_budget_floor=float(bundle.get("config", {}).get("pairperm_consistency_budget_floor", 0.24)),
            consistency_budget_ceiling=float(bundle.get("config", {}).get("pairperm_consistency_budget_ceiling", 0.84)),
        )
        pairperm_base_model.load_state_dict(model_state)
        pairperm_base_model = pairperm_base_model.to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            pairperm_model: nn.Module = nn.DataParallel(pairperm_base_model)
        else:
            pairperm_model = pairperm_base_model
        pairperm_model.eval()
        pairwise_permission_models[str(variant)] = {
            "model": pairperm_model,
            "feature_keys": variant_feature_keys,
            "feature_mean": variant_feature_mean,
            "feature_std": variant_feature_std,
        }

    pairwise_escapecert_models: dict[str, dict[str, object]] = {}
    for variant, variant_bundle in pairwise_escapecert_bundle.items():
        variant_feature_mean = np.asarray(
            variant_bundle.get("feature_mean", np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        )
        variant_feature_std = np.asarray(
            variant_bundle.get("feature_std", np.ones((0,), dtype=np.float32)),
            dtype=np.float32,
        )
        if variant_feature_std.size:
            variant_feature_std[variant_feature_std < 1e-6] = 1.0
        variant_feature_keys = list(variant_bundle.get("feature_keys", []))
        model_state = variant_bundle.get("model_state")
        if model_state is None or not variant_feature_keys:
            continue
        escapecert_base_model = EscapeCertStage1AnchoredPairwiseRouter(
            input_dim=len(variant_feature_keys),
            hidden_dim=int(variant_bundle.get("hidden_dim", 64)),
            dropout=float(variant_bundle.get("dropout", 0.05)),
            risk_budget_floor=float(bundle.get("config", {}).get("pairperm_risk_budget_floor", 0.10)),
            risk_budget_ceiling=float(bundle.get("config", {}).get("pairperm_risk_budget_ceiling", 0.52)),
            consistency_budget_floor=float(bundle.get("config", {}).get("pairperm_consistency_budget_floor", 0.24)),
            consistency_budget_ceiling=float(bundle.get("config", {}).get("pairperm_consistency_budget_ceiling", 0.84)),
        )
        escapecert_base_model.load_state_dict(model_state)
        escapecert_base_model = escapecert_base_model.to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            escapecert_model: nn.Module = nn.DataParallel(escapecert_base_model)
        else:
            escapecert_model = escapecert_base_model
        escapecert_model.eval()
        pairwise_escapecert_models[str(variant)] = {
            "model": escapecert_model,
            "feature_keys": variant_feature_keys,
            "feature_mean": variant_feature_mean,
            "feature_std": variant_feature_std,
        }

    pairwise_frontier_models: dict[str, dict[str, object]] = {}
    for variant, variant_bundle in pairwise_frontier_bundle.items():
        variant_feature_mean = np.asarray(
            variant_bundle.get("feature_mean", np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        )
        variant_feature_std = np.asarray(
            variant_bundle.get("feature_std", np.ones((0,), dtype=np.float32)),
            dtype=np.float32,
        )
        if variant_feature_std.size:
            variant_feature_std[variant_feature_std < 1e-6] = 1.0
        variant_feature_keys = list(variant_bundle.get("feature_keys", []))
        model_state = variant_bundle.get("model_state")
        if model_state is None or not variant_feature_keys:
            continue
        frontier_base_model = FrontierStage1AnchoredPairwiseRouter(
            input_dim=len(variant_feature_keys),
            hidden_dim=int(variant_bundle.get("hidden_dim", 72)),
            dropout=float(variant_bundle.get("dropout", 0.05)),
            risk_budget_floor=float(bundle.get("config", {}).get("pairperm_risk_budget_floor", 0.10)),
            risk_budget_ceiling=float(bundle.get("config", {}).get("pairperm_risk_budget_ceiling", 0.52)),
            consistency_budget_floor=float(bundle.get("config", {}).get("pairperm_consistency_budget_floor", 0.24)),
            consistency_budget_ceiling=float(bundle.get("config", {}).get("pairperm_consistency_budget_ceiling", 0.84)),
        )
        frontier_base_model.load_state_dict(model_state)
        frontier_base_model = frontier_base_model.to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            frontier_model: nn.Module = nn.DataParallel(frontier_base_model)
        else:
            frontier_model = frontier_base_model
        frontier_model.eval()
        pairwise_frontier_models[str(variant)] = {
            "model": frontier_model,
            "feature_keys": variant_feature_keys,
            "feature_mean": variant_feature_mean,
            "feature_std": variant_feature_std,
        }

    runtime = {
        "model": model,
        "feature_keys": list(bundle["feature_keys"]),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "expert_names": list(bundle["expert_names"]),
        "prototype_features": np.asarray(bundle["prototype_features"], dtype=np.float32),
        "prototype_gates": np.asarray(bundle["prototype_gates"], dtype=np.float32),
        "config": bundle["config"],
        "bank_model": bank_model,
        "bank_feature_keys": list(bundle.get("bank_feature_keys", [])),
        "bank_feature_mean": bank_feature_mean,
        "bank_feature_std": bank_feature_std,
        "bank_candidate_names": list(bundle.get("bank_candidate_names", [])),
        "curr_model": curr_model,
        "curr_feature_keys": list(bundle.get("curr_feature_keys", [])),
        "curr_feature_mean": curr_feature_mean,
        "curr_feature_std": curr_feature_std,
        "pairwise_model": pairwise_model,
        "pairwise_feature_keys": list(bundle.get("pairwise_feature_keys", [])),
        "pairwise_feature_mean": pairwise_feature_mean,
        "pairwise_feature_std": pairwise_feature_std,
        "pairwise_cal_models": pairwise_cal_models,
        "pairwise_permission_models": pairwise_permission_models,
        "pairwise_escapecert_models": pairwise_escapecert_models,
        "pairwise_frontier_models": pairwise_frontier_models,
    }
    _RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _evaluate_gate_loss(
    *,
    model: nn.Module,
    x_val: np.ndarray,
    y_val: np.ndarray,
    pref_val: np.ndarray,
    anti_val: np.ndarray,
    heuristic_val: np.ndarray,
    baseline_val: np.ndarray,
    pair_margin_val: np.ndarray,
    ranking_scale_val: np.ndarray,
    baseline_keep_val: np.ndarray,
    w_val: np.ndarray,
    device: torch.device,
    config: GateLearningConfig,
) -> float:
    if len(x_val) == 0:
        return 0.0

    model.eval()
    with torch.no_grad():
        batch_x = torch.from_numpy(x_val).to(device, non_blocking=torch.cuda.is_available())
        batch_y = torch.from_numpy(y_val).to(device, non_blocking=torch.cuda.is_available())
        batch_pref = torch.from_numpy(pref_val).to(device, non_blocking=torch.cuda.is_available())
        batch_anti = torch.from_numpy(anti_val).to(device, non_blocking=torch.cuda.is_available())
        batch_heuristic = torch.from_numpy(heuristic_val).to(device, non_blocking=torch.cuda.is_available())
        batch_baseline = torch.from_numpy(baseline_val).to(device, non_blocking=torch.cuda.is_available())
        batch_pair_margin = torch.from_numpy(pair_margin_val).to(device, non_blocking=torch.cuda.is_available())
        batch_ranking_scale = torch.from_numpy(ranking_scale_val).to(device, non_blocking=torch.cuda.is_available())
        batch_baseline_keep = torch.from_numpy(baseline_keep_val).to(device, non_blocking=torch.cuda.is_available())
        batch_w = torch.from_numpy(w_val).to(device, non_blocking=torch.cuda.is_available())

        delta_logits = model(batch_x)
        prob = _compose_gate_from_delta_torch(
            delta_logits=delta_logits,
            heuristic_gate=batch_heuristic,
            residual_scale=config.residual_scale,
        )
        loss = _gate_training_objective(
            prob=prob,
            target=batch_y,
            preferred=batch_pref,
            anti=batch_anti,
            heuristic=batch_heuristic,
            baseline=batch_baseline,
            weights=batch_w,
            preference_weight=config.preference_weight,
            anchor_weight=config.anchor_weight,
            baseline_ce_weight=config.baseline_ce_weight,
            pair_margin=batch_pair_margin,
            ranking_scale=batch_ranking_scale,
            baseline_keep=batch_baseline_keep,
            entropy_bonus=config.entropy_bonus,
        )
    return float(loss.detach().cpu())


def _gate_training_objective(
    *,
    prob: torch.Tensor,
    target: torch.Tensor,
    preferred: torch.Tensor,
    anti: torch.Tensor,
    heuristic: torch.Tensor,
    baseline: torch.Tensor,
    weights: torch.Tensor,
    preference_weight: float,
    anchor_weight: float,
    baseline_ce_weight: float,
    pair_margin: torch.Tensor,
    ranking_scale: torch.Tensor,
    baseline_keep: torch.Tensor,
    entropy_bonus: float,
) -> torch.Tensor:
    prob = torch.clamp(prob, min=1e-8)
    log_prob = torch.log(prob)
    heuristic = torch.clamp(heuristic, min=1e-8)
    log_heuristic = torch.log(heuristic)

    supervised = -(target * log_prob).sum(dim=1)
    pref_sim = F.cosine_similarity(prob, preferred, dim=1)
    anti_sim = F.cosine_similarity(prob, anti, dim=1)
    preference = F.softplus(-(pref_sim - anti_sim - pair_margin))
    anchor = (prob * (log_prob - log_heuristic)).sum(dim=1)
    baseline_ce = -(baseline * log_prob).sum(dim=1)
    entropy = -(prob * log_prob).sum(dim=1)

    total = (
        supervised
        + preference_weight * ranking_scale * preference
        + anchor_weight * anchor
        + baseline_ce_weight * baseline_keep * baseline_ce
    )
    return (total * weights).mean() - entropy_bonus * entropy.mean()


def _adjust_reward_from_metrics(
    *,
    metrics: dict[str, float],
    reward: float,
    ari_floor: float,
    nmi_floor: float,
    structure_floor: float,
    baseline_metrics: dict[str, float],
    batch_tradeoff_pressure: float,
    floor_weight: float,
    tradeoff_penalty_weight: float,
) -> tuple[float, float, float, float, float]:
    ari_gap = max(0.0, ari_floor - float(metrics["ari"]))
    nmi_gap = max(0.0, nmi_floor - float(metrics["nmi"]))
    structure_gap = max(0.0, structure_floor - structure_preservation_score(metrics))
    mixing_gain = max(0.0, float(metrics.get("batch_mixing", 0.0)) - float(baseline_metrics.get("batch_mixing", 0.0)))
    tradeoff_penalty = batch_tradeoff_pressure * mixing_gain * (
        1.25 * ari_gap + 1.05 * nmi_gap + 0.85 * structure_gap
    )
    adjusted_reward = (
        reward
        - floor_weight * (1.60 * ari_gap + 1.25 * nmi_gap + 0.95 * structure_gap)
        - tradeoff_penalty_weight * tradeoff_penalty
    )
    return adjusted_reward, ari_gap, nmi_gap, structure_gap, tradeoff_penalty


def _compose_gate_from_delta_torch(
    *,
    delta_logits: torch.Tensor,
    heuristic_gate: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    heuristic_gate = torch.clamp(heuristic_gate, min=1e-8)
    heuristic_logits = torch.log(heuristic_gate)
    return torch.softmax(heuristic_logits + residual_scale * delta_logits, dim=1)


def _baseline_metric_floors(
    candidate_records: list[dict[str, object]],
    *,
    floor_ratio: float,
    ari_margin: float,
    nmi_margin: float,
    structure_margin: float,
    baseline_names: set[str] | None = None,
) -> tuple[dict[str, object], float, float, float]:
    if baseline_names is None:
        baseline_names = {"uniform", "heuristic_blend"}
    baseline_records = [record for record in candidate_records if str(record["candidate_name"]) in baseline_names]
    if not baseline_records:
        baseline_records = list(candidate_records)
    baseline_record = max(baseline_records, key=lambda record: float(record["reward"]))
    baseline_metrics = baseline_record["metrics"]  # type: ignore[assignment]
    baseline_structure = structure_preservation_score(baseline_metrics)
    baseline_ari = float(baseline_metrics["ari"])
    baseline_nmi = float(baseline_metrics["nmi"])
    ari_floor = max(floor_ratio * baseline_ari, baseline_ari - ari_margin)
    nmi_floor = max(floor_ratio * baseline_nmi, baseline_nmi - nmi_margin)
    structure_floor = max(floor_ratio * baseline_structure, baseline_structure - structure_margin)
    return baseline_record, ari_floor, nmi_floor, structure_floor


def _batch_tradeoff_pressure(dataset_stats: dict[str, float]) -> float:
    batch_strength = float(dataset_stats.get("batch_strength", 0.0))
    trajectory_strength = float(dataset_stats.get("trajectory_strength", 0.0))
    cluster_strength = float(dataset_stats.get("cluster_strength", 0.0))
    return float(1.0 + 1.40 * batch_strength + 0.45 * trajectory_strength + 0.25 * cluster_strength)


def _blend_heuristic_gate(routed_gate: np.ndarray) -> np.ndarray:
    uniform = np.ones(len(EXPERT_NAMES), dtype=np.float64) / len(EXPERT_NAMES)
    return _normalize_gate_np(0.40 * routed_gate + 0.60 * uniform)


def _prototype_memory_gate(
    *,
    normalized_feature: np.ndarray,
    prototype_features: np.ndarray,
    prototype_gates: np.ndarray,
    k: int,
) -> tuple[np.ndarray, float]:
    if len(prototype_features) == 0:
        return np.ones(len(EXPERT_NAMES), dtype=np.float64) / len(EXPERT_NAMES), 10.0

    distances = np.linalg.norm(prototype_features - normalized_feature[None, :], axis=1)
    order = np.argsort(distances)
    local_radius = float(distances[order[0]] + 1.25)
    local_idx = order[distances[order] <= local_radius]
    k = min(k, max(1, len(local_idx)))
    idx = local_idx[:k]
    local_dist = distances[idx]
    weights = np.exp(-2.00 * local_dist)
    weights = weights / np.maximum(weights.sum(), 1e-8)
    gate = (weights[:, None] * prototype_gates[idx]).sum(axis=0)
    return gate.astype(np.float64), float(local_dist.mean())


def _prototype_memory_bank(
    *,
    normalized_feature: np.ndarray,
    prototype_features: np.ndarray,
    prototype_gates: np.ndarray,
    k: int,
    candidate_count: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if len(prototype_features) == 0:
        uniform = np.ones(len(EXPERT_NAMES), dtype=np.float64) / len(EXPERT_NAMES)
        neighbor_gates = np.repeat(uniform[None, :], candidate_count, axis=0)
        neighbor_distances = np.full(candidate_count, 10.0, dtype=np.float64)
        return uniform, 10.0, neighbor_gates, neighbor_distances

    distances = np.linalg.norm(prototype_features - normalized_feature[None, :], axis=1)
    order = np.argsort(distances)
    local_radius = float(distances[order[0]] + 1.25)
    local_idx = order[distances[order] <= local_radius]
    k = min(k, max(1, len(local_idx)))
    avg_idx = local_idx[:k]
    avg_dist = distances[avg_idx]
    avg_weights = np.exp(-2.00 * avg_dist)
    avg_weights = avg_weights / np.maximum(avg_weights.sum(), 1e-8)
    avg_gate = (avg_weights[:, None] * prototype_gates[avg_idx]).sum(axis=0)

    candidate_count = max(candidate_count, 1)
    neighbor_idx = order[: min(candidate_count, len(order))]
    neighbor_gates = prototype_gates[neighbor_idx].astype(np.float64)
    neighbor_distances = distances[neighbor_idx].astype(np.float64)
    if len(neighbor_idx) < candidate_count:
        fill_count = candidate_count - len(neighbor_idx)
        fill_gates = np.repeat(avg_gate[None, :], fill_count, axis=0)
        fill_distances = np.full(fill_count, float(avg_dist.mean() if len(avg_dist) else 10.0), dtype=np.float64)
        neighbor_gates = np.vstack([neighbor_gates, fill_gates])
        neighbor_distances = np.concatenate([neighbor_distances, fill_distances])
    return avg_gate.astype(np.float64), float(avg_dist.mean()), neighbor_gates, neighbor_distances


def _build_inference_gate_bank(
    *,
    dataset_stats: dict[str, float],
    heuristic_gate: np.ndarray,
    residual_gate: np.ndarray,
    stage1_gate: np.ndarray,
    prototype_gate: np.ndarray,
    prototype_neighbor_gates: np.ndarray,
    prototype_neighbor_distances: np.ndarray,
    runtime_config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_names = gate_bank_candidate_names(
        prototype_candidates=int(runtime_config.get("bank_prototype_candidates", 3)),
    )
    candidate_gates: list[np.ndarray] = []
    candidate_distances: list[float] = []
    heuristic_gate = _normalize_gate_np(heuristic_gate)

    def add(name: str, gate: np.ndarray, prototype_distance: float) -> None:
        projected = _apply_gate_safety_projection(
            gate=_normalize_gate_np(gate),
            dataset_stats=dataset_stats,
            heuristic_gate=heuristic_gate,
            safety_floor_strength=float(runtime_config.get("safety_floor_strength", 0.18)),
        )
        candidate_gates.append(projected.astype(np.float64))
        candidate_distances.append(float(prototype_distance))

    add("heuristic", heuristic_gate, float(np.mean(prototype_neighbor_distances)))
    add("stage1_residual", residual_gate, float(np.mean(prototype_neighbor_distances)))
    add("stage1_blended", stage1_gate, float(np.mean(prototype_neighbor_distances)))
    add("prototype_mean", prototype_gate, float(np.mean(prototype_neighbor_distances)))

    for idx in range(int(runtime_config.get("bank_prototype_candidates", 3))):
        add(
            f"prototype_{idx + 1}",
            prototype_neighbor_gates[idx],
            float(prototype_neighbor_distances[idx]),
        )

    boost_scale = float(runtime_config.get("bank_boost_scale", 0.58))
    for expert_name in BANK_BOOST_EXPERTS:
        one_hot = np.zeros(len(EXPERT_NAMES), dtype=np.float64)
        one_hot[EXPERT_NAMES.index(expert_name)] = 1.0
        add(
            f"boosted_{expert_name}",
            boost_scale * one_hot + (1.0 - boost_scale) * stage1_gate,
            float(np.mean(prototype_neighbor_distances)),
        )

    return (
        np.asarray(candidate_names, dtype=object),
        _normalize_gate_matrix_np(np.stack(candidate_gates, axis=0)),
        np.asarray(candidate_distances, dtype=np.float64),
    )


def _apply_gate_safety_projection(
    *,
    gate: np.ndarray,
    dataset_stats: dict[str, float],
    heuristic_gate: np.ndarray,
    safety_floor_strength: float,
) -> np.ndarray:
    gate = _normalize_gate_np(gate)
    heuristic_gate = _normalize_gate_np(heuristic_gate)

    batch_strength = float(dataset_stats.get("batch_strength", 0.0))
    cluster_strength = float(dataset_stats.get("cluster_strength", 0.0))
    trajectory_strength = float(dataset_stats.get("trajectory_strength", 0.0))

    structure_floor = 0.05 + safety_floor_strength * max(
        cluster_strength,
        trajectory_strength,
    ) + 0.04 * batch_strength
    structure_floor = float(np.clip(structure_floor, 0.08, 0.24))
    batch_floor = 0.02 + safety_floor_strength * (0.30 * batch_strength + 0.10 * trajectory_strength)
    batch_floor = float(np.clip(batch_floor, 0.02, 0.10))
    structure_strength = 0.5 * (cluster_strength + trajectory_strength) + 0.10
    share_denom = cluster_strength + trajectory_strength + structure_strength + 0.30
    cluster_share = (cluster_strength + 0.10) / share_denom
    trajectory_share = (trajectory_strength + 0.10) / share_denom
    structure_share = structure_strength / share_denom

    rare_bonus = max(0.18 - float(dataset_stats.get("rare_fraction", 0.20)), 0.0)
    rare_floor = float(np.clip(0.02 + 0.25 * rare_bonus, 0.02, 0.08))
    general_floor = 0.04

    floor = np.zeros(len(EXPERT_NAMES), dtype=np.float64)
    floor[EXPERT_NAMES.index("cluster")] = structure_floor * cluster_share
    floor[EXPERT_NAMES.index("trajectory")] = structure_floor * trajectory_share
    floor[EXPERT_NAMES.index("batch_robust")] = batch_floor
    floor[EXPERT_NAMES.index("structure")] = structure_floor * structure_share
    floor[EXPERT_NAMES.index("rare")] = rare_floor
    floor[EXPERT_NAMES.index("general")] = general_floor

    adjusted = np.maximum(gate, floor)
    excess = adjusted.sum() - 1.0
    if excess > 0:
        free_priority = np.maximum(adjusted - floor, 0.0)
        if free_priority.sum() < 1e-8:
            free_priority = np.maximum(heuristic_gate - floor, 0.0)
        if free_priority.sum() < 1e-8:
            free_priority = np.ones_like(free_priority)
        adjusted -= excess * free_priority / free_priority.sum()
        adjusted = np.maximum(adjusted, floor)
    return _normalize_gate_np(adjusted)


def _normalize_gate_np(gate: np.ndarray) -> np.ndarray:
    gate = np.asarray(gate, dtype=np.float64)
    gate = np.clip(gate, 1e-8, None)
    return gate / gate.sum()


def _normalize_gate_matrix_np(gates: np.ndarray) -> np.ndarray:
    gates = np.asarray(gates, dtype=np.float64)
    normalized = np.zeros_like(gates, dtype=np.float64)
    for idx in range(gates.shape[0]):
        normalized[idx] = _normalize_gate_np(gates[idx])
    return normalized


def _softmax_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.max()
    exp_values = np.exp(values)
    return exp_values / np.maximum(exp_values.sum(), 1e-12)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _fast_weak_metrics(
    *,
    counts: np.ndarray,
    selected_genes: np.ndarray,
    labels: np.ndarray,
    batches: np.ndarray,
    random_state: int,
) -> dict[str, float]:
    x_all = normalize_log1p(counts)
    x_sel = x_all[:, selected_genes]

    full_embedding = _pca_embedding(x_all, random_state=random_state)
    selected_embedding = _pca_embedding(x_sel, random_state=random_state)
    n_clusters = len(np.unique(labels))
    pred = KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state).fit_predict(selected_embedding)

    batch_sil = _silhouette_safe_local(selected_embedding, batches)
    return {
        "ari": adjusted_rand_score(labels, pred),
        "nmi": normalized_mutual_info_score(labels, pred),
        "label_silhouette": _silhouette_safe_local(selected_embedding, labels),
        "cluster_silhouette": _silhouette_safe_local(selected_embedding, pred),
        "batch_mixing": 1.0 - max(batch_sil, 0.0),
        "neighbor_preservation": neighbor_preservation(
            reference_embedding=full_embedding,
            test_embedding=selected_embedding,
            n_neighbors=min(15, max(5, counts.shape[0] // 30)),
        ),
        "stability": 0.0,
    }


def _pca_embedding(x: np.ndarray, random_state: int) -> np.ndarray:
    pca_dim = min(20, x.shape[1], x.shape[0] - 1)
    if pca_dim < 2:
        pca_dim = 2
    return PCA(n_components=pca_dim, random_state=random_state).fit_transform(x)


def _silhouette_safe_local(embedding: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or embedding.shape[0] <= len(unique):
        return 0.0
    try:
        return float(silhouette_score(embedding, labels))
    except ValueError:
        return 0.0
