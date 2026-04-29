from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None


MetricDirection = Literal["higher_is_better", "lower_is_better"]
MetricGroup = Literal["protected", "target", "other"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: MetricDirection = "higher_is_better"
    group: MetricGroup = "other"
    epsilon: float = 1e-9
    gain_threshold: float = 1e-9
    loss_tolerance: float = 1e-9


@dataclass(frozen=True)
class MetricRegistry:
    name: str
    metrics: tuple[MetricSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.metrics)

    @property
    def target_metrics(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.metrics if spec.group == "target")

    @property
    def protected_metrics(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.metrics if spec.group == "protected")

    def spec_for(self, metric_name: str) -> MetricSpec:
        for spec in self.metrics:
            if spec.name == metric_name:
                return spec
        raise KeyError(f"Metric '{metric_name}' is not registered in '{self.name}'.")

    def direction_for(self, metric_name: str) -> MetricDirection:
        return self.spec_for(metric_name).direction

    def group_for(self, metric_name: str) -> MetricGroup:
        return self.spec_for(metric_name).group

    def epsilon_for(self, metric_name: str) -> float:
        return float(self.spec_for(metric_name).epsilon)

    def gain_threshold_for(self, metric_name: str) -> float:
        return float(self.spec_for(metric_name).gain_threshold)

    def loss_tolerance_for(self, metric_name: str) -> float:
        return float(self.spec_for(metric_name).loss_tolerance)

    def orient_value(self, metric_name: str, value: Any) -> float:
        numeric = float(value)
        if self.direction_for(metric_name) == "lower_is_better":
            return -numeric
        return numeric

    def orient_delta(self, metric_name: str, delta: Any) -> float:
        numeric = float(delta)
        if self.direction_for(metric_name) == "lower_is_better":
            return -numeric
        return numeric

    def delta(self, metric_name: str, baseline_value: Any, candidate_value: Any) -> float:
        raw_delta = float(candidate_value) - float(baseline_value)
        return self.orient_delta(metric_name, raw_delta)

    def available_metrics(self, metric_names: Sequence[str]) -> tuple[str, ...]:
        return tuple(metric_name for metric_name in self.metric_names if metric_name in metric_names)


def build_metric_registry(
    name: str,
    *,
    protected_metrics: Sequence[str],
    target_metrics: Sequence[str],
    directions: Mapping[str, MetricDirection] | None = None,
    epsilons: Mapping[str, float] | None = None,
    gain_thresholds: Mapping[str, float] | None = None,
    loss_tolerances: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MetricRegistry:
    directions = directions or {}
    epsilons = epsilons or {}
    gain_thresholds = gain_thresholds or {}
    loss_tolerances = loss_tolerances or {}

    metric_specs: list[MetricSpec] = []
    for group_name, metric_names in (("protected", protected_metrics), ("target", target_metrics)):
        for metric_name in metric_names:
            metric_specs.append(
                MetricSpec(
                    name=metric_name,
                    direction=directions.get(metric_name, "higher_is_better"),
                    group=group_name,
                    epsilon=float(epsilons.get(metric_name, 1e-9)),
                    gain_threshold=float(gain_thresholds.get(metric_name, epsilons.get(metric_name, 1e-9))),
                    loss_tolerance=float(loss_tolerances.get(metric_name, epsilons.get(metric_name, 1e-9))),
                )
            )
    return MetricRegistry(name=name, metrics=tuple(metric_specs), metadata=metadata or {})


def get_runtime_device(*, prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


DISCRETE_REGISTRY = build_metric_registry(
    "discrete",
    protected_metrics=("var_ratio", "lisi"),
    target_metrics=("ari", "nmi"),
    epsilons={
        "var_ratio": 1e-9,
        "lisi": 1e-9,
        "ari": 1e-9,
        "nmi": 1e-9,
    },
    gain_thresholds={
        "ari": 0.0030,
        "nmi": 0.0030,
    },
    loss_tolerances={
        "var_ratio": 0.0100,
        "lisi": 0.0050,
        "ari": 0.0020,
        "nmi": 0.0020,
    },
)


CONTINUOUS_REGISTRY = build_metric_registry(
    "continuous",
    protected_metrics=("var_ratio", "dist_cor", "knn_ratio", "three_nn"),
    target_metrics=("max_ari", "max_nmi"),
    epsilons={
        "var_ratio": 1e-9,
        "dist_cor": 1e-9,
        "knn_ratio": 1e-9,
        "three_nn": 1e-9,
        "max_ari": 1e-9,
        "max_nmi": 1e-9,
    },
    gain_thresholds={
        "max_ari": 0.0050,
        "max_nmi": 0.0050,
    },
    loss_tolerances={
        "var_ratio": 0.0200,
        "dist_cor": 0.0015,
        "knn_ratio": 0.0030,
        "three_nn": 15.0,
        "max_ari": 0.0030,
        "max_nmi": 0.0030,
    },
)


DEFAULT_METRIC_REGISTRIES: dict[str, MetricRegistry] = {
    DISCRETE_REGISTRY.name: DISCRETE_REGISTRY,
    CONTINUOUS_REGISTRY.name: CONTINUOUS_REGISTRY,
}


def registry_for_evaluation_mode(evaluation_mode: str) -> MetricRegistry:
    normalized = evaluation_mode.strip().lower()
    if normalized not in DEFAULT_METRIC_REGISTRIES:
        raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")
    return DEFAULT_METRIC_REGISTRIES[normalized]

