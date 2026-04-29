from __future__ import annotations

from typing import Any, Mapping, Sequence

from .metrics import (
    CONTINUOUS_REGISTRY,
    DISCRETE_REGISTRY,
    MetricRegistry,
    get_runtime_device,
    registry_for_evaluation_mode,
)

ParetoProfile = MetricRegistry
DISCRETE_PROFILE = DISCRETE_REGISTRY
CONTINUOUS_PROFILE = CONTINUOUS_REGISTRY


def profile_for_evaluation_mode(evaluation_mode: str) -> ParetoProfile:
    return registry_for_evaluation_mode(evaluation_mode)


def _as_float_map(metric_values: Mapping[str, Any]) -> dict[str, float]:
    return {metric_name: float(metric_value) for metric_name, metric_value in metric_values.items()}


def compute_metric_deltas(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    profile: ParetoProfile,
    *,
    metric_names: Sequence[str] | None = None,
) -> dict[str, float]:
    resolved_metric_names = [
        metric_name
        for metric_name in (metric_names or profile.metric_names)
        if metric_name in baseline_metrics and metric_name in candidate_metrics
    ]
    if not resolved_metric_names:
        raise ValueError("No overlapping registered metrics were found between baseline and candidate values.")
    return {
        metric_name: profile.delta(metric_name, baseline_metrics[metric_name], candidate_metrics[metric_name])
        for metric_name in resolved_metric_names
    }


def classify_config(metric_deltas: Mapping[str, Any], profile: ParetoProfile) -> dict[str, Any]:
    deltas = _as_float_map(metric_deltas)
    all_metric_names = [metric_name for metric_name in profile.metric_names if metric_name in deltas]
    protected_metric_names = [metric_name for metric_name in profile.protected_metrics if metric_name in deltas]
    target_metric_names = [metric_name for metric_name in profile.target_metrics if metric_name in deltas]

    if not all_metric_names:
        raise ValueError("No usable metric deltas provided for classification.")

    win_metrics = [metric_name for metric_name in all_metric_names if deltas[metric_name] > profile.epsilon_for(metric_name)]
    loss_metrics = [metric_name for metric_name in all_metric_names if deltas[metric_name] < -profile.epsilon_for(metric_name)]
    protected_loss_metrics = [
        metric_name
        for metric_name in protected_metric_names
        if deltas[metric_name] < -profile.epsilon_for(metric_name)
    ]
    target_gain_metrics = [
        metric_name
        for metric_name in target_metric_names
        if deltas[metric_name] > profile.epsilon_for(metric_name)
    ]
    target_loss_metrics = [
        metric_name
        for metric_name in target_metric_names
        if deltas[metric_name] < -profile.epsilon_for(metric_name)
    ]

    clean_win = (
        bool(target_gain_metrics)
        and all(deltas[metric_name] >= -profile.epsilon_for(metric_name) for metric_name in protected_metric_names)
        and all(deltas[metric_name] >= -profile.epsilon_for(metric_name) for metric_name in target_metric_names)
    )

    near_clean = (
        bool(
            [
                metric_name
                for metric_name in target_metric_names
                if deltas[metric_name] >= profile.gain_threshold_for(metric_name)
            ]
        )
        and all(
            deltas[metric_name] >= -profile.loss_tolerance_for(metric_name)
            for metric_name in protected_metric_names
        )
        and all(
            deltas[metric_name] >= -profile.loss_tolerance_for(metric_name)
            for metric_name in target_metric_names
        )
        and not clean_win
    )

    mixed_tradeoff = bool(win_metrics) and bool(loss_metrics)
    dominated = not bool(win_metrics) and bool(loss_metrics)

    if clean_win:
        pareto_label = "clean_win"
    elif near_clean:
        pareto_label = "near_clean"
    elif mixed_tradeoff:
        pareto_label = "mixed_tradeoff"
    elif dominated:
        pareto_label = "dominated"
    else:
        pareto_label = "flat_or_tied"

    return {
        "clean_win": clean_win,
        "near_clean": near_clean,
        "mixed_tradeoff": mixed_tradeoff,
        "dominated": dominated,
        "pareto_label": pareto_label,
        "win_count": len(win_metrics),
        "loss_count": len(loss_metrics),
        "protected_loss_count": len(protected_loss_metrics),
        "target_gain_count": len(target_gain_metrics),
        "target_loss_count": len(target_loss_metrics),
        "win_metrics": ",".join(win_metrics),
        "loss_metrics": ",".join(loss_metrics),
        "protected_loss_metrics": ",".join(protected_loss_metrics),
        "target_gain_metrics": ",".join(target_gain_metrics),
        "target_loss_metrics": ",".join(target_loss_metrics),
    }


def classify_candidate_against_baseline(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    profile: ParetoProfile,
    *,
    metric_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    deltas = compute_metric_deltas(
        baseline_metrics,
        candidate_metrics,
        profile,
        metric_names=metric_names,
    )
    result = classify_config(deltas, profile)
    result["runtime_device"] = get_runtime_device()
    for metric_name, delta in deltas.items():
        result[f"delta_{metric_name}"] = delta
    return result


def classify_frame(
    frame: Any,
    *,
    evaluation_mode_col: str = "evaluation_mode",
    metric_name_col: str = "metric_name",
    delta_col: str = "delta",
    config_id_col: str = "config_id",
) -> Any:
    import pandas as pd

    if frame.empty:
        return pd.DataFrame()

    grouped_rows: list[dict[str, Any]] = []
    device = get_runtime_device()

    for config_id, group in frame.groupby(config_id_col, sort=False):
        evaluation_mode = str(group.iloc[0][evaluation_mode_col])
        profile = profile_for_evaluation_mode(evaluation_mode)
        metric_deltas = {
            str(metric_name): profile.orient_delta(str(metric_name), delta)
            for metric_name, delta in zip(group[metric_name_col], group[delta_col], strict=False)
            if str(metric_name) in profile.metric_names
        }
        result = classify_config(metric_deltas, profile)
        result["config_id"] = config_id
        result["evaluation_mode"] = evaluation_mode
        result["pareto_profile"] = profile.name
        result["runtime_device"] = device
        grouped_rows.append(result)

    summary = pd.DataFrame(grouped_rows)
    if summary.empty:
        return summary

    if device == "cuda":
        try:
            import torch
        except Exception:  # pragma: no cover - torch may be unavailable
            torch = None
        if torch is not None and torch.cuda.is_available():
            counts = torch.tensor(
                summary[
                    [
                        "win_count",
                        "loss_count",
                        "protected_loss_count",
                        "target_gain_count",
                        "target_loss_count",
                    ]
                ].to_numpy(),
                dtype=torch.float32,
                device="cuda",
            )
            summary["target_gain_minus_protected_loss"] = (
                counts[:, 3] - counts[:, 2]
            ).detach().cpu().numpy()
            return summary

    summary["target_gain_minus_protected_loss"] = summary["target_gain_count"] - summary["protected_loss_count"]
    return summary

